#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <iostream>
#include <fstream>
#include <optional>
#include <string>
#include <filesystem>

#include <argparse/argparse.hpp>

#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <string.h>

#define CHECK_CUDA_ERR(fcall)                                                        \
    if (cudaError_t err = (fcall); err != cudaError_t::cudaSuccess)                  \
    {                                                                                \
        std::cerr << "Line " << __LINE__ << " Got CUDA error " << err                \
                  << " " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) \
                  << std::endl;                                                      \
        std::exit(EXIT_FAILURE);                                                     \
    }

void processNPPResult(NppStatus status, int lineno) {
    if (status < 0) {
        std::cerr << "Line " << lineno << " Got NPP error " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (status > 0)
    {
        std::cerr << "Line " << lineno << " Got NPP warning " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_NPP_ERR(fcall) processNPPResult((fcall), __LINE__)

int cudaDeviceInit()
{
    int deviceCount;
    CHECK_CUDA_ERR(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }
    const int dev = 0;
    std::cout << "Selecting CUDA device " << dev << " by default" << std::endl;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    CHECK_CUDA_ERR(cudaSetDevice(dev));

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    std::cout << "CUDA Driver Version: " << (driverVersion / 1000) << '.' << ((driverVersion % 100) / 10) << std::endl;
    std::cout << "CUDA Runtime Version: " << (runtimeVersion / 1000) << ' ' << ((runtimeVersion % 100) / 10) << std::endl;

    return dev;
}

void initNPPLib(NppStreamContext &nppStreamCtx)
{

    nppStreamCtx.hStream = 0; // The NULL stream by default, set this to whatever your stream ID is if not the NULL stream.

    CHECK_CUDA_ERR(cudaGetDevice(&nppStreamCtx.nCudaDeviceId));

    const NppLibraryVersion *libVer = nppGetLibVersion();

    std::cout << "NPP Library Version " << libVer->major << '.'<< libVer->minor
        << '.' << libVer->build << std::endl;

    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
                                         cudaDevAttrComputeCapabilityMajor,
                                         nppStreamCtx.nCudaDeviceId));

    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
                                         cudaDevAttrComputeCapabilityMinor,
                                         nppStreamCtx.nCudaDeviceId));
    CHECK_CUDA_ERR(cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags));

    cudaDeviceProp oDeviceProperties;
    CHECK_CUDA_ERR(cudaGetDeviceProperties(&oDeviceProperties, nppStreamCtx.nCudaDeviceId));

    nppStreamCtx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor;
    nppStreamCtx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
    nppStreamCtx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;
}

void processImage(NppStreamContext &nppStreamCtx, const std::string &input, const std::string &output) {

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(input, oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};

    // create struct with ROI size
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

    int nBufferSize = 0;
    Npp8u *pScratchBufferNPP = 0;

    // get necessary scratch buffer size and allocate that much device memory
    CHECK_NPP_ERR(nppiFilterCannyBorderGetBufferSize(oSizeROI, &nBufferSize));

    cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

    // now run the canny edge detection filter
    // Using nppiNormL2 will produce larger magnitude values allowing for finer
    // control of threshold values while nppiNormL1 will be slightly faster.
    // Also, selecting the sobel gradient filter allows up to a 5x5 kernel size
    // which can produce more precise results but is a bit slower. Commonly
    // nppiNormL2 and sobel gradient filter size of 3x3 are used. Canny
    // recommends that the high threshold value should be about 3 times the low
    // threshold value. The threshold range will depend on the range of
    // magnitude values that the sobel gradient filter generates for a
    // particular image.

    Npp16s nLowThreshold = 72;
    Npp16s nHighThreshold = 256;

    if ((nBufferSize > 0) && (pScratchBufferNPP != 0))
    {
        CHECK_NPP_ERR(nppiFilterCannyBorder_8u_C1R_Ctx(
            oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
            oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, NPP_FILTER_SOBEL,
            NPP_MASK_SIZE_3_X_3, nLowThreshold, nHighThreshold, nppiNormL2,
            NPP_BORDER_REPLICATE, pScratchBufferNPP, nppStreamCtx));
    }

    // free scratch buffer memory
    cudaFree(pScratchBufferNPP);

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(output, oHostDst);
    std::cout << "Saved image: " << output << std::endl;

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
}

int main(int argc, char *argv[])
{
    using path_t = std::filesystem::path;

    printf("%s Starting...\n\n", argv[0]);

    try {
        std::string sFilename, output;
        char       *filePath;

        cudaDeviceInit();
        NppStreamContext nppStreamCtx;
        initNPPLib(nppStreamCtx);

        argparse::ArgumentParser program("edge_detect");
        program.add_argument("input")
            .help("input file or directory (automatically detected)");
        program.add_argument("-o")
            .default_value(".")
            .help("output file or directory (depending on input)");

        program.parse_args(argc, argv);
        path_t infile = path_t(program.get<std::string>("input"));
        path_t outdir = path_t(program.get<std::string>("-o"));
        path_t outfile = outdir / ("boxed_" + infile.filename().native());

        std::cout << "infile " << infile << "\n"
                            << "outfile " << outfile << std::endl;

        processImage(nppStreamCtx, infile, outfile);
    }
    catch (npp::Exception &rException) {
        std::cerr << "Program error! An NPP exception occurred: \n";
        std::cerr << rException << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    catch (...) {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}

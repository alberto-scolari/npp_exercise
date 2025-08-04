#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <string.h>

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

    cudaStream_t streams;
    CHECK_CUDA_ERR(cudaStreamCreate(&streams));
    nppStreamCtx.hStream = streams; // The NULL stream by default, set this to whatever your stream ID is if not the NULL stream.

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

void closeNPPLib(NppStreamContext &nppStreamCtx)
{
    if (nppStreamCtx.hStream != 0) {
        CHECK_CUDA_ERR(cudaStreamDestroy(nppStreamCtx.hStream));
        nppStreamCtx.hStream = 0;
    }
}

struct ImageProcessData
{
    NppStreamContext nppStreamCtx;
    npp::ImageCPU_8u_C1 oHostSrc;
    npp::ImageNPP_8u_C1 oDeviceSrc;
    npp::ImageNPP_8u_C1 oDeviceDst;
    NppiSize oSizeROI;
    int nBufferSize = 0;
    Npp8u *pScratchBufferNPP = nullptr;
    npp::ImageCPU_8u_C1 oHostDst;

    ~ImageProcessData() {
        if (pScratchBufferNPP != nullptr) {
            CHECK_CUDA_ERR(cudaFree(pScratchBufferNPP));
        }
    }
};

void DeviceToHostCopy2DAsync(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight, cudaStream_t stream)
{
    CHECK_CUDA_ERR(cudaMemcpy2DAsync(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp8u), nHeight, cudaMemcpyDeviceToHost, stream));
};

void DeviceToHostCopy2DAsync(Npp8u *pDst, size_t nDstPitch, npp::ImageNPP_8u_C1 &oDeviceSrc, cudaStream_t stream)
{
    DeviceToHostCopy2DAsync(pDst, nDstPitch, oDeviceSrc.data(), oDeviceSrc.pitch(), oDeviceSrc.width(), oDeviceSrc.height(), stream);
}

void initData(ImageProcessData &data, const std::string &input) {
    npp::loadImage(input, data.oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    data.oDeviceSrc = npp::ImageNPP_8u_C1(data.oHostSrc);

    // allocate device image of appropriately reduced size
    data.oDeviceDst = npp::ImageNPP_8u_C1((int)data.oDeviceSrc.width(), (int)data.oDeviceSrc.height());
    // declare a host image for the result
    data.oHostDst = npp::ImageCPU_8u_C1(data.oDeviceDst.size());

    std::cout << "image is " << data.oDeviceSrc.width() << 'x' << (int)data.oDeviceSrc.height()
              << " pitch is " << data.oDeviceSrc.pitch() << std::endl;

    // create struct with ROI size
    data.oSizeROI = {(int)data.oDeviceSrc.width(), (int)data.oDeviceSrc.height()};
}

void computeData(ImageProcessData &data) {
    // get necessary scratch buffer size and allocate that much device memory
    int old_size = data.nBufferSize, new_size;
    CHECK_NPP_ERR(nppiFilterCannyBorderGetBufferSize(data.oSizeROI, &new_size));

    if (old_size < new_size)
    {
        if (old_size > 0) {
            CHECK_CUDA_ERR(cudaFreeAsync(data.pScratchBufferNPP, data.nppStreamCtx.hStream));
        }
        data.nBufferSize = new_size;
        CHECK_CUDA_ERR(cudaMallocAsync((void **)&data.pScratchBufferNPP, new_size, data.nppStreamCtx.hStream));
    }

    NppiSize oSrcSize = {(int)data.oDeviceSrc.width(), (int)data.oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};

    const Npp16s nLowThreshold = 72;
    const Npp16s nHighThreshold = 256;

    if ((data.nBufferSize > 0) && (data.pScratchBufferNPP != nullptr))
    {
        CHECK_NPP_ERR(nppiFilterCannyBorder_8u_C1R_Ctx(
            data.oDeviceSrc.data(), data.oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
            data.oDeviceDst.data(), data.oDeviceDst.pitch(), data.oSizeROI, NPP_FILTER_SOBEL,
            NPP_MASK_SIZE_3_X_3, nLowThreshold, nHighThreshold, nppiNormL2,
            NPP_BORDER_REPLICATE, data.pScratchBufferNPP, data.nppStreamCtx));
    }

    // and copy the device result data into it
    DeviceToHostCopy2DAsync(data.oHostDst.data(), data.oHostDst.pitch(), data.oDeviceDst, data.nppStreamCtx.hStream);
}

void outputData(ImageProcessData &data, const std::string &output)
{
    CHECK_CUDA_ERR(cudaStreamSynchronize(data.nppStreamCtx.hStream));
    saveImage(output, data.oHostDst);
    std::cout << "Saved image: " << output << std::endl;
}


void processImage(NppStreamContext & nppStreamCtx, const std::string &input, const std::string &output)
{
    ImageProcessData data;
    data.nppStreamCtx = nppStreamCtx;

    initData(data, input);
    computeData(data);
    outputData(data, output);

}

int main(int argc, char *argv[])
{
    using path_t = std::filesystem::path;

    printf("%s Starting...\n\n", argv[0]);

        try {
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
        closeNPPLib(nppStreamCtx);
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

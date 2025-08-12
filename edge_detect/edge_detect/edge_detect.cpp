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
#include <functional>
#include <vector>
#include <algorithm>

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

// enable before compiling to log batched execution
constexpr bool DO_LOG = false;

#define LOG(args) if (DO_LOG) { std::cout << args << std::endl; }

#define LOG_ERR(cond, args)             \
    if ((cond))                         \
    {                                   \
        std::cerr << args << std::endl; \
        std::exit(EXIT_FAILURE);        \
    }

void processNPPResult(NppStatus, int);

#define CHECK_NPP_ERR(fcall) processNPPResult((fcall), __LINE__)

void processNPPResult(NppStatus status, int lineno)
{
    LOG_ERR(status < 0, "Line " << lineno << " Got NPP error " << status);
    LOG_ERR(status > 0, "Line " << lineno << " Got NPP warning " << status);
}

using path_t = std::filesystem::path;

int cudaDeviceInit(cudaDeviceProp &deviceProp)
{
    int deviceCount;
    CHECK_CUDA_ERR(cudaGetDeviceCount(&deviceCount));
    LOG_ERR(deviceCount == 0, "CUDA error: no devices supporting CUDA.");
    const int dev = 0;
    std::cout << "Selecting CUDA device " << dev << " by default" << std::endl;
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
    CHECK_CUDA_ERR(cudaGetDevice(&nppStreamCtx.nCudaDeviceId));
    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
                                         cudaDevAttrComputeCapabilityMajor,
                                         nppStreamCtx.nCudaDeviceId));
    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
                                         cudaDevAttrComputeCapabilityMinor,
                                         nppStreamCtx.nCudaDeviceId));
    cudaDeviceProp oDeviceProperties;
    CHECK_CUDA_ERR(cudaGetDeviceProperties(&oDeviceProperties, nppStreamCtx.nCudaDeviceId));
    nppStreamCtx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor;
    nppStreamCtx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
    nppStreamCtx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;
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

    ImageProcessData() {
        initNPPLib(this->nppStreamCtx);
        CHECK_CUDA_ERR(cudaStreamCreate(&nppStreamCtx.hStream));
        CHECK_CUDA_ERR(cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags));
    }

    ImageProcessData(const ImageProcessData &) = delete;

    ~ImageProcessData()
    {
        if (pScratchBufferNPP != nullptr) {
            CHECK_CUDA_ERR(cudaFree(pScratchBufferNPP));
        }
        if (nppStreamCtx.hStream != 0)
        {
            CHECK_CUDA_ERR(cudaStreamDestroy(nppStreamCtx.hStream));
            nppStreamCtx.hStream = 0;
        }
    }
};

void DeviceToHostCopy2DAsync(Npp8u *pDst, size_t nDstPitch, const Npp8u *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight, cudaStream_t stream)
{
    CHECK_CUDA_ERR(cudaMemcpy2DAsync(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp8u), nHeight, cudaMemcpyDeviceToHost, stream));
}

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
    LOG("Saved image: " << output);
}

void processImage(ImageProcessData &data, const std::string &input, const std::string &output)
{
    initData(data, input);
    computeData(data);
    outputData(data, output);
}

void processImageDoubleBuffered(const std::vector<path_t> &inputs,
    std::vector<ImageProcessData> &data, std::function<path_t(const path_t &)> in2out)
{
    const int num_files = static_cast<int>(inputs.size());
    const int batch_size = static_cast<int>(data.size() / 2);
    const int num_batches = (num_files + batch_size - 1) / batch_size; // round up
    ImageProcessData *dataA = data.data(), *dataB = dataA + batch_size;

    // prologue
    int input_i = 0;
    for (; input_i < std::min(batch_size, num_files); input_i++)
    {
        LOG("prologue " << input_i);
        initData(dataA[input_i], inputs[input_i].native());
        computeData(dataA[input_i]);
    }
    // main loop for double buffering
    // the first batch has already started in the prologue
    int batchB_size, batchA_size = input_i;
    for (int batch = 1; batch < num_batches; batch++)
    {
        batchB_size = std::min(batch_size, num_files - input_i);
        // init and run current batch
        for (int j = 0; j < batchB_size; j++)
        {
            LOG("  batched init " << (input_i + j));
            initData(dataB[j], inputs[input_i + j].native());
            computeData(dataB[j]);
        }
        // wait and process output for previous batch
        for (int j = 0; j < batchA_size; j++)
        {
            LOG("  batched out " << (input_i - batchA_size + j) << ": " << in2out(inputs[input_i - batchA_size + j]));
            outputData(dataA[j], in2out(inputs[input_i - batchA_size + j]).native());
        }
        input_i += batchB_size;
        batchA_size = batchB_size;
        std::swap(dataA, dataB);
        LOG("-------------------------");
    }
    // epilogue
    for (int j = 0; j < batchA_size; j++)
    {
        LOG("epilogue " << (input_i - batchA_size + j) << ": " << in2out(inputs[input_i - batchA_size + j]));
        outputData(dataA[j], in2out(inputs[input_i - batchA_size + j]).native());
    }
}

int main(int argc, char *argv[])
{
    using dirent_t = std::filesystem::directory_entry;
    using diriter_t = std::filesystem::directory_iterator;

    std::cout << argv[0] << " Starting...\n\n";

    try
    {
        cudaDeviceProp deviceProp;
        cudaDeviceInit(deviceProp);

        argparse::ArgumentParser program("edge_detect", "1.0", argparse::default_arguments::help);
        program.add_argument("-o")
            .default_value(".")
            .help("output file or directory (depending on input)");
        program.add_argument("--batch")
            .scan<'u', unsigned>()
            .help("batch size (none means decide from hardware)");
        program.add_argument("--dir")
            .help("input is a directory")
            .default_value(false)
            .implicit_value(true);
        program.add_argument("input")
            .help("input file or directory (see '--dir' option)");
        program.parse_args(argc, argv);

        path_t infile = path_t(program.get<std::string>("input"));
        path_t outdir = path_t(program.get<std::string>("-o"));
        std::function<path_t(const path_t &)> in2out = [=](const path_t &inf) {
            return outdir / ("boxed_" + inf.filename().native());
        };
        std::vector<path_t> paths;
        if (program["--dir"] == false) {
            paths.push_back(infile);
        } else {
            diriter_t indir(infile);
            std::copy_if(std::filesystem::begin(indir), std::filesystem::end(indir),
                std::back_inserter(paths), [](const dirent_t &e) { return e.is_regular_file(); });
        }

        const unsigned concurrentKernels = std::max(
            static_cast<unsigned>(deviceProp.concurrentKernels), // could be 0
            1U);
        const unsigned concurrency = program.present<unsigned>("--batch").value_or(concurrentKernels);
        std::cout << "Concurrency: " << concurrency << std::endl;

        std::vector<ImageProcessData> data(concurrency * 2);
        std::cout << "Starting processing " << paths.size() << " images..." << std::endl;
        processImageDoubleBuffered(paths, data, in2out);
        std::cout << "Processing completed." << std::endl;
    }
    catch (npp::Exception &rException) {
        LOG_ERR(true, "Program error! An NPP exception occurred: \n" << rException);
    }
    catch (const std::exception &err) {
        LOG_ERR(true, "Program error! An exception occurred: \n" << err.what());
    }
    catch (...) {
        LOG_ERR(true, "Program error! An unknow type of exception occurred.\nAborting.");
    }

    return 0;
}

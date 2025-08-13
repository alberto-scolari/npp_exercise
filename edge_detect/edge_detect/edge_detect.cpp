/// @file edge_detect.cpp
/// Full implementation of Canny Border Filter with double buffering via CUDA streams.


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

// clang-format off
#include <cstring>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
// clang-format on
#include <cuda_runtime.h>
#include <npp.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

/* UTILITIES FOR ERROR HANDLING AND LOGGING */

/// @brief check for a CUDA error as a result from a function call; if error,
///   warn and terminate
#define CHECK_CUDA_ERR(fcall)                                             \
  if (cudaError_t err = (fcall); err != cudaError_t::cudaSuccess) {       \
    std::cerr << "Line " << __LINE__ << " Got CUDA error " << err << " "  \
              << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) \
              << std::endl;                                               \
    std::exit(EXIT_FAILURE);                                              \
  }

/// @brief enable before compiling to log batched execution
constexpr bool DO_LOG = false;

/// @brief print args as stream inputs
#define LOG(args)                   \
  if (DO_LOG) {                     \
    std::cout << args << std::endl; \
  }

/// @brief if condition @p cond is met, print args as stream inputs into stderr
///   and exit abnormally
#define LOG_ERR(cond, args)         \
  if ((cond)) {                     \
    std::cerr << args << std::endl; \
    std::exit(EXIT_FAILURE);        \
  }

/// @brief check the result of a call to a routing of the NPP lib: in case of
///   error, report it and the given line number
// forward declaration for processNPPResult().
static void processNPPResult(NppStatus, int);

/// @brief check the result of a call to a routing of the NPP lib
#define CHECK_NPP_ERR(fcall) processNPPResult((fcall), __LINE__)

static void processNPPResult(NppStatus status, int lineno) {
  LOG_ERR(status < 0, "Line " << lineno << " Got NPP error " << status);
  LOG_ERR(status > 0, "Line " << lineno << " Got NPP warning " << status);
}

/// @brief shorter declaration for filesystem path objects
using path_t = std::filesystem::path;

/// @brief split a CUDA-encoded version number into major and minor
/// @param ver CUDA-encoded version number
constexpr std::tuple<int, int> getMajorMinor(int ver) {
  constexpr int MAJOR_DIV = 1000;
  constexpr int MINOR_DIV = 10;
  constexpr int MINOR_MOD = 100;
  return {ver / MAJOR_DIV, (ver % MINOR_MOD) / MINOR_DIV};
}

/* ROUTINES FOR CUDA AND NPP INITIALIZATION */

/// @brief initialize a CUDA device by selecting it
/// @param deviceProp reference of `cudaDeviceProp` object to store device
/// information into
/// @return 0-based index of initialized device
static int cudaDeviceInit(cudaDeviceProp &deviceProp) {
  int deviceCount = -1;
  CHECK_CUDA_ERR(cudaGetDeviceCount(&deviceCount));
  LOG_ERR(deviceCount == 0, "CUDA error: no devices supporting CUDA.");
  const int dev = 0;
  std::cout << "Selecting CUDA device " << dev << " by default" << std::endl;
  cudaGetDeviceProperties(&deviceProp, dev);
  CHECK_CUDA_ERR(cudaSetDevice(dev));

  int driverVersion = -1;
  cudaDriverGetVersion(&driverVersion);
  auto [dmajor, dminor] = getMajorMinor(driverVersion);
  std::cout << "CUDA Driver Version: " << dmajor << '.' << dminor << std::endl;
  int runtimeVersion = -1;
  cudaRuntimeGetVersion(&runtimeVersion);
  auto [rmajor, rminor] = getMajorMinor(runtimeVersion);
  std::cout << "CUDA Runtime Version: " << rmajor << ' ' << rminor << std::endl;
  return dev;
}

/// @brief initialize the NPP library
/// @param nppStreamCtx reference to `NppStreamContext` object to store NPP
/// information into
static void initNPPLib(NppStreamContext &nppStreamCtx) {
  CHECK_CUDA_ERR(cudaGetDevice(&nppStreamCtx.nCudaDeviceId));
  CHECK_CUDA_ERR(cudaDeviceGetAttribute(
      &nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
      cudaDevAttrComputeCapabilityMajor, nppStreamCtx.nCudaDeviceId));
  CHECK_CUDA_ERR(cudaDeviceGetAttribute(
      &nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
      cudaDevAttrComputeCapabilityMinor, nppStreamCtx.nCudaDeviceId));
  cudaDeviceProp oDeviceProperties;
  CHECK_CUDA_ERR(
      cudaGetDeviceProperties(&oDeviceProperties, nppStreamCtx.nCudaDeviceId));
  nppStreamCtx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
  nppStreamCtx.nMaxThreadsPerMultiProcessor =
      oDeviceProperties.maxThreadsPerMultiProcessor;
  nppStreamCtx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
  nppStreamCtx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;
}

/* DATA STRUCTURES AND ROUTINES FOR DOUBLE BUFFERING */

/// @brief data structure to track all objects and information needed to
///   process an image across all phases
struct ImageProcessData {
  NppStreamContext nppStreamCtx;
  npp::ImageCPU_8u_C1 oHostSrc;
  npp::ImageNPP_8u_C1 oDeviceSrc;
  npp::ImageNPP_8u_C1 oDeviceDst;
  NppiSize oSizeROI;
  int nBufferSize = 0;
  Npp8u *pScratchBufferNPP = nullptr;
  npp::ImageCPU_8u_C1 oHostDst;

  /// @brief constructs and object by creating a new CUDA stream
  ImageProcessData() {
    initNPPLib(this->nppStreamCtx);
    CHECK_CUDA_ERR(cudaStreamCreate(&nppStreamCtx.hStream));
    CHECK_CUDA_ERR(
        cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags));
  }

  ImageProcessData(const ImageProcessData &) = delete;

  ImageProcessData &operator=(const ImageProcessData &) = delete;

  /// @brief ensures the internal buffer for CUDA input data is big enough to
  /// accomodate @p new_size bytes
  void ensureScratchBufferNPPSize(int new_size) {
    if (nBufferSize >= new_size) {
      return;
    }
    if (nBufferSize > 0) {
      CHECK_CUDA_ERR(cudaFreeAsync(pScratchBufferNPP, nppStreamCtx.hStream));
    }
    nBufferSize = new_size;
    CHECK_CUDA_ERR(cudaMallocAsync((void **)&pScratchBufferNPP, new_size,
                                   nppStreamCtx.hStream));
  }

  /// @brief destructs an object by releasing the CUDA stream
  ~ImageProcessData() {
    if (pScratchBufferNPP != nullptr) {
      CHECK_CUDA_ERR(cudaFree(pScratchBufferNPP));
    }
    if (nppStreamCtx.hStream != 0) {
      CHECK_CUDA_ERR(cudaStreamDestroy(nppStreamCtx.hStream));
      nppStreamCtx.hStream = 0;
    }
  }
};

/// @brief asynchronous CUDA 2D memcpy between source and destination raw
/// addresses
static void DeviceToHostCopy2DAsync(Npp8u *pDst, size_t nDstPitch,
                                    const Npp8u *pSrc, size_t nSrcPitch,
                                    size_t nWidth, size_t nHeight,
                                    cudaStream_t stream) {
  CHECK_CUDA_ERR(cudaMemcpy2DAsync(pDst, nDstPitch, pSrc, nSrcPitch,
                                   nWidth * sizeof(Npp8u), nHeight,
                                   cudaMemcpyDeviceToHost, stream));
}

/// @brief asynchronous CUDA 2D memcpy between `npp::ImageNPP_8u_C1` source and
///   destination as raw addresse
static void DeviceToHostCopy2DAsync(Npp8u *pDst, size_t nDstPitch,
                                    const npp::ImageNPP_8u_C1 &oDeviceSrc,
                                    cudaStream_t stream) {
  DeviceToHostCopy2DAsync(pDst, nDstPitch, oDeviceSrc.data(),
                          oDeviceSrc.pitch(), oDeviceSrc.width(),
                          oDeviceSrc.height(), stream);
}

/// @brief load image from file into buffer data structure
/// @param data the destination buffer
/// @param input the path to the image
static void initData(ImageProcessData &data, const std::string &input) {
  npp::loadImage(input, data.oHostSrc);
  // declare a device image and copy construct from the host image,
  // i.e. upload host to device
  data.oDeviceSrc = npp::ImageNPP_8u_C1(data.oHostSrc);
  // allocate device image of appropriately reduced size
  data.oDeviceDst = npp::ImageNPP_8u_C1((int)data.oDeviceSrc.width(),
                                        (int)data.oDeviceSrc.height());
  // declare a host image for the result
  data.oHostDst = npp::ImageCPU_8u_C1(data.oDeviceDst.size());
  // create struct with ROI size
  data.oSizeROI = {(int)data.oDeviceSrc.width(), (int)data.oDeviceSrc.height()};
}

/// @brief run the computation asynchronously on a single image to the CUDA
///   device: send data, compute and load results into main memory buffer (all
///   async)
/// @param data the buffer with input and output data
static void computeData(ImageProcessData &data) {
  // get necessary scratch buffer size and allocate device memory if needed
  int new_size;
  CHECK_NPP_ERR(nppiFilterCannyBorderGetBufferSize(data.oSizeROI, &new_size));
  data.ensureScratchBufferNPPSize(new_size);
  NppiSize oSrcSize = {(int)data.oDeviceSrc.width(),
                       (int)data.oDeviceSrc.height()};
  NppiPoint oSrcOffset = {0, 0};
  const Npp16s nLowThreshold = 72;
  const Npp16s nHighThreshold = 256;

  if ((data.nBufferSize > 0) && (data.pScratchBufferNPP != nullptr)) {
    CHECK_NPP_ERR(nppiFilterCannyBorder_8u_C1R_Ctx(
        data.oDeviceSrc.data(), data.oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
        data.oDeviceDst.data(), data.oDeviceDst.pitch(), data.oSizeROI,
        NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, nLowThreshold, nHighThreshold,
        nppiNormL2, NPP_BORDER_REPLICATE, data.pScratchBufferNPP,
        data.nppStreamCtx));
  }
  // and copy the device result data into it
  DeviceToHostCopy2DAsync(data.oHostDst.data(), data.oHostDst.pitch(),
                          data.oDeviceDst, data.nppStreamCtx.hStream);
}

/// @brief store the image into a file
/// @param data buffer with output data
/// @param output path to output file
static void outputData(ImageProcessData &data, const std::string &output) {
  // first make sure data has arrived into main memory; if not, wait until it's
  // come
  CHECK_CUDA_ERR(cudaStreamSynchronize(data.nppStreamCtx.hStream));
  // then store into file
  saveImage(output, data.oHostDst);
  LOG("Saved image: " << output);
}

/// @brief **double-buffered implementation** of the load-process-store
/// pipeline:
///   handles batching, issuing of operations and buffer swapping
/// @param inputs vector if input images
/// @param data vector of buffers (for both LOAD-PROCESS and STORE)
/// @param in2out functor to compute the output path from the input path for
///   each image
static void processImageDoubleBuffered(
    const std::vector<path_t> &inputs, std::vector<ImageProcessData> &data,
    std::function<path_t(const path_t &)> in2out) {
  const int num_files = static_cast<int>(inputs.size());
  const int batch_size = static_cast<int>(data.size() / 2);
  const int num_batches =
      (num_files + batch_size - 1) / batch_size;  // round up
  ImageProcessData *dataA = data.data(), *dataB = dataA + batch_size;

  // prologue
  int input_i = 0;
  for (; input_i < std::min(batch_size, num_files); input_i++) {
    LOG("prologue " << input_i);
    initData(dataA[input_i], inputs[input_i].native());
    computeData(dataA[input_i]);
  }
  // main loop for double buffering
  // the first batch has already started in the prologue
  int batchB_size, batchA_size = input_i;
  for (int batch = 1; batch < num_batches; batch++) {
    batchB_size = std::min(batch_size, num_files - input_i);
    // init and run current batch
    for (int j = 0; j < batchB_size; j++) {
      LOG("  batched init " << (input_i + j));
      initData(dataB[j], inputs[input_i + j].native());
      computeData(dataB[j]);
    }
    // wait and process output for previous batch
    for (int j = 0; j < batchA_size; j++) {
      LOG("  batched out " << (input_i - batchA_size + j) << ": "
                           << in2out(inputs[input_i - batchA_size + j]));
      outputData(dataA[j], in2out(inputs[input_i - batchA_size + j]).native());
    }
    input_i += batchB_size;
    batchA_size = batchB_size;
    std::swap(dataA, dataB);
    LOG("-------------------------");
  }
  // epilogue
  for (int j = 0; j < batchA_size; j++) {
    LOG("epilogue " << (input_i - batchA_size + j) << ": "
                    << in2out(inputs[input_i - batchA_size + j]));
    outputData(dataA[j], in2out(inputs[input_i - batchA_size + j]).native());
  }
}

/* ENTRY POINT: PARSE CMDLINE, SETUP AND RUN */

/// @brief main function for the executable: get cmdline args, set things up and
///   run the computation
int main(int argc, char *argv[]) {
  using dirent_t = std::filesystem::directory_entry;
  using diriter_t = std::filesystem::directory_iterator;

  try {
    // first parse arguments from command line
    argparse::ArgumentParser program("edge_detect", "1.0",
                                     argparse::default_arguments::help);
    program.add_argument("-o").default_value(".").help(
        "output file or directory (depending on input)");
    program.add_argument("--batch").scan<'u', unsigned>().help(
        "batch size (none means decide from hardware)");
    program.add_argument("--dir")
        .help("input is a directory")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("input").help(
        "input file or directory (see '--dir' option)");
    program.parse_args(argc, argv);

    path_t infile = path_t(program.get<std::string>("input"));
    path_t outdir = path_t(program.get<std::string>("-o"));
    // create functor to give output paths
    std::function<path_t(const path_t &)> in2out = [=](const path_t &inf) {
      return outdir / ("boxed_" + inf.filename().native());
    };
    // store all input images into vector
    std::vector<path_t> paths;
    if (program["--dir"] == false) {
      // cmdline input is file
      paths.push_back(infile);
    } else {
      // cmdline input if folder: navigate all stored file
      diriter_t indir(infile);
      std::copy_if(std::filesystem::begin(indir), std::filesystem::end(indir),
                   std::back_inserter(paths),
                   [](const dirent_t &e) { return e.is_regular_file(); });
    }

    // initialize CUDA device and NPP lib
    cudaDeviceProp deviceProp;
    cudaDeviceInit(deviceProp);

    // initialize concurrency value from hardware or user's input
    const unsigned concurrentKernels = std::max(
        static_cast<unsigned>(deviceProp.concurrentKernels),  // could be 0
        1U);  // must be at least 1
    const unsigned concurrency =
        program.present<unsigned>("--batch").value_or(concurrentKernels);
    std::cout << "Concurrency: " << concurrency << std::endl;

    // initialize memory buffers for double buffering (hence * 2 below)
    std::vector<ImageProcessData> data(concurrency * 2);
    // start pushing tasks to device
    std::cout << "Starting processing " << paths.size() << " images..."
              << std::endl;
    processImageDoubleBuffered(paths, data, in2out);
    std::cout << "Processing completed." << std::endl;

  } catch (npp::Exception &rException) {
    LOG_ERR(true, "Program error! An NPP exception occurred: \n" << rException);
  } catch (const std::exception &err) {
    LOG_ERR(true, "Program error! An exception occurred: \n" << err.what());
  } catch (...) {
    LOG_ERR(true,
            "Program error! An unknow type of exception occurred.\nAborting.");
  }
  return 0;
}

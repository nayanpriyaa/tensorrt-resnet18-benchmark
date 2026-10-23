// TrtPipeline.cpp
#include "TrtPipeline.h"

#include <NvOnnxParser.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

// BuilderFlag::kFP16 is [[deprecated]] since TensorRT 10.12 and removed in 11.0. It is still the only
// FP16 switch on every TensorRT 10.x release, so it is used deliberately behind #if and with the warning muted.
#if defined(__GNUC__) || defined(__clang__)
#define TRTP_SUPPRESS_DEPRECATED_BEGIN                                                                    \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define TRTP_SUPPRESS_DEPRECATED_END _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER)
#define TRTP_SUPPRESS_DEPRECATED_BEGIN __pragma(warning(push)) __pragma(warning(disable : 4996))
#define TRTP_SUPPRESS_DEPRECATED_END __pragma(warning(pop))
#else
#define TRTP_SUPPRESS_DEPRECATED_BEGIN
#define TRTP_SUPPRESS_DEPRECATED_END
#endif

namespace fs = std::filesystem;

namespace trtp
{

// ======================================================================================================
// detail: error handling + RAII deleters
// ======================================================================================================
namespace detail
{

void throwCudaError(cudaError_t err, const char* expr, const char* file, int line)
{
    std::ostringstream os;
    os << "CUDA error " << cudaGetErrorName(err) << " (" << static_cast<int>(err)
       << "): " << cudaGetErrorString(err) << "\n  at " << file << ':' << line << "\n  in " << expr;
    throw std::runtime_error(os.str());
}

void logCudaError(cudaError_t err, const char* expr, const char* file, int line) noexcept
{
    try
    {
        std::cerr << "[CUDA][E] " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << " at "
                  << file << ':' << line << " in " << expr << '\n';
    }
    catch (...)
    {
        // Logging must never escape a destructor.
    }
}

void DeviceFree::operator()(void* p) const noexcept
{
    CUDA_CHECK_NOTHROW(cudaFree(p));
}

void PinnedFree::operator()(void* p) const noexcept
{
    CUDA_CHECK_NOTHROW(cudaFreeHost(p));
}

void StreamDestroy::operator()(cudaStream_t s) const noexcept
{
    CUDA_CHECK_NOTHROW(cudaStreamDestroy(s));
}

void EventDestroy::operator()(cudaEvent_t e) const noexcept
{
    CUDA_CHECK_NOTHROW(cudaEventDestroy(e));
}

} // namespace detail

// ======================================================================================================
// Logger
// ======================================================================================================
const char* toString(Precision p) noexcept
{
    switch (p)
    {
    case Precision::kFP32: return "FP32";
    case Precision::kTF32: return "TF32";
    case Precision::kFP16: return "FP16";
    }
    return "?";
}

const char* toString(IoMode m) noexcept
{
    switch (m)
    {
    case IoMode::kPinned: return "pinned";
    case IoMode::kPageable: return "pageable";
    case IoMode::kZeroCopy: return "zerocopy";
    }
    return "?";
}

void Logger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept
{
    if (severity > mMinSeverity)
    {
        return;
    }
    const char* tag = "V";
    switch (severity)
    {
    case Severity::kINTERNAL_ERROR: tag = "F"; break;
    case Severity::kERROR: tag = "E"; break;
    case Severity::kWARNING: tag = "W"; break;
    case Severity::kINFO: tag = "I"; break;
    case Severity::kVERBOSE: tag = "V"; break;
    }
    try
    {
        (severity <= Severity::kWARNING ? std::cerr : std::cout) << "[TRT][" << tag << "] " << msg << '\n';
    }
    catch (...)
    {
    }
}

// ======================================================================================================
// local helpers
// ======================================================================================================
namespace
{

using Clock = std::chrono::steady_clock;

std::string readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
    {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamsize size = f.tellg();
    std::string buf(static_cast<std::size_t>(size), '\0');
    f.seekg(0);
    if (!f.read(buf.data(), size))
    {
        throw std::runtime_error("short read on " + path);
    }
    return buf;
}

void writeFile(const std::string& path, const std::string& data)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.write(data.data(), static_cast<std::streamsize>(data.size())))
    {
        throw std::runtime_error("cannot write " + path);
    }
}

// Engine cache is stale if the ONNX it came from was re-exported afterwards.
bool cachedEngineUsable(const std::string& enginePath, const std::string& onnxPath)
{
    std::error_code ec;
    if (enginePath.empty() || !fs::exists(enginePath, ec))
    {
        return false;
    }
    if (!fs::exists(onnxPath, ec))
    {
        return true; // deploy-only target: engine shipped without the ONNX
    }
    return fs::last_write_time(enginePath, ec) >= fs::last_write_time(onnxPath, ec);
}

std::size_t volume(const nvinfer1::Dims& d, const std::string& name)
{
    std::size_t v = 1;
    for (int32_t i = 0; i < d.nbDims; ++i)
    {
        if (d.d[i] < 0)
        {
            throw std::runtime_error("tensor '" + name + "' has an unresolved dynamic dimension");
        }
        v *= static_cast<std::size_t>(d.d[i]);
    }
    return v;
}

std::string dimsToString(const nvinfer1::Dims& d)
{
    std::ostringstream os;
    os << '[';
    for (int32_t i = 0; i < d.nbDims; ++i)
    {
        os << (i ? "," : "") << d.d[i];
    }
    os << ']';
    return os.str();
}

const char* dataTypeName(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT: return "FP32";
    case nvinfer1::DataType::kHALF: return "FP16";
    case nvinfer1::DataType::kBF16: return "BF16";
    case nvinfer1::DataType::kINT8: return "INT8";
    case nvinfer1::DataType::kINT32: return "INT32";
    case nvinfer1::DataType::kINT64: return "INT64";
    case nvinfer1::DataType::kBOOL: return "BOOL";
    case nvinfer1::DataType::kUINT8: return "UINT8";
    default: return "other";
    }
}

#if NV_TENSORRT_MAJOR >= 11
// Strongly typed networks take precision from the graph. Used to refuse benchmarking an FP32 graph as "FP16".
bool networkHasHalfTensors(const nvinfer1::INetworkDefinition& network)
{
    for (int32_t i = 0; i < network.getNbLayers(); ++i)
    {
        const nvinfer1::ILayer* layer = network.getLayer(i);
        for (int32_t o = 0; o < layer->getNbOutputs(); ++o)
        {
            if (layer->getOutput(o)->getType() == nvinfer1::DataType::kHALF)
            {
                return true;
            }
        }
    }
    return false;
}
#endif

} // namespace

// ======================================================================================================
// TrtPipeline
// ======================================================================================================
TrtPipeline::TrtPipeline(PipelineConfig config)
    : mConfig(std::move(config))
    , mLogger(mConfig.logSeverity)
{
    // Linked library vs. compiled headers: a major-version mismatch is an ABI break (e.g. headers from
    // TRT 11 at compile time, but an older nvinfer_10.dll found first on PATH at run time).
    const int32_t libVer = getInferLibVersion();
    if (libVer / 10000 != NV_TENSORRT_MAJOR)
    {
        throw std::runtime_error("TensorRT header/library major version mismatch: headers "
            + std::to_string(NV_TENSORRT_MAJOR) + ".x, libnvinfer " + std::to_string(libVer / 10000) + ".x");
    }

    CUDA_CHECK(cudaSetDevice(mConfig.deviceId));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, mConfig.deviceId));
    mDeviceName = prop.name;
    mComputeCapability = prop.major * 10 + prop.minor;
    mIntegratedGpu = prop.integrated != 0;

    if (mConfig.ioMode == IoMode::kZeroCopy && prop.canMapHostMemory == 0)
    {
        throw std::runtime_error("IoMode::kZeroCopy requested but this device cannot map host memory");
    }

    // ---- engine: load cached plan, else build from ONNX -------------------------------------------------
    const auto t0 = Clock::now();
    std::string plan;
    if (!mConfig.forceRebuild && cachedEngineUsable(mConfig.enginePath, mConfig.onnxPath))
    {
        try
        {
            plan = readFile(mConfig.enginePath);
            deserializeEngine(plan);
        }
        catch (const std::exception& e)
        {
            // Typical cause: plan built for another SM architecture or another TensorRT version.
            std::cerr << "[TrtPipeline][W] cached engine rejected (" << e.what() << "), rebuilding\n";
            mEngine.reset();
        }
    }
    if (!mEngine)
    {
        plan = buildSerializedEngine();
        mBuiltThisRun = true;
        if (!mConfig.enginePath.empty())
        {
            writeFile(mConfig.enginePath, plan);
        }
        deserializeEngine(plan);
    }
    mPlanBytes = plan.size();
    mSetupSeconds = std::chrono::duration<double>(Clock::now() - t0).count();

    // The execution context owns the activation memory (getDeviceMemorySizeV2() bytes of device memory).
    // One context == one in-flight inference; concurrent streams need one context each.
    mContext.reset(mEngine->createExecutionContext());
    if (!mContext)
    {
        throw std::runtime_error("createExecutionContext failed (out of device memory?)");
    }

    // ---- stream + timing events ------------------------------------------------------------------------
    // Non-blocking stream: no implicit synchronisation with the legacy default stream (stream 0), so
    // unrelated default-stream work in the process (OpenCV CUDA, other libs) cannot serialise with us.
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    mStream.reset(stream);

    for (EventPtr* ev : {&mEvStart, &mEvAfterH2D, &mEvAfterCompute, &mEvAfterD2H})
    {
        cudaEvent_t e = nullptr;
        CUDA_CHECK(cudaEventCreate(&e));
        ev->reset(e);
    }

    bindIoTensors();
}

TrtPipeline::~TrtPipeline()
{
    // Drain in-flight async work before member destructors free the buffers the GPU may still be using.
    if (mStream)
    {
        CUDA_CHECK_NOTHROW(cudaStreamSynchronize(mStream.get()));
    }
}

std::string TrtPipeline::buildSerializedEngine()
{
    std::cout << "[TrtPipeline] building " << toString(mConfig.precision) << " engine from " << mConfig.onnxPath
              << " (one-off: TensorRT times candidate kernels per layer, expect tens of seconds)\n";

    std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(mLogger)};
    if (!builder)
    {
        throw std::runtime_error("createInferBuilder failed");
    }

    // Flags = 0. Explicit batch is the only mode since TRT 10 (kEXPLICIT_BATCH is deprecated and ignored),
    // and on TRT 11 every network is strongly typed, so no kSTRONGLY_TYPED flag either.
    std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0U)};
    if (!network)
    {
        throw std::runtime_error("createNetworkV2 failed");
    }

    std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, mLogger)};
    if (!parser)
    {
        throw std::runtime_error("nvonnxparser::createParser failed");
    }
    if (!parser->parseFromFile(mConfig.onnxPath.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::ostringstream os;
        os << "failed to parse ONNX '" << mConfig.onnxPath << "'";
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            os << "\n  " << parser->getError(i)->desc();
        }
        throw std::runtime_error(os.str());
    }

    // Static shapes only: TensorRT then picks the single best tactic for exactly 1x3x224x224 instead of
    // a tactic valid across an optimisation-profile range.
    for (int32_t i = 0; i < network->getNbInputs(); ++i)
    {
        const nvinfer1::ITensor* in = network->getInput(i);
        const nvinfer1::Dims d = in->getDimensions();
        for (int32_t k = 0; k < d.nbDims; ++k)
        {
            if (d.d[k] < 0)
            {
                throw std::runtime_error(std::string("input '") + in->getName() + "' is dynamic " + dimsToString(d)
                    + "; re-export with a static shape or add an IOptimizationProfile");
            }
        }
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> cfg{builder->createBuilderConfig()};
    if (!cfg)
    {
        throw std::runtime_error("createBuilderConfig failed");
    }
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, mConfig.workspaceBytes);

    // kTF32 is ON by default. Clear it unless TF32 was asked for, so kFP32 is a true FP32 baseline and the
    // FP32 fallback layers inside a kFP16 engine are not silently TF32 either.
    if (mConfig.precision == Precision::kTF32)
    {
        cfg->setFlag(nvinfer1::BuilderFlag::kTF32);
    }
    else
    {
        cfg->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }

#if NV_TENSORRT_MAJOR < 11
    if (mConfig.precision == Precision::kFP16)
    {
        // Weak typing: TensorRT may choose an FP16 kernel per layer when it times faster, with FP32 fallback.
        // The speedup comes from Tensor Cores (FP16 operands) and halved activation/weight bandwidth.
        // Network I/O stays FP32; TensorRT inserts reformat layers at the boundaries, so the host-side
        // buffers and the H2D/D2H transfer sizes are identical across precisions.
        TRTP_SUPPRESS_DEPRECATED_BEGIN
        cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
        TRTP_SUPPRESS_DEPRECATED_END
    }
#else
    // Strong typing (TRT 11): the ONNX graph dictates every tensor's type; there is no FP16 builder flag.
    const bool graphIsHalf = networkHasHalfTensors(*network);
    if (mConfig.precision == Precision::kFP16 && !graphIsHalf)
    {
        throw std::runtime_error("Precision::kFP16 on TensorRT " + std::to_string(NV_TENSORRT_MAJOR)
            + ".x needs an FP16 ONNX graph (python export_to_onnx.py --fp16-onnx); '" + mConfig.onnxPath
            + "' is pure FP32");
    }
    if (mConfig.precision != Precision::kFP16 && graphIsHalf)
    {
        std::cerr << "[TrtPipeline][W] " << toString(mConfig.precision) << " requested but '" << mConfig.onnxPath
                  << "' contains FP16 tensors; the strongly typed build will run them in FP16\n";
    }
#endif

    std::unique_ptr<nvinfer1::IHostMemory> serialized{builder->buildSerializedNetwork(*network, *cfg)};
    if (!serialized || serialized->size() == 0)
    {
        throw std::runtime_error("buildSerializedNetwork failed (see [TRT] log above)");
    }
    return std::string(static_cast<const char*>(serialized->data()), serialized->size());
}

void TrtPipeline::deserializeEngine(const std::string& plan)
{
    if (!mRuntime)
    {
        mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
        if (!mRuntime)
        {
            throw std::runtime_error("createInferRuntime failed");
        }
    }
    mEngine.reset(mRuntime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!mEngine)
    {
        throw std::runtime_error("deserializeCudaEngine failed (plan from a different GPU arch or TensorRT version?)");
    }
}

void TrtPipeline::allocate(TensorBinding& t)
{
    void* raw = nullptr;
    switch (mConfig.ioMode)
    {
    case IoMode::kZeroCopy:
    {
        // Page-locked AND mapped into the GPU's virtual address space: kernels dereference host RAM directly,
        // each access a PCIe transaction. With UVA (any 64-bit OS) the device pointer equals the host pointer,
        // but cudaHostGetDevicePointer is the portable way to get it.
        CUDA_CHECK(cudaHostAlloc(&raw, t.bytes, cudaHostAllocMapped));
        t.pinned.reset(raw);
        void* mapped = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&mapped, raw, 0));
        t.devicePtr = mapped;
        break;
    }
    case IoMode::kPinned:
    case IoMode::kPageable:
    {
        if (mConfig.ioMode == IoMode::kPinned)
        {
            // Pinned staging buffer: the OS can neither swap nor move it, so the copy engine can DMA from it
            // and cudaMemcpyAsync is truly asynchronous. Pinning is expensive (cudaMallocHost/cudaFreeHost
            // take milliseconds and may synchronise the device) and takes RAM away from the OS: allocate once
            // at init, never per frame, and keep it small.
            CUDA_CHECK(cudaMallocHost(&raw, t.bytes));
            t.pinned.reset(raw);
        }
        else
        {
            // Ordinary heap memory: exactly what a naive pipeline (std::vector, cv::Mat) hands to cudaMemcpy.
            t.pageable = std::make_unique<float[]>(t.count);
            raw = t.pageable.get();
        }

        // GPU global memory: the card's own GDDR6, on the far side of the PCIe link.
        void* dev = nullptr;
        CUDA_CHECK(cudaMalloc(&dev, t.bytes));
        t.device.reset(dev);
        t.devicePtr = dev;
        break;
    }
    }
    t.host = static_cast<float*>(raw);
    std::memset(t.host, 0, t.bytes);
}

void TrtPipeline::bindIoTensors()
{
    int32_t nInputs = 0;
    int32_t nOutputs = 0;

    // Name-based I/O (TRT >= 8.5): the old binding-index API and enqueueV2 were removed in TRT 10.
    for (int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        const char* name = mEngine->getIOTensorName(i);
        const bool isInput = mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        (isInput ? nInputs : nOutputs)++;
        if (nInputs > 1 || nOutputs > 1)
        {
            throw std::runtime_error("TrtPipeline handles single-input/single-output networks only");
        }

        const nvinfer1::DataType dt = mEngine->getTensorDataType(name);
        if (dt != nvinfer1::DataType::kFLOAT)
        {
            throw std::runtime_error(std::string("I/O tensor '") + name + "' is " + dataTypeName(dt)
                + "; this pipeline stages FP32 I/O (re-run AutoCast with keep_io_types=True)");
        }

        TensorBinding& t = isInput ? mInput : mOutput;
        t.name = name;
        t.dims = mContext->getTensorShape(name);
        t.count = volume(t.dims, t.name);
        t.bytes = t.count * sizeof(float);
        allocate(t);

        // Binds the device address once. Re-binding per frame is legal but unnecessary: the buffers are fixed.
        if (!mContext->setTensorAddress(name, t.devicePtr))
        {
            throw std::runtime_error(std::string("setTensorAddress failed for '") + name + "'");
        }
    }

    if (nInputs != 1 || nOutputs != 1)
    {
        throw std::runtime_error("expected exactly 1 input and 1 output tensor");
    }
    if (!mContext->allInputDimensionsSpecified())
    {
        throw std::runtime_error("input dimensions not fully specified");
    }
}

float* TrtPipeline::inputHostBuffer() noexcept
{
    return mInput.host;
}

const float* TrtPipeline::outputHostBuffer() const noexcept
{
    return mOutput.host;
}

void TrtPipeline::infer(const float* input, std::size_t inputCount, float* output, std::size_t outputCount)
{
    if (input == nullptr || output == nullptr)
    {
        throw std::invalid_argument("infer: null buffer");
    }
    if (inputCount != mInput.count || outputCount != mOutput.count)
    {
        throw std::invalid_argument("infer: expected " + std::to_string(mInput.count) + " input / "
            + std::to_string(mOutput.count) + " output elements, got " + std::to_string(inputCount) + " / "
            + std::to_string(outputCount));
    }

    if (mConfig.ioMode == IoMode::kPageable)
    {
        // Naive baseline: hand the caller's pageable pointers straight to the copy engine path.
        run(input, output);
        return;
    }

    // Caller memory -> pinned: a CPU memcpy of 602 KB (tens of microseconds). It is the same bounce copy the
    // driver performs in kPageable mode; inferInPlace() removes it by having the producer write into pinned
    // memory in the first place.
    std::memcpy(mInput.host, input, mInput.bytes);
    run(mInput.host, mOutput.host);
    std::memcpy(output, mOutput.host, mOutput.bytes);
}

void TrtPipeline::inferInPlace()
{
    run(mInput.host, mOutput.host);
}

void TrtPipeline::run(const void* hostIn, void* hostOut)
{
    cudaStream_t s = mStream.get();
    const bool copies = mConfig.ioMode != IoMode::kZeroCopy;

    if (mStageTiming)
    {
        CUDA_CHECK(cudaEventRecord(mEvStart.get(), s));
    }

    if (copies)
    {
        // H2D: 602,112 B. At ~12 GB/s (PCIe 4.0 x8, common on laptops) that is ~50 us; ~25 us on x16. Laptop
        // links drop to PCIe Gen1 when idle to save power and only come back up under load, which is one
        // reason the warm-up loop exists.
        //
        // hostIn pinned  : the copy is queued on mStream and the call returns immediately.
        // hostIn pageable: the driver first memcpys hostIn into an internal pinned buffer on the CPU, and this
        //                  call blocks until that staging is done.
        //
        // Every copy stays on mStream, never a plain cudaMemcpy: that runs on the legacy default stream, which
        // does not synchronise with a cudaStreamNonBlocking stream. For a pageable source it can also return
        // before the DMA has landed, so enqueueV3 below could read a half-written input.
        CUDA_CHECK(cudaMemcpyAsync(mInput.devicePtr, hostIn, mInput.bytes, cudaMemcpyHostToDevice, s));
    }

    if (mStageTiming)
    {
        CUDA_CHECK(cudaEventRecord(mEvAfterH2D.get(), s));
    }

    // Stream ordering guarantees the kernels start only after the H2D copy has landed; no explicit sync.
    if (!mContext->enqueueV3(s))
    {
        throw std::runtime_error("IExecutionContext::enqueueV3 failed");
    }

    if (mStageTiming)
    {
        CUDA_CHECK(cudaEventRecord(mEvAfterCompute.get(), s));
    }

    if (copies)
    {
        // D2H: 1000 FP32 = 4 KB. Bandwidth is irrelevant at this size; it is pure latency (DMA setup, ~5-10 us).
        // Into pageable memory this call blocks until the copy is complete, i.e. until the whole inference
        // has finished: the CPU loses the ability to do anything else while the GPU works.
        CUDA_CHECK(cudaMemcpyAsync(hostOut, mOutput.devicePtr, mOutput.bytes, cudaMemcpyDeviceToHost, s));
    }

    if (mStageTiming)
    {
        CUDA_CHECK(cudaEventRecord(mEvAfterD2H.get(), s));
    }

    // The single host<->device synchronisation point per frame. Asynchronous kernel faults surface here.
    CUDA_CHECK(cudaStreamSynchronize(s));

    if (mStageTiming)
    {
        CUDA_CHECK(cudaEventElapsedTime(&mTimings.h2dMs, mEvStart.get(), mEvAfterH2D.get()));
        CUDA_CHECK(cudaEventElapsedTime(&mTimings.computeMs, mEvAfterH2D.get(), mEvAfterCompute.get()));
        CUDA_CHECK(cudaEventElapsedTime(&mTimings.d2hMs, mEvAfterCompute.get(), mEvAfterD2H.get()));
    }
}

std::string TrtPipeline::describe() const
{
    const int32_t v = getInferLibVersion();
    const double mib = 1024.0 * 1024.0;
    std::ostringstream os;
    os << std::fixed << std::setprecision(2);
    os << "  TensorRT     : " << v / 10000 << '.' << (v / 100) % 100 << '.' << v % 100 << '\n'
       << "  GPU          : " << mDeviceName << " (sm_" << mComputeCapability << ", "
       << (mIntegratedGpu ? "integrated, shared DRAM)" : "discrete, PCIe)")
       << '\n'
       << "  Precision    : " << toString(mConfig.precision) << '\n'
       << "  Engine       : " << (mBuiltThisRun ? "built" : "loaded from cache") << " in " << mSetupSeconds
       << " s, plan " << static_cast<double>(mPlanBytes) / mib << " MiB, " << mEngine->getNbLayers()
       << " fused layers\n"
       << "  Activations  : " << static_cast<double>(mEngine->getDeviceMemorySizeV2()) / mib
       << " MiB device memory (context-owned)\n"
       << "  Input        : '" << mInput.name << "' " << dimsToString(mInput.dims) << " FP32, " << mInput.bytes
       << " B\n"
       << "  Output       : '" << mOutput.name << "' " << dimsToString(mOutput.dims) << " FP32, " << mOutput.bytes
       << " B\n"
       << "  I/O memory   : " << toString(mConfig.ioMode) << " - ";
    switch (mConfig.ioMode)
    {
    case IoMode::kPinned: os << "cudaMallocHost staging + cudaMalloc, async DMA"; break;
    case IoMode::kPageable: os << "heap memory + cudaMalloc, driver bounce-buffer copies"; break;
    case IoMode::kZeroCopy:
        os << "mapped pinned host memory, no copies"
           << (mIntegratedGpu ? " (shared DRAM)" : " (kernels read host RAM over PCIe)");
        break;
    }
    return os.str();
}

} // namespace trtp

// TrtPipeline.h
// ONNX -> TensorRT engine builder + single-stream inference wrapper (TensorRT 10.x / 11.x, C++17).
//
// Memory hierarchy on a discrete laptop GPU (e.g. RTX 3050 Laptop: Ampere sm_86, 4 GB GDDR6 behind PCIe 4.0):
//
//   host DDR4/5                                  PCIe link              GPU GDDR6               on-chip
//   pageable buffer --(driver bounce memcpy)--> pinned buffer ==DMA==> cudaMalloc buffer --> L2 / smem / regs
//                                                                      (I/O tensors + TRT activations)
//
//   * Only the network I/O crosses the bus: 602 KB in (1x3x224x224 FP32), 4 KB out. Activations are bigger
//     (conv1's output alone is 64x112x112 FP32 = 3.2 MB) but live in GDDR6, in scratch memory TensorRT owns
//     per execution context. They never touch the host.
//   * The bus is the narrow pipe: a laptop PCIe 4.0 link moves roughly 12-25 GB/s depending on lane count
//     (x8 or x16; check `nvidia-smi -q`), while the GPU reads its own GDDR6 at ~190 GB/s or more. Every
//     byte the GPU has to fetch from host RAM is about 10x more expensive than a byte already in VRAM.
//
// IoMode picks the path the I/O takes, so the cost of each can be measured on the same engine:
//   kPinned   : page-locked host buffers (cudaMallocHost) + cudaMemcpyAsync. The copy engine DMAs directly
//               and the call returns immediately. Production default.
//   kPageable : the naive path, DMA straight from ordinary new/malloc memory. The OS may move or swap
//               that memory, so the copy engine cannot touch it. The driver first memcpys it on the CPU
//               into an internal pinned bounce buffer, and the "async" copy blocks the host meanwhile.
//   kZeroCopy : pinned memory mapped into the GPU address space (cudaHostAllocMapped). No copy at all: the
//               kernels read and write host RAM over PCIe. A win on Jetson-class integrated GPUs, where CPU
//               and GPU share one DRAM pool and there is no bus. A loss on a discrete GPU, where the first
//               layer pulls its input across PCIe instead of from GDDR6.

#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

// --------------------------------------------------------------------------------------------------------
// Error checking. Every CUDA runtime call in this project goes through one of these two macros.
//   CUDA_CHECK          -> throws std::runtime_error (constructors, hot path: fail loudly, never silently)
//   CUDA_CHECK_NOTHROW  -> logs to stderr (destructors / deleters: throwing there calls std::terminate)
// Note: kernel faults inside enqueueV3 are asynchronous and surface at the next synchronising call
// (cudaStreamSynchronize), which is also checked.
// --------------------------------------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                                  \
    do                                                                                                    \
    {                                                                                                     \
        const cudaError_t trtp_err_ = (call);                                                             \
        if (trtp_err_ != cudaSuccess)                                                                     \
        {                                                                                                 \
            ::trtp::detail::throwCudaError(trtp_err_, #call, __FILE__, __LINE__);                         \
        }                                                                                                 \
    } while (0)

#define CUDA_CHECK_NOTHROW(call)                                                                          \
    do                                                                                                    \
    {                                                                                                     \
        const cudaError_t trtp_err_ = (call);                                                             \
        if (trtp_err_ != cudaSuccess)                                                                     \
        {                                                                                                 \
            ::trtp::detail::logCudaError(trtp_err_, #call, __FILE__, __LINE__);                           \
        }                                                                                                 \
    } while (0)

namespace trtp
{

namespace detail
{
[[noreturn]] void throwCudaError(cudaError_t err, const char* expr, const char* file, int line);
void logCudaError(cudaError_t err, const char* expr, const char* file, int line) noexcept;

// RAII owners for raw CUDA resources. They make the constructor exception-safe: if any allocation throws
// halfway through, everything already acquired is released by the member destructors.
struct DeviceFree
{
    void operator()(void* p) const noexcept;
};
struct PinnedFree
{
    void operator()(void* p) const noexcept;
};
struct StreamDestroy
{
    void operator()(cudaStream_t s) const noexcept;
};
struct EventDestroy
{
    void operator()(cudaEvent_t e) const noexcept;
};
} // namespace detail

using DevicePtr = std::unique_ptr<void, detail::DeviceFree>;
using PinnedPtr = std::unique_ptr<void, detail::PinnedFree>;
using StreamPtr = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, detail::StreamDestroy>;
using EventPtr = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, detail::EventDestroy>;

// The three points of the precision/throughput curve for an FP32-trained model on Tensor-Core GPUs.
//   kFP32 : strict IEEE FP32 math. TF32 explicitly disabled (TensorRT enables it by default on sm_80+,
//           so an unqualified "FP32" engine on any RTX 30-series or newer is actually TF32 - a silently
//           wrong baseline).
//   kTF32 : FP32 storage, Tensor-Core GEMMs with inputs rounded to a 10-bit mantissa, FP32 accumulate.
//           Requires sm_80+ (RTX 30-series and newer); identical to kFP32 on older GPUs.
//   kFP16 : FP16 storage and math where TensorRT (TRT 10) or the AutoCast graph (TRT 11) allows it.
enum class Precision
{
    kFP32,
    kTF32,
    kFP16
};
const char* toString(Precision p) noexcept;

enum class IoMode
{
    kPinned,
    kPageable,
    kZeroCopy
};
const char* toString(IoMode m) noexcept;

class Logger final : public nvinfer1::ILogger
{
public:
    explicit Logger(Severity minSeverity = Severity::kWARNING) noexcept : mMinSeverity(minSeverity) {}
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;

private:
    Severity mMinSeverity;
};

struct PipelineConfig
{
    // FP32 graph for TRT 10.x (all precisions) and for TRT 11.x FP32/TF32.
    // On TRT >= 11, Precision::kFP16 requires the ModelOpt AutoCast graph (export_to_onnx.py --fp16-onnx),
    // because the BuilderFlag::kFP16 weak-typing switch no longer exists.
    std::string onnxPath = "resnet18.onnx";

    // Serialized plan cache. A plan is specific to the GPU architecture AND the exact TensorRT version;
    // a mismatch is detected on deserialisation and triggers a rebuild. Empty string disables caching.
    std::string enginePath;

    Precision precision = Precision::kFP32;

    // Upper bound on tactic scratch memory during build and inference. A 4 GB laptop GPU shares its VRAM
    // with everything else on it (browser, and the desktop compositor if the laptop runs dGPU-only/MUX mode).
    std::size_t workspaceBytes = std::size_t{256} << 20;

    bool forceRebuild = false;

    IoMode ioMode = IoMode::kPinned;

    int deviceId = 0;
    Logger::Severity logSeverity = Logger::Severity::kWARNING;
};

// GPU-side durations of the last inference, measured with CUDA events on the pipeline stream.
// These isolate each stage. Host-side std::chrono latency additionally includes launch overhead, CPU
// memcpys and the stream-synchronisation wake-up, so it is the number that matters end to end.
struct StageTimings
{
    float h2dMs = 0.F;
    float computeMs = 0.F;
    float d2hMs = 0.F;
};

class TrtPipeline
{
public:
    explicit TrtPipeline(PipelineConfig config);
    ~TrtPipeline();

    // Owns CUDA streams/events and TensorRT objects bound to raw device addresses: non-copyable, non-movable.
    TrtPipeline(const TrtPipeline&) = delete;
    TrtPipeline& operator=(const TrtPipeline&) = delete;
    TrtPipeline(TrtPipeline&&) = delete;
    TrtPipeline& operator=(TrtPipeline&&) = delete;

    // Blocking end-to-end inference from arbitrary (pageable) caller memory.
    //   kPinned / kZeroCopy : memcpy into the internal pinned buffer, run, memcpy the logits back out.
    //                         (kZeroCopy runs no H2D/D2H copies; the kernels read that buffer over the bus.)
    //   kPageable           : DMA straight from/to the caller's buffers (the driver does the bounce copy).
    void infer(const float* input, std::size_t inputCount, float* output, std::size_t outputCount);

    // Same pipeline without the caller-side memcpys: the producer (camera decode / preprocessing) writes
    // straight into inputHostBuffer() and the consumer reads outputHostBuffer(). This is the production path,
    // and the only way to get the full benefit of pinned memory.
    void inferInPlace();
    float* inputHostBuffer() noexcept;
    const float* outputHostBuffer() const noexcept;

    std::size_t inputCount() const noexcept { return mInput.count; }
    std::size_t outputCount() const noexcept { return mOutput.count; }
    std::size_t inputBytes() const noexcept { return mInput.bytes; }

    // CUDA-event stage timing adds four cudaEventRecord calls (~1-2 us each) per inference.
    void setStageTiming(bool enabled) noexcept { mStageTiming = enabled; }
    const StageTimings& lastStageTimings() const noexcept { return mTimings; }

    int computeCapability() const noexcept { return mComputeCapability; } // 86 for an RTX 3050 Laptop GPU
    bool integratedGpu() const noexcept { return mIntegratedGpu; }
    bool engineBuiltThisRun() const noexcept { return mBuiltThisRun; }
    double engineSetupSeconds() const noexcept { return mSetupSeconds; }
    const PipelineConfig& config() const noexcept { return mConfig; }
    std::string describe() const;

private:
    struct TensorBinding
    {
        std::string name;
        nvinfer1::Dims dims{};
        std::size_t count = 0;      // elements
        std::size_t bytes = 0;              // count * sizeof(float)
        PinnedPtr pinned;                   // kPinned: DMA staging buffer; kZeroCopy: the mapped buffer itself
        std::unique_ptr<float[]> pageable;  // kPageable: ordinary heap memory
        DevicePtr device;                   // cudaMalloc'd GDDR6; empty in zero-copy mode
        float* host = nullptr;              // whichever host buffer is active
        void* devicePtr = nullptr;          // address handed to TensorRT: device.get() or the mapped alias
    };

    std::string buildSerializedEngine();
    void deserializeEngine(const std::string& plan);
    void bindIoTensors();
    void allocate(TensorBinding& t);
    void run(const void* hostIn, void* hostOut); // copies (if any) + enqueueV3 + sync, all on mStream

    PipelineConfig mConfig;
    Logger mLogger; // must outlive every TensorRT object below (declared first => destroyed last)

    // Destruction runs bottom-up: buffers, events, stream, context, engine, runtime.
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;

    StreamPtr mStream;
    EventPtr mEvStart;
    EventPtr mEvAfterH2D;
    EventPtr mEvAfterCompute;
    EventPtr mEvAfterD2H;

    TensorBinding mInput;
    TensorBinding mOutput;

    std::string mDeviceName;
    int mComputeCapability = 0;
    bool mIntegratedGpu = false;
    bool mStageTiming = false;
    StageTimings mTimings;
    bool mBuiltThisRun = false;
    double mSetupSeconds = 0.0;
    std::size_t mPlanBytes = 0;
};

} // namespace trtp

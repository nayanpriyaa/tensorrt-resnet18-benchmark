// main.cpp
// ResNet-18 TensorRT benchmark on a discrete GPU:
//   * precision  : FP32 vs TF32 vs FP16 latency + accuracy against the PyTorch FP32 reference
//   * I/O path   : pinned vs pageable vs zero-copy host memory, i.e. what crossing PCIe actually costs
//
//   ./trt_resnet18                              # all precisions, pinned I/O
//   ./trt_resnet18 --io all                     # every precision x every I/O mode
//   ./trt_resnet18 --precision fp16 --io pinned,pageable --iters 2000
//
// Exit codes: 0 = OK, 1 = runtime error, 2 = an engine failed its accuracy gate.

#include "TrtPipeline.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace
{

constexpr std::size_t kChannels = 3;
constexpr std::size_t kHeight = 224;
constexpr std::size_t kWidth = 224;
constexpr std::size_t kInputPerImage = kChannels * kHeight * kWidth;
constexpr std::size_t kNumClasses = 1000;
constexpr float kMean[kChannels] = {0.485F, 0.456F, 0.406F};
constexpr float kStd[kChannels] = {0.229F, 0.224F, 0.225F};

struct Options
{
    std::string onnx = "resnet18.onnx";
    std::string onnxFp16 = "resnet18_fp16.onnx"; // only read on TensorRT >= 11 (strongly typed FP16)
    std::string inputBin = "input.bin";
    std::string refBin = "ref_output.bin";
    std::vector<trtp::Precision> precisions{trtp::Precision::kFP32, trtp::Precision::kTF32, trtp::Precision::kFP16};
    std::vector<trtp::IoMode> ioModes{trtp::IoMode::kPinned};
    int batch = 1;
    int warmup = 200;
    int iters = 1000;
    std::size_t workspaceMiB = 256;
    bool rebuild = false;
    bool verbose = false;
};

void printUsage(const char* argv0)
{
    std::cout << "usage: " << argv0 << " [options]\n"
              << "  --precision all|fp32|tf32|fp16[,..]  engines to benchmark (default: all)\n"
              << "  --io all|pinned|pageable|zerocopy[,..] host I/O memory paths (default: pinned)\n"
              << "  --onnx PATH            FP32 ONNX (default resnet18.onnx)\n"
              << "  --onnx-fp16 PATH       AutoCast FP16 ONNX, TensorRT >= 11 only (default resnet18_fp16.onnx)\n"
              << "  --batch N              batch size (default 1)\n"
              << "  --input PATH           raw float32 NCHW input (default input.bin; synthetic frame if missing)\n"
              << "  --ref PATH             raw float32 PyTorch logits for --input (default ref_output.bin)\n"
              << "  --warmup N             untimed iterations (default 200)\n"
              << "  --iters N              timed iterations (default 1000)\n"
              << "  --workspace-mib N      builder workspace limit (default 256)\n"
              << "  --rebuild              ignore cached .engine files\n"
              << "  --verbose              TensorRT INFO logging\n";
}

trtp::Precision parsePrecision(const std::string& s)
{
    if (s == "fp32") return trtp::Precision::kFP32;
    if (s == "tf32") return trtp::Precision::kTF32;
    if (s == "fp16") return trtp::Precision::kFP16;
    throw std::invalid_argument("unknown precision '" + s + "'");
}

trtp::IoMode parseIoMode(const std::string& s)
{
    if (s == "pinned") return trtp::IoMode::kPinned;
    if (s == "pageable") return trtp::IoMode::kPageable;
    if (s == "zerocopy") return trtp::IoMode::kZeroCopy;
    throw std::invalid_argument("unknown I/O mode '" + s + "'");
}

template <typename T, typename Parse>
std::vector<T> parseList(const std::string& v, Parse parse, std::vector<T> all)
{
    if (v == "all")
    {
        return all;
    }
    std::vector<T> out;
    std::stringstream ss(v);
    for (std::string tok; std::getline(ss, tok, ',');)
    {
        out.push_back(parse(tok));
    }
    return out;
}

Options parseArgs(int argc, char** argv)
{
    Options o;
    auto next = [&](int& i) -> std::string {
        if (i + 1 >= argc)
        {
            throw std::invalid_argument(std::string("missing value for ") + argv[i]);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") { printUsage(argv[0]); std::exit(0); }
        else if (a == "--precision")
            o.precisions = parseList<trtp::Precision>(next(i), parsePrecision,
                {trtp::Precision::kFP32, trtp::Precision::kTF32, trtp::Precision::kFP16});
        else if (a == "--io")
            o.ioModes = parseList<trtp::IoMode>(next(i), parseIoMode,
                {trtp::IoMode::kPinned, trtp::IoMode::kPageable, trtp::IoMode::kZeroCopy});
        else if (a == "--onnx") o.onnx = next(i);
        else if (a == "--onnx-fp16") o.onnxFp16 = next(i);
        else if (a == "--input") o.inputBin = next(i);
        else if (a == "--ref") o.refBin = next(i);
        else if (a == "--warmup") o.warmup = std::stoi(next(i));
        else if (a == "--iters") o.iters = std::stoi(next(i));
        else if (a == "--workspace-mib") o.workspaceMiB = static_cast<std::size_t>(std::stoul(next(i)));
        else if (a == "--rebuild") o.rebuild = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--batch") o.batch = std::stoi(next(i));
        else throw std::invalid_argument("unknown option '" + a + "' (see --help)");
    }
    if (o.iters <= 0 || o.warmup < 0 || o.batch <= 0 ||
    o.precisions.empty() || o.ioModes.empty())
{
    throw std::invalid_argument("invalid benchmark options");
}

return o;
}

// Raw little-endian float32 blob written by export_to_onnx.py (numpy.tofile). Empty vector if absent.
std::vector<float> loadFloats(const std::string& path, std::size_t expectedCount)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
    {
        return {};
    }
    const auto bytes = static_cast<std::size_t>(f.tellg());
    if (bytes != expectedCount * sizeof(float))
    {
        throw std::runtime_error(path + ": expected " + std::to_string(expectedCount * sizeof(float)) + " bytes, got "
            + std::to_string(bytes));
    }
    std::vector<float> v(expectedCount);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
    return v;
}


std::vector<float> makeSyntheticFrame(int batch)
{
    std::vector<float> v(
        static_cast<std::size_t>(batch) * kInputPerImage
    );

    for (int b = 0; b < batch; ++b)
    {
        for (std::size_t c = 0; c < kChannels; ++c)
        {
            for (std::size_t y = 0; y < kHeight; ++y)
            {
                for (std::size_t x = 0; x < kWidth; ++x)
                {
                    const float pixel =
                        0.5F + 0.5F * std::sin(
                            0.07F * static_cast<float>(x)
                            + 0.11F * static_cast<float>(y)
                            + 1.3F * static_cast<float>(c)
                        );

                    const std::size_t index =
                        static_cast<std::size_t>(b) * kInputPerImage
                        + (c * kHeight + y) * kWidth + x;

                    v[index] = (pixel - kMean[c]) / kStd[c];
                }
            }
        }
    }

    return v;
}


struct LatencyStats
{
    double mean = 0, p50 = 0, p90 = 0, p99 = 0, min = 0, max = 0;
};

LatencyStats summarize(std::vector<double> ms)
{
    std::sort(ms.begin(), ms.end());
    auto pct = [&](double p) {
        const auto idx = static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(ms.size()))) - 1;
        return ms[std::min(idx, ms.size() - 1)]; // nearest-rank percentile
    };
    LatencyStats s;
    s.mean = std::accumulate(ms.begin(), ms.end(), 0.0) / static_cast<double>(ms.size());
    s.p50 = pct(50);
    s.p90 = pct(90);
    s.p99 = pct(99);
    s.min = ms.front();
    s.max = ms.back();
    return s;
}


struct Accuracy
{
    double maxAbs = 0, meanAbs = 0, cosine = 0;

    std::size_t top1 = 0, refTop1 = 0;
    std::size_t top1Matches = 0;
    std::size_t batchSize = 0;
};


Accuracy compareLogits(
    const std::vector<float>& out,
    const std::vector<float>& ref)
{
    Accuracy a;

    if (out.size() != ref.size() ||
        out.empty() ||
        out.size() % kNumClasses != 0)
    {
        throw std::runtime_error(
            "Output and reference logits have incompatible sizes"
        );
    }

    a.batchSize = out.size() / kNumClasses;

    double dot = 0, nOut = 0, nRef = 0, sumAbs = 0;

    for (std::size_t i = 0; i < out.size(); ++i)
    {
        const double d =
            std::abs(static_cast<double>(out[i]) - ref[i]);

        a.maxAbs = std::max(a.maxAbs, d);
        sumAbs += d;

        dot += static_cast<double>(out[i]) * ref[i];
        nOut += static_cast<double>(out[i]) * out[i];
        nRef += static_cast<double>(ref[i]) * ref[i];
    }

    a.meanAbs = sumAbs / static_cast<double>(out.size());

    a.cosine =
        dot / (std::sqrt(nOut) * std::sqrt(nRef) + 1e-30);

    for (std::size_t b = 0; b < a.batchSize; ++b)
    {
        const std::size_t start = b * kNumClasses;
        const std::size_t end = start + kNumClasses;

        const auto outMax =
            std::max_element(out.begin() + start, out.begin() + end);

        const auto refMax =
            std::max_element(ref.begin() + start, ref.begin() + end);

        const std::size_t outIndex =
            static_cast<std::size_t>(outMax - (out.begin() + start));

        const std::size_t refIndex =
            static_cast<std::size_t>(refMax - (ref.begin() + start));

        if (b == 0)
        {
            a.top1 = outIndex;
            a.refTop1 = refIndex;
        }

        if (outIndex == refIndex)
        {
            ++a.top1Matches;
        }
    }

    return a;
}


// Accuracy gates vs. PyTorch FP32 logits (O(10) magnitude). FP32 differs only by kernel summation order;
// TF32/FP16 round operands to a 10-bit mantissa, so they are gated on direction (cosine) plus a looser bound.
bool passesGate(trtp::Precision p, const Accuracy& a)
{
    const bool top1Passed =
        a.top1Matches == a.batchSize;

    if (p == trtp::Precision::kFP32)
    {
        return top1Passed && a.maxAbs <= 1e-2;
    }

    return top1Passed &&
           a.cosine >= 0.999 &&
           a.maxAbs <= 0.25;
}

struct RunResult
{
    trtp::Precision precision{};
    trtp::IoMode io{};
    int batchSize = 1;
    LatencyStats e2e;
    trtp::StageTimings gpuMean;
    double h2dGBps = 0; // effective H2D bandwidth; 0 when no copy happens (zero-copy)
    std::vector<float> logits;
    std::optional<Accuracy> vsRef;
    bool gatePassed = true;
};

std::string lower(trtp::Precision p)
{
    std::string s = trtp::toString(p);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

RunResult benchmark(trtp::Precision precision, trtp::IoMode io, bool rebuild, const Options& opt,
    const std::vector<float>& input, const std::vector<float>& ref)
{
    trtp::PipelineConfig cfg;
#if NV_TENSORRT_MAJOR >= 11
    cfg.onnxPath = precision == trtp::Precision::kFP16 ? opt.onnxFp16 : opt.onnx;
#else
    cfg.onnxPath = opt.onnx;
#endif
    cfg.enginePath = "resnet18_" + lower(precision)
    + "_b" + std::to_string(opt.batch) + ".engine"; 
    cfg.precision = precision;
    cfg.workspaceBytes = opt.workspaceMiB << 20;
    cfg.forceRebuild = rebuild;
    cfg.ioMode = io;
    cfg.logSeverity = opt.verbose ? trtp::Logger::Severity::kINFO : trtp::Logger::Severity::kWARNING;

    std::cout << "\n==================== " << trtp::toString(precision) << " / " << trtp::toString(io)
              << " ====================\n";
    trtp::TrtPipeline pipe(cfg);
    std::cout << pipe.describe() << '\n';
    if (precision == trtp::Precision::kTF32 && pipe.computeCapability() < 80)
    {
        std::cout << "  note: TF32 needs sm_80+; on sm_" << pipe.computeCapability() << " this engine is plain FP32\n";
    }

    RunResult r;
    r.precision = precision;
    r.io = io;
    r.batchSize = opt.batch;
    r.logits.resize(pipe.outputCount());

    // Fair comparison of memory types: the producer (camera decode / preprocessing) writes the frame once,
    // outside the timed loop, either into the pipeline's own host buffer (pinned, zero-copy) or into an
    // ordinary std::vector (pageable, the naive pipeline). The timed loop then measures only what it costs
    // to get that frame through the GPU and back from where it sits.
    const bool inPlace = io != trtp::IoMode::kPageable;
    if (inPlace)
    {
        std::copy(input.begin(), input.end(), pipe.inputHostBuffer());
    }
    auto step = [&] {
        if (inPlace)
        {
            pipe.inferInPlace();
        }
        else
        {
            pipe.infer(input.data(), input.size(), r.logits.data(), r.logits.size());
        }
    };

    // Warm-up, never timed. The first calls pay one-off costs: lazy CUDA module loading of each kernel,
    // first-touch page mapping, and the GPU boost clock and PCIe link (parked at Gen1 when a laptop dGPU idles)
    // ramping up out of their power-saving states.
    for (int i = 0; i < opt.warmup; ++i)
    {
        step();
    }

    // Pass 1 - end-to-end latency, host wall clock (std::chrono::steady_clock: monotonic, unaffected by NTP).
    // Every step ends in cudaStreamSynchronize, so each interval brackets H2D -> kernels -> D2H -> sync.
    // Timing an async enqueue without a sync measures only the CPU-side launch cost (tens of microseconds),
    // which is the most common GPU benchmarking mistake.
    pipe.setStageTiming(false);
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(opt.iters));
    for (int i = 0; i < opt.iters; ++i)
    {
        const auto t0 = std::chrono::steady_clock::now();
        step();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    r.e2e = summarize(std::move(samples));

    // Pass 2 - per-stage GPU time with CUDA events, kept separate so event overhead cannot skew pass 1.
    // Pageable: the H2D window also contains the driver's CPU-side bounce copy, so the "GB/s" figure is the
    // effective rate of the whole naive path, not of the PCIe DMA alone.
    pipe.setStageTiming(true);
    double h2d = 0, comp = 0, d2h = 0;
    for (int i = 0; i < opt.iters; ++i)
    {
        step();
        const auto& t = pipe.lastStageTimings();
        h2d += t.h2dMs;
        comp += t.computeMs;
        d2h += t.d2hMs;
    }
    const double n = static_cast<double>(opt.iters);
    r.gpuMean = {static_cast<float>(h2d / n), static_cast<float>(comp / n), static_cast<float>(d2h / n)};

    if (inPlace)
    {
        std::copy(pipe.outputHostBuffer(), pipe.outputHostBuffer() + r.logits.size(), r.logits.begin());
    }
    if (io != trtp::IoMode::kZeroCopy && r.gpuMean.h2dMs > 0.F)
    {
        r.h2dGBps = static_cast<double>(pipe.inputBytes()) / (static_cast<double>(r.gpuMean.h2dMs) * 1e6);
    }

    const double gpuTotal = r.gpuMean.h2dMs + r.gpuMean.computeMs + r.gpuMean.d2hMs;
    std::cout << std::fixed << std::setprecision(3)
              << "  Forward pass (end-to-end, std::chrono, n=" << opt.iters << "):\n"
              << "    mean " << r.e2e.mean << " ms | p50 " << r.e2e.p50 << " | p90 " << r.e2e.p90 << " | p99 "
              << r.e2e.p99 << " | min " << r.e2e.min << " | max " << r.e2e.max << "  -> "
              << std::setprecision(1)
              << static_cast<double>(r.batchSize) * 1000.0 / r.e2e.mean
              << " images/s\n"
              << std::setprecision(3) << "  GPU stages (CUDA events, mean):\n    ";
    if (io == trtp::IoMode::kZeroCopy)
    {
        std::cout << "H2D/D2H none (the PCIe reads happen inside the kernels) | compute " << r.gpuMean.computeMs
                  << " ms";
    }
    else
    {
        std::cout << "H2D " << r.gpuMean.h2dMs << " ms (" << std::setprecision(1) << r.h2dGBps << " GB/s effective)"
                  << std::setprecision(3) << " | compute " << r.gpuMean.computeMs << " ms | D2H " << r.gpuMean.d2hMs
                  << " ms";
    }
    std::cout << " | host-side overhead " << std::max(0.0, r.e2e.mean - gpuTotal) << " ms (launch, sync wake-up)\n";

    if (!ref.empty())
    {
        r.vsRef = compareLogits(r.logits, ref);
        r.gatePassed = passesGate(precision, *r.vsRef);
        std::cout << std::scientific << std::setprecision(2) << "  vs PyTorch FP32: max|diff| " << r.vsRef->maxAbs
                  << " | mean|diff| " << r.vsRef->meanAbs << std::fixed << std::setprecision(6) << " | cosine "
                  << r.vsRef->cosine << " | top-1 "<< r.vsRef->top1Matches << "/" << r.vsRef->batchSize
                  << " images | gate " << (r.gatePassed ? "PASS" : "FAIL") << '\n';
    }
    return r;
}

void printSummary(const std::vector<RunResult>& runs)
{
    auto find = [&runs](trtp::Precision p, trtp::IoMode io) -> const RunResult* {
        for (const auto& r : runs)
        {
            if (r.precision == p && r.io == io) return &r;
        }
        return nullptr;
    };

    std::cout << "\n==================================== Summary ====================================\n"
              << std::left << std::setw(6) << "prec" << std::setw(10) << "io" << std::right << std::setw(9) << "mean ms"
              << std::setw(9) << "p99 ms" << std::setw(9) << "H2D ms" << std::setw(7) << "GB/s" << std::setw(11)
              << "compute ms" << std::setw(8) << "FPS" << std::setw(9) << "vs FP32" << std::setw(11) << "max|diff|"
              << std::setw(7) << "top-1" << '\n';
    for (const auto& r : runs)
    {
        std::cout << std::left << std::setw(6) << trtp::toString(r.precision) << std::setw(10) << trtp::toString(r.io)
                  << std::right << std::fixed << std::setprecision(3) << std::setw(9) << r.e2e.mean << std::setw(9)
                  << r.e2e.p99;
        if (r.io == trtp::IoMode::kZeroCopy)
        {
            std::cout << std::setw(9) << "-" << std::setw(7) << "-";
        }
        else
        {
            std::cout << std::setw(9) << r.gpuMean.h2dMs << std::setprecision(1) << std::setw(7) << r.h2dGBps;
        }
        std::cout << std::setprecision(3) << std::setw(11) << r.gpuMean.computeMs << std::setprecision(1) << std::setw(8)
                  << static_cast<double>(r.batchSize) * 1000.0 / r.e2e.mean;
        // Precision speedup is only meaningful between runs that use the same memory path.
        if (const RunResult* base = find(trtp::Precision::kFP32, r.io))
        {
            std::cout << std::setprecision(2) << std::setw(8) << base->e2e.mean / r.e2e.mean << 'x';
        }
        else
        {
            std::cout << std::setw(9) << "-";
        }
        if (r.vsRef)
        {
            std::cout << std::scientific << std::setprecision(2) << std::setw(11) << r.vsRef->maxAbs << std::setw(7)
                      << (r.vsRef->top1 == r.vsRef->refTop1 ? "match" : "DIFF") << std::fixed;
        }
        std::cout << '\n';
    }

    // What each memory path costs on top of pinned, per engine. This is the host<->device bottleneck, measured.
    bool printedHeader = false;
    for (const auto& r : runs)
    {
        const RunResult* pinned = find(r.precision, trtp::IoMode::kPinned);
        if (r.io == trtp::IoMode::kPinned || pinned == nullptr) continue;
        if (!printedHeader)
        {
            std::cout << "\nI/O path cost vs pinned (end-to-end mean):\n";
            printedHeader = true;
        }
        const double delta = r.e2e.mean - pinned->e2e.mean;
        std::cout << "  " << std::left << std::setw(6) << trtp::toString(r.precision) << std::setw(10)
                  << trtp::toString(r.io) << std::right << std::fixed << std::showpos << std::setprecision(3) << delta
                  << " ms (" << std::setprecision(1) << 100.0 * delta / pinned->e2e.mean << "%)" << std::noshowpos
                  << '\n';
    }

    // Cross-precision agreement does not need the PyTorch reference: compare reduced precision to TRT FP32.
    // Logits do not depend on the I/O path, so one I/O mode is enough.
    const trtp::IoMode io0 = runs.front().io;
    const RunResult* base = find(trtp::Precision::kFP32, io0);
    if (base != nullptr)
    {
        for (const auto& r : runs)
        {
            if (&r == base || r.io != io0) continue;
            const Accuracy a = compareLogits(r.logits, base->logits);
            std::cout << std::scientific << std::setprecision(2) << trtp::toString(r.precision)
                      << " vs TRT FP32: max|diff| " << a.maxAbs << std::fixed << std::setprecision(6) << ", cosine "
                      << a.cosine << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options opt = parseArgs(argc, argv);

        const std::size_t inputCount =
           static_cast<std::size_t>(opt.batch) * kInputPerImage;

        std::vector<float> input = loadFloats(opt.inputBin, inputCount);
        std::vector<float> ref;
        if (!input.empty())
        {
            // The reference logits are only meaningful for the exact tensor they were computed from.
            const std::size_t refCount =
                static_cast<std::size_t>(opt.batch) * kNumClasses;

        ref = loadFloats(opt.refBin, refCount);
            std::cout << "Input: " << opt.inputBin << (ref.empty() ? " (no reference logits -> accuracy check skipped)"
                                                                    : " + reference logits " + opt.refBin)
                      << '\n';
        }
        else
        {
           input = makeSyntheticFrame(opt.batch);
            std::cout << "Input: synthetic normalised frame (" << opt.inputBin
                      << " not found -> latency only, accuracy check skipped)\n";
        }

        std::vector<RunResult> runs;
        for (const trtp::Precision p : opt.precisions)
        {
            for (const trtp::IoMode io : opt.ioModes)
            {
                // --rebuild applies once per precision; later I/O modes reuse the freshly cached engine.
                const bool rebuild = opt.rebuild && io == opt.ioModes.front();
                runs.push_back(benchmark(p, io, rebuild, opt, input, ref));
            }
        }
        printSummary(runs);

        const bool allPassed = std::all_of(runs.begin(), runs.end(), [](const RunResult& r) { return r.gatePassed; });
        return allPassed ? 0 : 2;
    }
    catch (const std::exception& e)
    {
        std::cerr << "fatal: " << e.what() << '\n';
        return 1;
    }
}

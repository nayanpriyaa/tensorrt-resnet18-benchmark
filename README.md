# ResNet-18 → ONNX → TensorRT (C++) on an RTX 3050 laptop

PyTorch ResNet-18 exported to ONNX, compiled into a TensorRT engine through the C++ API, and benchmarked on Windows with an NVIDIA RTX 3050 6GB Laptop GPU. The current implementation supports configurable static batch sizes and compares TensorRT logits against PyTorch FP32 reference logits.

1. **Precision:** strict FP32 vs TF32 vs FP16. Each run reports latency and accuracy against the PyTorch FP32 logits.
2. **Host↔device memory path:** pinned vs pageable vs zero-copy host memory. This measures what crossing PCIe actually costs on a discrete GPU, instead of asserting it.

| File | Role |
|---|---|
| `export_to_onnx.py` | ResNet-18 → static `[N,3,224,224]` ONNX (`--batch N`), ONNX Runtime parity check, dumps `input.bin` + `ref_output.bin` |
| `TrtPipeline.h/.cpp` | ONNX parse, engine build + on-disk cache, pinned / pageable / zero-copy I/O, `enqueueV3` inference |
| `main.cpp` | Batch-aware warm-up and inference, latency percentiles, CUDA-event H2D/compute/D2H breakdown, effective PCIe GB/s, batch-aware top-1 and numerical accuracy gates, images/s throughput |
| `CMakeLists.txt` | Links `CUDA::cudart`, `nvinfer`, `nvonnxparser`; Windows (MSVC) and Linux/WSL2 |

## Hardware context

- **GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (sm_86), connected to the CPU over PCIe.
- **Why the I/O experiment exists:** the GPU reads its own GDDR6 at roughly 10× the speed of the PCIe link. Every byte that has to come from host RAM is the expensive kind.
- **Edge relevance:** Jetson Orin's GPU is the same Ampere generation (sm_87), so the precision results and Tensor Core behaviour carry over. The memory system does not carry over (PCIe + GDDR6 here, shared LPDDR5 on Jetson). That is exactly what the `--io` experiment exposes, and why zero-copy flips from a loss here to a win on Jetson.
- **Robot framing:** TurtleBot 3/4 carry a Raspberry Pi with no CUDA GPU. In the standard setup, the robot streams sensor topics over Wi-Fi via ROS 2 to a "Remote PC", which is this laptop. Moving a camera frame over Wi-Fi usually costs an order of magnitude more than this inference, so at the system level the bottleneck is transport, not the engine.

## Pick a TensorRT version

Both versions compile from the same source.

| | TensorRT 10.x | TensorRT 11.x |
|---|---|---|
| CUDA | 12.x | 13.x (driver R580+); the C++ zip/tar packages target CUDA 13 |
| FP16 | `BuilderFlag::kFP16` (weak typing) | Removed. FP16 comes from the graph: `python export_to_onnx.py --fp16-onnx` (needs `pip install "nvidia-modelopt[onnx]"`) |
| Verdict | Fewest moving parts. Start here. | Current API. Rebuild against it afterwards; "same code, both typing models" is a good interview point |

## Build and run: Windows (native)

Prerequisites:
- Visual Studio 2022 with the "Desktop development with C++" workload
- CUDA Toolkit
- TensorRT zip, extracted to e.g. `C:\TensorRT-10.x.y.z`
- CMake ≥ 3.18
- Python 3

```powershell
pip install torch torchvision onnx onnxscript onnxruntime
python export_to_onnx.py

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DTENSORRT_ROOT=C:/TensorRT-10.x.y.z
cmake --build build --config Release

# TensorRT DLLs must be on PATH (bin\ on recent releases, lib\ on some older 10.x zips)
$env:PATH = "C:\TensorRT-10.x.y.z\bin;C:\TensorRT-10.x.y.z\lib;" + $env:PATH
.\build\Release\trt_resnet18.exe --io all     # run from this folder: it reads resnet18.onnx / *.bin from here
```

## Build and run: Linux or WSL2

- **WSL2:** install only the Windows NVIDIA driver. Inside WSL, install `cuda-toolkit-12-x` from NVIDIA's WSL-Ubuntu repo. Never install the `cuda`, `cuda-12-x` or `cuda-drivers` meta-packages there: they try to install a Linux driver. `nvidia-smi` lives at `/usr/lib/wsl/lib/nvidia-smi`.
- **Native Linux laptop:** keep the dGPU powered. Use `sudo prime-select on-demand` or `nvidia`, not `intel`, which unloads the NVIDIA driver so CUDA sees no device. CUDA programs need no PRIME offload variables.
- **TensorRT:** `sudo apt install tensorrt-dev` from NVIDIA's CUDA apt repo, or a tar build matching your CUDA version (then pass `-DTENSORRT_ROOT=...`).

```bash
pip install torch torchvision onnx onnxscript onnxruntime
python export_to_onnx.py
cmake -S . -B build && cmake --build build -j
./build/trt_resnet18 --io all
./build/trt_resnet18 --help
```

The first run builds and caches engines; later runs deserialize them. The current engine cache filename includes precision and batch size (for example, `resnet18_fp16_b1.engine` and `resnet18_fp16_b2.engine`). An engine is tied to the GPU architecture and exact TensorRT version, and a cache older than its ONNX is rebuilt automatically.

## Benchmarking a laptop GPU without fooling yourself

- **Power:** plug in AC and select the vendor's performance mode. On battery the GPU power limit is cut hard, and the numbers mean nothing.
- **Watch clocks and the PCIe link while it runs.** Laptop links drop to Gen1 when idle, and the GPU throttles on temperature and power:
  ```bash
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv -l 1
  ```
  Write down the link generation and width under load: they set the ceiling for the H2D GB/s column.
- **Optional:** lock clocks with `nvidia-smi --lock-gpu-clocks=<MHz>,<MHz>` (admin/root; some laptop SKUs refuse). Undo with `--reset-gpu-clocks`.
- **OS overhead:** native Windows and WSL2 both submit work through the Windows WDDM driver. Expect higher per-launch overhead and more jitter than native Linux. Report p99, not only the mean, and state the OS next to every number.
- **Repeatability:** run 3 times and report the median.
- **Cross-check against NVIDIA's tool** (`trtexec` ships in TensorRT's `bin`):
  ```bash
  trtexec --onnx=resnet18.onnx --fp16 --noDataTransfers --useCudaGraph       # TensorRT 10.x
  trtexec --onnx=resnet18_fp16.onnx                                          # TensorRT 11.x (--fp16 removed;
                                                                             # no transfers + CUDA graphs are default)
  ```
  `trtexec` times the GPU work only, with CUDA graphs and no transfers. Expect it to beat this harness's end-to-end mean; the gap is your copy + launch overhead.

## Observed results (Windows, TensorRT 10.15.1)

Measurements are from the Windows native build on an NVIDIA GeForce RTX 3050 6GB Laptop GPU (sm_86), using FP16, pinned host I/O, 100 warmup iterations, and 500 measured iterations. Batch-specific ONNX and binary files are stored under `artifacts\batch1` and `artifacts\batch2`.

These are repeat-run measurements, not a formal multi-run statistical benchmark. The first batch-1 run was unusually noisy, so the table below uses the subsequent repeat runs.

| Precision | I/O | Batch | Mean batch ms | p50 ms | p90 ms | p99 ms | H2D ms | H2D GB/s | Compute ms | Throughput (images/s) | Accuracy gate | max|Δ| |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| FP16 | pinned | 1 | 0.507 | 0.504 | 0.546 | 0.612 | 0.078 | 7.7 | 0.441 | 1973.3 | PASS (1/1 top-1) | 1.44e-2 |
| FP16 | pinned | 2 | 0.808 | 0.795 | 0.843 | 0.891 | 0.124 | 9.7 | 0.684 | 2475.2 | PASS (2/2 top-1) | 1.27e-2 |

Additional accuracy details:
- Batch 1: mean absolute difference `3.13e-3`, cosine similarity `0.999997`.
- Batch 2: mean absolute difference `2.68e-3`, cosine similarity `0.999998`.
- Both batch sizes passed the configured accuracy gate.

Interpretation:
- Batch 1 has lower single-inference latency in these runs.
- Batch 2 provides approximately 25.4% higher throughput than batch 1, while mean batch latency is approximately 59.4% higher.
- Throughput is reported in **images/s**, not batches/s.
- The first batch-1 run (mean 0.761 ms, p99 1.462 ms) was an outlier compared with its repeat (mean 0.507 ms, p99 0.612 ms); it is excluded from the repeat-run comparison.
- The batch-2 repeat had a maximum latency of 1.495 ms and p99 of 0.891 ms. More repetitions are needed before claiming robust stability.

## Design decisions

- **Static batch size per exported ONNX.** The exporter accepts `--batch N` and emits a fixed `[N,3,224,224]` input. Each batch size uses its own ONNX, reference logits, and TensorRT engine cache.
- **TF32 is its own mode.** TensorRT enables TF32 by default on sm_80+, so an unqualified "FP32" engine on any RTX 30-series card is really TF32. `kFP32` clears the flag to give a true baseline.
- **FP32 I/O at every precision.** Host buffers and transfer sizes are identical across precisions, so the comparison isolates compute.
- **A fair I/O comparison.** The frame is written once, outside the timed loop: into the pipeline's pinned/mapped buffer, or into a plain `std::vector` for pageable. Only the memory type changes between runs.
- **Every copy runs on one non-blocking stream, with one sync per frame.** A plain `cudaMemcpy` would run on the legacy default stream, which does not order against a non-blocking stream. From pageable memory it can return before the DMA lands, so the kernels could read a half-written input.

## Reproduce the current Windows benchmarks

Run these from the project root after exporting the batch-specific ONNX and binary files:

```bat
build-win\Release\trt_resnet18.exe --batch 1 --precision fp16 --io pinned --onnx artifacts\batch1\resnet18.onnx --input artifacts\batch1\input.bin --ref artifacts\batch1\ref_output.bin --warmup 100 --iters 500

build-win\Release\trt_resnet18.exe --batch 2 --precision fp16 --io pinned --onnx artifacts\batch2\resnet18.onnx --input artifacts\batch2\input.bin --ref artifacts\batch2\ref_output.bin --warmup 100 --iters 500
```

The `--batch` value must match the ONNX batch dimension and the sizes of both binary files. Current generated files:
- `artifacts\batch1\input.bin` (602,112 bytes), `ref_output.bin` (4,000 bytes), `resnet18.onnx`
- `artifacts\batch2\input.bin` (1,204,224 bytes), `ref_output.bin` (8,000 bytes), `resnet18.onnx`

## Next steps

- **CUDA Graphs.** Capture H2D + `enqueueV3` + D2H once and replay it. Under WDDM, launch overhead is a large share of a batch-1 ResNet-18, so this is the highest-value optimization on this laptop.
- **GPU preprocessing.** Do resize, HWC→CHW and normalization in a CUDA kernel, or send `uint8` and fold normalization into the graph. Sending `uint8` also cuts H2D bytes by 4×, which the GB/s column will show directly.
- **Batch sweep.** Use a dynamic-batch ONNX and an optimization profile, then plot latency and throughput against batch size per precision. FP16's advantage grows with batch size as kernels become compute-bound.
- **INT8.** Use ModelOpt post-training quantization (explicit Q/DQ). Implicit calibration is deprecated in TensorRT 10 and removed in 11.
- **Pipelining.** Use two contexts and two streams with double-buffered pinned I/O, so the copy for frame N+1 overlaps compute for frame N.
- **ROS 2 node.** Subscribe to the TurtleBot's compressed image topic and publish detections, then measure end-to-end latency from camera to result.

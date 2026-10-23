#!/usr/bin/env python3
"""
Export torchvision ResNet-18 (ImageNet-1k) to ONNX for TensorRT.

Artifacts written to --out-dir:
  resnet18.onnx        FP32 graph, static shape input[1,3,224,224] -> logits[1,1000]
  input.bin            the exact NCHW float32 tensor used for verification (raw, little-endian)
  ref_output.bin       PyTorch FP32 logits for input.bin -> ground truth for the C++ accuracy check
  resnet18_fp16.onnx   (--fp16-onnx only) ModelOpt AutoCast FP16 graph with FP32 I/O.
                       Required on TensorRT >= 11, which removed BuilderFlag::kFP16 (weak typing).
                       Not needed on TensorRT 10.x, where the C++ builder flag still works.

Usage:
  pip install torch torchvision onnx onnxscript onnxruntime
  python export_to_onnx.py
  python export_to_onnx.py --fp16-onnx          # additionally: pip install "nvidia-modelopt[onnx]"
"""

from __future__ import annotations

import argparse
import inspect
import pathlib
import sys

import numpy as np
import torch
import torchvision

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
BATCH_SIZE = 1
INPUT_SHAPE = (BATCH_SIZE, 3, 224, 224)  # static batch=1: single camera stream, lets TRT specialise every tactic
NUM_CLASSES = 1000
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("."))
    p.add_argument("--opset", type=int, default=18, help="ONNX opset (18 is parsed by TensorRT 10.x and 11.x)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=1,
               help="batch size for inference")
    p.add_argument("--fp16-onnx", action="store_true",
                   help="also emit resnet18_fp16.onnx via ModelOpt AutoCast (needed for TensorRT >= 11)")
    p.add_argument("--legacy-exporter", action="store_true",
                   help="use the deprecated TorchScript exporter (dynamo=False) instead of torch.export")
    p.add_argument("--random-weights", action="store_true",
                   help="skip the pretrained download (offline/CI smoke test; accuracy numbers become meaningless)")
    return p.parse_args()


def load_model(random_weights: bool) -> torch.nn.Module:
    weights = None if random_weights else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    # eval() is not optional: in train mode BatchNorm normalises with batch statistics and the exported
    # graph would differ. In eval mode BN is an affine op on running stats, which TensorRT folds into
    # the preceding Conv's weights (Conv+BN+ReLU becomes one fused kernel).
    return model.eval()


def make_input(seed: int, batch_size: int) -> torch.Tensor:
    """Image-like tensor: uniform [0,1] pixels normalised with ImageNet stats.

    Deliberately not torch.randn: FP16 error depends on activation magnitudes, so the verification
    input must sit in the same value range as a real preprocessed camera frame (~[-2.1, 2.6]).
    """
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(input_shape(batch_size), generator=g, dtype=torch.float32)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return ((img - mean) / std).contiguous()


def export_onnx(model: torch.nn.Module, x: torch.Tensor, path: pathlib.Path, opset: int, legacy: bool) -> None:
    kwargs: dict = dict(input_names=[INPUT_NAME], output_names=[OUTPUT_NAME], opset_version=opset)
    params = inspect.signature(torch.onnx.export).parameters

    if legacy:
        kwargs.update(dynamo=False, do_constant_folding=True)
    else:
        # torch.export-based exporter (default since PyTorch 2.9). No dynamic_shapes -> fully static graph.
        kwargs["dynamo"] = True
        if "external_data" in params:
            # Default is True, which splits weights into resnet18.onnx.data. 45 MB is far below the
            # 2 GB protobuf limit, so keep a single self-contained file that is trivial to move between machines.
            kwargs["external_data"] = False

    torch.onnx.export(model, (x,), str(path), **kwargs)


def check_onnx(path: pathlib.Path, batch_size: int) -> None:
    import onnx

    onnx.checker.check_model(str(path), full_check=True)
    m = onnx.load(str(path))
    opset = next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))

    def dims(vi) -> list:
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in vi.type.tensor_type.shape.dim]

    ins = {i.name: dims(i) for i in m.graph.input}
    outs = {o.name: dims(o) for o in m.graph.output}
    print(f"[onnx] {path.name}: opset={opset} nodes={len(m.graph.node)} inputs={ins} outputs={outs}")

    expected_input = list(input_shape(batch_size))

    if ins.get(INPUT_NAME) != expected_input:
       raise RuntimeError(
        f"expected static input {expected_input}, got {ins}"
    )
    expected_output = [batch_size, NUM_CLASSES]

    if outs.get(OUTPUT_NAME) != expected_output:
       raise RuntimeError(
        f"expected static output {expected_output}, got {outs}"
    )


def run_ort(path: pathlib.Path, x: np.ndarray) -> np.ndarray | None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ort] onnxruntime not installed -> skipping numerical check")
        return None
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]


def compare(tag: str, out: np.ndarray, ref: np.ndarray, atol: float, require_top1: bool = True) -> None:
    max_abs = float(np.max(np.abs(out - ref)))
    cos = float(np.dot(out.ravel(), ref.ravel()) / (np.linalg.norm(out) * np.linalg.norm(ref)))
    top1 = int(out.argmax()) == int(ref.argmax())
    # A synthetic input can have near-tied top-2 logits, so a reduced-precision argmax flip is reported,
    # not treated as failure; max|diff| is the real gate there.
    ok = max_abs <= atol and (top1 or not require_top1)
    status = "OK" if ok else "FAIL"
    print(f"[{tag}] vs PyTorch FP32: max|diff|={max_abs:.3e} cos={cos:.6f} top1_match={top1} -> {status}")
    if status == "FAIL":
        raise RuntimeError(f"{tag} output diverges from PyTorch reference (atol={atol})")


def autocast_fp16(src: pathlib.Path, dst: pathlib.Path, calib_npz: pathlib.Path) -> None:
    try:
        import onnx
        from modelopt.onnx.autocast import convert_to_mixed_precision
    except ImportError as e:
        raise SystemExit(f"--fp16-onnx needs ModelOpt: pip install 'nvidia-modelopt[onnx]' ({e})")

    # keep_io_types=True: graph I/O stays FP32, so the C++ host buffers and the H2D copy are identical
    # for both precisions and the benchmark isolates compute. AutoCast keeps numerically sensitive nodes
    # (|activation| > data_max, calibrated on our image-like input) in FP32 and inserts Cast ops.
    model = convert_to_mixed_precision(
        onnx_path=str(src),
        low_precision_type="fp16",
        keep_io_types=True,
        calibration_data=str(calib_npz),
    )
    onnx.save(model, str(dst))

def input_shape(batch_size: int) -> tuple[int, int, int, int]:
    return (batch_size, 3, 224, 224)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    onnx_path = args.out_dir / "resnet18.onnx"
    model = load_model(args.random_weights)
    x = make_input(args.seed, args.batch)
    with torch.inference_mode():
        ref = model(x).numpy().astype(np.float32)

    print(f"[torch] {torch.__version__} | torchvision {torchvision.__version__} | exporting -> {onnx_path}")
    export_onnx(model, x, onnx_path, args.opset, args.legacy_exporter)
    check_onnx(onnx_path, args.batch)

    x_np = x.numpy().astype(np.float32)
    ort_out = run_ort(onnx_path, x_np)
    if ort_out is not None:
        compare("ort-fp32", ort_out, ref, atol=1e-3)

    # Raw little-endian float32 blobs: zero-dependency to read from C++ (std::ifstream::read).
    x_np.tofile(args.out_dir / "input.bin")
    ref.tofile(args.out_dir / "ref_output.bin")
    top5 = np.argsort(ref[0])[::-1][:5]
    print(f"[ref] wrote input.bin ({x_np.nbytes} B) + ref_output.bin ({ref.nbytes} B); top-5 classes {top5.tolist()}")

    if args.fp16_onnx:
        calib = args.out_dir / "calib.npz"
        np.savez(calib, **{INPUT_NAME: x_np})
        fp16_path = args.out_dir / "resnet18_fp16.onnx"
        autocast_fp16(onnx_path, fp16_path, calib)
        check_onnx(fp16_path, args.batch)
        try:
            out16 = run_ort(fp16_path, x_np)
        except Exception as e:  # ORT's CPU EP lacks FP16 kernels for some ops on some builds
            print(f"[ort-fp16] CPU check skipped: {e}")
            out16 = None
        if out16 is not None:
            # FP16 tolerance: 10-bit mantissa accumulated over 18 conv layers; logits are O(10).
            compare("ort-fp16", out16, ref, atol=5e-2, require_top1=False)
        print(f"[fp16] wrote {fp16_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Step 2: Post-Training Static Quantization (PTQ), FP32 -> INT8.

Pipeline: load FP32 model -> fuse Conv/BN/ReLU -> set backend + qconfig ->
insert observers (prepare) -> calibrate on a training subset -> convert to INT8
-> evaluate -> save (TorchScript for deployment + state_dict).

Usage:
    python src/ptq.py --calib-images 2000 --backend x86
    # bonus: quantize the structured-pruned model (pruning + INT8 combined)
    python src/ptq.py --input models/structured_pruned_resnet18.pth \
                      --output models/structured_pruned_ptq_int8_resnet18.pt
"""
import argparse
import copy
import time

import torch
import torch.ao.quantization as tq

import common as C
from calibration import calibrate, get_calibration_loader


def build_ptq_model(fp32_model, calib_loader, backend="x86"):
    model = copy.deepcopy(fp32_model).cpu().eval()

    # 1) Fuse Conv+BN+ReLU / Conv+BN (BN gets folded into the conv weights)
    model.fuse_model(qat=False)

    # 2) Backend + quantization config
    #    x86/fbgemm: per-channel symmetric int8 weights, histogram observer for activations
    torch.backends.quantized.engine = backend
    model.qconfig = tq.get_default_qconfig(backend)
    print(f"  backend: {backend}\n  qconfig: {model.qconfig}")

    # 3) Insert observers
    tq.prepare(model, inplace=True)

    # 4) Calibration
    calibrate(model, calib_loader)

    # 5) Convert observed FP32 model to INT8
    tq.convert(model, inplace=True)
    return model


def save_quantized(model, path, example):
    """TorchScript is used for deployment: the INT8 graph can be loaded with
    torch.jit.load without rebuilding/re-preparing the Python model."""
    traced = torch.jit.trace(model, example)
    traced = torch.jit.freeze(traced)
    torch.jit.save(traced, path)
    torch.save(model.state_dict(), path.replace(".pt", "_state_dict.pth"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-images", type=int, default=2000)
    ap.add_argument("--backend", default="x86", choices=["x86", "fbgemm", "qnnpack"])
    ap.add_argument("--input", default=C.FP32_PATH, help="FP32 checkpoint to quantize")
    ap.add_argument("--output", default=C.PTQ_PATH, help="where to save the INT8 TorchScript model")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    C.set_seed(0)
    C.ensure_dirs()
    fp32 = C.load_fp32_model(args.input)
    print(f"Input model: {args.input}")
    _, test_loader = C.get_loaders(fake=args.fake)
    calib_loader = get_calibration_loader(args.calib_images, fake=args.fake)

    print("FP32 accuracy ...")
    fp32_acc = C.evaluate(fp32, test_loader)
    print(f"  FP32 test accuracy: {fp32_acc:.2f}%")

    print("Building PTQ INT8 model ...")
    t0 = time.time()
    qmodel = build_ptq_model(fp32, calib_loader, args.backend)
    print(f"  done in {time.time() - t0:.1f}s")
    print(qmodel.layer1[0])  # show quantized modules (useful for the report)

    acc = C.evaluate(qmodel, test_loader)
    print(f"  PTQ INT8 test accuracy: {acc:.2f}%  (change {acc - fp32_acc:+.2f} pts)")

    save_quantized(qmodel, args.output, torch.randn(1, 3, 32, 32))
    print(f"Saved PTQ model -> {args.output}  ({C.file_size_mb(args.output):.2f} MB, "
          f"FP32 input was {C.file_size_mb(args.input):.2f} MB)")

    # Deployment check: reload the saved file and re-test
    reloaded = torch.jit.load(args.output)
    print(f"  reloaded TorchScript accuracy: {C.evaluate(reloaded, test_loader):.2f}%")


if __name__ == "__main__":
    main()

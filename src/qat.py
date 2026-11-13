"""
Step 3: Quantization-Aware Training (QAT), FP32 -> fine-tuned INT8.

Pipeline: load pretrained FP32 model -> fuse (QAT-style, BN kept) -> QAT qconfig
-> prepare_qat (inserts FakeQuantize modules) -> fine-tune -> freeze observers
and BN statistics late in training -> convert to INT8 -> evaluate -> save.

Usage:
    python src/qat.py --epochs 5 --lr 1e-3
"""
import argparse
import copy

import torch
import torch.ao.quantization as tq

import common as C
from ptq import save_quantized


def prepare_qat_model(fp32_model, backend="x86"):
    model = copy.deepcopy(fp32_model).cpu()
    model.train()
    # QAT fusion keeps BatchNorm inside ConvBn(ReLU)2d so it still trains
    model.fuse_model(qat=True)
    torch.backends.quantized.engine = backend
    # version=0: standard FakeQuantize + MovingAverage observers. (The default
    # version=1 uses a fused CUDA/CPU op that cannot be exported to ONNX.)
    model.qconfig = tq.get_default_qat_qconfig(backend, version=0)
    # Inserts FakeQuantize (quantize -> dequantize) on weights and activations
    tq.prepare_qat(model, inplace=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--backend", default="x86", choices=["x86", "fbgemm", "qnnpack"])
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    C.set_seed(0)
    C.ensure_dirs()
    device = C.get_device()
    train_loader, test_loader = C.get_loaders(args.batch_size, args.fake)

    fp32 = C.load_fp32_model()
    print(f"FP32 test accuracy: {C.evaluate(fp32, test_loader):.2f}%")

    model = prepare_qat_model(fp32, args.backend)
    n_fq = sum(isinstance(m, tq.FakeQuantize) for m in model.modules())
    print(f"Inserted {n_fq} FakeQuantize modules")
    print(model.layer1[0].conv1)  # shows ConvBnReLU2d with weight_fake_quant

    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    # Schedule: freeze BN running stats and quantizer ranges near the end so
    # the last epochs fine-tune against fixed quantization parameters.
    freeze_bn_epoch = max(1, int(args.epochs * 0.6))
    freeze_obs_epoch = max(1, int(args.epochs * 0.8))

    for ep in range(args.epochs):
        if ep == freeze_bn_epoch:
            model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
            print("  froze BatchNorm statistics")
        if ep == freeze_obs_epoch:
            model.apply(tq.disable_observer)
            print("  froze quantizer observers (scale / zero-point)")

        loss, tr_acc = C.train_one_epoch(model, train_loader, opt, device, sched, log_every=0)

        # Monitoring only: evaluate the real INT8 model produced at this point (CPU)
        int8 = tq.convert(copy.deepcopy(model).cpu().eval(), inplace=False)
        acc = C.evaluate(int8, test_loader)
        print(f"Epoch {ep + 1}/{args.epochs}  loss {loss:.4f}  train(fake-quant) {tr_acc:.2f}%  "
              f"test INT8 {acc:.2f}%")

    # Final weights are used (fixed schedule, no selection on the test set)
    model.cpu().eval()
    torch.save(model.state_dict(), C.QAT_PATH.replace(".pt", "_qat_prepared_state_dict.pth"))

    int8 = tq.convert(model, inplace=False)
    acc = C.evaluate(int8, test_loader)
    print(f"QAT INT8 test accuracy: {acc:.2f}%")

    save_quantized(int8, C.QAT_PATH, torch.randn(1, 3, 32, 32))
    print(f"Saved QAT model -> {C.QAT_PATH}  ({C.file_size_mb(C.QAT_PATH):.2f} MB)")
    reloaded = torch.jit.load(C.QAT_PATH)
    print(f"  reloaded TorchScript accuracy: {C.evaluate(reloaded, test_loader):.2f}%")


if __name__ == "__main__":
    main()

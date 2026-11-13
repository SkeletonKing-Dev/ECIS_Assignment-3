"""
PTQ calibration: feeds a fixed random subset of the CIFAR-10 training set
(no augmentation) through a model that has observers inserted, so the
observers record activation ranges for INT8 scale / zero-point.

Used by ptq.py. Can also be run on its own to inspect the calibration
result (scale / zero-point of some activation observers):
    python src/calibration.py --calib-images 2000
"""
import argparse
import copy

import torch
import torch.ao.quantization as tq

import common as C


def get_calibration_loader(num_images=2000, batch_size=64, fake=False, seed=0):
    return C.get_calibration_loader(num_images, batch_size, fake, seed)


@torch.no_grad()
def calibrate(model, loader):
    """Run data through the observed model (eval mode, no gradients)."""
    model.eval()
    n_batches = 0
    for x, _ in loader:
        model(x)
        n_batches += 1
    print(f"  calibrated on {len(loader.dataset)} images ({n_batches} batches)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-images", type=int, default=2000)
    ap.add_argument("--backend", default="x86")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    model = copy.deepcopy(C.load_fp32_model()).eval()
    model.fuse_model(qat=False)
    torch.backends.quantized.engine = args.backend
    model.qconfig = tq.get_default_qconfig(args.backend)
    tq.prepare(model, inplace=True)
    calibrate(model, get_calibration_loader(args.calib_images, fake=args.fake))

    print("Activation quantization parameters after calibration (first observers):")
    shown = 0
    for name, m in model.named_modules():
        if isinstance(m, tq.HistogramObserver):
            scale, zp = m.calculate_qparams()
            print(f"  {name:<45} min {m.min_val.item():8.4f}  max {m.max_val.item():8.4f}  "
                  f"scale {scale.item():.5f}  zero_point {zp.item()}")
            shown += 1
            if shown == 8:
                break


if __name__ == "__main__":
    main()

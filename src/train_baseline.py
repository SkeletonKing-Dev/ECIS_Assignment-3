"""
Step 1: train the baseline FP32 ResNet-18 on CIFAR-10 and save the checkpoint.

Usage:
    python src/train_baseline.py --epochs 30
"""
import argparse
import time

import torch

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--fake", action="store_true", help="random data, for smoke tests only")
    args = ap.parse_args()

    C.set_seed(42)
    C.ensure_dirs()
    device = C.get_device()
    print(f"Device: {device}")

    train_loader, test_loader = C.get_loaders(args.batch_size, args.fake)
    model = C.ResNet18().to(device)
    opt, sched = C.make_optimizer(model, args.lr, args.epochs, len(train_loader))

    # Fixed schedule (One-Cycle); the final weights are saved. No checkpoint is
    # selected on the test set, so the test accuracy stays an unbiased estimate.
    t0 = time.time()
    for ep in range(args.epochs):
        loss, tr_acc = C.train_one_epoch(model, train_loader, opt, device, sched, log_every=0)
        acc = C.evaluate(model, test_loader, device)
        print(f"Epoch {ep + 1:3d}/{args.epochs}  loss {loss:.4f}  train {tr_acc:.2f}%  test {acc:.2f}%")

    C.save_fp32_model(model, C.FP32_PATH, test_acc=acc, epoch=args.epochs)
    print(f"Training time: {(time.time() - t0) / 60:.1f} min")
    print(f"Final test accuracy: {acc:.2f}%  -> saved to {C.FP32_PATH}")


if __name__ == "__main__":
    main()

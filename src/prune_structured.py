"""
Q2 - Structured channel pruning with architecture reconstruction.

Which channels: the intermediate channels of every ResNet BasicBlock
(output of conv1 -> bn1 -> relu -> input of conv2).

Why these: the block output channels are tied to the residual addition
(out + shortcut(x)), so removing them would force the same channels to be
removed in the shortcut, the previous blocks and the next blocks. The middle
channels are only connected to conv1 (as output), bn1 and conv2 (as input),
so pruning them and slicing all three tensors consistently keeps every
tensor dimension valid.

Importance score per channel: L1 norm of the conv1 filter * |BN gamma|
(the filter magnitude and its BN scaling factor together decide how much the
channel contributes). The lowest scoring channels are removed.

The result is a genuinely smaller dense network (new Conv/BN layers with fewer
channels), not a masked one, so it runs faster without sparse kernels.

Usage:
    python src/prune_structured.py --ratio 0.5 --ft-epochs 10
"""
import argparse

import torch

import common as C


def channel_importance(block):
    w = block.conv1.weight.detach()                    # [mid, in, 3, 3]
    l1 = w.abs().sum(dim=(1, 2, 3))
    gamma = block.bn1.weight.detach().abs()
    return l1 * gamma


def select_channels(model, ratio, min_channels=8):
    """Returns the indices to keep for each block (per-block ratio)."""
    keep = []
    for b in model.blocks():
        score = channel_importance(b)
        n = score.numel()
        n_keep = max(min_channels, int(round(n * (1 - ratio))))
        idx = torch.argsort(score, descending=True)[:n_keep]
        keep.append(torch.sort(idx).values)
    return keep


@torch.no_grad()
def rebuild_pruned(model, keep):
    """Builds a new, smaller ResNet18 and copies the surviving weights."""
    cfg = [len(k) for k in keep]
    new = C.ResNet18(channel_cfg=cfg)
    old_sd = model.state_dict()
    new_sd = new.state_dict()

    # Start with all tensors that are unchanged in shape
    for k in new_sd:
        if new_sd[k].shape == old_sd[k].shape:
            new_sd[k].copy_(old_sd[k])

    # Slice the tensors connected to the pruned channels of each block
    names = [f"layer{s}.{i}" for s in range(1, 5) for i in range(2)]
    for name, idx in zip(names, keep):
        new_sd[f"{name}.conv1.weight"].copy_(old_sd[f"{name}.conv1.weight"][idx])        # output channels
        for p in ("weight", "bias", "running_mean", "running_var"):
            new_sd[f"{name}.bn1.{p}"].copy_(old_sd[f"{name}.bn1.{p}"][idx])
        new_sd[f"{name}.conv2.weight"].copy_(old_sd[f"{name}.conv2.weight"][:, idx])     # input channels
    new.load_state_dict(new_sd)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.5, help="fraction of mid channels removed per block")
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    C.set_seed(0)
    C.ensure_dirs()
    device = C.get_device()
    train_loader, test_loader = C.get_loaders(fake=args.fake)

    model = C.load_fp32_model().eval()
    base_acc = C.evaluate(model, test_loader)
    base_params, _ = C.count_params(model)
    base_macs = C.count_macs(model)
    print(f"Original: acc {base_acc:.2f}%  params {base_params:,}  MACs {base_macs / 1e6:.1f}M")

    keep = select_channels(model, args.ratio)
    pruned = rebuild_pruned(model, keep)
    print("Channel config (mid channels per block):")
    for b_old, b_new in zip(model.blocks(), pruned.blocks()):
        print(f"  {b_old.conv1.out_channels:4d} -> {b_new.conv1.out_channels:4d}")

    # Sanity check: forward pass works with the new shapes
    out = pruned(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)

    params, _ = C.count_params(pruned)
    macs = C.count_macs(pruned)
    acc0 = C.evaluate(pruned, test_loader)
    print(f"Pruned (before fine-tune): acc {acc0:.2f}%  params {params:,} "
          f"({100 * params / base_params:.1f}%)  MACs {macs / 1e6:.1f}M ({100 * macs / base_macs:.1f}%)")

    acc = C.finetune(pruned, train_loader, test_loader, device, args.ft_epochs, args.lr, "structured")
    pruned.cpu()
    print(f"Pruned (after fine-tune): acc {acc:.2f}%")

    # The checkpoint stores channel_cfg so the smaller architecture can be rebuilt
    C.save_fp32_model(pruned, C.STRUCT_PATH, test_acc=acc, prune_ratio=args.ratio)
    print(f"Saved -> {C.STRUCT_PATH} ({C.file_size_mb(C.STRUCT_PATH):.2f} MB, "
          f"original {C.file_size_mb(C.FP32_PATH):.2f} MB)")
    reloaded = C.load_fp32_model(C.STRUCT_PATH)
    print(f"Reloaded accuracy: {C.evaluate(reloaded, test_loader):.2f}%")


if __name__ == "__main__":
    main()

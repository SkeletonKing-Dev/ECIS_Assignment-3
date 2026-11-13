"""
Q2 - Unstructured pruning: iterative global magnitude (L1) pruning of all Conv
and Linear weights, with fine-tuning after each iteration, then the masks are
made permanent and the model is exported.

Usage:
    python src/prune_unstructured.py --target-sparsity 0.8 --iterations 4 --ft-epochs 2 --final-ft-epochs 5
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import common as C


def prunable_params(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def sparsity_report(model):
    total, zeros = 0, 0
    for m, name in prunable_params(model):
        w = getattr(m, name)
        total += w.numel()
        zeros += (w == 0).sum().item()
    return 100.0 * zeros / total


def per_layer_sparsity(model, path):
    """Global pruning chooses one threshold for all layers, so each layer ends
    up with a different sparsity. Saved to CSV for the report."""
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight
            rows.append((name, type(m).__name__, w.numel(), int((w == 0).sum()),
                         100.0 * (w == 0).float().mean().item()))
    import pandas as pd
    df = pd.DataFrame(rows, columns=["layer", "type", "weights", "zeros", "sparsity_%"])
    df.to_csv(path, index=False, float_format="%.2f")
    return df


def save_sparse(model, path):
    """Compressed checkpoint. Each pruned weight tensor is stored as
    (shape, 1-bit packed mask, non-zero values). At 80% sparsity this needs
    ~0.2*4 + 1/8 bytes per weight instead of 4 bytes (about 4x smaller).
    (COO format would be larger: it stores an int64 index per dimension.)"""
    sd = {}
    for k, v in model.state_dict().items():
        if v.dim() > 1 and (v == 0).float().mean() > 0.3:
            flat = v.flatten()
            mask = flat != 0
            sd[k] = {"shape": tuple(v.shape),
                     "mask": torch.from_numpy(np.packbits(mask.numpy())),
                     "values": flat[mask].clone()}
        else:
            sd[k] = v
    torch.save({"state_dict": sd, "channel_cfg": None, "format": "bitmask"}, path)


def load_sparse(path):
    ckpt = torch.load(path, map_location="cpu")
    sd = {}
    for k, v in ckpt["state_dict"].items():
        if isinstance(v, dict):
            n = int(np.prod(v["shape"]))
            mask = torch.from_numpy(np.unpackbits(v["mask"].numpy())[:n].astype(bool))
            flat = torch.zeros(n, dtype=v["values"].dtype)
            flat[mask] = v["values"]
            sd[k] = flat.view(v["shape"])
        else:
            sd[k] = v
    model = C.ResNet18()
    model.load_state_dict(sd)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-sparsity", type=float, default=0.8)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--ft-epochs", type=int, default=2, help="fine-tune epochs after each iteration")
    ap.add_argument("--final-ft-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    C.set_seed(0)
    C.ensure_dirs()
    device = C.get_device()
    train_loader, test_loader = C.get_loaders(fake=args.fake)

    model = C.load_fp32_model().to(device)
    base_acc = C.evaluate(model, test_loader, device)
    print(f"Original FP32 accuracy: {base_acc:.2f}%")

    # Each iteration removes the same fraction of the *remaining* weights so that
    # after N iterations total sparsity == target:  (1 - a)^N = 1 - target
    per_iter = 1 - (1 - args.target_sparsity) ** (1.0 / args.iterations)
    params = prunable_params(model)

    for it in range(1, args.iterations + 1):
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=per_iter)
        sp = sparsity_report(model)
        acc_before = C.evaluate(model, test_loader, device)
        print(f"Iteration {it}/{args.iterations}: sparsity {sp:.2f}%  acc after pruning {acc_before:.2f}%")
        epochs = args.final_ft_epochs if it == args.iterations else args.ft_epochs
        acc = C.finetune(model, train_loader, test_loader, device, epochs, args.lr, f"iter {it}")
        print(f"  -> after fine-tuning: {acc:.2f}%  (masks keep pruned weights at zero)")

    # Make pruning permanent: weight = weight_orig * weight_mask, drop the mask buffers
    for m, name in params:
        prune.remove(m, name)
    model.cpu()
    final_sp = sparsity_report(model)
    acc = C.evaluate(model, test_loader)
    total, nz = C.count_params(model)
    print(f"Final: sparsity {final_sp:.2f}%  non-zero params {nz:,}/{total:,}  accuracy {acc:.2f}%")

    df = per_layer_sparsity(model, os.path.join(C.RESULT_DIR, "unstructured_layer_sparsity.csv"))
    print("Per-layer sparsity (global threshold):")
    for _, r in df.iterrows():
        print(f"  {r['layer']:<22} {r['weights']:>9,d} weights  {r['sparsity_%']:6.2f}% zero")

    C.save_fp32_model(model, C.UNSTRUCT_PATH, test_acc=acc, sparsity=final_sp)
    save_sparse(model, C.UNSTRUCT_SPARSE_PATH)
    print(f"Saved dense  -> {C.UNSTRUCT_PATH} ({C.file_size_mb(C.UNSTRUCT_PATH):.2f} MB)")
    print(f"Saved sparse -> {C.UNSTRUCT_SPARSE_PATH} ({C.file_size_mb(C.UNSTRUCT_SPARSE_PATH):.2f} MB)")
    print(f"Sparse checkpoint reload accuracy: {C.evaluate(load_sparse(C.UNSTRUCT_SPARSE_PATH), test_loader):.2f}%")


if __name__ == "__main__":
    main()

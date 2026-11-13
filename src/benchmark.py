"""
Automated benchmark of all models on the full CIFAR-10 test set (CPU).

Measures: top-1 accuracy, checkpoint size, parameter / non-zero counts,
MACs, batch-1 latency (mean and p95), batched throughput.

Outputs (results/):
  benchmark_quantization.csv   FP32 vs PTQ-INT8 vs QAT-INT8           (Q1)
  benchmark_pruning.csv        FP32 vs unstructured vs structured     (Q2)
  benchmark_onnx.csv           the same models in ONNX Runtime        (deployment)
  benchmark_all.csv            everything in one table
  predictions_all_models.csv   per-image prediction of every model
  per_class_accuracy.csv       accuracy per class per model
  *.png                        comparison plots

Usage:
    python src/benchmark.py --threads 1
"""
import argparse
import os
import platform
import time

import numpy as np
import pandas as pd
import torch

import common as C
from infer import available_models, load_model

SIZE_FILES = {
    "fp32": C.FP32_PATH,
    "ptq": C.PTQ_PATH,
    "qat": C.QAT_PATH,
    "unstructured": C.UNSTRUCT_SPARSE_PATH,   # compressed (bitmask) checkpoint
    "structured": C.STRUCT_PATH,
    "structured_ptq": C.STRUCT_PTQ_PATH,
}
INT8_MODELS = ("ptq", "qat", "structured_ptq")
LABELS = {"fp32": "FP32 baseline", "ptq": "PTQ INT8", "qat": "QAT INT8",
          "unstructured": "Unstructured pruned", "structured": "Structured pruned",
          "structured_ptq": "Structured pruned + PTQ INT8 (bonus)"}


@torch.no_grad()
def run_testset(fn, loader):
    preds, n, t = [], 0, 0.0
    for x, _ in loader:
        t0 = time.perf_counter()
        out = fn(x)
        t += time.perf_counter() - t0
        preds.append(np.asarray(out).argmax(1))
        n += x.shape[0]
    return np.concatenate(preds), n / t


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def float_stats(name):
    if name in INT8_MODELS:
        # same architecture / MACs as the float model it was quantized from
        m = C.load_fp32_model(C.STRUCT_PATH if name == "structured_ptq" else C.FP32_PATH)
        total, nz = C.count_params(m)
        return total, nz, C.count_macs(m)
    m = load_model(name)
    total, nz = C.count_params(m)
    return total, nz, C.count_macs(m)


def bench_torch(name, loader, labels, args):
    model = load_model(name)
    fn = lambda x: model(x).numpy()
    preds, thr = run_testset(fn, loader)
    lat, p95 = C.measure_latency(lambda x: model(x), torch.randn(1, 3, 32, 32),
                                 args.warmup, args.runs)
    total, nz, macs = float_stats(name)
    return preds, {
        "model": LABELS[name], "key": name, "runtime": "PyTorch",
        "precision": "INT8" if name in INT8_MODELS else "FP32",
        "accuracy_%": 100 * (preds == labels).mean(),
        "size_mb": C.file_size_mb(SIZE_FILES[name]),
        "params": total, "nonzero_params": nz,
        "sparsity_%": 100 * (1 - nz / total), "macs_M": macs / 1e6,
        "latency_b1_ms": lat, "latency_b1_p95_ms": p95, "throughput_img_s": thr,
    }


def bench_onnx(name, loader, labels, args):
    from export_onnx import ONNX_PATHS
    from onnx_infer import make_session
    sess = make_session(name, args.threads)
    fn = lambda x: sess.run(None, {"input": x.numpy()})[0]
    preds, thr = run_testset(fn, loader)
    x1 = np.random.randn(1, 3, 32, 32).astype(np.float32)
    lat, p95 = C.measure_latency(lambda x: sess.run(None, {"input": x}), x1, args.warmup, args.runs)
    return preds, {
        "model": LABELS[name], "key": name, "runtime": "ONNX Runtime",
        "precision": "INT8" if name in INT8_MODELS else "FP32",
        "accuracy_%": 100 * (preds == labels).mean(),
        "size_mb": C.file_size_mb(ONNX_PATHS[name]),
        "latency_b1_ms": lat, "latency_b1_p95_ms": p95, "throughput_img_s": thr,
    }


def add_relative(df, base_key="fp32"):
    base = df[df.key == base_key].iloc[0]
    df["acc_change_pts"] = df["accuracy_%"] - base["accuracy_%"]
    df["compression_x"] = base["size_mb"] / df["size_mb"]
    df["speedup_b1_x"] = base["latency_b1_ms"] / df["latency_b1_ms"]
    return df


SHORT = {"fp32": "FP32", "ptq": "PTQ\nINT8", "qat": "QAT\nINT8", "unstructured": "Unstruct.\npruned",
         "structured": "Struct.\npruned", "structured_ptq": "Struct.+\nINT8"}


def make_plots(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pt = df[df.runtime == "PyTorch"]
    labels = [SHORT[k] for k in pt.key]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, col, title, fmt in zip(axes, ["accuracy_%", "size_mb", "latency_b1_ms"],
                                   ["Test accuracy (%)", "Model size (MB)", "CPU latency, batch 1 (ms)"],
                                   ["{:.2f}", "{:.1f}", "{:.2f}"]):
        bars = ax.bar(labels, pt[col], color="#4C72B0")
        ax.set_title(title)
        for bar, v in zip(bars, pt[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=8)
        if col == "accuracy_%":
            # zoomed axis so small accuracy differences are visible (axis does not start at 0)
            ax.set_ylim(pt[col].min() - 1.0, pt[col].max() + 0.5)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.RESULT_DIR, "comparison.png"), dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, r in pt.iterrows():
        ax.scatter(r["size_mb"], r["accuracy_%"], s=60)
        ax.annotate(SHORT[r["key"]].replace("\n", " "), (r["size_mb"], r["accuracy_%"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs size")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.RESULT_DIR, "accuracy_vs_size.png"), dpi=130)
    plt.close()

    ox = df[df.runtime == "ONNX Runtime"]
    if len(ox):
        keys = [k for k in pt.key if k in set(ox.key)]
        x = np.arange(len(keys))
        lat_pt = [float(pt.loc[pt.key == k, "latency_b1_ms"].iloc[0]) for k in keys]
        lat_ox = [float(ox.loc[ox.key == k, "latency_b1_ms"].iloc[0]) for k in keys]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar(x - 0.2, lat_pt, 0.4, label="PyTorch", color="#4C72B0")
        ax.bar(x + 0.2, lat_ox, 0.4, label="ONNX Runtime", color="#DD8452")
        ax.set_xticks(x, [SHORT[k] for k in keys])
        ax.set_ylabel("ms / image (batch 1, 1 thread)")
        ax.set_title("CPU latency: PyTorch vs ONNX Runtime")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(C.RESULT_DIR, "latency_pytorch_vs_onnx.png"), dpi=130)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1, help="CPU threads (1 = edge-like, fair comparison)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    torch.set_num_threads(args.threads)
    _, _, test_set = C.get_datasets(args.fake)
    loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=False, num_workers=0)
    labels = np.array([int(y) for y in getattr(test_set, "targets", [test_set[i][1] for i in range(len(test_set))])])
    print(f"Test images: {len(labels)}   CPU threads: {args.threads}   torch {torch.__version__}")
    print(f"CPU: {cpu_name()}")
    names = available_models()

    rows, preds_table = [], {"image_index": np.arange(len(labels)), "true_label": labels,
                             "true_class": [C.CLASSES[i] for i in labels]}
    for name in names:
        print(f"Benchmarking PyTorch {name} ...")
        preds, row = bench_torch(name, loader, labels, args)
        rows.append(row)
        preds_table[f"pred_{name}"] = preds
        preds_table[f"correct_{name}"] = (preds == labels).astype(int)
        print(f"  acc {row['accuracy_%']:.2f}%  size {row['size_mb']:.2f} MB  latency {row['latency_b1_ms']:.3f} ms")

    onnx_rows = []
    if not args.no_onnx:
        from export_onnx import ONNX_PATHS
        for name in names:
            if not os.path.exists(ONNX_PATHS[name]):
                print(f"  (skip ONNX {name}: run export_onnx.py first)")
                continue
            print(f"Benchmarking ONNX Runtime {name} ...")
            preds, row = bench_onnx(name, loader, labels, args)
            onnx_rows.append(row)
            preds_table[f"pred_onnx_{name}"] = preds
            print(f"  acc {row['accuracy_%']:.2f}%  size {row['size_mb']:.2f} MB  latency {row['latency_b1_ms']:.3f} ms")

    df = add_relative(pd.DataFrame(rows))
    q = df[df.key.isin(["fp32", "ptq", "qat"])]
    p = df[df.key.isin(["fp32", "unstructured", "structured", "structured_ptq"])]
    q.to_csv(os.path.join(C.RESULT_DIR, "benchmark_quantization.csv"), index=False, float_format="%.4f")
    p.to_csv(os.path.join(C.RESULT_DIR, "benchmark_pruning.csv"), index=False, float_format="%.4f")
    all_df = df
    if onnx_rows:
        odf = add_relative(pd.DataFrame(onnx_rows))
        odf.to_csv(os.path.join(C.RESULT_DIR, "benchmark_onnx.csv"), index=False, float_format="%.4f")
        all_df = pd.concat([df, odf], ignore_index=True)
    for col in ("params", "nonzero_params"):          # keep integer columns as integers
        if col in all_df:
            all_df[col] = all_df[col].astype("Int64")
    all_df.to_csv(os.path.join(C.RESULT_DIR, "benchmark_all.csv"), index=False, float_format="%.4f")

    pred_df = pd.DataFrame(preds_table)
    pred_df.to_csv(os.path.join(C.RESULT_DIR, "predictions_all_models.csv"), index=False)

    per_class = {"class": list(C.CLASSES)}
    for name in names:
        per_class[name] = [100 * pred_df.loc[pred_df.true_label == c, f"correct_{name}"].mean()
                           for c in range(10)]
    pd.DataFrame(per_class).to_csv(os.path.join(C.RESULT_DIR, "per_class_accuracy.csv"),
                                   index=False, float_format="%.2f")
    with open(os.path.join(C.RESULT_DIR, "benchmark_env.txt"), "w") as f:
        f.write(f"cpu: {cpu_name()}\nthreads: {args.threads}\ntorch: {torch.__version__}\n"
                f"python: {platform.python_version()}\nos: {platform.platform()}\n")
    make_plots(all_df)

    cols = ["model", "runtime", "precision", "accuracy_%", "acc_change_pts", "size_mb",
            "compression_x", "latency_b1_ms", "speedup_b1_x", "throughput_img_s"]
    pd.set_option("display.width", 200)
    print("\n" + all_df[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nCSV files and plots written to {C.RESULT_DIR}/")


if __name__ == "__main__":
    main()

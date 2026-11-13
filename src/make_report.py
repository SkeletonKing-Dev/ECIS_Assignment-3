"""
Builds REPORT.md (technical report) from the logs/ and results/ produced by
run_all.sh, so every number in the report comes from the actual run.

Usage:
    python src/make_report.py
"""
import math
import os
import re
import platform

import pandas as pd
import torch

import common as C

LOG = C.LOG_DIR


def read_log(name):
    p = os.path.join(LOG, name)
    return open(p, encoding="utf-8", errors="ignore").read() if os.path.exists(p) else ""


def md_table(df, float_fmt="{:.2f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            elif isinstance(v, (int,)) and abs(v) >= 10000:
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mcnemar(base_ok, model_ok):
    """Paired comparison of two classifiers on the same test images.
    n01 = base correct / model wrong, n10 = base wrong / model correct.
    Chi-square with continuity correction, 1 d.o.f.; returns (n01, n10, p)."""
    n01 = int(((base_ok == 1) & (model_ok == 0)).sum())
    n10 = int(((base_ok == 0) & (model_ok == 1)).sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return n01, n10, math.erfc(math.sqrt(chi2 / 2))


def training_curve(log_text):
    rows = re.findall(r"Epoch\s+(\d+)/\d+\s+loss ([\d.]+)\s+train ([\d.]+)%\s+test ([\d.]+)%", log_text)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["epoch", "loss", "train", "test"]).astype(float)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(df.epoch, df.loss)
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(df.epoch, df.train, label="train")
    ax[1].plot(df.epoch, df.test, label="test")
    ax[1].set_title("Accuracy (%)")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(C.RESULT_DIR, "baseline_training_curve.png")
    plt.savefig(path, dpi=130)
    return df


def main():
    q = pd.read_csv(os.path.join(C.RESULT_DIR, "benchmark_quantization.csv"))
    p = pd.read_csv(os.path.join(C.RESULT_DIR, "benchmark_pruning.csv"))
    onnx_path = os.path.join(C.RESULT_DIR, "benchmark_onnx.csv")
    o = pd.read_csv(onnx_path) if os.path.exists(onnx_path) else None
    per_class = pd.read_csv(os.path.join(C.RESULT_DIR, "per_class_accuracy.csv"))

    base_log = read_log("1_train_baseline.log")
    curve = training_curve(base_log)
    u_log = read_log("4_prune_unstructured.log")
    s_log = read_log("5_prune_structured.log")
    qat_log = read_log("3_qat.log")
    ptq_log = read_log("2_ptq.log")
    sptq_log = read_log("5b_structured_ptq.log")
    onnx_log = read_log("6_export_onnx.log")
    preds = pd.read_csv(os.path.join(C.RESULT_DIR, "predictions_all_models.csv"))
    env_path = os.path.join(C.RESULT_DIR, "benchmark_env.txt")
    env = dict(l.split(": ", 1) for l in open(env_path).read().splitlines() if ": " in l) \
        if os.path.exists(env_path) else {}
    has_bonus = "structured_ptq" in set(p.key)

    def get(df, key, col):
        return float(df.loc[df.key == key, col].iloc[0])

    fp32_acc = get(q, "fp32", "accuracy_%")
    out = []
    w = out.append

    w("# Assignment 3 - Edge Computing & Intelligent Systems\n")
    w("## Model Optimization: Quantization (PTQ, QAT) and Pruning (Unstructured, Structured) of ResNet-18 on CIFAR-10\n")
    w("Name: Syed M. Alyun  Roll No: 23I-0022  Section: AI-A7  \n")

    w("## 1. Environment\n")
    w(f"- Python {platform.python_version()}, PyTorch {torch.__version__}, OS: {platform.system()} {platform.machine()}")
    w(f"- Training device: {'CUDA GPU' if 'cuda' in base_log.lower() else 'see logs/1_train_baseline.log'}; "
      "all benchmarks run on CPU (1 thread, to resemble an edge device and keep the comparison fair)")
    if env.get("cpu"):
        w(f"- Benchmark CPU: {env['cpu']}")
    w("- Quantization backend: x86 (FBGEMM / oneDNN kernels), INT8 weights and activations\n")

    w("## 2. Baseline FP32 model\n")
    w("**Architecture.** ResNet-18 adapted for 32x32 inputs: the first layer is a 3x3 conv with stride 1 "
      "and the max-pool is removed (the ImageNet 7x7/stride-2 stem would shrink a 32x32 image to 8x8 too early). "
      "Four stages of two BasicBlocks with 64/128/256/512 channels, global average pooling and a 10-way linear layer "
      "(11.2 M parameters).\n")
    w("To make the same network quantizable in eager mode it contains a `QuantStub` / `DeQuantStub` at input/output, "
      "a separate `nn.ReLU` module for every activation (so it can be fused), and the residual addition is done with "
      "`nn.quantized.FloatFunctional` so the add itself gets an observer and an INT8 kernel.\n")
    w("**Training.** SGD (momentum 0.9, Nesterov, weight decay 5e-4), One-Cycle learning-rate schedule (max LR 0.1), "
      "batch size 128, augmentation: random crop with 4-pixel padding + horizontal flip, normalisation with CIFAR-10 "
      "mean/std. 30 epochs, and the weights after the final epoch are saved to `models/fp32_resnet18.pth`.\n")
    w("**Model selection.** No checkpoint is selected using the test set in any script: every training / fine-tuning "
      "run uses a fixed schedule and keeps the final weights. Test accuracy is printed per epoch only to monitor "
      "progress, so the reported test accuracies remain unbiased estimates.\n")
    if curve is not None:
        w(f"Result: test accuracy **{curve.test.iloc[-1]:.2f}%** after {int(curve.epoch.max())} epochs.\n")
        w("![training curve](results/baseline_training_curve.png)\n")

    w("## 3. Post-Training Static Quantization (PTQ) - `src/ptq.py`, `src/calibration.py`\n")
    w("1. **Fusion** - `Conv+BN+ReLU` and `Conv+BN` groups (stem, every block, every 1x1 shortcut) are fused with "
      "`torch.ao.quantization.fuse_modules`. BatchNorm is folded into the conv weights/bias, which removes layers and "
      "avoids quantization error between conv and BN.")
    w("2. **Backend / qconfig** - `torch.backends.quantized.engine = 'x86'` and `get_default_qconfig('x86')`: "
      "weights per-channel symmetric INT8 (`PerChannelMinMaxObserver`), activations per-tensor affine UINT8 with a "
      "`HistogramObserver` (chooses the clipping range that minimises quantization error instead of using raw min/max).")
    w("3. **Prepare** - `prepare()` inserts observers after every quantizable op.")
    w("4. **Calibration** - 2000 randomly chosen *training* images (no augmentation, fixed seed) are passed through "
      "the observed model in eval mode; the observers record activation ranges. The test set is never used for calibration.")
    w("5. **Convert** - `convert()` replaces modules with INT8 kernels (`QuantizedConvReLU2d`, `QuantizedLinear`, "
      "`QFunctional` add) using the computed scale / zero-point.")
    w("6. **Save / deploy** - the INT8 model is traced and frozen to TorchScript (`models/ptq_int8_resnet18.pt`), "
      "so inference only needs `torch.jit.load()`; the script reloads the file and re-checks accuracy.\n")
    m = re.search(r"PTQ INT8 test accuracy: ([\d.]+)%", ptq_log)
    if m:
        w(f"Result: PTQ INT8 accuracy **{float(m.group(1)):.2f}%** vs FP32 {fp32_acc:.2f}%.\n")

    w("## 4. Quantization-Aware Training (QAT) - `src/qat.py`\n")
    w("1. **Initialisation** from the trained FP32 checkpoint (not from scratch).")
    w("2. **QAT fusion** with `fuse_modules_qat` (model in train mode): produces `ConvBnReLU2d` modules that keep "
      "BatchNorm trainable while simulating the folded weights.")
    w("3. **Fake quantization** - `get_default_qat_qconfig('x86')` + `prepare_qat()` inserts `FakeQuantize` modules on "
      "weights and activations. In the forward pass values are quantized and immediately de-quantized, so the network "
      "*sees* INT8 rounding error; in the backward pass the straight-through estimator lets gradients flow.")
    w("4. **Fine-tuning** - SGD, LR 1e-3 with cosine decay. BatchNorm statistics are frozen at 60% of the epochs and "
      "the observers (scale/zero-point) at 80%, so the last epochs train against fixed quantization parameters. "
      "After every epoch the model is converted to a real INT8 model and evaluated (monitoring only); the weights "
      "after the final epoch are used.")
    w("5. **Convert + export** - `convert()` to INT8 and saved as TorchScript (`models/qat_int8_resnet18.pt`).\n")
    qrows = re.findall(r"Epoch (\d+)/\d+\s+loss ([\d.]+)\s+train\(fake-quant\) ([\d.]+)%\s+test INT8 ([\d.]+)%", qat_log)
    if qrows:
        w(md_table(pd.DataFrame(qrows, columns=["QAT epoch", "loss", "train acc (fake-quant) %", "test acc INT8 %"])))
        w("")

    w("## 5. Unstructured pruning - `src/prune_unstructured.py`\n")
    w("- **Criterion:** global L1 magnitude pruning (`prune.global_unstructured` with `L1Unstructured`) over the weights "
      "of all 20 Conv layers and the FC layer. A global threshold lets layers with many redundant weights be pruned "
      "more than sensitive ones.")
    w("- **Iterative schedule:** the target sparsity (80%) is reached over N iterations. Each iteration removes the "
      "same fraction `a` of the *remaining* weights, with `(1-a)^N = 1 - target`, then fine-tunes. Gradual pruning "
      "lets the network adapt between steps and recovers more accuracy than pruning 80% in one shot.")
    w("- **Masking:** PyTorch keeps `weight_orig` and a binary `weight_mask` buffer; the forward uses "
      "`weight_orig * weight_mask`, so pruned weights stay zero during fine-tuning.")
    w("- **Permanent:** `prune.remove()` writes the masked weight back and deletes the mask/orig tensors.")
    w("- **Compressed checkpoint:** a dense tensor with 80% zeros is still the same size on disk, so the pruned weights "
      "are stored as a 1-bit packed mask + the non-zero values (`*_sparse.pth`). The loader rebuilds the dense tensors.\n")
    urows = re.findall(r"Iteration (\d+)/\d+: sparsity ([\d.]+)%\s+acc after pruning ([\d.]+)%\s*\n(?:.*\n)*?\s*-> after fine-tuning: ([\d.]+)%", u_log)
    if urows:
        w(md_table(pd.DataFrame(urows, columns=["iteration", "sparsity %", "acc right after pruning %", "acc after fine-tune %"])))
        w("")
    lsp = os.path.join(C.RESULT_DIR, "unstructured_layer_sparsity.csv")
    if os.path.exists(lsp):
        ls = pd.read_csv(lsp)
        w("Per-layer sparsity after pruning (`results/unstructured_layer_sparsity.csv`). The single global threshold "
          "prunes the large, redundant layers much harder than the small, sensitive first conv and the classifier:\n")
        w(md_table(ls[["layer", "weights", "sparsity_%"]]))
        w("")

    w("## 6. Structured channel pruning - `src/prune_structured.py`\n")
    w("- **What is pruned:** the intermediate channels of each BasicBlock (output of `conv1`, `bn1`, input of `conv2`).")
    w("- **Keeping dimensions valid:** the output channels of a block are tied together by the residual addition "
      "(`out + shortcut(x)`), so they must match the shortcut and the neighbouring blocks. The middle channels are "
      "only connected to three tensors, so for the kept indices `k` the script slices `conv1.weight[k]`, "
      "`bn1.{weight,bias,running_mean,running_var}[k]` and `conv2.weight[:, k]` consistently.")
    w("- **Importance:** `score(c) = ||W_conv1[c]||_1 * |gamma_bn1[c]|` - the L1 norm of the filter times its BatchNorm "
      "scale; the lowest-scoring 50% of channels in every block are removed.")
    w("- **Reconstruction:** a new `ResNet18(channel_cfg=[...])` with physically smaller Conv/BN layers is built and the "
      "surviving weights are copied in. No masks or zero channels remain, so the model is faster with normal dense "
      "kernels. The checkpoint stores `channel_cfg` so the architecture can be rebuilt at load time.")
    w("- **Fine-tuning:** SGD with One-Cycle LR (max 0.02), 10 epochs, final weights kept.\n")
    ch = re.findall(r"^\s+(\d+) ->\s+(\d+)$", s_log, re.M)
    if ch:
        names = [f"layer{s}.{i}" for s in range(1, 5) for i in range(2)]
        w(md_table(pd.DataFrame([(n, a, b) for n, (a, b) in zip(names, ch)],
                                columns=["block", "mid channels before", "after"])))
        w("")
    for pat in (r"Original: .*", r"Pruned \(before fine-tune\): .*", r"Pruned \(after fine-tune\): .*"):
        mm = re.search(pat, s_log)
        if mm:
            w(f"- `{mm.group(0).strip()}`")
    w("")
    if has_bonus:
        w("**Bonus - pruning + quantization combined.** The structured-pruned model is a normal dense network, so the "
          "same PTQ pipeline (`src/ptq.py --input models/structured_pruned_resnet18.pth`) quantizes it to INT8 "
          f"(`models/structured_pruned_ptq_int8_resnet18.pt`): {get(p, 'structured_ptq', 'accuracy_%'):.2f}% accuracy, "
          f"{get(p, 'structured_ptq', 'size_mb'):.2f} MB ({get(p, 'structured_ptq', 'compression_x'):.1f}x smaller than "
          f"FP32), {get(p, 'structured_ptq', 'latency_b1_ms'):.2f} ms per image "
          f"({get(p, 'structured_ptq', 'speedup_b1_x'):.2f}x faster).\n")

    w("## 7. ONNX deployment - `src/export_onnx.py`, `src/onnx_infer.py`\n")
    w("- FP32, unstructured and structured models: `torch.onnx.export` (opset 17, dynamic batch).")
    w("- PTQ INT8: PyTorch's converted INT8 modules cannot be exported to ONNX directly, so the FP32 ONNX graph is "
      "statically quantized with ONNX Runtime (`quantize_static`, QDQ format, per-channel INT8 weights, UINT8 "
      "activations) using the same 2000 calibration images. Before quantization the graph goes through ORT's "
      "`quant_pre_process` (shape inference + graph optimization).")
    cm = re.search(r"calibration: (\w+)", onnx_log)
    w(f"- Activation calibration method: **{cm.group(1) if cm else 'see logs'}**. ORT's default MinMax uses the raw "
      "min/max of each activation, so a few outliers stretch the scale and waste INT8 levels; Percentile (99.999%) "
      "and Entropy clip outliers, which is closer to PyTorch's `HistogramObserver`. Calibration runs in chunks of "
      "100 images (`CalibStridedMinMax`) so the histogram methods do not keep every activation in memory.")
    w("- QAT INT8: exporting the fake-quant QAT graph directly keeps BatchNorm unfolded (the QAT ConvBn modules "
      "compute conv -> rescale -> BN), so ONNX Runtime cannot map it to INT8 conv kernels and the file keeps FP32 "
      "weights. Instead, the converted QAT INT8 weights and biases are de-quantized into a BN-folded float copy of "
      "the network (identical values, already on the INT8 grid) and that graph is statically quantized by ONNX Runtime. "
      "The weights quantize back onto the same grid; activation ranges are re-calibrated on the same 2000 images.")
    if has_bonus:
        w("- Bonus: the structured-pruned ONNX graph is quantized the same way (`structured_ptq_resnet18.onnx`).")
    w("- All models are run with the ONNX Runtime CPU execution provider.\n")

    w("## 8. Benchmark results (CPU, full 10,000-image CIFAR-10 test set) - `src/benchmark.py`\n")
    cols = ["model", "precision", "accuracy_%", "acc_change_pts", "size_mb", "compression_x",
            "latency_b1_ms", "speedup_b1_x", "throughput_img_s"]
    w("### 8.1 Quantization (Q1)\n")
    w(md_table(q[cols]))
    w("")
    w("### 8.2 Pruning (Q2)\n")
    w(md_table(p[cols + ["sparsity_%", "nonzero_params", "macs_M"]]))
    w("\nSize of the unstructured model is the compressed (bitmask) checkpoint. Sparsity / MACs of the INT8 bonus "
      "model refer to its (structured-pruned) architecture.\n")
    if o is not None:
        w("### 8.3 ONNX Runtime\n")
        w(md_table(o[["model", "precision", "accuracy_%", "size_mb", "latency_b1_ms", "speedup_b1_x", "throughput_img_s"]]))
        w("")
    w("### 8.4 Per-class accuracy (%)\n")
    w(md_table(per_class))
    w("")

    # Paired significance test against the FP32 baseline
    w("### 8.5 Are the accuracy differences significant?\n")
    w("All models are evaluated on the same 10,000 images, so they can be compared image by image. The table counts "
      "images where the baseline is right and the model is wrong (n01) and the opposite (n10), plus prediction "
      "agreement with FP32. McNemar's test (chi-square, continuity correction) gives the p-value for "
      "\"same accuracy as FP32\"; p > 0.05 means the difference is within noise.\n")
    rows = []
    pred_cols = [c for c in preds.columns if c.startswith("pred_") and c != "pred_fp32"]
    for c in pred_cols:
        key = c[len("pred_"):]
        ok = (preds[c] == preds.true_label).astype(int)
        n01, n10, pv = mcnemar(preds.correct_fp32, ok)
        name = ("ONNX " if key.startswith("onnx_") else "PyTorch ") + key.replace("onnx_", "")
        rows.append((name, 100 * ok.mean(), 100 * (preds[c] == preds.pred_fp32).mean(), n01, n10, pv,
                     "yes" if pv < 0.05 else "no"))
    sig = pd.DataFrame(rows, columns=["model", "accuracy_%", "agreement_with_fp32_%", "n01", "n10",
                                      "p_value", "significant"])
    w(md_table(sig, "{:.3f}"))
    w("")
    w("![comparison](results/comparison.png)\n")
    w("![accuracy vs size](results/accuracy_vs_size.png)\n")
    if os.path.exists(os.path.join(C.RESULT_DIR, "latency_pytorch_vs_onnx.png")):
        w("![latency pytorch vs onnx](results/latency_pytorch_vs_onnx.png)\n")

    # ---- Discussion generated from the numbers ----
    w("## 9. Discussion\n")
    ptq_d, qat_d = get(q, "ptq", "acc_change_pts"), get(q, "qat", "acc_change_pts")
    sig_of = dict(zip(sig.model, sig.significant))
    noise = (sig_of.get("PyTorch ptq") == "no" and sig_of.get("PyTorch qat") == "no")
    w(f"- **PTQ vs QAT.** PTQ changed accuracy by {ptq_d:+.2f} points and QAT by {qat_d:+.2f} points relative to FP32 "
      f"({fp32_acc:.2f}%). Both models are {get(q, 'ptq', 'compression_x'):.1f}x smaller (32-bit -> 8-bit weights) and "
      f"run {get(q, 'ptq', 'speedup_b1_x'):.2f}x / {get(q, 'qat', 'speedup_b1_x'):.2f}x faster at batch 1. "
      + ("McNemar's test (8.5) shows neither difference from FP32 is statistically significant, so both INT8 models "
         "should be read as matching FP32 accuracy rather than QAT being 'better'. ResNet-18 with per-channel weights "
         "and histogram calibration quantizes very well, which leaves PTQ little to lose; QAT pays off more on compact "
         "models (e.g. MobileNet) or at lower bit-widths (4-bit). " if noise else
         ("QAT recovered accuracy that PTQ lost, because the weights were trained to tolerate the rounding error. "
          if qat_d > ptq_d else "QAT did not improve on PTQ here. "))
      + "PTQ takes seconds and needs only unlabeled calibration data; QAT needs labeled data and extra GPU training.")
    us = get(p, "unstructured", "sparsity_%")
    w(f"- **Unstructured pruning** removed {us:.1f}% of the weights with a {get(p, 'unstructured', 'acc_change_pts'):+.2f} "
      f"point accuracy change and a {get(p, 'unstructured', 'compression_x'):.1f}x smaller (compressed) checkpoint, but "
      f"latency changed only {get(p, 'unstructured', 'speedup_b1_x'):.2f}x: the zeros are still multiplied by dense "
      "conv kernels. Speed-ups need sparse kernels or hardware support (e.g. 2:4 sparsity on NVIDIA Ampere).")
    w(f"- **Structured pruning** kept a normal dense network with {get(p, 'structured', 'macs_M'):.0f}M MACs instead of "
      f"{get(p, 'fp32', 'macs_M'):.0f}M, giving a real {get(p, 'structured', 'speedup_b1_x'):.2f}x CPU speed-up and "
      f"{get(p, 'structured', 'compression_x'):.1f}x smaller file, at {get(p, 'structured', 'acc_change_pts'):+.2f} "
      "points. Removing whole channels is coarser than removing single weights, so it usually costs more accuracy "
      "for the same parameter reduction, but it is the approach that helps on ordinary edge CPUs.")
    if has_bonus:
        w(f"- **Combining methods.** Structured pruning + INT8 gave the smallest and fastest model: "
          f"{get(p, 'structured_ptq', 'size_mb'):.2f} MB and {get(p, 'structured_ptq', 'speedup_b1_x'):.2f}x faster "
          f"than FP32 at {get(p, 'structured_ptq', 'acc_change_pts'):+.2f} points, because the two methods remove "
          "different redundancy (fewer channels vs fewer bits per value).")
    if o is not None:
        w(f"- **ONNX Runtime.** The FP32 ONNX model gives identical predictions to PyTorch and runs "
          f"{get(q, 'fp32', 'latency_b1_ms') / get(o, 'fp32', 'latency_b1_ms'):.2f}x faster (graph optimizations, "
          f"fused kernels). The ONNX INT8 models changed accuracy by {get(o, 'ptq', 'acc_change_pts'):+.2f} (PTQ) and "
          f"{get(o, 'qat', 'acc_change_pts'):+.2f} (QAT) points: ORT re-calibrates the activation ranges with its own "
          "calibrator, while the PyTorch INT8 models use the HistogramObserver ranges (PTQ) or the ranges learned "
          "during QAT. The unstructured ONNX model is stored dense, so it has no size benefit there.")
    w("- **Fine-tuning** was essential for both pruning methods; the tables above show the accuracy right after pruning "
      "and after recovery.\n")

    # ---- 10. Screenshots  ----
        # ---- 10. Screenshots: 10.1-10.2 as images, 10.3-10.10 as pasted logs ----
    w("## 10. Screenshots\n")

    # 10.1: images (markdown alt text doubles as fallback if missing)
    for heading, filename in [
        ("10.1 Baseline Training",                "1_train_baseline.png"),
    ]:
        w(f"### {heading}\n")
        w(f"![{heading} — put screenshot in `logs/screenshots/{filename}`](logs/screenshots/{filename})\n")

    # 10.2 - 10.10: paste the terminal log directly (same basename, .log)
    for heading, logname in [
        ("10.2 Post-Training Quantization (PTQ)",             "2_ptq.log"),
        ("10.3 Quantization-Aware Training (QAT)",            "3_qat.log"),
        ("10.4 Unstructured Pruning",                         "4_prune_unstructured.log"),
        ("10.5 Structured Pruning",                           "5_prune_structured.log"),
        ("10.6 ONNX Export",                                  "6_export_onnx.log"),
        ("10.7 PyTorch Inference (`infer.py --model all`)",   "7_infer.log"),
        ("10.8 ONNX Inference (`onnx_infer.py --model all`)", "7_onnx_infer.log"),
        ("10.9 Final Benchmark Table",                        "8_benchmark.log"),
    ]:
        w(f"### {heading}\n")
        body = read_log(logname).rstrip()
        w(f"```\n{body}\n```\n" if body else f"_`logs/{logname}` not found._\n")
    
    w("## 11. Reproducibility\n")
    w("Fixed random seeds, fixed calibration subset (seed 0), all hyper-parameters as command-line arguments with the "
      "values used in `run_all.sh`, and full logs of this run in `logs/`. See README.md for step-by-step commands.\n")

    with open(os.path.join(C.ROOT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()

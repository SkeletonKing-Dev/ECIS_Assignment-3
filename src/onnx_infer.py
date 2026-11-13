"""
CPU inference with ONNX Runtime for the exported models.

Usage:
    python src/onnx_infer.py --model ptq --image my_cat.png
    python src/onnx_infer.py --model all --test-samples 5
"""
import argparse
import os

import numpy as np
import onnxruntime as ort

import common as C
from export_onnx import ONNX_PATHS
from infer import preprocess_image


def make_session(name, threads=0):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(ONNX_PATHS[name], opts, providers=["CPUExecutionProvider"])


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fp32", choices=list(ONNX_PATHS) + ["all"])
    ap.add_argument("--image", nargs="*")
    ap.add_argument("--test-samples", type=int, default=8)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    if args.image:
        x = np.concatenate([preprocess_image(p).numpy() for p in args.image])
        labels, ids = [None] * len(args.image), args.image
    else:
        _, _, test_set = C.get_datasets(args.fake)
        items = [test_set[i] for i in range(args.test_samples)]
        x = np.stack([it[0].numpy() for it in items])
        labels = [int(it[1]) for it in items]
        ids = [f"test[{i}]" for i in range(args.test_samples)]

    names = [n for n in ONNX_PATHS if os.path.exists(ONNX_PATHS[n])] if args.model == "all" else [args.model]
    for name in names:
        sess = make_session(name)
        probs = softmax(sess.run(None, {"input": x})[0])
        print(f"\n== ONNX {name} ==")
        for i, row in enumerate(probs):
            p = int(row.argmax())
            true = f"  true: {C.CLASSES[labels[i]]}" if labels[i] is not None else ""
            print(f"  {ids[i]:<12} pred: {C.CLASSES[p]:<10} conf {row[p]:.3f}{true}")


if __name__ == "__main__":
    main()

"""
Inference pipeline for all five PyTorch models (and the ONNX models).

Models: fp32, ptq, qat, unstructured, structured

Usage:
    python src/infer.py --model ptq --image my_cat.png
    python src/infer.py --model structured --test-samples 10
    python src/infer.py --model all --test-samples 5
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image

import common as C

MODEL_NAMES = ["fp32", "ptq", "qat", "unstructured", "structured"]
EXTRA_MODELS = ["structured_ptq"]   # bonus: structured pruning + INT8 PTQ


def available_models():
    """The five required models plus any bonus model that has been built."""
    return MODEL_NAMES + [n for n in EXTRA_MODELS if os.path.exists(C.STRUCT_PTQ_PATH)]


def load_model(name):
    """Returns a ready-to-run CPU model in eval mode."""
    if name in ("ptq", "qat", "structured_ptq"):
        torch.backends.quantized.engine = "x86" if "x86" in torch.backends.quantized.supported_engines \
            else torch.backends.quantized.supported_engines[-1]
        path = {"ptq": C.PTQ_PATH, "qat": C.QAT_PATH, "structured_ptq": C.STRUCT_PTQ_PATH}[name]
        model = torch.jit.load(path, map_location="cpu")
    elif name == "fp32":
        model = C.load_fp32_model(C.FP32_PATH)
    elif name == "unstructured":
        model = C.load_fp32_model(C.UNSTRUCT_PATH)
    elif name == "structured":
        model = C.load_fp32_model(C.STRUCT_PATH)
    else:
        raise ValueError(f"unknown model '{name}', choose from {MODEL_NAMES + EXTRA_MODELS}")
    model.eval()
    return model


def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    return C.get_transforms(train=False)(img).unsqueeze(0)


@torch.no_grad()
def predict(model, x):
    probs = F.softmax(model(x), dim=1)
    conf, idx = probs.max(1)
    return idx.tolist(), conf.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fp32", choices=MODEL_NAMES + EXTRA_MODELS + ["all"])
    ap.add_argument("--image", nargs="*", help="image file(s) to classify")
    ap.add_argument("--test-samples", type=int, default=8, help="if no image given, use N CIFAR-10 test images")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    names = available_models() if args.model == "all" else [args.model]

    if args.image:
        x = torch.cat([preprocess_image(p) for p in args.image])
        labels, ids = [None] * len(args.image), args.image
    else:
        _, _, test_set = C.get_datasets(args.fake)
        items = [test_set[i] for i in range(args.test_samples)]
        x = torch.stack([it[0] for it in items])
        labels = [int(it[1]) for it in items]
        ids = [f"test[{i}]" for i in range(args.test_samples)]

    for name in names:
        model = load_model(name)
        predict(model, x)   # warm-up run (TorchScript optimizes on the first calls)
        t0 = time.perf_counter()
        preds, confs = predict(model, x)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n== {name} ==  ({ms:.1f} ms for {len(preds)} images)")
        for i, (p, c) in enumerate(zip(preds, confs)):
            true = f"  true: {C.CLASSES[labels[i]]}" if labels[i] is not None else ""
            mark = "" if labels[i] is None else ("  ok" if p == labels[i] else "  WRONG")
            print(f"  {ids[i]:<12} pred: {C.CLASSES[p]:<10} conf {c:.3f}{true}{mark}")


if __name__ == "__main__":
    main()

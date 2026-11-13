"""
Export the models to ONNX for CPU deployment with ONNX Runtime.

  fp32, unstructured, structured -> torch.onnx.export of the float model
  ptq  -> FP32 ONNX graph statically quantized by ONNX Runtime (QDQ, INT8)
          using the same calibration subset as PyTorch PTQ
  qat  -> QAT-trained INT8 weights placed in a BN-folded float graph, then
          statically quantized by ONNX Runtime (QDQ, INT8). (Exporting the
          fake-quant QAT graph directly keeps BatchNorm unfolded, so ORT
          cannot use INT8 conv kernels for it.)

Note: PyTorch's converted INT8 eager modules (quantized::conv2d etc.) cannot be
exported to ONNX directly, which is why the standard QDQ route is used.

Usage:
    python src/export_onnx.py                          # percentile calibration (default)
    python src/export_onnx.py --calib-method entropy   # or minmax
"""
import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

import common as C
from qat import prepare_qat_model

ONNX_PATHS = {n: os.path.join(C.ONNX_DIR, f"{n}_resnet18.onnx")
              for n in ("fp32", "ptq", "qat", "unstructured", "structured", "structured_ptq")}

CALIB_METHODS = {"minmax": CalibrationMethod.MinMax,
                 "percentile": CalibrationMethod.Percentile,
                 "entropy": CalibrationMethod.Entropy}


def ort_quantize(float_path, out_path, calib_images, method, fake):
    """ONNX Runtime static INT8 quantization (QDQ format).
    1) quant_pre_process: shape inference + graph optimization (recommended by ORT)
    2) calibration on the same training subset as PyTorch PTQ
    3) per-channel INT8 weights, UINT8 activations
    MinMax uses the raw activation range, so a few outliers stretch the scale.
    Percentile (99.999%) / Entropy clip outliers, which is closer to what
    PyTorch's HistogramObserver does."""
    pre = out_path.replace(".onnx", "_pre_tmp.onnx")
    quant_pre_process(float_path, pre, skip_symbolic_shape=True)
    reader = CifarCalibReader(C.get_calibration_loader(calib_images, batch_size=1, fake=fake))
    chunk = next(c for c in (100, 50, 25, 10, 5, 1) if len(reader) % c == 0)
    quantize_static(pre, out_path, reader,
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CALIB_METHODS[method],
                    extra_options={"CalibStridedMinMax": chunk})
    os.remove(pre)
    print(f"  {out_path}  ({C.file_size_mb(out_path):.2f} MB, calibration: {method})")


def export(model, path, opset=17):
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, (x,), path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    print(f"  {path}  ({C.file_size_mb(path):.2f} MB)")


@torch.no_grad()
def qat_weights_to_folded_float():
    """Rebuild the QAT INT8 model, then copy its (de-quantized) INT8 weights and
    biases into a BN-folded float copy of the network. The weights are therefore
    exactly the QAT-learned values on the INT8 grid; ONNX Runtime then re-applies
    INT8 quantization (weights map back onto the same grid) and calibrates the
    activation ranges. BN is already folded, so ORT can fuse every
    Conv(+ReLU) into a real INT8 QLinearConv kernel."""
    qat = prepare_qat_model(C.load_fp32_model(C.FP32_PATH))
    qat.load_state_dict(torch.load(C.QAT_PATH.replace(".pt", "_qat_prepared_state_dict.pth"),
                                   map_location="cpu"))
    qat.eval()
    int8 = torch.ao.quantization.convert(qat, inplace=False)

    folded = C.load_fp32_model(C.FP32_PATH).eval()
    folded.fuse_model(qat=False)                 # same module names as the INT8 model
    n = 0
    for name, qm in int8.named_modules():
        if hasattr(qm, "weight") and callable(qm.weight) and hasattr(qm, "bias") and callable(qm.bias):
            fm = folded.get_submodule(name)
            if isinstance(fm, torch.nn.Sequential):  # ConvReLU2d -> [Conv2d, ReLU]
                fm = fm[0]
            fm.weight.copy_(qm.weight().dequantize())
            b = qm.bias()
            if fm.bias is None:
                fm.bias = torch.nn.Parameter(b.clone())
            else:
                fm.bias.copy_(b)
            n += 1
    print(f"  copied QAT weights into {n} folded layers")
    return folded


class CifarCalibReader(CalibrationDataReader):
    """Feeds the calibration images one by one. __len__ and set_range let
    ONNX Runtime calibrate in chunks (CalibStridedMinMax), so the histogram
    methods do not keep the activations of all images in memory at once."""

    def __init__(self, loader):
        self.data = [{"input": x.numpy()} for x, _ in loader]
        self.set_range(0, len(self.data))

    def __len__(self):
        return len(self.data)

    def set_range(self, start_index, end_index):
        self.it = iter(self.data[start_index:end_index])

    def get_next(self):
        return next(self.it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-images", type=int, default=2000)
    ap.add_argument("--calib-method", default="percentile", choices=list(CALIB_METHODS),
                    help="ONNX Runtime activation calibration method")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    C.ensure_dirs()

    print("Exporting float models:")
    export(C.load_fp32_model(C.FP32_PATH), ONNX_PATHS["fp32"])
    export(C.load_fp32_model(C.UNSTRUCT_PATH), ONNX_PATHS["unstructured"])
    export(C.load_fp32_model(C.STRUCT_PATH), ONNX_PATHS["structured"])

    print("PTQ INT8 (ONNX Runtime static quantization, QDQ):")
    ort_quantize(ONNX_PATHS["fp32"], ONNX_PATHS["ptq"], args.calib_images, args.calib_method, args.fake)

    print("QAT INT8 (QAT-trained INT8 weights in a BN-folded graph -> QDQ):")
    folded = qat_weights_to_folded_float()
    tmp = ONNX_PATHS["qat"].replace(".onnx", "_float_tmp.onnx")
    export(folded, tmp)
    ort_quantize(tmp, ONNX_PATHS["qat"], args.calib_images, args.calib_method, args.fake)
    os.remove(tmp)

    print("Bonus - structured pruned + INT8:")
    ort_quantize(ONNX_PATHS["structured"], ONNX_PATHS["structured_ptq"], args.calib_images,
                 args.calib_method, args.fake)

    # Quick check that every model loads and runs in ONNX Runtime on CPU
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    for name, p in ONNX_PATHS.items():
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": x})[0]
        print(f"  ORT check {name:<13} output {out.shape}")


if __name__ == "__main__":
    main()

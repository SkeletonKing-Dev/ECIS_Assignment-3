# Assignment 3: Model Optimization for Edge Deployment

This project uses ResNet-18 on CIFAR-10 and covers:

- Q1: Post-Training Static Quantization (PTQ) and Quantization-Aware Training (QAT) to INT8
- Q2: Unstructured (magnitude) pruning and structured (channel) pruning with architecture reconstruction
- ONNX export and CPU inference with ONNX Runtime
- Automated benchmarking (CSV results and plots) and an auto-generated technical report

## 1. Project structure

```
.
├── data/                   CIFAR-10 (data/cifar10/cifar-10-batches-py), not submitted
├── logs/                   log of every pipeline step
├── models/                 PyTorch checkpoints (FP32, PTQ, QAT, unstructured, structured)
├── onnx_models/            ONNX models
├── results/                benchmark CSV files and plots
├── screenshots/            screenshots for the report
├── scripts/
│   ├── run_all.sh          reproducible end-to-end pipeline
│   └── make_submission.sh  builds ROLL-NUM_SECTION.zip
├── src/
│   ├── common.py           dataset, quantization-ready ResNet-18, train/eval and measurement helpers
│   ├── train_baseline.py   baseline FP32 training
│   ├── calibration.py      PTQ calibration (training subset through the observers)
│   ├── ptq.py              PTQ: fusion, backend/qconfig, calibration, INT8 conversion, TorchScript export
│   ├── qat.py              QAT: fake-quant insertion, fine-tuning, INT8 conversion, export
│   ├── prune_unstructured.py  iterative global L1 pruning, masks, prune.remove, compressed checkpoint
│   ├── prune_structured.py    channel importance, channel selection, network rebuild, fine-tuning
│   ├── export_onnx.py      ONNX export of all 5 models (INT8 models in QDQ format)
│   ├── infer.py            PyTorch inference pipeline for all 5 models
│   ├── onnx_infer.py       ONNX Runtime CPU inference
│   ├── benchmark.py        automated evaluation of all models, writes the CSV files and plots
│   └── make_report.py      generates REPORT.md from logs/ and results/
├── README.md
└── requirements.txt
```

## 2. Installation (uv)

Requirements: Python 3.9 or newer and [uv](https://docs.astral.sh/uv/). An NVIDIA GPU is recommended for training. Quantized inference and benchmarking run on the CPU.

```
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

The default PyPI `torch` wheel on Linux includes CUDA support. On a machine without a GPU, install the smaller CPU build first:

```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

Tested with PyTorch 2.15 and ONNX Runtime 1.29.

## 3. Dataset

The code looks for an existing `cifar-10-batches-py` folder in these places, in order:

1. `$CIFAR_ROOT` (environment variable)
2. `data/cifar10/`
3. `data/`

If it finds the folder, torchvision verifies the files and skips the download. Otherwise CIFAR-10 (about 170 MB) is downloaded to `data/cifar10/`. A symlink such as `data/cifar10 -> ../../Assignment-2/data/cifar10` works.

## 4. Run everything (one command)

Run these from the project root. The scripts use `.venv/bin/python` automatically, so you don't need to activate the environment first.

```
bash scripts/run_all.sh            # full run, about 45-60 min on a T4-class GPU
bash scripts/run_all.sh --quick    # fewer epochs, about half the time
bash scripts/run_all.sh --fake     # 5 min smoke test on random data (checks the code only)
bash scripts/run_all.sh --skip-baseline   # reuse an existing models/fp32_resnet18.pth (skip step 1)
```

You can combine flags, for example `--quick --skip-baseline`.

Each step's log is saved in `logs/`. The CSV files and plots go in `results/`, and `REPORT.md` is written to the project root.

## 5. Step-by-step execution

Activate the environment first (`source .venv/bin/activate`) and run each command from the project root.

| Step | Command | Output |
|---|---|---|
| 1. Train baseline | `python src/train_baseline.py --epochs 30` | `models/fp32_resnet18.pth` |
| 2a. Calibration (optional, inspect) | `python src/calibration.py --calib-images 2000` | console |
| 2b. PTQ | `python src/ptq.py --calib-images 2000 --backend x86` | `models/ptq_int8_resnet18.pt` |
| 3. QAT | `python src/qat.py --epochs 5 --lr 1e-3` | `models/qat_int8_resnet18.pt` |
| 4. Unstructured pruning + fine-tune | `python src/prune_unstructured.py --target-sparsity 0.8 --iterations 4 --ft-epochs 2 --final-ft-epochs 5` | `models/unstructured_pruned_resnet18.pth`, `..._sparse.pth` |
| 5. Structured pruning + fine-tune | `python src/prune_structured.py --ratio 0.5 --ft-epochs 10` | `models/structured_pruned_resnet18.pth` |
| 5b. Bonus: structured pruning + INT8 | `python src/ptq.py --input models/structured_pruned_resnet18.pth --output models/structured_pruned_ptq_int8_resnet18.pt` | `models/structured_pruned_ptq_int8_resnet18.pt` |
| 6. ONNX export | `python src/export_onnx.py --calib-method percentile` | `onnx_models/*.onnx` |
| 7a. Inference (PyTorch) | `python src/infer.py --model all --test-samples 8` | console |
| 7b. Inference (ONNX, CPU) | `python src/onnx_infer.py --model all --test-samples 8` | console |
| 8. Benchmark | `python src/benchmark.py --threads 1` | `results/*.csv`, `results/*.png` |
| 9. Report | `python src/make_report.py` | `REPORT.md` |

Steps 2 to 5 all need the baseline from step 1. Step 6 needs steps 1 to 5. Without activating the environment, use `uv run python src/<script>.py` instead.

To classify your own image with any model:

```
python src/infer.py --model qat --image cat.jpg
python src/onnx_infer.py --model structured --image cat.jpg
```

## 6. Result files

- `results/benchmark_quantization.csv`: FP32 vs PTQ-INT8 vs QAT-INT8 (accuracy, size, latency, throughput, speed-up, compression)
- `results/benchmark_pruning.csv`: FP32 vs unstructured vs structured (plus sparsity, non-zero parameters, MACs)
- `results/benchmark_onnx.csv`: the same models in ONNX Runtime
- `results/benchmark_all.csv`: all of the above in one table
- `results/predictions_all_models.csv`: prediction and correct flag of every model for each of the 10,000 test images
- `results/per_class_accuracy.csv`: accuracy per class for each model
- `results/unstructured_layer_sparsity.csv`: sparsity of each layer after global pruning
- `results/benchmark_env.txt`: CPU, thread count and library versions used for the benchmark
- `results/*.png`: comparison chart, accuracy vs size, PyTorch vs ONNX latency, baseline training curve

## 7. Submission

```
bash scripts/make_submission.sh 23i-XXXX_A
```

This creates `23i-XXXX_A.zip` in the project root. It leaves out `data/`, `.venv/` and caches.

## 8. Notes

- Model selection: no checkpoint is chosen using the test set. Every training and fine-tuning run follows a fixed schedule and keeps the final weights. Test accuracy is printed per epoch for monitoring only.
- ONNX INT8 calibration uses the Percentile method by default (`--calib-method minmax|percentile|entropy`). It runs in chunks of 100 images, so memory use stays low.
- The bonus model (structured pruning + INT8) is optional. Inference, the benchmark and the report include it automatically when it exists.

- Latency is measured on the CPU with 1 thread by default, which is closer to an edge device. Use `--threads N` to change this.
- Quantized models are saved as TorchScript so they can be loaded with `torch.jit.load` without rebuilding the Python model.
- The structured-pruned checkpoint stores `channel_cfg`, so `common.load_fp32_model()` rebuilds the smaller architecture automatically.
- On recent PyTorch versions, eager-mode quantization prints deprecation warnings. These are hidden and do not affect results.

## 9. Running on Google Colab (step by step)

Use this if your own machine has no NVIDIA GPU. Colab gives you a free GPU in the browser. Each line in a code block below goes into a notebook **cell**: type it, then press **Shift+Enter** to run it. Lines that start with `!` run shell commands, and `%cd` changes the folder.

### 9.1 Prepare the upload on your computer

From the project root, zip only the code (plus the trained baseline, if you want to skip retraining it):

```
zip -r assignment3.zip src scripts README.md requirements.txt models/fp32_resnet18.pth
```

If you don't have `models/fp32_resnet18.pth` yet, leave it out of the command.

### 9.2 Create the notebook and turn on the GPU

1. Go to https://colab.research.google.com and sign in with your Google account.
2. Click **New notebook**.
3. Open the menu **Runtime > Change runtime type**, choose **T4 GPU**, then click **Save**.
4. Check that the GPU works. Run this in the first cell:
   ```
   !nvidia-smi
   ```
   You should see a table that mentions "Tesla T4".

### 9.3 Upload and unzip the project

Run this in a new cell. Click **Choose Files** and pick `assignment3.zip`. Wait until the upload reaches 100%.

```
from google.colab import files
files.upload()
```

Then unzip it and move into the folder:

```
!unzip -q -o assignment3.zip -d Assignment3
%cd /content/Assignment3
!ls
```

`ls` should list `src`, `scripts`, `README.md` and `requirements.txt`.

### 9.4 Install the requirements

Colab already has PyTorch with GPU support, so only the ONNX packages are needed. Don't use uv or a virtual environment here.

```
!pip install -q onnx onnxruntime onnxscript
```

### 9.5 Run the pipeline

CIFAR-10 downloads automatically on the first run (about 1 minute). Choose one of these:

```
!bash scripts/run_all.sh                    # full run, including baseline training
!bash scripts/run_all.sh --skip-baseline    # if you uploaded models/fp32_resnet18.pth
```

The output appears under the cell while it runs. Keep the browser tab open: Colab disconnects idle sessions, and **all files are deleted when the session ends**. If it disconnects, repeat steps 9.3 to 9.5.

### 9.6 Download the results

When the last line says `Done.`, zip the outputs and download them:

```
!zip -r -q outputs.zip models onnx_models results logs REPORT.md
from google.colab import files
files.download("outputs.zip")
```

The file is about 250 MB, so the download can take a few minutes. If the browser download fails, copy it to your Google Drive instead. Click **Connect to Google Drive** in the window that opens:

```
from google.colab import drive
drive.mount("/content/drive")
!cp outputs.zip /content/drive/MyDrive/
```

Then download `outputs.zip` from https://drive.google.com.

### 9.7 Put the results into your project

On your computer, extract `outputs.zip` into the project root. This puts the files into `models/`, `onnx_models/`, `results/` and `logs/`, and adds `REPORT.md`. Then take the screenshots, fill in your details in `REPORT.md`, and run:

```
bash scripts/make_submission.sh 23i-XXXX_A
```

#!/usr/bin/env bash
# Reproducible end-to-end pipeline. Logs go to logs/.
# Usage (from any directory):
#   bash scripts/run_all.sh            full run on CIFAR-10
#   bash scripts/run_all.sh --quick    fewer epochs, about half the time, slightly lower accuracy
#   bash scripts/run_all.sh --fake     smoke test on random data (checks the code only)
#   bash scripts/run_all.sh --skip-baseline   reuse models/fp32_resnet18.pth (skip step 1)
# Flags can be combined, e.g.  --quick --skip-baseline
set -e -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Python from the uv virtual environment (.venv), unless PY is set
if [ -z "$PY" ]; then
  if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python"; fi
fi
echo "Using Python: $PY"

FAKE=""; SKIP_BASE=0
EP=30; QAT_EP=5; U_IT=4; U_FT=2; U_FINAL=5; S_FT=10
for arg in "$@"; do
  case "$arg" in
    --quick) EP=15; QAT_EP=3; U_IT=3; U_FT=1; U_FINAL=3; S_FT=5 ;;
    --fake)  FAKE="--fake"; EP=2; QAT_EP=2; U_IT=2; U_FT=1; U_FINAL=1; S_FT=1 ;;
    --skip-baseline) SKIP_BASE=1 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done
export PYTHONWARNINGS=ignore

mkdir -p logs models onnx_models results
if [ "$SKIP_BASE" == "1" ]; then
  [ -f models/fp32_resnet18.pth ] || { echo "models/fp32_resnet18.pth not found, run without --skip-baseline"; exit 1; }
  echo "== 1. Baseline FP32 training == (skipped, using models/fp32_resnet18.pth)"
else
  echo "== 1. Baseline FP32 training ==";  $PY src/train_baseline.py --epochs $EP $FAKE | tee logs/1_train_baseline.log
fi
echo "== 2. PTQ ==";                     $PY src/ptq.py --calib-images 2000 $FAKE | tee logs/2_ptq.log
echo "== 3. QAT ==";                     $PY src/qat.py --epochs $QAT_EP $FAKE | tee logs/3_qat.log
echo "== 4. Unstructured pruning ==";    $PY src/prune_unstructured.py --target-sparsity 0.8 --iterations $U_IT --ft-epochs $U_FT --final-ft-epochs $U_FINAL $FAKE | tee logs/4_prune_unstructured.log
echo "== 5. Structured pruning ==";      $PY src/prune_structured.py --ratio 0.5 --ft-epochs $S_FT $FAKE | tee logs/5_prune_structured.log
echo "== 5b. Bonus: structured pruned + PTQ INT8 =="
$PY src/ptq.py --input models/structured_pruned_resnet18.pth --output models/structured_pruned_ptq_int8_resnet18.pt $FAKE | tee logs/5b_structured_ptq.log
echo "== 6. ONNX export ==";             $PY src/export_onnx.py $FAKE | tee logs/6_export_onnx.log
echo "== 7. Inference demo ==";          $PY src/infer.py --model all --test-samples 5 $FAKE | tee logs/7_infer.log
                                         $PY src/onnx_infer.py --model all --test-samples 5 $FAKE | tee logs/7_onnx_infer.log
echo "== 8. Benchmark ==";               $PY src/benchmark.py --threads 1 $FAKE | tee logs/8_benchmark.log
echo "== 9. Report ==";                  $PY src/make_report.py | tee logs/9_report.log
echo "Done. See results/, logs/ and REPORT.md."

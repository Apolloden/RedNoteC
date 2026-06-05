#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rednote_aigt_mpl}"
mkdir -p "$MPLCONFIGDIR" outputs/reports

TRANSFORMER_LOG="outputs/reports/transformer_mps_smoke_train.log"

echo "1/7 Device diagnostics"
python3 scripts/check_device.py || exit 1

echo "2/7 Running unit tests"
python3 -m pytest || exit 1

echo "3/7 Training TF-IDF smoke model"
python3 scripts/train.py \
  --model tfidf_logreg \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --force || exit 1

echo "4/7 Evaluating TF-IDF smoke model"
python3 scripts/evaluate.py \
  --model-dir models/tfidf_logreg \
  --max-test-samples 128 \
  --device auto || exit 1

echo "5/7 Training transformer Apple Silicon smoke model"
if python3 scripts/train.py \
  --model transformer_roberta \
  --max-train-samples 64 \
  --max-val-samples 32 \
  --max-steps 3 \
  --batch-size 2 \
  --max-length 128 \
  --device auto \
  --force 2>&1 | tee "$TRANSFORMER_LOG"; then
  echo "6/7 Evaluating transformer Apple Silicon smoke model"
  if python3 scripts/evaluate.py \
    --model-dir models/transformer_roberta \
    --max-test-samples 32 \
    --batch-size 2 \
    --max-length 128 \
    --device auto; then
    echo "Transformer MPS/CPU smoke evaluation passed"
  else
    cat > outputs/reports/transformer_mps_smoke_status.md <<'EOF'
# Transformer MPS Smoke Status

Transformer training completed, but transformer evaluation failed.

Rerun after resolving the environment issue:

```bash
python3 scripts/evaluate.py --model-dir models/transformer_roberta --max-test-samples 32 --batch-size 2 --max-length 128 --device auto
```
EOF
  fi
else
  {
    cat <<'EOF'
# Transformer MPS Smoke Status

Transformer smoke training failed in this environment. This is usually caused by
missing internet access for Hugging Face model downloads, an unavailable cached
model, or local compute/runtime limits.

TF-IDF smoke training and evaluation completed before this step.

Run locally once internet/GPU or cached Hugging Face models are available:

```bash
python3 scripts/train.py --model transformer_roberta --max-train-samples 64 --max-val-samples 32 --max-steps 3 --batch-size 2 --max-length 128 --device auto --force
python3 scripts/evaluate.py --model-dir models/transformer_roberta --max-test-samples 32 --batch-size 2 --max-length 128 --device auto
```
EOF
    echo
    echo "## Captured Error Log"
    echo
    echo "Full log: \`outputs/reports/transformer_mps_smoke_train.log\`"
    echo
    echo '```text'
    tail -n 100 "$TRANSFORMER_LOG"
    echo '```'
  } > outputs/reports/transformer_mps_smoke_status.md
  echo "Transformer MPS smoke training failed; wrote outputs/reports/transformer_mps_smoke_status.md"
fi

echo "7/7 Smoke test outputs:"
echo "- TF-IDF reports: outputs/reports/tfidf_logreg"
echo "- TF-IDF figures: outputs/figures/tfidf_logreg"
echo "- Transformer reports: outputs/reports/transformer_roberta"
echo "- Transformer figures: outputs/figures/transformer_roberta"

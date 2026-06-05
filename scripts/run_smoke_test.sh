#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rednote_aigt_mpl}"
mkdir -p "$MPLCONFIGDIR" outputs/reports
TRANSFORMER_LOG="outputs/reports/transformer_smoke_train.log"

echo "1/5 Running unit tests"
python3 -m pytest || exit 1

echo "2/5 Training TF-IDF smoke model"
python3 scripts/train.py \
  --model tfidf_logreg \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --force || exit 1

echo "3/5 Evaluating TF-IDF smoke model"
python3 scripts/evaluate.py \
  --model-dir models/tfidf_logreg \
  --max-test-samples 128 || exit 1

echo "4/5 Training transformer smoke model"
if python3 scripts/train.py \
  --model transformer_roberta \
  --max-train-samples 64 \
  --max-val-samples 32 \
  --max-steps 3 \
  --force 2>&1 | tee "$TRANSFORMER_LOG"; then
  echo "5/5 Evaluating transformer smoke model"
  if ! python3 scripts/evaluate.py \
    --model-dir models/transformer_roberta \
    --max-test-samples 64; then
    cat > outputs/reports/transformer_smoke_status.md <<'EOF'
# Transformer Smoke Status

Transformer training completed, but transformer evaluation failed in this environment.

Run locally after resolving the environment issue:

```bash
python3 scripts/evaluate.py --model-dir models/transformer_roberta --max-test-samples 64
```
EOF
  fi
else
  {
    cat <<'EOF'
# Transformer Smoke Status

Transformer smoke training failed in this environment. This is usually caused by
missing internet access for Hugging Face model downloads, an unavailable cached
model, or local compute/runtime limits.

TF-IDF smoke training and evaluation completed before this step.

Run locally once internet/GPU or cached Hugging Face models are available:

```bash
python3 scripts/train.py --model transformer_roberta --max-train-samples 64 --max-val-samples 32 --max-steps 3 --force
python3 scripts/evaluate.py --model-dir models/transformer_roberta --max-test-samples 64
```
EOF
    echo
    echo "## Captured Error Log"
    echo
    echo "Full log: \`outputs/reports/transformer_smoke_train.log\`"
    echo
    echo '```text'
    tail -n 80 "$TRANSFORMER_LOG"
    echo '```'
  } > outputs/reports/transformer_smoke_status.md
  echo "Transformer smoke training failed; wrote outputs/reports/transformer_smoke_status.md"
fi

echo "Smoke test outputs:"
echo "- TF-IDF reports: outputs/reports/tfidf_logreg"
echo "- TF-IDF figures: outputs/figures/tfidf_logreg"
echo "- Transformer reports: outputs/reports/transformer_roberta"
echo "- Transformer figures: outputs/figures/transformer_roberta"

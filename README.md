# RedNoteC

Small, reproducible NLP research project for detecting AI-generated Chinese
RedNote/Xiaohongshu posts from RedNote-Vibe.

This repository currently implements the data phase only: loading, conservative
cleaning, auditing, deduplication, and fixed train/validation/test splits.
It also includes the first model layer: one TF-IDF baseline, one transformer
classifier, shared evaluation, and smoke-test commands.

## Data

Large dataset files are intentionally not committed. The repository tracks only
the folder placeholders; local raw, interim, and processed data are ignored by
`.gitignore`.

These local artifact folders are ignored and should not be committed:

```text
data/
docs/
outputs/
models/
checkpoints/
```

Download RedNote-Vibe from:

```text
https://github.com/ydli-ai/RedNote-Vibe/tree/main
```

Upload or copy the supervised training files into this local folder:

```text
data/raw/training_set_human.jsonl
data/raw/training_set_aigc.jsonl
```

`data/raw/exploration_set.jsonl` may also be stored locally, but it is not used
for supervised training or metrics because it is unlabeled/post-LLM analysis
data.

To reproduce the current processed data exactly, use the same raw files. The
current local files have these SHA-256 checksums:

```text
c737fe5d8b40dc21a3c61657e42e5359942a6abdf6837f31ad1dcef696eb1054  data/raw/training_set_human.jsonl
288d6ec75512301dd5accaa22205204ce626d13b61753bc3e7ff005061108659  data/raw/training_set_aigc.jsonl
```

## Research Rule

The main classifier input is text-only:

```text
text = note_title + note_content
```

Metadata such as `local_time`, `likes`, `collections`, `comments`, `domain`,
`model_family`, and `model` is retained only for audit, stratification,
subgroup evaluation, and error analysis. These columns must not be used as
model features for the main binary classification task.

Label convention:

```text
0 = human
1 = AI
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development, including tests:

```bash
python3 -m pip install -e ".[dev]"
```

## Reproduce Current Processed Data

Follow these steps from the repository root.

1. Create the raw data folder if it does not already exist:

```bash
mkdir -p data/raw
```

2. Put the downloaded RedNote-Vibe files here:

```text
data/raw/training_set_human.jsonl
data/raw/training_set_aigc.jsonl
```

3. Optional but recommended: verify the raw file checksums match the current
local data:

```bash
shasum -a 256 data/raw/training_set_human.jsonl data/raw/training_set_aigc.jsonl
```

Expected:

```text
c737fe5d8b40dc21a3c61657e42e5359942a6abdf6837f31ad1dcef696eb1054  data/raw/training_set_human.jsonl
288d6ec75512301dd5accaa22205204ce626d13b61753bc3e7ff005061108659  data/raw/training_set_aigc.jsonl
```

4. Install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

5. Generate the full processed dataset. Do not pass `--sample` for the real
experiment:

```bash
python3 scripts/prepare_data.py --config configs/data.yaml --force
```

6. Confirm the output counts:

```bash
python3 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("data/processed/dataset_manifest.json").read_text(encoding="utf-8"))
print(manifest["counts"])
print(manifest["dedupe_report"])
PY
```

Expected:

```text
{'train': 41217, 'val': 8832, 'test': 8833}
{'input_rows': 59132, 'conflict_text_values': 0, 'conflict_rows_removed': 0, 'within_label_duplicate_rows_removed': 250, 'output_rows': 58882, 'label_conflicts_path': 'data/interim/label_conflicts.csv'}
```

The split is controlled by [configs/data.yaml](configs/data.yaml): seed `42`,
train/validation/test ratio `70/15/15`, and label+domain stratification when
feasible.

## Audit Data

```bash
python3 scripts/audit_data.py --config configs/data.yaml
```

Audit outputs:

```text
outputs/reports/data_audit.json
outputs/reports/data_audit.md
```

## Prepare Data

```bash
python3 scripts/prepare_data.py --config configs/data.yaml
```

Use a quick sample run while developing:

```bash
python3 scripts/prepare_data.py --config configs/data.yaml --sample 100 --force
```

Processed outputs:

```text
data/interim/cleaned_all.csv
data/interim/label_conflicts.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/dataset_manifest.json
outputs/reports/prepare_summary.json
```

The processed split files contain `text` and `label` for future model training,
plus metadata columns for analysis only.

## Models

### TF-IDF Baseline

The required baseline is `tfidf_logreg`: character n-gram TF-IDF followed by
class-weighted Logistic Regression.

Character n-grams are a strong first baseline for Chinese social media because
they do not require jieba or any other word segmenter, and they preserve useful
signals from slang, emojis, hashtags, punctuation, and mixed-language text.
Logistic Regression is used because it is stable, fast, interpretable, and
produces probabilities for AUROC and AUPRC. `class_weight="balanced"` addresses
the strong human/AI imbalance.

The default solver is `liblinear` because it is stable for binary logistic
regression on sparse TF-IDF features and works well for this baseline size.

### Transformer

The required transformer is `transformer_roberta`, using
`hfl/chinese-roberta-wwm-ext` with `AutoTokenizer` and
`AutoModelForSequenceClassification`.

This is a Chinese BERT/RoBERTa-style base model with whole word masking. It is a
better fit than English or generic multilingual baselines for Chinese contextual
classification, while remaining more practical than a large model under limited
compute.

Full transformer training may require a GPU and internet access to download the
Hugging Face model. Smoke tests may fall back to a tiny Hugging Face test model
only to verify code paths; fallback smoke-test performance is not a research
result.

## Apple Silicon

The transformer code is optimized for Apple Silicon MacBooks, including M1 Pro.
It uses PyTorch MPS automatically when available and falls back to CPU when MPS
is unavailable. CUDA is not assumed.

Recommended local setup:

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/check_device.py
bash scripts/run_mps_smoke_test.sh
```

The device selector checks:

```text
torch.backends.mps.is_built()
torch.backends.mps.is_available()
```

Default behavior:

```text
MPS available -> device=mps
MPS unavailable -> device=cpu
```

The first transformer run may need internet access to download
`hfl/chinese-roberta-wwm-ext` from Hugging Face. Full transformer training on an
M1 Pro can still take a while. If memory errors occur, reduce `--batch-size` or
`--max-length`; the default config uses a small per-device batch size and
gradient accumulation for Apple Silicon.

## Smoke Test

Run the full local smoke test:

```bash
bash scripts/run_smoke_test.sh
```

This runs unit tests, trains/evaluates a small TF-IDF model, then attempts a
tiny transformer training/evaluation run. If transformer download or runtime is
blocked by the environment, the script writes:

```text
outputs/reports/transformer_smoke_status.md
```

Smoke-test metrics only verify that the architecture, training code, evaluation
metrics, and plots work. Do not report them as research results.

## Train And Evaluate

Train TF-IDF:

```bash
python3 scripts/train.py --model tfidf_logreg --config configs/models.yaml --force
```

Evaluate TF-IDF:

```bash
python3 scripts/evaluate.py --model-dir models/tfidf_logreg --test-path data/processed/test.csv
```

Train the transformer:

```bash
python3 scripts/train.py --model transformer_roberta --config configs/models.yaml --device auto --force
```

Evaluate the transformer:

```bash
python3 scripts/evaluate.py --model-dir models/transformer_roberta --test-path data/processed/test.csv --device auto
```

Training always selects only:

```text
X = df["text"]
y = df["label"]
```

Metadata columns are never used as model features.

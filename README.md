# RedNoteC

Small, reproducible NLP research project for detecting AI-generated Chinese
RedNote/Xiaohongshu posts from RedNote-Vibe.

This repository currently implements the data phase only: loading, conservative
cleaning, auditing, deduplication, and fixed train/validation/test splits.
Model training is intentionally not implemented yet.

## Data

Large dataset files are intentionally not committed. The repository tracks only
the folder placeholders; local raw, interim, and processed data are ignored by
`.gitignore`.

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

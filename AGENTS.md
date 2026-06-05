# AGENTS.md

## Project goal

Build a small, reproducible NLP research project for detecting AI-generated Chinese RedNote/Xiaohongshu posts. The primary experiment compares:

1. a TF-IDF baseline, and
2. one transformer classifier.

The project must prioritize clean data handling, reproducible splits, comparable evaluation, and interpretable analysis over adding many models.

## Non-negotiable research rule

For the main binary classification task, model inputs must be derived only from:

- `note_title`
- `note_content`

Do not train the main classifiers on `local_time`, `likes`, `collections`, `comments`, `domain`, `model_family`, or `model`.

Allowed uses of metadata:

- `domain`: stratified splitting if possible, subgroup reporting, error analysis.
- `model_family` and `model`: AI-generator subgroup reporting and optional held-out-generator experiments.
- `local_time`: dataset audit and temporal analysis only; do not use for training.
- `likes`, `collections`, `comments`: engagement analysis only; do not use for training.

Reason: the goal is to detect AI-generated writing from text, not exploit metadata leakage or artifacts that would be unavailable, unfair, or non-generalizable in real deployment.

## Expected raw dataset files

Store downloaded RedNote-Vibe files under `data/raw/`:

```text
data/raw/
  training_set_human.jsonl
  training_set_aigc.jsonl
```

Only `training_set_human.jsonl` and `training_set_aigc.jsonl` are used for supervised binary classification.

## Recommended project structure

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── tfidf.yaml
│   └── transformer.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   │   ├── rednote_clean.parquet
│   │   └── audit_report.json
│   └── processed/
│       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       └── split_manifest.json
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_rednote.py
│   │   ├── clean.py
│   │   ├── split.py
│   │   └── audit.py
│   ├── features/
│   │   ├── text_building.py
│   │   └── simple_text_stats.py
│   ├── models/
│   │   ├── base.py
│   │   ├── tfidf_linear.py
│   │   ├── transformer_cls.py
│   │   └── registry.py
│   ├── training/
│   │   ├── train_tfidf.py
│   │   └── train_transformer.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   └── compare_runs.py
│   ├── analysis/
│   │   ├── error_analysis.py
│   │   └── subgroup_analysis.py
│   └── utils/
│       ├── io.py
│       ├── logging.py
│       └── seed.py
├── scripts/
│   ├── 01_audit_data.py
│   ├── 02_prepare_splits.py
│   ├── 03_train_tfidf.py
│   ├── 04_train_transformer.py
│   ├── 05_evaluate.py
│   └── 06_error_analysis.py
├── outputs/
│   ├── runs/
│   ├── predictions/
│   ├── metrics/
│   ├── figures/
│   └── models/
├── notebooks/
│   └── exploratory_analysis.ipynb
└── tests/
    ├── test_cleaning.py
    ├── test_splits.py
    └── test_metrics.py
```

## Canonical processed dataset schema

Every processed split file must contain at least:

```text
id                 stable row id created by us
text               final input text = title + separator + content
label              0 = human, 1 = AI
note_title         cleaned title
note_content       cleaned content
domain             retained for stratification and subgroup analysis only
model_family       retained for AI subgroup analysis only; null for human
model              retained for AI subgroup analysis only; null for human
source_file        original file name
split              train, val, or test
text_len_chars     character length of final text
```

The training code must read only `text` and `label` unless a specific ablation explicitly says otherwise.

## Data pipeline requirements

### `src/data/load_rednote.py`

Responsibilities:

- Load JSONL files safely.
- Attach labels: human = 0, AI = 1.
- Preserve raw fields needed for audit and subgroup analysis.
- Never silently drop malformed rows without counting and reporting them.

### `src/features/text_building.py`

Responsibilities:

- Build the canonical `text` field from title and content.
- Use a clear separator, for example:

```text
标题：{note_title}\n正文：{note_content}
```

- If title is empty, use only content.
- If content is empty, drop the row unless the title alone is long enough and this decision is documented.

### `src/data/clean.py`

Responsibilities:

- Normalize whitespace.
- Remove empty or near-empty posts.
- Remove exact duplicate `text` values.
- Preserve emojis, hashtags, punctuation, informal wording, and code-switching unless there is a documented reason to normalize them.
- Do not remove stopwords for transformer input.
- Any stronger normalization for TF-IDF must be controlled by config and reported.

### `src/data/split.py`

Responsibilities:

- Create train/val/test splits once and save them.
- Use fixed random seed.
- Stratify by `label`; also stratify by `domain` when feasible.
- Prevent exact duplicate texts from appearing across splits.
- Save `split_manifest.json` with counts by label, domain, and AI model.
- Prefer 70/15/15 or 80/10/10. Use one split consistently for all models.

Optional stricter split:

- If original human-to-AI pairing information exists or can be reconstructed, use group-aware splitting so related human seeds and AI variants do not cross train/test boundaries.

## Model interface

Every model should follow the same interface so models can be swapped easily.

```python
class TextClassifier:
    def fit(self, train_df, val_df=None): ...
    def predict(self, texts): ...
    def predict_proba(self, texts): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

`predict_proba` must return a score for the AI class when possible. AUROC should be computed from this score, not from hard labels.

## Initial models

### TF-IDF baseline

Implement in `src/models/tfidf_linear.py`.

Recommended default:

- `TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=200000)`
- `LogisticRegression(max_iter=2000, class_weight="balanced")`

Alternative:

- `LinearSVC(class_weight="balanced")` with calibrated probabilities if AUROC is needed.

Character n-grams are preferred for the first baseline because Chinese social media text may include slang, emojis, hashtags, and irregular segmentation.

### Transformer classifier

Implement in `src/models/transformer_cls.py`.

Recommended default:

- `bert-base-chinese` for speed and availability, or `hfl/chinese-roberta-wwm-ext` if available.
- max length: 256 or 384.
- epochs: 1-3.
- save best model by validation F1 or AUROC.
- evaluate on the same fixed test split as TF-IDF.

## Evaluation requirements

Evaluation code belongs in `src/evaluation/`, not inside model files.

Report at least:

- accuracy
- precision
- recall
- F1
- AUROC
- confusion matrix

Also save:

```text
outputs/predictions/{run_name}_test_predictions.csv
outputs/metrics/{run_name}_metrics.json
outputs/figures/{run_name}_confusion_matrix.png
```

Prediction CSV should include:

```text
id, text, label, pred_label, score_ai, split, domain, model_family, model
```

## Analysis requirements

Keep analysis separate from training.

Minimum analysis:

- dataset audit: counts, lengths, duplicates, class balance, domain balance.
- subgroup metrics by domain.
- subgroup metrics by AI model or model family, if enough examples exist.
- false positives and false negatives sampled for qualitative inspection.
- simple text statistics by correctness group: length, emoji count, hashtag count, punctuation ratio, repetition rate.

Optional ablations, only after main comparison works:

1. content only vs title + content.
2. title only vs content only.
3. char TF-IDF vs word/jieba TF-IDF.
4. with vs without hashtags/emojis.
5. held-out AI model family evaluation.
6. domain-held-out evaluation.

## Reproducibility rules

- Set all random seeds.
- All scripts must be runnable from the repository root.
- Use config files for hyperparameters.
- Never overwrite processed splits unless the user explicitly asks.
- Every run should create a run folder under `outputs/runs/{timestamp}_{model_name}/`.
- Save the exact config used for each run.

## Suggested command flow

```bash
python scripts/01_audit_data.py --config configs/data.yaml
python scripts/02_prepare_splits.py --config configs/data.yaml
python scripts/03_train_tfidf.py --config configs/tfidf.yaml
python scripts/04_train_transformer.py --config configs/transformer.yaml
python scripts/05_evaluate.py --predictions outputs/predictions
python scripts/06_error_analysis.py --predictions outputs/predictions
```

## Coding style

- Prefer simple, readable Python over clever abstractions.
- Use type hints for public functions.
- Use `pathlib.Path` instead of raw path strings.
- Use logging instead of print statements in library code.
- Avoid notebooks for core logic; notebooks are only for exploration and figures.
- Put reusable code in `src/`, not in `scripts/`.

## What not to do

- Do not train on `model`, `model_family`, engagement counts, timestamps, or domain labels in the main experiment.
- Do not use the exploration set for supervised metrics.
- Do not create different random splits for different models.
- Do not report accuracy alone.
- Do not tune on the test set.
- Do not add new models until the TF-IDF and transformer comparison is complete.

## Deadline strategy

Priority order:

1. Clean data and fixed splits.
2. TF-IDF baseline.
3. Transformer classifier.
4. Shared evaluation table.
5. Error and subgroup analysis.
6. Optional ablations.

A small, reproducible comparison with honest limitations is better than many incomplete models.

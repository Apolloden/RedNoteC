# Model Choice

## TF-IDF Character N-Grams + Logistic Regression

The baseline is character n-gram TF-IDF with class-weighted Logistic Regression.
Chinese RedNote/Xiaohongshu posts contain informal language, emojis, hashtags,
punctuation, slang, and inconsistent segmentation. Character n-grams avoid a
hard dependency on jieba or word segmentation while still capturing local
surface patterns.

Logistic Regression is used because it is fast, stable, and produces calibrated
enough probability scores for AUROC and AUPRC. `class_weight="balanced"` is used
because the processed data is strongly imbalanced, with far more human rows than
AI rows.

The default solver is `liblinear`; for this binary sparse-feature baseline it is
stable and deterministic enough for a first reproducible comparison.

## Transformer

The transformer model is `hfl/chinese-roberta-wwm-ext` through Hugging Face
`AutoTokenizer` and `AutoModelForSequenceClassification` with `num_labels=2`.
It is a Chinese pretrained BERT/RoBERTa-style base model with whole word
masking, making it a practical contextual classifier for Chinese text. The base
model is selected rather than a large model because the project prioritizes a
small reproducible comparison under limited time and compute.

If the default model cannot be downloaded or loaded during smoke testing, the
code may fall back to a tiny Hugging Face test model or `bert-base-chinese`.
That fallback is only for verifying the software path and is not the final
research model.

## Metrics

This is binary classification, not text generation, so BLEU, ROUGE, and
BERTScore are not appropriate. The evaluation reports accuracy, balanced
accuracy, per-class precision/recall/F1, macro F1, weighted F1, AUROC, AUPRC,
confusion matrix, and subgroup metrics.

Accuracy alone is misleading because the data is imbalanced. AI recall matters
because missing AI-generated posts is central to the task. Macro F1 treats human
and AI labels equally. AUROC and AUPRC evaluate ranking quality from model
scores.

Two conventions make the numbers comparable across slices:

- Every metric is computed over the fixed label set `[0, 1]`. Without this,
  scikit-learn averages over the labels it happens to observe, and a subgroup
  containing only AI rows would report a two-class macro score that is not
  comparable to the same metric on the full split.
- AUROC and AUPRC are reported as empty when only one class is present, rather
  than as a substituted number.

Threshold metrics (precision, recall, F1, accuracy) depend on both the model and
its calibration. `class_weight="balanced"` shifts the baseline's probabilities
toward the positive class, so at a shared 0.5 threshold the two models sit at
different operating points on their own curves. AUROC and AUPRC are threshold-
free and are the cleaner head-to-head comparison; the threshold metrics answer
"what happens if you deploy this as configured".

## Generator holdout

The random split measures detection of generators the model was trained on. It
cannot separate "this text was written by a machine" from "this text was written
by one of the six machines in my training set", because both hypotheses predict
the same in-distribution score.

`scripts/holdout_generator.py` separates them. For each generator family, the
model is retrained with that family's posts removed from training *and*
validation, then scored on the standard test split reduced to human posts plus
that family's posts. Because the test rows are the same ones the main model was
scored on in `subgroup_metrics_model_family.csv`, the two recalls differ only in
whether the family was in training.

Read `recall_ai` across folds. Precision and AUPRC are not comparable between
folds: each fold keeps all 7,747 human test rows but only one family's AI rows,
so prevalence changes with the family. AUROC and the false-positive count stay
comparable.

Removing a family also removes 10-24% of the AI training rows, so a recall drop
could in principle be data volume rather than novelty.
`build_size_matched_control` isolates that: it removes the same number of AI
rows sampled at random across all families while keeping the target family in
training, which splits the drop into a volume component and a novelty
component.

## Limitations

- Class imbalance means accuracy can look high even when AI recall is weak.
- Title presence is label-skewed: 14.29% of human rows have no title against 0%
  of AI rows, so the `标题：` line in the canonical text is itself weak label
  evidence. `scripts/audit_data.py` reports `empty_title_counts_by_label`.
- The transformer truncates input at 256 tokens while TF-IDF sees the full post,
  so the two models are not given identical information on long posts.
- Text length differs by label; human rows are longer on average.
- Domain labels are noisy and some examples appear domain-inconsistent.

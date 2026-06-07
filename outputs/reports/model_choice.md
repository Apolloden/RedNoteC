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

## Limitations

- Class imbalance means accuracy can look high even when AI recall is weak.
- Title presence is label-skewed: many human rows have empty titles while AI rows
  in the processed data do not.
- Text length differs by label; human rows are longer on average.
- Domain labels are noisy and some examples appear domain-inconsistent.
- The current split is a fixed random label+domain stratified split, not a
  held-out-generator evaluation.

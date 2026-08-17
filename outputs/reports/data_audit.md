# Data Audit

- Total rows: 59132
- Label counts: `{'0': 51878, '1': 7254}`
- Duplicate texts: `{'duplicate_rows_overall': 490, 'duplicate_text_values_overall': 250, 'duplicate_rows_by_label': {'0': 490}}`
- Human/AIGC exact text overlap: `{'overlap_text_values': 0, 'rows_in_overlap': 0}`
- Empty titles by label (0=human, 1=AI): `{'0': {'rows': 51878, 'empty_title_rows': 7411, 'empty_title_share': 0.142854}, '1': {'rows': 7254, 'empty_title_rows': 0, 'empty_title_share': 0.0}}`
- Suspicious columns excluded from model features: `['local_time', 'likes', 'collections', 'comments', 'domain', 'model_family', 'model']`

## Missing Fields

```json
{
  "note_title": 0,
  "note_content": 0,
  "label": 0,
  "source_file": 0,
  "source_line": 0,
  "note_id": 7254,
  "local_time": 7254,
  "likes": 7254,
  "collections": 7254,
  "comments": 7254,
  "domain": 0,
  "model_family": 51878,
  "model": 51878,
  "text": 0,
  "title_only": 0,
  "text_len_chars": 0,
  "id": 0
}
```

## Domain Counts By Label

```json
{
  "0": {
    "健康": 9440,
    "学习": 3744,
    "宠物": 1686,
    "心理": 4241,
    "情感": 6825,
    "旅行": 6195,
    "穿搭": 5101,
    "美食": 8006,
    "职场": 5468,
    "运动": 1172
  },
  "1": {
    "健康": 1122,
    "学习": 243,
    "宠物": 169,
    "心理": 764,
    "情感": 1099,
    "旅行": 837,
    "穿搭": 1141,
    "美食": 939,
    "职场": 843,
    "运动": 97
  }
}
```

## AIGC Model Counts

```json
{
  "model_family": {
    "gpt": 1730,
    "deepseek": 1593,
    "qwen": 1250,
    "gemini": 1153,
    "glm": 792,
    "anthropic": 736
  },
  "model": {
    "deepseek-r1-250528": 653,
    "deepseek-r1-250120": 556,
    "gemini-1.5": 473,
    "gemini-2.0": 462,
    "gpt-o3": 449,
    "gpt-4.1": 443,
    "qwen2.5": 439,
    "glm-3": 424,
    "gpt-4o": 423,
    "gpt-o4": 415,
    "qwen2": 409,
    "qwen3": 402,
    "claude-sonnet-4": 385,
    "deepseek-v3": 384,
    "glm-4": 368,
    "claude-3-7-sonnet": 351,
    "gemini-2.5": 218
  }
}
```

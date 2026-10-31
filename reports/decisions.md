# Decisions Log

## 1. Project scope
- Task: detect harmful comments in a moderation pipeline that handles about 10,000 comments per day.
- Architecture: hybrid. Rules handle clear cases, a calibrated TF-IDF + logistic regression classifier handles most comments, and Claude Haiku reviews only ambiguous comments.
- Human review: any outcome that could lead to an account penalty goes to a human. No automated penalties.
- Baselines for comparison: (a) classifier only, (b) LLM on every comment, (c) hybrid.

## 2. Datasets
- **Jigsaw Toxic Comment Classification (Kaggle):** Wikipedia talk page comments, 159,571 rows, six binary labels. Used for training and validation. Source: Kaggle competition download, `train.csv`.
- **Measuring Hate Speech (Hugging Face, ucberkeley-dlab):** Reddit, Twitter, and YouTube comments, 135,556 annotation rows for 39,565 unique comments. Used for the stress test and identity audit.
- **Civil Comments:** Considered and dropped. Reason: adds TensorFlow dependency and overlaps with Measuring Hate Speech's identity labels.

## 3. Label definitions
- **Jigsaw primary label:** `toxic` (binary, as provided).
- **Jigsaw secondary labels:** `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. Reported separately because of small sample sizes.
- **Measuring Hate Speech label:** `hate = hate_speech_score > 0.5`. Threshold confirmed from the dataset card.
- **Ambiguous band:** `hate_speech_score` between -1 and 0.5 per the dataset card. Used for slice-level evaluation.
- **Counter/supportive speech:** `hate_speech_score` below -1. Treated as not hateful.

## 4. Aggregation
- Measuring Hate Speech has multiple annotator rows per comment. Aggregated to one row per `comment_id` using the mean score.
- Verified that `hate_speech_score` is identical across annotators for each comment (std = 0 for all comments with 2+ rows). Aggregation therefore has no effect on the result.
- Identity-related definition: a comment counts as identity-targeted if any annotator flagged any `target_*` group. This measures what the comment is about, not whether it contains an identity word.

## 5. Splits
- Jigsaw: duplicates removed on `comment_text`. Split 70/15/15 with `random_state=42`, stratified on `toxic`.
  - Train 111,699 / Validation 23,936 / Test 23,936. Toxic rate 9.58% in each.
- Measuring Hate Speech: aggregated first, then split 70/15/15 with `random_state=42`, stratified on the hate label.
  - Train 27,695 / Validation 5,935 / Test 5,935. Hate rate 26.17% in each.
- Splits are saved to `data/processed/` and are fixed. Never regenerate them mid-experiment.

## 6. Evaluation design
- **Main evaluation:** Jigsaw test split, primary metric F1 and recall on the toxic class. Accuracy reported only as a secondary figure, since 90% of comments are not toxic.
- **Stress test:** Train on Jigsaw, test on Measuring Hate Speech test split. Expected to drop because the datasets differ in source and label definition. Reported as a generalization result, not as a failure.
- **Identity audit:** Run on Measuring Hate Speech using the `target_*` columns. Metrics: false positive and false negative rates per group, counterfactual swap test, and manual review of 50 to 100 identity-term false positives.

## 7. Known limitations (to report)
- Both datasets are English only.
- Jigsaw is from 2017–2018 and Wikipedia talk pages. Measuring Hate Speech is from 2020 and Reddit, Twitter, and YouTube. Neither reflects current platform language.
- Rare labels (`threat` 0.3%, `identity_hate` 0.9%, `severe_toxic` 1.0%) have too few examples for reliable scores.
- Labels come from crowd annotators and can disagree.

## 8. Open decisions
- Final thresholds for routing (low and high confidence bands): to be chosen on the Jigsaw validation split only.
- LLM prompt and cost assumptions: to be set in Phase 3.
- Whether to use the card's `binary` config for Measuring Hate Speech: not needed, since the score-based label already matches the card's recommended cutoff.
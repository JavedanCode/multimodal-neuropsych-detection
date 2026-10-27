# Stage 4 target and evaluation protocol

Status: provisional development baselines, fixed before real-data model fitting
on 2026-10-27. These experiments reproduce the released PHQ_Binary field. They
must not be described as confirmed PHQ-8 >=10 screening or clinical diagnoses.

## Label investigation

The official E-DAIC manual lists PHQ_Binary and PHQ_Score, but does not specify a
binary derivation rule or explain the observed disagreement. Locally, 20 of 275
records have score >=10 and released binary 0 (13 train, 3 dev, 4 test). In the
training set, detailed_lables.csv Depression_label agrees with the split binary
for all 163 participants; the eight item sums agree with PHQ_Score for all 163.
Thus the discrepancy is not resolved by using that supplementary file. The
previously observed 2/219 disagreement with Detailed_PHQ8_Labels remains separate.

A published study also reports 20 discordant E-DAIC labels and chooses a score-
derived outcome. That is a distinct target policy, not an official correction we
can silently apply. We retain supplied labels and record their unresolved meaning.
A score-derived sensitivity experiment needs a separate protocol and output path;
it must never be selected because its scores look better.

Sources inspected on 2026-10-27:

- USC E-DAIC manual: https://dcapswoz.ict.usc.edu/wwwedaic/E-DAIC%20Manual.pdf
- Leakage-Free Multimodal Depression Screening: Controlled Evaluation of Text,
  Facial Behavior, and Prosodic Fusion: https://www.mdpi.com/2306-5354/13/9/1009
  (publisher search-index excerpt corroborates the discrepancy; full-page fetch
  returned HTTP 429 in this session).
- The requested Mathematics benchmark page also returned HTTP 429. Its reported
  scores and exact protocol have not been verified and will not be compared here.

## Predeclared experiment

- Data: unnormalized Stage 3 participant summaries, unchanged official partitions.
- Target: PHQ_Binary exactly as released; no scores, demographics, IDs or quality
  indicators enter the classifier features.
- Models: audio-only, facial-only, concatenation logistic regression, plus a
  training-majority DummyClassifier reference.
- Selection: stratified five-fold training-only CV, positive-class F1. Median
  imputation, constant/duplicate-column removal, and scaling refit inside each fold.
- Candidates: C in [0.01, 0.1, 1.0], class_weight in [null, balanced]. Balanced
  weights are calculated from the training fold by the estimator.
- Seeds: 42 (primary), 137, 2026. Report all three; never choose the best seed.
- Threshold: fixed 0.5; no threshold tuning or probability calibration.
- Final fit for each seed: official training participants only.
- Evaluation: development set only. The official test is not predicted or scored.
- Report positive-class precision/recall/F1, macro-F1, ROC-AUC, accuracy, specificity,
  confusion matrix, class counts, feature counts, timing, selected parameters,
  complete candidate scores and CV participant membership.
- CV selection scores are not unbiased estimates of generalization. Across-seed
  variation reflects changed selection folds, not independent cohorts or CIs.
- No benchmark-superiority or novelty claim. Stage 5 is not part of this run.

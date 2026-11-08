# Corrected voicing aggregation and common-scale late fusion — 2026-11-08

Freeze this protocol before building features or fitting. This is a correction
of two code defects found by review, not a new search. It reuses the 161-person
training cohort, unchanged released labels, original 98 facial summaries, the two
facial timing quarantines, the existing nine quality fields and the existing
joint label/domain splits (5 outer folds, seeds 42, 137, 2026; 3 inner folds with
seed = outer seed + fold + 1). No development/test evaluation, relabeling,
threshold search, new C grid, feature selection or text predictor.

## Defect 1: unvoiced placeholders in supplied eGeMAPS summaries

The supplied `OpenSMILE2.3.0_egemaps.csv` low-level descriptors write 0 for F0,
jitter, shimmer, HNR, H1-H2 and H1-A3, and -201 for F1-F3 amplitudeLogRelF0, on
unvoiced frames. `tables.aggregate` averaged those placeholders with real values,
so voicing-dependent summaries mostly encoded the voiced fraction.

Corrected rule, computed per participant without labels:

- A frame is voiced when `F0semitoneFrom27.5Hz_sma3nz > 0`.
- Every `*_sma3nz` descriptor (13 descriptors) is summarized over voiced frames
  only. Genuine zeros inside voiced frames are kept.
- Any value of exactly -201 is treated as missing.
- The 10 `*_sma3` (non-voicing) descriptors keep all frames, unchanged.
- Same mean and population standard deviation; a participant with no voiced
  frames gets missing values for training-fold median imputation.
- Add one explicit feature, `audio__voiced_fraction` (voiced / all frames).

This gives 47 audio features. The same pass recomputes the legacy all-frame
summary and must reproduce `data/processed/features.parquet` audio columns
(maximum absolute difference recorded); otherwise the build stops.

## Defect 2: late fusion on unequal score scales

`fuse_scores` averaged `expit(raw margin)` from separately fitted branches whose
margin scales differed up to ~30x, so the nominal weight was not the applied
weight. Corrected rule: divide each branch's margin by its scale, defined as the
root-mean-square of that branch's pooled inner out-of-fold margins on the outer
training participants (1 if zero). Fuse standardized margins with the normalized
(and, for the quality variant, reliability-discounted) weights; threshold zero.
This preserves each branch's own decision boundary and uses no labels and no
held-out data. Unavailable modalities get zero weight; no available input gives
the outer-training prevalence logit.

## Fixed models

Every branch is the matched historical recipe: BranchPreprocessor (fold-only
median imputation, constant/duplicate removal, scaling) and balanced L2 logistic
regression, liblinear, C=.1, all features, margin threshold zero.

Policies: `original_audio`, `voiced_audio`, `face`, `original_concat` (historical
control, must reproduce earlier scores), `voiced_concat`, and late fusion of the
`voiced_audio` and `face` branches: `late_equal_raw` (previous unscaled method,
w=.5), `late_equal_scaled` (w=.5), `late_weighted_scaled` (audio weight from
[.5, .25, .75] by pooled inner-OOF F1 on scaled scores, first on ties) and
`late_quality_scaled` (identical branches, scales and weight, plus the unchanged
quality discount: audio poor if clipping >.001, zero fraction >.01 or coverage
<.95; face poor if retained <.8 or confidence <.8; poor weight x.5). Add
always-negative/always-positive controls. 9 fitted policies x 15 folds.

## Comparisons and decision

Primary, against `original_concat`: `voiced_concat`, `late_equal_scaled`,
`late_weighted_scaled`, `late_quality_scaled`. Isolating comparisons:
`voiced_audio` vs `original_audio`; `late_equal_scaled` vs `late_equal_raw`;
`late_weighted_scaled` vs `late_equal_scaled`; `late_quality_scaled` vs
`late_weighted_scaled`; `late_weighted_scaled` vs `face` and vs `voiced_audio`.
Use the existing screen unchanged: mean F1 +3 points, no mean balanced-accuracy
or fold-AUC loss, recall loss <=5 points, F1 gain in at least 2/3 repeats. These
are engineering screens, not multiplicity-adjusted tests. Report accuracy,
balanced accuracy, precision, recall, F1, fold AUC/AP, domain metrics, applied
weights and every failure, alongside always-positive F1.

The cohort has been reused across many studies, so any pass is exploratory and
needs a fresh holdout. A pass of `voiced_concat` would replace the original
audio summaries as the retained audio representation; otherwise keep the
original control and report the corrected results as the valid test of these
two ideas. Stop after this run.

# Follow-up audio study v1: results

Completed 2026-10-28. Run: `experiments/runs/audio_study_20261028T181901228265Z`.
This study is separate from the original Stage 9 test evaluation; it does not
replace that report. Stage 10 remains pending.

## What was tested

Only the 163 official training participants entered this study. Three repetitions
of five outer folds evaluated 13 predeclared procedures. Each fitting partition
used inner folds to choose hyperparameters and, where applicable, a decision
cutoff. The original Stage 4 audio recipe was rerun within these same outer folds
as the reference. Development and test predictions were not generated.

Richer summaries extended 46 mean/std features to 230 raw descriptors using
quantiles, IQR, adjacent-frame change and variability of 10-second window means.
Extraction verified the old means/stds against Stage 3. There were no missing
aggregate values; per-participant row counts ranged from 41,468 to 196,597.
23 raw columns were constant across the entire training cohort, mostly low
percentiles of zero-heavy measurements. This diagnostic did not control feature
removal: each fold fitted its own preprocessing. Rich models retained 198-201
columns after fold-specific constant/duplicate removal. Zeros were not discarded.

Models were logistic regression, linear SVM and RBF SVM. Default and inner-selected
cutoff procedures were compared separately. SVM scores are uncalibrated margins.
All feature engineering and search settings were written before study results.

## Complete comparison

F1 is pooled across outer-held-out predictions within each repeat; the table shows
mean and population SD across three repeats. AUC is averaged across outer folds
to avoid combining incompatible SVM margin scales. Repeats share participants,
so SD is not a confidence interval and does not imply statistical significance.

| Features / model / cutoff | F1 mean +/- SD | Precision | Recall | Mean fold AUC |
|---|---:|---:|---:|---:|
| Original Stage 4 recipe | 0.324 +/- 0.037 | 0.242 | 0.495 | 0.557 |
| Base / logistic / default | 0.307 +/- 0.047 | 0.230 | 0.468 | 0.555 |
| Base / logistic / tuned | 0.333 +/- 0.030 | 0.222 | 0.676 | 0.540 |
| Base / linear SVM / default | 0.316 +/- 0.057 | 0.242 | 0.459 | 0.547 |
| Base / linear SVM / tuned | 0.347 +/- 0.028 | 0.231 | 0.703 | 0.547 |
| Base / RBF SVM / default | 0.179 +/- 0.052 | 0.170 | 0.198 | 0.492 |
| Base / RBF SVM / tuned | 0.366 +/- 0.008 | 0.235 | 0.829 | 0.496 |
| Rich / logistic / default | 0.320 +/- 0.033 | 0.247 | 0.459 | 0.548 |
| Rich / logistic / tuned | 0.371 +/- 0.027 | 0.247 | 0.748 | 0.544 |
| Rich / linear SVM / default | 0.342 +/- 0.036 | 0.279 | 0.441 | 0.586 |
| Rich / linear SVM / tuned | 0.341 +/- 0.044 | 0.235 | 0.631 | 0.553 |
| Rich / RBF SVM / default | 0.128 +/- 0.015 | 0.180 | 0.099 | 0.474 |
| Rich / RBF SVM / tuned | 0.363 +/- 0.018 | 0.234 | 0.811 | 0.498 |

## Interpretation

Rich logistic regression with a selected cutoff had the largest observed mean
F1, approximately 0.047 above the original recipe. Its F1 exceeded that reference
in each of the three repeats. However, the higher recall (0.748 versus 0.495)
came with low precision (0.247) and lower average specificity. Its mean fold AUC
was 0.544 versus the reference 0.557, so this is not clear evidence of improved
ranking discrimination. Cutoffs were below zero in 14 of its 15 outer fits.

For context, an always-positive prediction has F1 0.370 on this cohort (37 of
163 participants are positive), with precision 0.227, recall 1 and specificity 0.
This is an analytical sanity check added when interpreting the results, not an
additional tuned candidate. The near equality of F1 shows why the largest F1
alone is insufficient to declare a useful improvement. It does not imply that
the fitted model makes exactly the same predictions as the trivial rule.

Rich linear SVM at the default cutoff showed a more balanced exploratory signal:
mean F1 0.342, precision 0.279, recall 0.441, mean fold AUC 0.586. Relative to the
original recipe, its F1 and fold-averaged AUC were higher in each repeat, but
recall was lower. Relative to the base linear SVM, its F1 was slightly lower in
seed 42 and higher in the other two seeds. The effect is modest and not uniformly
beneficial. No independent confirmation, significance or superiority is claimed.

RBF SVM showed weak ranking performance, around chance AUC; its tuned-cutoff F1
largely reflects more positive predictions. Expanding this search just to raise
F1 is not supported by this pilot. Do not compare these training-only nested-CV
numbers directly with the earlier 0.563 test F1: the participants and evaluation
procedures differ. Released label semantics and whole-interview summaries remain
limitations. No new final model was selected or deployed.

## Verification and reproduction

89 tests pass without warnings. All 195 outer model checkpoints reproduce their
predictions after restoration. Independently verified 39 pooled repeat metric
rows, 195 fold AUCs, all inner/outer participant partitions and recorded selection
choices. The primary rich-logistic inner search and outer predictions reproduced
exactly. All 6,357 prediction records belong to official training participants.
Input/source/checkpoint hashes verified, and the comparison figure was visually
checked. The model study took about 83 seconds locally, excluding extraction;
this is a local elapsed time, not a controlled performance benchmark.

```powershell
.venv\Scripts\python.exe -m src.dataset.audio_summary
.venv\Scripts\python.exe -m src.experiments.audio_study
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/audio-study-check --tb=short
```

Extraction caches live under `data/processed/audio_study_v1/`. A study run contains
the frozen protocol, participant fold membership, every inner candidate/cutoff
score, all outer models and predictions, fold/repeat metrics, paired differences,
PNG/SVG figure and report. The completed run additionally has verification.json
and environment-freeze.txt. All participant data and artifacts stay ignored in
Git. Existing Stage 9 evaluation files were preserved.

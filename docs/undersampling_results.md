# MPNet undersampling results

Completed and verified 2026-11-03. Run
`experiments/runs/undersampling_20261103T113703489563Z`.
The [protocol](undersampling_protocol.md) preceded fitting. Same 161 training
participants, unchanged released labels, five folds repeated with three seeds.
Evaluation folds retain all participants and their natural class distribution.

**None of the undersampling procedures improved on the existing MPNet SVM.**
Retain the current research checkpoint. All values below are percentages;
F1 SD describes overlapping repeats, not a confidence interval.

| Procedure | F1 +/- SD | Precision | Recall | Specificity | Accuracy | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original reference | 58.1 +/- 5.1 | 47.8 | 73.9 | 76.1 | 75.6 | 75.0 | 80.4 |
| Balanced-weight-only control | 58.1 +/- 5.1 | 47.8 | 73.9 | 76.1 | 75.6 | 75.0 | 80.4 |
| Single subset, 1:1 negatives:positives | 54.0 +/- 2.8 | 41.7 | 76.6 | 68.0 | 70.0 | 72.3 | 77.5 |
| Single subset, 2:1 | 44.0 +/- 9.9 | 43.0 | 45.0 | 81.7 | 73.3 | 63.4 | 76.6 |
| Seven-model ensemble, 1:1 | 53.6 +/- 4.1 | 41.5 | 75.7 | 68.0 | 69.8 | 71.8 | 79.1 |
| Seven-model ensemble, 2:1 | 51.3 +/- 5.7 | 50.0 | 53.2 | 84.1 | 77.0 | 68.6 | 77.9 |
| Sampling procedure selected inside training folds | 55.1 +/- 4.1 | 44.1 | 73.9 | 71.5 | 72.0 | 72.7 | 79.7 |

The primary inner-selected sampling procedure loses 2.99 F1 percentage points.
Single 1:1 sampling detects slightly more positives but increases false positives.
The 2:1 ensemble raises precision slightly while losing substantial recall.
No variant passes the preregistered improvement screen. These results support
retaining class weighting and all fitting participants for this recipe, not a
general claim that undersampling never helps other models or datasets.

## Implementation and checks

Sampling occurs separately inside every inner fitting fold and the final outer
fitting fold. All positives remain. Negative sampling is without replacement
within a member and proportional to AI/WoZ counts, with integer quota rounding.
Ensemble members use different fixed seeds, may overlap, and average SVM margins.
PCA/scaling use all fitting-fold participants before classifier undersampling;
held-out participants never contribute to preprocessing or sampling. C is chosen
by inner pooled F1, with no threshold search or outer-score selection.

The independent verifier restored 105 saved bundles, checked 3,381 predictions
and 21 repeat metric rows, audited 315 inner candidates and their sampling IDs,
and reproduced all 483 previous MPNet reference scores. Source/input/checkpoint
hashes, nested split membership, quotas, positive retention and selection maxima
passed. The full 122-test suite passed, with the existing pandas regex warning.
An initial attempt stopped before completing a fold because of DataFrame row
indexing; it produced no reported results. The indexing fix has a regression test.

No development/test prediction, relabeling, new encoder extraction or deployment
was performed. These are exploratory training-CV results on reused folds.
Existing AI-domain and label-definition limitations remain unresolved.

```powershell
.venv\Scripts\python.exe -m src.experiments.undersampling
.venv\Scripts\python.exe -m src.experiments.verify_undersampling experiments/runs/undersampling_20261103T113703489563Z
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/undersampling_full --tb=short
```

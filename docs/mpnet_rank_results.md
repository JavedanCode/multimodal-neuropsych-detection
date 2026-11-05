# MPNet compression follow-up

Verified 2026-11-02. Run: `experiments/runs/mpnet_rank_20261102T171723915064Z`.
The [protocol](mpnet_rank_protocol.md) preceded fitting. Same 161 training people,
released labels, screened MPNet means and matching nested folds. No dev/test use.

Neither the primary inner-selected rank procedure nor secondary rank/threshold
procedure improves the reference. Keep the original MPNet PCA32 SVM recipe.
Every fixed-rank diagnostic also has lower mean F1 than the previous reference.

All numbers are percentages; SD is across overlapping repeats, not a CI.

| Procedure | F1 +/- SD | Recall | Balanced accuracy | Mean fold AUC |
|---|---:|---:|---:|---:|
| Previous PCA32, original C grid | 58.1 +/- 5.1 | 73.9 | 75.0 | 80.4 |
| Rank8, default cutoff | 52.4 +/- 2.1 | 65.8 | 70.1 | 78.0 |
| Rank8, tuned cutoff | 48.5 +/- 0.7 | 59.5 | 67.0 | 77.9 |
| Rank16, default cutoff | 54.8 +/- 3.7 | 69.4 | 72.3 | 79.7 |
| Rank16, tuned cutoff | 57.4 +/- 2.7 | 70.3 | 74.1 | 80.1 |
| Rank32, expanded C grid, default cutoff | 56.8 +/- 6.6 | 72.1 | 73.9 | 79.3 |
| Rank32, expanded C grid, tuned cutoff | 52.5 +/- 4.8 | 64.9 | 70.1 | 78.2 |
| Rank64, default cutoff | 48.5 +/- 1.9 | 56.8 | 66.8 | 75.3 |
| Rank64, tuned cutoff | 48.2 +/- 3.8 | 61.3 | 66.8 | 75.0 |
| Inner-selected rank, default cutoff (primary) | 51.9 +/- 6.9 | 64.0 | 69.9 | 76.9 |
| Inner-selected rank and cutoff (secondary) | 51.7 +/- 5.1 | 62.2 | 69.4 | 77.6 |
| Always negative | 0.0 +/- 0.0 | 0.0 | 50.0 | 50.0 |
| Always positive | 37.4 +/- 0.0 | 100.0 | 50.0 | 50.0 |

The new rank32 procedure differs from the old reference because its C grid also
contains 10. A broader inner search can choose settings that perform worse on
outer participants; identical rank does not imply identical fitted classifier.
The restricted old-control recipe reproduces all prior scores. No favorable
fixed rank or repeat was substituted for the unsuccessful primary selection rule.

Independent verification restored 165 saved bundles, recomputed 6,279 predictions
and 39 metric rows, checked 1,440 inner candidate records, verified folds/hashes,
and matched 483 prior MPNet SVM control predictions including scores. All 119
tests pass (one existing cosmetic regex warning). Complete metrics, SD, subgroup
summaries and screening decisions remain in the ignored run directory.

These are exploratory reused training folds. They do not establish a global
optimum, independent test performance, or clinical readiness. The stronger
reference is documented in [MPNet results](mpnet_results.md).

```powershell
.venv\Scripts\python.exe -m src.experiments.mpnet_rank
.venv\Scripts\python.exe -m src.experiments.verify_mpnet_rank experiments/runs/mpnet_rank_20261102T171723915064Z
```

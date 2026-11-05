# Precision experiment series: completed comparison

Completed 2026-11-03. All four experiments in the
[frozen experiment list and protocol](precision_protocol.md) were implemented,
run sequentially, and independently verified. No production deployment occurred.

## Results

Percentages are means across the same three repeated five-fold training
comparisons (161 participants, 37 positives). Each experiment is a constrained
selection policy: inner validation selects its candidate or retains the reference.
Thus these are not results from forcing the new model in every fold.

| Experiment | Precision | Recall | F1 | F1 SD | Specificity | Accuracy | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing MPNet SVM reference | 47.8 | 73.9 | 58.1 | 5.1 | 76.1 | 75.6 | 75.0 | 80.4 |
| 1. Higher decision threshold | 48.7 | 72.1 | 58.1 | 4.0 | 77.4 | 76.2 | 74.7 | 80.4 |
| 2. Class-penalty adjustment | 49.2 | 73.0 | 58.8 | 4.3 | 77.7 | 76.6 | 75.3 | 80.1 |
| 3. Nonlinear RBF SVM | 45.9 | 72.1 | 56.0 | 4.9 | 74.7 | 74.1 | 73.4 | 80.1 |
| 4. SVM/logistic agreement | 49.3 | 71.2 | 58.2 | 5.5 | 78.2 | 76.6 | 74.7 | 80.1 |

**No experiment passes the declared precision improvement screen.** Agreement
has the highest observed precision, +1.49 percentage points, but loses 2.70 recall
points and does not increase precision in two repeats. Class-penalty adjustment
offers the more promising small tradeoff: +1.40 precision points, +0.69 F1 points,
and -0.90 recall points. Neither reaches the +5-point precision target. The small
changes do not justify claiming a reliable improvement or replacing the retained
reference checkpoint. The 58.8% F1 is the highest observed mean in this series,
not a newly validated deployed model.

| Policy | Mean false positives / 161 | Mean false negatives / 161 | Folds using new candidate / 15 |
|---|---:|---:|---:|
| Reference | 29.67 | 9.67 | -- |
| Threshold | 28.00 | 10.33 | 6 |
| Class penalties | 27.67 | 10.00 | 4 |
| RBF | 31.33 | 10.33 | 6 |
| Agreement | 27.00 | 10.67 | 6 |

Counts are averages over repeat predictions for the same people, not fractional
people or independent cohorts. In all remaining folds the policy retained the
reference, either because it won constrained selection or because no candidate
met the constraints. F1 SD is repeat variation, not a confidence interval.

## Domain limits and past comparisons

| Policy | AI precision | AI recall | AI F1 | WoZ F1 |
|---|---:|---:|---:|---:|
| Reference | 33.3 | 46.7 | 38.9 | 60.9 |
| Threshold | 30.2 | 40.0 | 34.3 | 61.5 |
| Class penalties | 34.9 | 46.7 | 39.9 | 61.5 |
| RBF | 30.4 | 46.7 | 36.8 | 59.0 |
| Agreement | 34.9 | 46.7 | 39.9 | 61.0 |

The AI subgroup remains very small (18 people, five positives), and these
procedures do not resolve the domain gap. The historical test cohort is all AI.
See [earlier model comparisons](research_progress_20261102.md) and
[undersampling results](undersampling_results.md). Prior undersampling reached
55.1% F1 for its inner-selected policy; previous MiniLM text reached 52.8% F1.
All comparisons here concern matched training CV, not new test performance.

## Verification and reproduction

| Experiment | Run directory suffix |
|---|---|
| Threshold | precision_threshold_20261103T114210866132Z |
| Class penalties | precision_cost_20261103T114250191419Z |
| RBF | precision_rbf_20261103T114339132822Z |
| Agreement | precision_agreement_20261103T114408710725Z |

All directories are under `experiments/runs/`, local and ignored by Git. Each
contains model bundles, inner scores and selections, fold memberships, input and
source hashes, dependency versions, predictions, repeat/domain metrics, a
screening decision and an independent verification record.

Across four runs, the verifier restored 120 bundles and checked 3,864 predictions
and 24 metric rows. It checked 1,095 candidate records (including reference
records repeated within selection lists), refitted 120 selected inner procedures,
verified the training-only scaler means, and reproduced 483 historical reference
scores in each run. All 124 tests passed; the existing pandas regex warning remains.
The verifier initially compared a float32 mean with the scaler's float64 mean;
using float64 for the independent calculation resolved numerical rounding without
changing model fitting, predictions or tolerances.

```powershell
.venv\Scripts\python.exe -m src.experiments.precision_series threshold
.venv\Scripts\python.exe -m src.experiments.precision_series cost
.venv\Scripts\python.exe -m src.experiments.precision_series rbf
.venv\Scripts\python.exe -m src.experiments.precision_series agreement
# Substitute the corresponding run directory for each verification:
.venv\Scripts\python.exe -m src.experiments.verify_precision experiments/runs/precision_agreement_20261103T114408710725Z
```

No labels, splits or held-out participant distributions were changed; no new
development/test predictions were generated. Reused-fold exploratory selection
cannot establish independent superiority. The next substantive hypothesis is
better utterance representation/pooling rather than continued threshold or
penalty searches on these same folds. That follow-up is not part of this series.

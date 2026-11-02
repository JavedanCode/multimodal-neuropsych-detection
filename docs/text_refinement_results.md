# Text refinement results

Verified 2026-11-02. Run:
`experiments/runs/text_refinement_20261030T232042036662Z`.
The [protocol](text_refinement_protocol.md) preceded fitting. Same 161 training
participants, released provisional labels and matching outer folds. No new
development/test evaluation. All searches and threshold choices used inner folds.

The previous PCA32 text baselines remain strongest. Retaining all dimensions,
adding within-interview variation, threshold tuning, and conservative late fusion
did not improve mean F1 over either previous control. Do not promote a new model.
The broad inner-selected procedure underperformed the simpler existing recipe.

| Procedure | Mean F1 | Recall | Balanced accuracy | Mean fold AUC |
|---|---:|---:|---:|---:|
| Previous text SVM | 52.8% | 66.7% | 70.4% | 79.4% |
| Previous text logistic | 52.5% | 70.3% | 70.5% | 78.2% |
| Full mean, standard scaling, default cutoff | 49.3% | 68.5% | 68.0% | 76.5% |
| Full mean, standard scaling, tuned cutoff | 47.5% | 64.0% | 66.4% | 74.8% |
| Full mean, L2 normalization, default cutoff | 50.4% | 77.5% | 69.4% | 75.0% |
| Full mean, L2 normalization, tuned cutoff | 50.4% | 77.5% | 69.4% | 75.0% |
| Rich text, standard scaling, default cutoff | 47.2% | 65.8% | 66.1% | 71.7% |
| Rich text, standard scaling, tuned cutoff | 48.4% | 64.9% | 67.1% | 72.2% |
| Rich text, L2 normalization, default cutoff | 33.2% | 39.6% | 55.4% | 67.5% |
| Rich text, L2 normalization, tuned cutoff | 42.8% | 94.6% | 60.3% | 67.3% |
| Late fusion text-only branch | 49.3% | 68.5% | 68.0% | 76.5% |
| Selected late fusion | 47.6% | 65.8% | 66.6% | 75.8% |
| Inner-selected text search | 46.1% | 64.9% | 65.2% | 72.5% |
| Inner-selected text-or-fusion strategy | 44.7% | 61.3% | 64.1% | 73.2% |
| Always negative | 0.0% | 0.0% | 50.0% | 50.0% |
| Always positive | 37.4% | 100.0% | 50.0% | 50.0% |

All candidate procedures fail the three-point F1 improvement requirement.
High recall alone is insufficient: the rich L2 tuned condition reaches 94.6%
recall but only 27.6% precision. The best new F1 is 50.4%, versus 52.8% previously.
This is evidence against these particular recipes, not against every possible
uncompressed representation, threshold or fusion method.

Full metrics, repeat SD, domain summaries, inner scores/selections, saved
classifiers and frozen hashes are in the run directory. Verification restored
210 bundles, recomputed 7,728 predictions and 48 metric rows, checked 3,780 inner
candidate records, and matched 966 prior-control predictions including scores.
All 114 tests passed (one existing cosmetic regex warning). The repeated training
folds are exploratory and have been used adaptively across studies; these results
are not new independent confirmation. The small AI subgroup and unresolved labels
remain limitations. Historical test scores are unchanged.

```powershell
.venv\Scripts\python.exe -m src.experiments.text_refinement
.venv\Scripts\python.exe -m src.experiments.verify_text_refinement experiments/runs/text_refinement_20261030T232042036662Z
```

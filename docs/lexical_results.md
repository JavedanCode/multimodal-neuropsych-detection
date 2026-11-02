# Screened lexical study results

Verified 2026-11-02. Run: `experiments/runs/lexical_20261102T165805634953Z`.
The [protocol](lexical_protocol.md) preceded extraction and fitting. Same 161
training participants, released provisional labels, screened utterances and
outer/inner folds. No development or test evaluation.

No new procedure beats the previous semantic baseline. The best new mean F1 is
47.8% for combined semantic/lexical logistic with default cutoff, versus 52.8%
for the existing semantic SVM. The primary inner-selected new procedure reaches
45.0% F1. Retain the previous semantic reference; no lexical model is promoted.

| Procedure | F1 | Precision | Recall | Balanced accuracy | Mean fold AUC |
|---|---:|---:|---:|---:|---:|
| Previous semantic SVM | 52.8% | 43.9% | 66.7% | 70.4% | 79.4% |
| Previous semantic logistic | 52.5% | 42.1% | 70.3% | 70.5% | 78.2% |
| Lexical SVM, default cutoff | 0.0% | 0.0% | 0.0% | 50.0% | 70.1% |
| Lexical SVM, tuned cutoff | 42.8% | 38.0% | 49.5% | 62.8% | 71.2% |
| Lexical logistic, default cutoff | 19.7% | 45.8% | 12.6% | 54.2% | 70.5% |
| Lexical logistic, tuned cutoff | 36.8% | 22.9% | 93.7% | 49.8% | 57.5% |
| Combined SVM, default cutoff | 46.5% | 39.5% | 56.8% | 65.3% | 77.9% |
| Combined SVM, tuned cutoff | 46.4% | 37.9% | 60.4% | 65.4% | 75.2% |
| Combined logistic, default cutoff | 47.8% | 40.1% | 59.5% | 66.3% | 76.6% |
| Combined logistic, tuned cutoff | 45.0% | 38.8% | 54.1% | 64.1% | 76.5% |
| Inner-selected new procedure | 45.0% | 37.5% | 56.8% | 64.1% | 74.5% |
| Always negative | 0.0% | 0.0% | 0.0% | 50.0% | 50.0% |
| Always positive | 37.4% | 23.0% | 100.0% | 50.0% | 50.0% |

Lexical default SVM ranks some positives above negatives but predicts no positives
at its selected zero-margin cutoff; AUC and F1 measure different behavior.
The high-recall tuned lexical logistic condition has poor precision and balanced
accuracy. It is not an improvement hidden by F1.

All 161 transcript hashes and screening counts match the previous semantic study.
Sparse counts use fixed nonnegative hashed unigrams/bigrams, without cross-utterance
bigrams. Hash collisions are possible. Hashing is not anonymization; counts remain
restricted derived data under ignored paths. No transcripts were uploaded or
exported as text. TF-IDF uses fitting participants only.

Verification restored 165 bundles, recomputed 6,279 predictions and 39 metric
rows, checked 3,060 inner candidate records, verified all 161 count caches and
every saved outer IDF vector, and reproduced 966 original semantic-control
predictions including scores. All 117 tests pass (one existing cosmetic regex
warning). All metrics, repeat SD, subgroup descriptions, settings and provenance
are in the run directory. Reused training folds are exploratory; no fresh external
validation or clinical claim follows. Historical test scores remain unchanged.

```powershell
.venv\Scripts\python.exe -m src.dataset.lexical
.venv\Scripts\python.exe -m src.experiments.lexical_study
.venv\Scripts\python.exe -m src.experiments.verify_lexical experiments/runs/lexical_20261102T165805634953Z
```

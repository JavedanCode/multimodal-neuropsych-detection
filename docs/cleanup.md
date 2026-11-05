# Pipeline consolidation — 2026-11-05

## Preserved history

The complete pre-cleanup source and notes are committed at:

`5f3e3f3c4f754c25e92f377cdfedcf822d561a71`

Inspect an original implementation with, for example:

```powershell
git show 5f3e3f3:src/experiments/branch_audit_v6.py
```

For an exact historical rerun, use that revision in a separate checkout with the
necessary local data. Original protocols, verification scripts and experimental
variants remain in Git rather than in the active source tree.

## Retained working pipeline

There are four maintained study runners: the retained text reference, the bounded
text/threshold comparison, the matched audio/facial comparison, and the explicit
fallback policy. Each includes its necessary controls. Retaining only an outer-fold
winner and discarding its failed comparisons would misrepresent the experiment.

Shared data loading, fitting and evaluation functions were extracted from obsolete
runners. Old experiment and verifier scripts were removed. Neural pooling variants
used only by retired studies were also removed. Binary HHO/L1 and explainability
remain reusable research components, without a performance-superiority claim.

| Tracked organization | Before | After |
|---|---:|---:|
| Python files under src | 91 | 65 |
| Files under src/experiments, including package marker | 49 | 5 |
| Test files | 30 | 18 |
| Markdown documentation files | 49 | 8 |

Code is consistently formatted, shared modules have responsibility descriptions,
and a single CLI dispatches training and verification. Empty scaffold directories
were removed. Superseded tests remain with their implementations in the snapshot;
maintained behavior and the cleanup contracts have 108 passing tests.

The experiment history consolidates original findings, including negative results.
`results.csv` contains 110 aggregate rows from nine runs with summary CSVs and
completion records. It is a supplement to the narrative log, not a complete list
of every historical run or an assertion that all rows are independently validated.
Raw data, caches, saved models and participant-level experiment evidence stay local
and ignored by Git. They were not deleted or committed during consolidation.

## Historical compatibility

Saved pickles contain module names. `training/checkpoints.py` resolves the small
explicit list of renamed model modules lazily when loading trusted local models.
Unsupported retired checkpoints should be used with their original Git revision.

Historical manifests contain source hashes. Formatting and moving code necessarily
changes those hashes. `training/provenance.py` first checks the current file, then
allows **source and method documents only** to match the recorded Git snapshot in
`configs/history.json`. It checks exact archived bytes, known Git line-ending
representations, and the recorded adaptive module's mixed final line ending.
Data and model files must always match their current on-disk hashes; they cannot
fall back to Git. Matching archived source proves provenance, not equivalence of
new code. New runs hash all maintained code, methods and shared configuration.

New verification reports use `verification_current.json`; original reports and
original manifests remain available. Cache directory versions are intentionally
unchanged. Fresh builds still require their prerequisite caches; the original full
preparation recipes are recoverable from the snapshot.

## Validation

- 108 maintained tests passed; one pre-existing transcript-regex warning remains.
- All 65 Python modules imported successfully.
- All 83 source/test Python files pass Black's formatting comparison.
- Historical checks restored 393 models and checked 30,966 prediction rows across
  the retained reference, audio/facial, text-candidate and fallback studies.
- The audio/facial verification replayed 540 inner candidate vectors and retrained
  45 neural models. Text candidates replayed 1,020 vectors; fallback replayed 180.
- A fresh reference run reproduced all 537 original predictions exactly:
  `baseline_v2_20261105T071535380455Z`.
- A fresh audio/facial run reproduced all 7,245 original prediction rows and its
  summary metrics exactly: `branch_audit_v6_20261105T071558696251Z`.

The CLI imports numerical libraries before setting thread limits. Verification
caught and corrected the reverse ordering, which could otherwise introduce numeric
variation during replay. Historical fallback serialization retains its previously
accepted maximum margin difference of about 3.48e-7, with identical classifications.
No score tolerance was widened to make the cleanup pass.

This is a behavior-preserving organization change, not a new model improvement or
new holdout evaluation. Local verification details are in `reports/cleanup`.

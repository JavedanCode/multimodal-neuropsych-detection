# Stage 9 completion: provisional final evaluation

Completed 2026-10-28. Run: `experiments/runs/stage9_20261028T180907085552Z`.
The user explicitly authorized evaluation against unchanged released PHQ_Binary.
The 20 label/score disagreements remain unresolved; this is not a confirmed
PHQ-8 >=10 screening experiment. No published benchmark comparison is justified.

## Evaluation and findings

37 models (nine classical, 27 neural, one majority reference) were evaluated on
56 development and 56 test participants. Every model used training participants
only; the new nine fixed-weight controls used the same selected columns and
training settings as Stage 7. The protocol, checkpoint hashes and source hashes
were frozen before any test prediction. No test-driven tuning, seed/model choice,
threshold change, ensembling or label correction occurred.

Positive-class F1 below is mean +/- population SD across seeds 42/137/2026, except
the deterministic majority reference. It is not accuracy or a confidence interval.

| Model family | Dev F1 mean | Test F1 mean +/- SD |
|---|---:|---:|
| Audio logistic | 0.362 | 0.563 +/- 0.010 |
| Facial logistic | 0.256 | 0.448 +/- 0.018 |
| Concatenation logistic | 0.240 | 0.485 +/- 0.027 |
| All descriptors, equal weights | 0.337 | 0.524 +/- 0.035 |
| All descriptors, adaptive | 0.327 | 0.538 +/- 0.045 |
| All descriptors, quality | 0.327 | 0.538 +/- 0.045 |
| L1, equal weights | 0.328 | 0.470 +/- 0.072 |
| L1, adaptive | 0.328 | 0.468 +/- 0.062 |
| L1, quality | 0.328 | 0.468 +/- 0.062 |
| HHO, equal weights | 0.321 | 0.485 +/- 0.060 |
| HHO, adaptive | 0.318 | 0.474 +/- 0.050 |
| HHO, quality | 0.318 | 0.474 +/- 0.050 |
| Majority | 0.000 | 0.000 |

The audio baseline has the highest observed mean test F1 among these families;
this is a descriptive result, not a model-selection decision or a superiority
claim. Quality inputs did not change thresholded predictions/F1 for matched
models. The gate's effect varies by seed. HHO adaptive/quality F1 was lower than
the corresponding all-feature model for each seed. L1 substantially reduces
inputs but its similar development F1 did not translate into similar mean test
F1 in this cohort. The central improvement hypothesis is not supported here.

All primary-seed paired F1 difference intervals include zero. Zero-width quality
F1 intervals reflect identical thresholded predictions, not proof the underlying
probabilities or model functions are equivalent. Primary all-feature adaptive
F1 is 0.476, with a 95% stratified bootstrap interval 0.316-0.632; primary audio
F1 is 0.556 with interval 0.462-0.642. These intervals condition on observed class
counts and fitted models; they omit retraining uncertainty and multiplicity.

## Deliverables and verification

The run includes frozen_manifest.json with checkpoint/input/source hashes and
historical training/selection timings; nine fixed-weight checkpoints; development
predictions; environment-freeze.txt; pre-evaluation and final verification files.
Its evaluation/ folder contains:

- all_metrics.csv: all 74 model/partition rows, every metric, confusion counts,
  primary F1 intervals, retained features, neural parameter counts and timings.
- paired_ablations.csv: 72 matched seed/partition comparisons, including primary
  paired F1 intervals.
- seed_summary.csv and report.md: separate development/test family summaries.
- predictions.parquet: local participant predictions (restricted; ignored by Git).
- test_f1.png/.svg and test_confusion.png/.svg: visually verified research figures.
- results.json: final status, frozen manifest hash and output hashes.

83 tests passed without warnings. All 27 historical checkpoints reproduced their
saved development predictions. All nine new checkpoints restored correctly;
the three primary controls reproduced identical trained weights. After scoring,
all 74 metric rows and confusion matrices were independently checked against
saved predictions, and one full 2,000-resample primary F1 interval was checked
against scikit-learn. Frozen input/model/source and output hashes verified.
All 4,144 prediction records cover the expected 37 models x 112 participants.

The evaluation command refuses an existing evaluation directory. Reproduction
requires a new prepare run and the same frozen settings; do not overwrite the
completed report or use observed test outcomes to tune a new model on this test.
Checkpoint-restoration time is included in new inference timing. CPU peak memory
was not instrumented for these controls; earlier CUDA measurements are not a
matched memory comparison. No runtime or memory superiority claim is made.

Stage 10 remains: final documentation, reproducibility checks and audit. The
test partition has now been evaluated and must not be described as still sealed.

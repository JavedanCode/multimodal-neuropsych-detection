# Existing-data improvement series — 2026-11-04

Completed the finite experiment series declared in `bounded_v3_protocol.md`.
No other dataset is available. Use unchanged released labels and the same 161
training participants, including 37 positives. Official development and test
participants were evaluated in historical Stages 4–9; no untouched internal
partition can be recovered by reshuffling. This study performs no new dev/test
evaluation and remains exploratory repeated nested cross-validation.

## Step status

| Step | Status and evidence |
|---|---|
| Speaker/timing review | All 24 initial human annotations validated; 11 cases now have full-utterance/context follow-up clips |
| Label provenance | Existing source discrepancies remain unresolved; no labels changed |
| Preserve simple reference | Original MPNet PCA32 linear SVM reproduced on identical folds |
| Fixed topic summaries | Both bounded alternatives implemented, evaluated, and independently replayed |
| Conservative fusion with text fallback | Implemented and evaluated under clean and three missing-input scenarios |
| Precision/recall objective | Inner-selected thresholds; recall floor 65%, limited F1 loss; fixed outer promotion criteria |
| Limit adaptive search | Finite series completed; no extra grid based on the observed winner |
| Uncertainty and final validation | Paired participant bootstrap reported with limitations; fresh independent validation unavailable |

## Human review evidence

Initial worksheet: eight participants, 24 clips, 16 transcript-covered intervals.
The reviewer marked covered clips as Participant 6, Uncertain 7, Interviewer 3.
Timing was Matches 15 and Uncertain 1. These are human flags from a small pilot,
not a reliable dataset-wide contamination estimate or definitive diarization.
Ten covered cases need follow-up by speaker/timing status. Notes also flagged
three transcripts longer than their eight-second preview; two overlap the ten
cases, resulting in 11 follow-ups.

The original worksheet displayed full transcript text beside previews capped at
eight seconds. This can impede timing and speaker judgments. The new worksheet
preserves the original annotations and supplies complete utterances (69.3 seconds
total; longest 22.4 seconds) plus five seconds of context on either side where
available. All 22 audio files were checked against expected durations and target
offsets. Judge the target utterance rather than all speakers in the context.

Local worksheet: `data/processed/aligned_v2/speaker_review/context_v3/index.html`.
Export `annotations_context.json` into that folder. Ingest with:

```powershell
.venv\Scripts\python.exe -m src.dataset.review_annotations_v3 --context
```

Initial annotation SHA256:
`595111b0f01bd03efc6b6bdf69a0baa730b1ed1cceaf0f056e0a5ce80da141d0`.
Participant-level notes, source records, audio and annotation details stay in
ignored local data/reports. No speech was uploaded. The three interviewer flags
will not automatically remove entire interviews or relabel other utterances.
Speaker-verified model features remain deferred until sufficient evidence exists.

## Matched training results

Run: `experiments/runs/bounded_v3_20261104T105104330112Z`.
Mean of three repeats; each repeat pools participant predictions from five outer
folds. Settings and cutoffs are selected with three inner folds. Topic/fusion
procedures include a reference fallback; they are selection policies, not pure
topic-only models. All six procedures were declared before training.

| Procedure | Precision | Recall | F1 | Balanced accuracy | Mean fold AUC | Mean fold AP |
|---|---:|---:|---:|---:|---:|---:|
| Original text reference | 47.3% | 73.9% | 57.5% | 74.4% | 0.805 | 0.607 |
| Inner-selected text cutoff | 49.9% | 72.1% | 58.5% | 74.8% | 0.805 | 0.607 |
| Topic-balanced policy | 44.4% | 69.4% | 53.9% | 71.4% | 0.782 | 0.587 |
| Concatenated-topic policy | 46.2% | 71.2% | 55.9% | 73.1% | 0.794 | 0.592 |
| Conservative quality residual | 49.4% | 68.5% | 57.1% | 73.3% | 0.802 | 0.595 |
| Inner-selected union | 46.6% | 66.7% | 54.4% | 71.5% | 0.782 | 0.591 |

Threshold selection gives +2.58 percentage points precision, -1.80 recall, and
+0.99 F1. It satisfies the other precision safeguards but misses the required
+5-point precision gain and the separate +3-point F1 screen. It remains a modest
candidate improvement, not a promoted replacement. Its ranking metrics do not
change: changing a cutoff changes decisions, not ranking within a fold.

Residual fusion's precision gain is +2.08 points, with -5.41 recall and -0.46 F1;
it fails both the precision-gain and recall-loss requirements. No candidate passes
either predeclared improvement screen. The baseline remains retained. Selecting
across more options inside inner folds did not improve held-out performance here.

Threshold choices across 15 outer fits: cutoff 0 in eight, .1 in six, .2 in one.
Topic-balanced policy fell back to original text in eight fits; concatenated topic
policy did so in thirteen. Residual fusion selected nonzero auxiliary weight in
four fits. These are recorded decisions, not post-hoc manual choices.

## Missing inputs and domains

| Procedure | Clean F1 | Audio missing | Face missing | Both missing |
|---|---:|---:|---:|---:|
| Reference | 57.5% | 57.5% | 57.5% | 57.5% |
| Threshold policy | 58.5% | 58.5% | 58.5% | 58.5% |
| Quality residual | 57.1% | 56.5% | 56.7% | 58.5% |
| Inner-selected union | 54.4% | 54.1% | 54.4% | 54.4% |

When no auxiliary modality qualifies, residual fusion exactly uses its text model
and selected cutoff. A different cutoff explains why its both-missing result can
differ from the cutoff-zero reference. The experiments provide no evidence that
audio/face residuals improve upon the threshold policy.

Reference subgroup F1: AI 42.2%, WoZ 60.2%. Threshold policy: AI 43.3%, WoZ 61.3%.
Only five AI positives exist in retained training data; the subgroup remains
uncertain and domain transfer remains unresolved. Domain is used for stratification
and audit, not as a predictor.

## Conditional uncertainty

2,000 paired bootstrap samples resampled participants within domain/label strata,
carrying all repeat predictions for each sampled participant together. Percentile
95% intervals for difference from reference, in percentage points:

| Procedure | Precision difference interval | F1 difference interval |
|---|---:|---:|
| Threshold policy | +0.28 to +5.35 | -1.26 to +3.00 |
| Topic-balanced policy | -5.73 to -0.18 | -6.66 to -0.71 |
| Concatenated-topic policy | -3.03 to +0.86 | -3.95 to +0.32 |
| Quality residual | -0.80 to +5.14 | -3.74 to +2.17 |
| Inner-selected union | -4.58 to +2.65 | -7.08 to +0.50 |

These are conditional on saved predictions. They omit training/refitting and
adaptive search uncertainty, are not corrected for all historical comparisons,
and are not independent confirmation. A positive conditional precision interval
does not override the frozen promotion rule or establish a generalizable gain.

## Verification and reproduction

All 90 saved bundles and 11,592 predictions were checked, together with all metrics,
domain outcomes, source/model hashes, fold membership, scaler fitting means,
candidate selections and promotion decisions. All 1,020 recorded inner candidate
score vectors were replayed. Topic features were independently recomputed for all
161 participants from original embeddings and fixed anchor similarities. Original
reference scores match the separately verified v2 baseline.
The complete suite passes 145 tests, including topic pooling, precision safeguards,
quality fallback, paired bootstrap sampling and annotation validation. One existing
regex warning remains in frozen historical code.

```powershell
.venv\Scripts\python.exe -m src.dataset.topics_v3
.venv\Scripts\python.exe -m src.experiments.bounded_v3 experiments/runs/baseline_v2_20261103T165509103723Z
.venv\Scripts\python.exe -m src.experiments.verify_bounded_v3 experiments/runs/bounded_v3_20261104T105104330112Z
.venv\Scripts\python.exe -m src.experiments.uncertainty_v3 experiments/runs/bounded_v3_20261104T105104330112Z
.venv\Scripts\python.exe -m src.dataset.review_annotations_v3
.venv\Scripts\python.exe -m src.dataset.speaker_context_v3
```

Use newly printed run paths for reproductions. Do not overwrite human annotations.
The series stops here as declared. The next evidence-gathering action is the
context review, followed by a separately declared correction study only if the
speech issue can be resolved. More score-driven searching is not the next step.

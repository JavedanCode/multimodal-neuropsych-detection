# Experiment history

Original source, protocols and reports are preserved in Git commit `5f3e3f3c4f754c25e92f377cdfedcf822d561a71`.

This log preserves favorable and unfavorable findings. Historical commands and paths below refer to that snapshot; use the current README for maintained commands. Numerical results are unchanged by the cleanup.

---

## Record: audio_study_results.md

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


---

## Record: bounded_v3_results.md

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


---

## Record: branch_audit_v6_results.md

# Primary branch audit and matched fusion results

Run: `branch_audit_v6_20261104T202915295134Z`. Protocol: [frozen methods](branch_audit_v6_protocol.md).

## Outcome

Aligned and richer facial representations each pass the predeclared improvement screen against original facial features, gaining about 3.8 percentage points F1. Neither audio revision passes. No selected-branch or neural fusion passes its comparison. Keep the prior reference; these results do not establish a reliability-aware fusion advantage.

The two facial variants reach roughly 37.5% F1, barely above the 37.4% always-positive baseline, although their balanced accuracy and ranking discriminate better than that constant. Passing a relative screen is not evidence of adequate absolute performance or statistical significance. Twelve comparisons were screened without a multiplicity-adjusted significance claim.

## Matched evaluation

161 training participants, 37 positives; five outer and three inner folds jointly balanced by label and interview domain, repeated with seeds 42, 137 and 2026. All preprocessing, classical model selection and representation selection use outer-training data only. Official development/test sets were not evaluated. Two existing facial timing quarantines apply to every facial representation, so original-feature results are not exact replays of historical runs.

Original, aligned and richer feature dimensions are respectively 46/92/230 for audio and 98/196/490 for face. Training-only median imputation, constant/duplicate removal and standardization are shared. Classical branches select balanced logistic regression or linear SVM with C in .01/.1/1. Concatenation uses fixed balanced logistic C=.1. Neural models use the same selected branches, matched initialization/dropout, 16-unit encoders and fixed 40-epoch training; quality gating adds nine measured acoustic/facial quality fields. No text predictor or fallback enters this study, although aligned intervals originate from transcripts.

## Results

Precision, recall, F1 and balanced accuracy are percentages, averaged over three repeat-level out-of-fold estimates. AUC and AP are means across the 15 held-out folds, on a 0–1 scale. Repeats reuse participants and are not independent samples.

| Condition | Precision | Recall | F1 | Balanced accuracy | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| aligned_audio | 24.59 | 38.74 | 30.06 | 51.63 | 0.506 | 0.315 |
| aligned_face | 27.83 | 57.66 | 37.52 | 56.52 | 0.586 | 0.382 |
| always_negative | 0.00 | 0.00 | 0.00 | 50.00 | 0.500 | 0.230 |
| always_positive | 22.98 | 100.00 | 37.37 | 50.00 | 0.500 | 0.230 |
| neural_adaptive | 24.80 | 32.43 | 28.10 | 51.43 | 0.524 | 0.326 |
| neural_equal | 25.55 | 35.14 | 29.58 | 52.24 | 0.522 | 0.339 |
| neural_quality | 25.11 | 32.43 | 28.30 | 51.70 | 0.520 | 0.321 |
| original_audio | 25.33 | 48.65 | 33.26 | 52.68 | 0.556 | 0.330 |
| original_concat | 30.42 | 47.75 | 37.16 | 57.61 | 0.600 | 0.401 |
| original_face | 26.65 | 45.95 | 33.71 | 53.89 | 0.559 | 0.311 |
| rich_audio | 25.98 | 46.85 | 33.38 | 53.53 | 0.561 | 0.322 |
| rich_face | 29.90 | 50.45 | 37.53 | 57.48 | 0.608 | 0.381 |
| selected_audio | 22.02 | 40.54 | 28.53 | 48.63 | 0.509 | 0.304 |
| selected_concat | 26.89 | 40.54 | 32.29 | 53.87 | 0.559 | 0.379 |
| selected_face | 25.27 | 47.75 | 32.98 | 52.64 | 0.552 | 0.345 |

## Predeclared comparisons

Screen: at least +3 percentage points F1, no mean balanced-accuracy or AUC loss, recall loss at most five points, and positive F1 change in at least two repeats.

| Candidate | Control | F1 change (points) | Positive repeats | Pass |
|---|---|---:|---:|---|
| aligned_audio | original_audio | -3.20 | 0/3 | False |
| rich_audio | original_audio | +0.12 | 1/3 | False |
| selected_audio | original_audio | -4.74 | 0/3 | False |
| aligned_face | original_face | +3.81 | 2/3 | True |
| rich_face | original_face | +3.82 | 2/3 | True |
| selected_face | original_face | -0.73 | 1/3 | False |
| selected_concat | original_concat | -4.87 | 0/3 | False |
| neural_adaptive | neural_equal | -1.48 | 1/3 | False |
| neural_quality | neural_adaptive | +0.20 | 2/3 | False |
| neural_equal | original_concat | -7.58 | 1/3 | False |
| neural_adaptive | original_concat | -9.06 | 0/3 | False |
| neural_quality | original_concat | -8.86 | 0/3 | False |

## Selection and computation

- selected_audio: aligned_audio in 6/15 folds, original_audio in 5/15 folds, rich_audio in 4/15 folds.
- selected_face: aligned_face in 6/15 folds, original_face in 6/15 folds, rich_face in 3/15 folds.

| Condition | Retained features (min–max) | Mean final fit seconds | Mean inference seconds |
|---|---:|---:|---:|
| aligned_audio | 92–92 | 0.0044 | 0.0006 |
| aligned_face | 196–196 | 0.0048 | 0.0006 |
| neural_adaptive | 144–625 | 0.0746 | 0.0019 |
| neural_equal | 144–625 | 0.2083 | 0.0018 |
| neural_quality | 144–625 | 0.0781 | 0.0019 |
| original_audio | 46–46 | 0.0035 | 0.0006 |
| original_concat | 144–144 | 0.0053 | 0.0009 |
| original_face | 98–98 | 0.0039 | 0.0006 |
| rich_audio | 200–201 | 0.0055 | 0.0007 |
| rich_face | 420–424 | 0.0092 | 0.0007 |
| selected_audio | 46–201 | 0.0043 | 0.0006 |
| selected_concat | 144–625 | 0.0086 | 0.0011 |
| selected_face | 98–424 | 0.0053 | 0.0006 |

Times cover final fits/inference; inner selection is separate in fold manifests. Full fold/repeat confusion counts, domain metrics, AP, choices and learning curves remain in the local run artifacts. Participant-level artifacts are not for publication.

## Comparison with previous work

The retained text reference remains at 57.5% F1; the earlier threshold candidate reached 58.5% but failed its own promotion rule. The v5 quality audio/facial primary scored 33.3%, versus 29.5% for its ordinary primary. Here original concatenation reaches 37.2%, while quality neural fusion reaches 28.3%. These are different recipes on the same reused cohort/folds, not fresh independent confirmations. The v5 within-study quality improvement does not establish that the v6 neural quality gate helps.

The evidence points to weak measured audio/facial predictive signal and unstable inner selection under these tested recipes. Aligned audio loses about 3.2 F1 points relative to whole-interview audio, while facial summaries improve. This does not prove why signal is weak, that the dataset is unusable, or that its performance ceiling has been reached. Selecting the highest outer-scoring feature combination for another run would make the follow-up adaptive and must be reported as such.

## Verification and decision

All 195 serialized models reproduced all 7,245 prediction rows with zero maximum score error. The verifier replayed 540 inner candidate score vectors, retrained all 45 neural models, and checked hashes, folds, preprocessing, selection, metrics and comparison decisions. All 154 tests passed; one existing regex warning remains.

This completes the frozen branch-audit series. Do not promote the fusion variants or discard unfavorable results. Preserve the reference and report the facial improvements as exploratory branch-level evidence. No claim of fresh test performance, clinical readiness, or a proven reliability-aware advantage follows from this run.


---

## Record: label_provider_inquiry.md

# Provider inquiry draft (not sent)

Public-documentation check: 2026-10-30. The official E-DAIC manual lists
PHQ_Binary and PHQ_Score but does not specify how to reconcile their disagreement.
No verified official correction was found in that manual. Existing local audit
remains the evidence for the 20 discrepancies; labels have not been modified.
https://dcapswoz.ict.usc.edu/wwwedaic/E-DAIC%20Manual.pdf

Subject: Clarification of E-DAIC released binary labels and transcript provenance

We are conducting participant-level depression-screening research using the
released E-DAIC training, development, and test splits. In our 275-participant
copy, 20 participants have PHQ_Binary=0 and PHQ_Score>=10 (13 train, 3 development,
4 test). The master binary labels agree with the split labels, and questionnaire
item sums agree with the split scores. No integer score cutoff reproduces all
released binary labels.

Could you confirm the intended definition of PHQ_Binary, the applicable release
version, and whether an official erratum or replacement label file exists?
Should released-label and score-threshold experiments be treated as distinct
targets? We can provide participant IDs through your approved private channel.

Could you also confirm whether the supplied Start_Time/End_Time/Text/Confidence
transcripts contain only participant speech, how confidence is defined, and
whether timestamps share the time origin of the supplied audio and OpenFace
features? Two training recordings have substantial WAV/feature endpoint
disagreement; we have quarantined them pending clarification.

Until clarified, we retain the unchanged released labels and explicitly report
the uncertainty. Thank you.

No email or other external message has been sent. This draft requires the
researcher's dataset access account/contact route and explicit sending instruction.


---

## Record: label_speech_investigation.md

# Label and speech-segmentation investigation

Investigation date: 2026-10-29. Scope: read-only reconciliation of supplied label
files and feasibility of transcript-aligned audio selection. No training, target
replacement or new development/test predictions. Participant-level audit outputs
remain under ignored reports/data_audit/. Stage 10 is still pending.

## Label evidence

All 275 master questionnaire item sums equal the split-file PHQ_Score values.
All 275 master Depression_label values equal split-file PHQ_Binary. Master split
assignments also agree. The older metadata table overlaps 219 participants and
agrees with their split scores and binary labels. Therefore the discrepancies are
present in the supplied files; they were not introduced by model preprocessing.

| Partition | Participants | Released positives | Score >=10 positives | Disagreements |
|---|---:|---:|---:|---:|
| Train | 163 | 37 | 50 | 13 |
| Development | 56 | 12 | 15 | 3 |
| Test | 56 | 17 | 21 | 4 |
| Total | 275 | 66 | 86 | 20 |

All 20 disagreements have released label zero and score at least 10. Their score
distribution is: ten at 10, four at 11, three at 12, and one each at 13, 15 and 16.
No integer score cutoff reproduces all released binary labels. There are no
released positive labels below score 10. The discrepancy is not explained simply
by changing >=10 to >10.

The 219-row Detailed_PHQ8_Labels.csv is internally consistent (item sums match its
totals), but two totals disagree with current split scores. Its existence does
not resolve which questionnaire version should define the target.

Ali et al. explicitly report the same 20 participant IDs and adopt a score-based
correction in their study [2]. We checked the ID sets match exactly without
publishing participant records here. This is independent corroboration, not an
official dataset correction notice. The E-DAIC manual lists the binary and score
fields but does not resolve the mismatch [1]. The older DAIC-WOZ manual explicitly
defines its PHQ8 binary field as score >=10 [3]; it describes a different release.
It is supporting context, not sufficient authority to overwrite E-DAIC labels.

Conclusion: keep the completed released-label experiments intact. A separately
named score>=10 sensitivity study is scientifically defensible if its target is
declared in advance, but it answers a different outcome definition. No automatic
relabeling or promise of better metrics is warranted. A definitive provenance
resolution still requires a release-specific clarification from the provider.

## Interview-condition difference

The manual identifies IDs 300-492 as human-controlled WoZ interviews and IDs
600-718 as AI-controlled interviews [1]. The supplied partitions confirm:

| Partition | WoZ participants (released positives) | AI participants (released positives) |
|---|---:|---:|
| Train | 143 (32) | 20 (5) |
| Development | 46 (10) | 10 (2) |
| Test | 0 (0) | 56 (17) |

Thus training-only stratified CV mostly measures performance on WoZ interviews,
while the original test measures AI interviews. This is a confirmed difference
in recording conditions, not proof it caused any particular prediction error.
Future training-only evaluation should record results by condition. The five AI
positives in training make precise AI-specific estimates difficult. Avoid an
AI-only training recipe or grand claims from this tiny subgroup. Condition/ID
metadata should remain audit metadata, not shortcut predictor inputs.

## Speech evidence and interpretation

All 275 archives contain WAV audio and CSV transcripts with Start_Time, End_Time,
Text and Confidence; none has an explicit speaker column. There are 241 mono
16 kHz recordings and 34 stereo 48 kHz recordings, all with 16-bit samples. Ten
stereo recordings are in training, four in development and twenty in test.
The timestamps can select corresponding feature
frames; they do not themselves identify a speaker. Ali et al. describe the
provided transcripts as participant answers and report transcribing the raw audio
to obtain both participant and interviewer speech [2]. That supports a candidate
participant-speech mask, but the present investigation does not claim manual
speaker validation or a measured diarization accuracy.

The audit merges overlapping nonempty transcript intervals, counts invalid
intervals, inspects WAV headers, and measures coverage on training feature
timestamps. Features outside transcript intervals are **untranscribed**, not
automatically silence or interviewer speech. Zero pitch is likewise not a
silence detector: unvoiced speech can also have zero pitch. No transcript text,
recordings or segment-level personal content is copied into tracked documentation.

Some feature timestamps repeat. The first audit attempt rejected such a sequence;
inspection showed a repeated timestamp rather than backwards time. The audit now
records repeats and gaps and keeps the original values. It does not silently
reconstruct timestamps from row numbers. This is important for later interval
alignment. Existing aggregated tables were not modified.

Across all 163 training sessions, transcript intervals cover a median 60.0% of
feature frames (range 15.0%-99.98%). These are interval-coverage diagnostics, not
verified participant-speech percentages. The median zero-pitch fraction is 57.9%
inside those intervals and 95.6% outside, reinforcing that zero pitch cannot be
used alone to identify silence. There are repeated timestamps in 126 sessions,
one repeat each; the median step is 0.01 seconds, but individual gaps reach 0.26
seconds. Filtering must use actual timestamps, not a reconstructed row clock.

Two training sessions, both from the AI condition, have feature/audio endpoint
mismatches greater than one second. One feature sequence extends 107.04 seconds
beyond its WAV; another stops 485.537 seconds before its WAV ends. Independently
reopened WAV headers agree with the actual archive member byte lengths, so this
is not merely an incorrectly inferred audio duration. The cause (different source
version, truncation, or another extraction issue) is unknown. Endpoint agreement
for the other sessions also does not prove detailed temporal synchronization.
IDs and diagnostics are saved only in the ignored local audit outputs.

Only one transcript interval across all 275 files was invalid or empty under the
audit's checks, in development; none was flagged in training. This validates
structural interval bounds, not transcription completeness or speaker identity.

A further local check streamed all sample frames of the ten stereo training
recordings in bounded chunks. Two had bit-identical channels; the other eight
had channel correlations between 0.9999905 and 0.9999988. These near-identical
signals do not demonstrate usable interviewer/participant channel separation.
No listening-based speaker annotation was performed. Stereo channel diagnostics
are saved separately in training_stereo_channel_audit.csv.

## Recommended next experiment, not executed here

1. Validate a small, fixed sample of transcript-covered and uncovered audio spans
   across both interview conditions. Check speaker identity, timing offsets and
   missing speech before calling the mask participant-only. Keep the review local.
   First resolve the two length anomalies, or explicitly quarantine them from a
   prospective segmentation experiment. Re-extract features from verified source
   WAVs with a documented matching configuration if needed; do not invent offsets
   or silently change the completed experiments' participants.
2. Freeze a comparison of full-session versus transcript-covered eGeMAPS features
   using the same simple classifier and matched nested training folds. Name the
   condition transcript-covered unless speaker checks justify a stronger claim.
   Keep outside-interval and zero-pitch statistics as diagnostics, not assumed
   silence labels. Reuse the actual frameTime values, including duplicates/gaps.
3. Keep the target unchanged for that comparison to isolate segmentation's effect.
   If a score-derived target study is authorized, register it separately and
   report both policies rather than selecting whichever gives higher scores.
4. Report precision, recall, specificity, balanced accuracy and fold AUC alongside
   F1; include simple constant references and condition-specific diagnostics.
   Preserve the already evaluated test set and seek independent confirmation.

This prioritizes a concrete input issue before a larger sequence model. It does
not establish that segmentation will improve prediction.

## Verification and reproduction

Completed audit: `reports/data_audit/label_speech_20261028T232459648689Z` (UTC
directory timestamp; local investigation date 2026-10-29). The folder contains
label reconciliation, legacy questionnaire discrepancies, audio/transcript timing,
the aggregate results and verification files. A separate local JSON records the
two WAV-header/file-size checks. No participant records are committed.

93 tests pass. All 275 archive records and all aggregate timing statistics were
checked. The 163 inspected training eGeMAPS content hashes match the preceding
audio-study extraction. Source/label hashes verified; the previous Stage 9 output
hashes and audio-study input/source hashes remain unchanged.

```powershell
.venv\Scripts\python.exe -m src.dataset.label_speech_audit
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/label-speech-check --tb=short
```

The main audit caches per-participant diagnostics by source hash and archive
size/mtime. It inspects all split labels and transcript/WAV headers, but acoustic
frame diagnostics are restricted to official training participants. The additional
stereo check reads PCM frames in chunks of 65,536, compares left/right sample
equality, and accumulates whole-recording Pearson correlation without exporting
waveforms. These diagnostics do not fit or evaluate a classifier.

## Sources and access limits

[1] USC [E-DAIC manual](https://dcapswoz.ict.usc.edu/wwwedaic/E-DAIC%20Manual.pdf),
accessed 2026-10-29. Its 219-directory description differs from the downloaded
275-participant release. It also describes 88 eGeMAPS measures, whereas our actual
LLD CSV schema contains 23 descriptors plus name/frameTime. Use the verified file
schema for implementation rather than assuming the manual's feature count.

[2] Ali et al., [Leveraging Audio and Text Modalities in Mental Health, v2](https://arxiv.org/html/2412.10417v2),
sections 3.1.1 and 4.4, accessed 2026-10-29. Used for label discrepancy and transcript
provenance evidence only. Its LLM results use a different evaluation setup and are
not benchmark comparisons for this project.

[3] USC [DAIC-WOZ documentation](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf),
accessed 2026-10-29. Different release from E-DAIC.

The official labels directory returned HTTP 403 and the previously referenced
MDPI study returned HTTP 429. No current replacement label package or official
erratum was verified. No provider correspondence was sent, and no dataset files
were downloaded or changed.


---

## Record: lexical_results.md

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


---

## Record: mpnet_rank_results.md

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


---

## Record: mpnet_results.md

# MPNet encoder results

Verified 2026-11-02. Run: `experiments/runs/mpnet_20261102T171156498575Z`.
The [protocol](mpnet_protocol.md) preceded extraction/fitting. Same 161 training
participants, 37 released positives, screened transcripts, pooling and classifier
recipe; no new development/test predictions or label changes.

**MPNet with linear SVM passes the predeclared exploratory improvement screen.**
Mean F1 rises from 52.8% to 58.1% (+5.23 points), with positive F1 differences in
all three repeats. Balanced accuracy, AUC and recall also improve. MPNet logistic
does not clear the three-point F1 requirement. MPNet SVM becomes the stronger
research reference; it is not an independently test-validated replacement.

All entries are percentages; F1 SD is across overlapping repeats, not a CI.

| Encoder / classifier | F1 +/- SD | Precision | Recall | Specificity | Accuracy | Balanced accuracy | Mean fold AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniLM / SVM | 52.8 +/- 3.4 | 43.9 | 66.7 | 74.2 | 72.5 | 70.4 | 79.4 |
| MiniLM / logistic | 52.5 +/- 2.5 | 42.1 | 70.3 | 70.7 | 70.6 | 70.5 | 78.2 |
| MPNet / SVM | 58.1 +/- 5.1 | 47.8 | 73.9 | 76.1 | 75.6 | 75.0 | 80.4 |
| MPNet / logistic | 54.0 +/- 5.9 | 45.4 | 66.7 | 76.3 | 74.1 | 71.5 | 79.2 |
| Always negative | 0.0 +/- 0.0 | 0.0 | 0.0 | 100.0 | 77.0 | 50.0 | 50.0 |
| Always positive | 37.4 +/- 0.0 | 23.0 | 100.0 | 0.0 | 23.0 | 50.0 | 50.0 |

The domain limitation remains: MPNet SVM averages 60.9% F1 in 143 WoZ interviews
but 38.9% in 18 AI interviews (only five positives). The historical test partition
contains AI interviews only; the pooled improvement cannot be assumed to transfer
there. Neither that subgroup nor the previously inspected test was used to tune
this encoder comparison. The 65%-70% pooled F1 aspiration remains unmet.

## Provenance and verification

Frozen local `sentence-transformers/all-mpnet-base-v2` revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`, safetensors, float32, eager attention,
TF32 disabled. The unused learned pooler is deliberately disabled; its two
checkpoint keys are unused, and no required encoder weights are missing.
The same encoding helper as MiniLM uses 254-token content chunks plus special
tokens, without dropping longer utterance remainders. No transcripts were uploaded.
The [model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
describes a general sentence encoder, not a depression-specific model.

Extraction covers all 14,724 screened utterances. Recorded per-participant
extraction time totals 307 seconds (5.1 minutes), excluding download/loading and
classifier fitting. Peak allocated CUDA memory is 0.818 GiB. All 161 caches and
pooled features passed hash/shape/finite checks; replaying the first participant
reproduced utterance embeddings exactly (maximum difference zero).

Independent checks restored 60 pipelines, recomputed 2,898 predictions and 18
repeat metric rows, verified folds/source/model hashes, and matched 966 prior
MiniLM control predictions and scores. The existing 117-test suite passes; the
new encoder is additionally covered by real-data extraction and replay checks.
Complete metrics, repeat/domain summaries and screening decisions are in the run.

An additional descriptive audit of the previous MiniLM errors found no obvious
false-positive concentration in the 12 retained label/score-disagreement cases:
SVM false-positive rates were 25.0% there versus 25.9% among the other 112 released
negatives; logistic rates were 22.2% versus 30.1%. These small-group observations
do not resolve the labels or prove they are harmless. Scores were used only for
this post-hoc diagnostic, never as predictors or replacement targets.

```powershell
.venv\Scripts\python.exe -m src.dataset.mpnet
.venv\Scripts\python.exe -m src.experiments.mpnet_study
.venv\Scripts\python.exe -m src.experiments.verify_mpnet experiments/runs/mpnet_20261102T171156498575Z
```


---

## Record: pipeline_v2_results.md

# Pipeline remediation results — completed 2026-11-04

This is an exploratory training-cohort comparison, with unchanged released binary
labels. Historical experiments and their source files remain frozen. The v2
entrypoints below are the recommended path for further work. No new official
development/test evaluation, production deployment, or provider message occurred.

## What was corrected

| Finding | Implemented response | Remaining limit |
|---|---|---|
| Unknown ASR confidence previously became perfect confidence | v2 uses zero plus an explicit known flag | Confidence is not calibrated diagnostic reliability |
| Label-only folds sometimes omitted the AI subgroup | Joint domain/label stratification, including inner selection | Only 18 AI participants, five positive, in retained training data |
| Neural fitting performance diverged from validation | Inner-selected epochs and regularization, saved learning curves, matched initialization | Neural models still underperform the text SVM |
| Coarse temporal bins discarded utterance boundaries | Rebuilt all 14,724 screened utterances using actual feature timestamps | Fusion currently pools these aligned summaries; it is not a full sequence model |
| Transcript row order was not always chronological | Stable timestamp sorting moves embeddings and metadata together | Timestamp agreement alone cannot establish speaker identity |
| Most original quality fields were constant; stress tests exposed severity | Measured waveform/tracker/ASR proxies and observable availability, no severity input | These are not validated SNR, diarization, or reliability probabilities |
| Temporal branch used weaker text representations | Aligned cache retains MPNet embeddings; fusion retains the original MPNet mean | Text remains a fixed encoder |
| Stacking could learn from optimistic fitted margins | Base-model cross-fitting nested within inner and outer training partitions | Repeated adaptive work on the same cohort still limits generalization claims |
| New cache metadata key was accidentally overwritten during sorting | Separate field variable and independent cache-key verification | Rebuilt before fusion fitting |

Two facial streams remain quarantined for endpoint timing mismatch. The participant
cohort is unchanged. The original screening policy is retained before sorting so
that transcript-to-embedding correspondence remains reproducible.

## Manual and unresolved data provenance

The supplied four-page `data/raw/E-DAIC Manual.pdf` was inspected. SHA256:
`eff6958ac918492b20cae9506e3f69e2a87f0fd5ce9cf1382ca5da16b634e706`.
It confirms mixed WoZ/AI train and development interviews and AI-only test
interviews. It lists PHQ_Binary and PHQ_Score but does not specify how to reconcile
their disagreements, nor guarantee participant-only transcript speech. Its stated
219 directories differ from this release's 275. It therefore does not resolve the
20 binary-versus-score discrepancies, including 12 in the retained training cohort.
No automatic relabeling or speaker-cleaning claim is justified.

A local listening worksheet contains 24 short clips from eight participants,
sampled across domain and class without model predictions. Review covered
intervals and gaps at `data/processed/aligned_v2/speaker_review/index.html`, then
export annotations locally. No speaker annotations have been invented. This
sample can identify problems; it cannot certify the entire dataset.

## Matched results on the revised folds

Mean of three repeats; percentages except AUC shown on a 0–1 scale. F1 is computed
from pooled participant predictions per repeat, while AUC is averaged across folds.

| Procedure | Precision | Recall | F1 | Mean fold AUC |
|---|---:|---:|---:|---:|
| MPNet SVM reference | 47.3% | 73.9% | 57.5% | 0.805 |
| Regularized mean pooling | 35.8% | 60.4% | 44.9% | 0.716 |
| Regularized attention pooling | 37.6% | 64.0% | 47.2% | 0.749 |
| Inner-selected neural procedure | 36.6% | 62.2% | 46.0% | 0.739 |

Mean fold average precision is 0.607 for the reference, 0.480 for mean pooling,
0.520 for attention, and 0.513 for the selected neural procedure. Average precision
summarizes ranking across thresholds; it is not the precision at cutoff zero.

The historical 58.1% reference and highest observed 58.8% class-penalty result used
different folds. They must not be ranked directly against these new numbers.
The reference recipe remains retained; none of the neural procedures passed the
predeclared improvement screen.

The WoZ-only fitting / AI-only evaluation diagnostic reaches 36.9% F1 and 0.621
AUC. It uses only the official training partition, not a fresh test cohort, and
demonstrates that domain transfer remains a substantial weakness.
Within the joint-fold comparison, the reference achieves 60.2% F1 on WoZ and
42.2% on AI interviews. These subgroup estimates are especially uncertain for AI.

Attention's mean inner fitting F1 rises from 76.3% at epoch 40 to 99.0% at epoch
100, while mean inner-validation F1 falls from 46.4% to 37.0%. These averages
cover all inner runs and regularization candidates, not selected outer scores.
The learning curves support controlling training duration; they do not establish
that learned attention is better than the simpler reference.
Attention selected epoch 40 in nine outer fits, epoch 20 in three, and epoch 10
in three; none selected epoch 100. Mean pooling selected epoch 100 in six fits,
illustrating why training duration is selected rather than universally shortened.

## Reproduction and evidence

### Aligned fusion comparison

Completed and independently verified on 2026-11-04. Run:
`experiments/runs/aligned_fusion_v2_20261104T104203391619Z`.

| Clean-input procedure | Precision | Recall | F1 | Mean fold AUC | Mean fold AP |
|---|---:|---:|---:|---:|---:|
| Retained text SVM | 47.3% | 73.9% | 57.5% | 0.805 | 0.607 |
| Text-margin calibration control | 40.1% | 80.2% | 53.5% | 0.805 | 0.618 |
| Ordinary aligned fusion | 39.3% | 81.1% | 52.9% | 0.782 | 0.550 |
| Measured-quality aligned fusion | 39.5% | 81.1% | 53.1% | 0.770 | 0.526 |
| Inner-selected procedure, including reference | 44.5% | 75.7% | 55.9% | 0.805 | 0.609 |

All four candidate procedures failed the predeclared clean improvement screen.
Measured-quality fusion increases recall but reduces precision, F1, balanced
accuracy, AUC, and AP relative to the reference. No replacement is promoted.

| Missing-input condition | Ordinary F1 | Quality-aware F1 | Text reference F1 |
|---|---:|---:|---:|
| Audio absent | 53.5% | 54.1% | 57.5% |
| Face absent | 52.4% | 52.4% | 57.5% |
| Audio and face absent | 52.9% | 53.0% | 57.5% |

Quality-aware minus ordinary fusion gains **0.28 percentage points** on average
over these three conditions, below the required +5 points. Clean F1 improves by
0.14 points over ordinary fusion. The robustness screen fails. Text-only scores
remain identical when audio/face are removed, as expected.

Quality-aware fusion's clean subgroup F1 is 45.8% for AI and 54.3% for WoZ,
versus the reference's 42.2% and 60.2%. The small AI increase does not compensate
for the overall reduction or establish transfer success. These subgroup results
are descriptive; they were not used to choose a new winner.

Verification restored all 75 saved fusion bundles and reproduced all 9,660
prediction rows, 60 repeat-metric rows, 135 inner candidate scores, domain metrics,
selection rules, comparison screens, and nested cross-fitting membership. The
reference predictions exactly match the separately verified baseline. Across the
three v2 studies, 138 bundles and 12,129 prediction rows were verified.

Alignment verification checked all cache hashes, shapes, confidence flags,
chronological ordering and cache keys, and independently recomputed source
intervals for one interview from each domain. A subsequent build reused all
161 caches without changing artifacts or the manifest. Nine of 11 quality fields
vary in these data; finite-audio fraction and confidence-known are constant, and
audio frame coverage varies very little. This is measured variation, not proof
of a predictive quality signal.

### Commands

Use the repository venv from PowerShell. These commands create local ignored
artifacts and never call the official development/test evaluation entrypoints.

```powershell
.venv\Scripts\python.exe -m src.experiments.baseline_v2
.venv\Scripts\python.exe -m src.experiments.verify_v2 experiments/runs/baseline_v2_20261103T165509103723Z
.venv\Scripts\python.exe -m src.experiments.pooling_v2 experiments/runs/baseline_v2_20261103T165509103723Z
.venv\Scripts\python.exe -m src.experiments.verify_v2 experiments/runs/pooling_v2_20261103T165631446124Z
.venv\Scripts\python.exe -m src.dataset.aligned_v2
.venv\Scripts\python.exe -m src.dataset.verify_aligned_v2
.venv\Scripts\python.exe -m src.dataset.speaker_review_v2
.venv\Scripts\python.exe -m src.experiments.aligned_fusion_v2 experiments/runs/baseline_v2_20261103T165509103723Z
.venv\Scripts\python.exe -m src.experiments.verify_aligned_fusion_v2 experiments/runs/aligned_fusion_v2_20261104T104203391619Z
```

New runs receive new timestamped paths: substitute their printed paths when
reproducing dependent commands. The listed baseline and pooling paths identify
the verified runs reported here. Earlier partial/superseded v2 runs are excluded.
Baseline verification restored 18 models and checked 537 predictions; neural
verification restored 45 bundles and checked 1,932 predictions. Protocols are
`pipeline_v2_protocol.md` and `aligned_fusion_v2_protocol.md`.
The complete suite passes 140 tests. One pre-existing regex capture-group warning
remains in frozen historical screening code; the v2 expression uses noncapturing
groups. A cache regression test builds unsorted synthetic transcript features and
proves that a second call reuses the cache without reopening the archive.

## Outstanding work requiring evidence

Provider clarification of labels, human speaker/timing review, calibrated quality
validation, and fresh independent evaluation remain open. Existing repeated CV
cannot become an untouched validation set by reshuffling it. Missing-input tests
do not prove robustness to realistic microphone noise or mixed speakers. No model
is promoted merely because a point estimate is the largest.


---

## Record: precision_results.md

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


---

## Record: primary_fallback_v5_results.md

# Audio/facial primary with text fallback — 2026-11-04

Completed run: `experiments/runs/primary_fallback_v5_20261104T201923176706Z`.
Protocol: `primary_fallback_v5_protocol.md`. Same 161 training participants,
37 released positives, joint domain/label folds and three seeds as the v2 reference.
No new official development/test evaluation, relabeling, or model deployment.

## What changed

The primary prediction uses recording-derived acoustic and facial measurements,
with no transcript embeddings, text margins or ASR confidence in its fusion design.
It uses the existing transcript-aligned acoustic/facial summaries, so preprocessing
still depends on transcript timestamps. It is not an end-to-end raw-waveform model.
The availability-only fallback consults text only when neither auxiliary is present.
The quality fallback consults text only when neither meets fixed pre-existing
quality checks. One usable auxiliary keeps primary inference active.

At inference the fallback encoder/classifier features are read only for routed
participants. Tests enforce primary independence from text scores/text quality
and ensure a primary-usable input never calls the text classifier. Missing text
on a required fallback raises an error. This experiment assumes fallback text is
available; it does not establish behavior when all three modalities are absent.

Base margins used to fit primary fusion are cross-fitted inside each training set,
including inside each inner selection fold. All preprocessing and C selection stay
within training partitions. Missing-input augmentation covers clean, audio absent
and face absent with equal total participant weight. No corruption severity,
participant target or known prediction error triggers the fallback.

## Clean-input results

Mean repeat metrics; F1 pools outer-fold participant decisions per repeat. AUC/AP
are means of fold rankings. Fixed fallback text and retained reference have different
model-selection recipes and are reported separately.

| Procedure | Precision | Recall | F1 | Accuracy | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| Audio/facial primary, no quality | 21.7% | 45.9% | 29.5% | 49.5% | 0.465 | 0.282 |
| Audio/facial primary, quality | 24.7% | 51.4% | 33.3% | 52.8% | 0.519 | 0.276 |
| Availability fallback | 21.7% | 45.9% | 29.5% | 49.5% | 0.465 | 0.282 |
| Quality fallback | 24.7% | 51.4% | 33.3% | 52.8% | 0.519 | 0.275 |
| Fixed text fallback model alone | 46.9% | 67.6% | 55.4% | 74.9% | 0.805 | 0.618 |
| Retained text reference | 47.3% | 73.9% | 57.5% | 74.7% | 0.805 | 0.607 |

Quality-primary F1 exceeds plain-primary by 3.88 percentage points, with positive
differences in all three repeats. This is a descriptive contrast between policies
that differ in eligibility masking and quality interactions; it does not isolate
which part helps. AP falls slightly. Both remain weak: always predicting positive
has F1 37.4% on this cohort (precision23.0%, recall100%, balanced accuracy50%).
The always-negative majority rule has accuracy77.0% but F1 zero. Neither accuracy
alone nor an improvement over the weakest fusion baseline establishes success.

All four primary/fallback conditions fail the predeclared clean improvement screen
against the retained reference. No replacement is promoted. Do not compare these
training-CV estimates directly to the historical Stage9 official-test scores.

## Fallback behavior

| Condition | Clean F1 | Audio absent | Face absent | Both absent |
|---|---:|---:|---:|---:|
| Plain primary | 29.5% | 29.3% | 30.2% | 7.7% |
| Availability fallback | 29.5% | 28.9% | 30.2% | 55.4% |
| Quality primary | 33.3% | 35.0% | 35.7% | 37.4% |
| Quality fallback | 33.3% | 35.0% | 37.5% | 55.4% |
| Fixed text alone | 55.4% | 55.4% | 55.4% | 55.4% |

Primary-only controls with both inputs absent predict from their intercept; those
scores are ablation controls without observed auxiliary evidence. Recovering text
performance in that scenario verifies fallback functionality, not a new quality
fusion advantage or robustness to realistic noisy recordings.

| Fallback policy | Clean routes | Audio absent routes | Face absent routes | Both absent routes |
|---|---:|---:|---:|---:|
| Availability only | 0/161 | 2/161 | 0/161 | 161/161 |
| Quality based | 1/161 | 2/161 | 78/161 | 161/161 |

Routes are identical across repeats because the rule uses observed availability
and fixed measured-quality thresholds, not fitted prediction confidence or labels.
On clean inputs only one person invokes quality fallback, leaving the weak primary
classifier responsible for almost all decisions. The main performance bottleneck
is thus still the audio/facial predictor under this recipe. The quality proxies
identify recording properties, not necessarily whether a depression prediction
will be correct. They must not be described as calibrated predictive reliability.

## Verification

All 152 tests pass (one historical regex warning). Verification restored 90 saved
bundles, checked 11,592 prediction rows, independently recomputed routes, replayed
180 inner candidate score vectors and checked nested cross-fitting membership,
source/model hashes, metrics, subgroup outcomes and comparison decisions.
The largest restored score drift was 3.48e-7, within an absolute1e-6 tolerance
for float32 PCA inputs after serialization; every predicted class was identical.
The retained-reference predictions matched the separately verified baseline.

```powershell
.venv\Scripts\python.exe -m src.experiments.primary_fallback_v5 experiments/runs/baseline_v2_20261103T165509103723Z
.venv\Scripts\python.exe -m src.experiments.verify_primary_fallback_v5 experiments/runs/primary_fallback_v5_20261104T201923176706Z
```

Substitute the new printed path when reproducing a run. Existing historical runs
and primary models remain preserved. The study implements the requested modality
priority, but does not justify a claim of improved overall predictive performance.


---

## Record: representation_followup_results.md

# Learned pooling and complementary encoders

Completed and independently verified 2026-11-03. Two additional structural
experiments tested alternatives to averaging MPNet embeddings and changing SVM
thresholds. Neither improved the retained reference. Keep all negative results.

| Training-CV procedure | Precision | Recall | F1 +/- repeat SD | Specificity | Accuracy | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Retained MPNet SVM reference | 47.8% | 73.9% | 58.1 +/- 5.1% | 76.1% | 75.6% | 75.0% | 80.4% |
| Previous class-penalty selection | 49.2% | 73.0% | 58.8 +/- 4.3% | 77.7% | 76.6% | 75.3% | 80.1% |
| Learned mean-pooling network | 38.6% | 57.7% | 46.2 +/- 4.6% | 72.3% | 68.9% | 65.0% | 71.9% |
| Learned attention-pooling network | 38.3% | 53.2% | 44.4 +/- 2.4% | 74.5% | 69.6% | 63.8% | 75.4% |
| Inner-selected pooling architecture/epoch | 40.2% | 52.3% | 45.2 +/- 6.0% | 77.2% | 71.4% | 64.7% | 73.0% |
| Inner-selected MPNet/MiniLM blend | 46.1% | 66.7% | 54.5 +/- 4.5% | 76.9% | 74.5% | 71.8% | 80.3% |

All use the same 161-person training cohort, released labels, five outer folds
and three repeats. SD is not a confidence interval, and repeats overlap. Previous
class-penalty results are from the [precision experiment series](precision_results.md).
They remain the highest observed mean F1 (58.8%) but did not clear its precision
promotion screen; their +0.69 F1-point difference is not a demonstrated reliable
improvement. The saved full-training research reference remains unchanged.

## Learned pooling

[Protocol](utterance_pooling_protocol.md), run
`experiments/runs/utterance_pooling_20261103T115821003735Z`.

Preserved all 14,724 screened utterances, with 41-197 per participant. Frozen
MPNet embeddings underwent a label-independent fixed random projection to64
dimensions. A small 16-unit learned representation feeds either masked mean
pooling or learned attention. Both use identical training schedules, with epoch
40/100 chosen on inner validation. One loss per participant avoids treating
utterances as independent labeled people. These networks are order-invariant;
they do not model temporal order. Attention adds 17 parameters to its control.

The inner-selected procedure loses 12.82 F1 points against the reference, with
losses in all three repeats. Attention has higher AUC than its neural mean
control but lower F1. It does not pass the clean or precision improvement screen.
The inner selector chose mean pooling in nine folds and attention in six.

This rules out this particular small projected architecture/training recipe as
an improvement. It does not prove all attention methods fail: the projection,
small hidden layer, training schedule and frozen encoder remain limitations.
No causal claim that the neural model overfit is made without a learning-curve
analysis. No encoder fine-tuning was performed.

## Complementary encoder blend

[Protocol](encoder_blend_protocol.md), run
`experiments/runs/encoder_blend_20261103T120034683682Z`.

Joint inner selection tested SVM settings for both encoders and five MPNet
weights (1/.75/.5/.25/0). Each encoder has independently fold-fitted PCA32.
Combining raw margins is not probability calibration. Selected MPNet weights:
.75 in six folds, .25 in four, 1 in three, .5 in two. The procedure loses 3.57 F1
points and 7.21 recall points against the reference, with F1 losses in all three
repeats. Its AUC remains similar, but its decisions do not improve. Do not choose
a different blend weight retrospectively from outer predictions.

## Domain checks

| Procedure | AI F1 | WoZ F1 |
|---|---:|---:|
| Reference | 38.9% | 60.9% |
| Neural mean pooling | 44.3% | 46.6% |
| Neural attention | 29.9% | 46.7% |
| Inner-selected pooling | 38.1% | 46.5% |
| Encoder blend | 40.9% | 56.4% |

AI has only 18 participants/five positives. Its apparent subgroup gains do not
justify post-hoc model selection; the historical test cohort is AI-only, and no
new test predictions were generated. Label and speaker-attribution uncertainties
remain unchanged.

## Verification

All 129 tests passed (existing pandas regex warning only). Pooling verification
restored 45 bundles, checked 1,932 predictions, 12 repeat metric rows and 60 inner
candidate records. It verified all participant caches and replayed 12 sampled
inner epoch checkpoints by retraining, as well as checking training-only scalers.
Encoder-blend verification restored 30 bundles, checked 966 predictions, six
metric rows and all 1,800 inner combinations, and refitted selected inner SVMs.
Both runs reproduced all 483 historical MPNet reference scores. Total new saved
bundles checked:75; prediction records:2,898. Bundles can duplicate a selected
architecture and are not a count of unique training fits.

```powershell
.venv\Scripts\python.exe -m src.experiments.utterance_pooling
.venv\Scripts\python.exe -m src.experiments.verify_utterance_pooling experiments/runs/utterance_pooling_20261103T115821003735Z
.venv\Scripts\python.exe -m src.experiments.encoder_blend
.venv\Scripts\python.exe -m src.experiments.verify_encoder_blend experiments/runs/encoder_blend_20261103T120034683682Z
```

These experiments establish negative findings for tested recipes, not the best
possible model. Further adaptive search on the same folds risks optimizing the
research process to this cohort. Fresh independent validation, a confirmed label
definition and verified participant speech remain necessary for stronger claims.
No model was deployed and the best-F1 reference checkpoint was not replaced.


---

## Record: representation_study_results.md

# Supplied pretrained audio and temporal facial results

Completed 2026-10-29. Run: `experiments/runs/representations_20261029T133130401813Z`.
See the [frozen protocol](representation_study_protocol.md). All results below are
exploratory nested training CV, not a fresh test evaluation or clinical validation.

None of the primary candidates passed the predeclared improvement rule. The
existing reference is retained. This negative result applies to supplied DenseNet
features pooled to whole-session mean/SD with fitting-fold PCA, and engineered
facial temporal summaries. It does not rule out speech-specific pretrained models,
text representations, or sequence models; none of those was tested here.

Same 161 training participants, 37 positives, released labels, two quarantined
timing anomalies, folds and classifiers as the preceding segmentation study.
All six original-feature controls reproduced all 2,898 prior predictions, including
decision scores to 1e-8. Differences therefore reflect feature conditions within
this protocol rather than a favorable change of cohort or folds.

| Condition | F1 mean +/- SD | Precision | Recall | Balanced accuracy | Mean fold AUC |
|---|---:|---:|---:|---:|---:|
| Original audio / logistic | .326 +/- .008 | .246 | .486 | .520 | .585 |
| Original facial / logistic | .367 +/- .016 | .281 | .532 | .561 | .550 |
| Original concatenation / logistic | .356 +/- .017 | .289 | .468 | .562 | .576 |
| Temporal facial / logistic | .325 +/- .060 | .264 | .423 | .533 | .577 |
| DenseNet audio / logistic | .303 +/- .036 | .224 | .468 | .494 | .550 |
| DenseNet + original facial / logistic | .344 +/- .026 | .276 | .459 | .547 | .562 |
| DenseNet + temporal facial / logistic | .355 +/- .058 | .290 | .459 | .560 | .586 |
| Original audio / linear SVM | .362 +/- .019 | .271 | .550 | .554 | .599 |
| Original facial / linear SVM | .351 +/- .005 | .283 | .459 | .556 | .571 |
| Original concatenation / linear SVM | .371 +/- .012 | .321 | .441 | .581 | .569 |
| Temporal facial / linear SVM | .323 +/- .051 | .300 | .351 | .555 | .554 |
| DenseNet audio / linear SVM | .312 +/- .015 | .242 | .441 | .515 | .550 |
| DenseNet + original facial / linear SVM | .275 +/- .066 | .256 | .297 | .522 | .548 |
| DenseNet + temporal facial / linear SVM | .216 +/- .086 | .218 | .216 | .497 | .563 |
| Always negative | .000 +/- .000 | .000 | .000 | .500 | .500 |
| Always positive | .374 +/- .000 | .230 | 1.000 | .500 | .500 |

SD is population SD across three overlapping repeats, not a confidence interval.
Specificity, accuracy, confusion counts, subgroup metrics and all paired differences
are available in the local CSV artifacts. No model exceeds the always-positive
reference's mean F1; that reference has zero specificity and is not useful screening.

Mean F1 changes in primary comparisons: DenseNet audio vs original audio -2.3
points for logistic and -5.0 for linear SVM; temporal facial vs original facial
-4.2 and -2.7; DenseNet+temporal facial vs original concatenation -0.2 and -15.5.
The small logistic AUC increases from temporal facial features did not produce
a corresponding F1/recall improvement. No post-result threshold search was added.

Extraction produced 3,840 DenseNet and 490 facial columns per participant. Existing
facial mean/SD were independently checked against the original feature table to
1e-8 tolerance. Temporal changes do not bridge rejected tracking frames, nonfinite
values or gaps longer than 0.1 seconds. These are variation summaries, not a full
sequence model. Supplied DenseNet source weights and detailed temporal alignment
were not independently reproduced. PCA was fitted only within each fitting fold.

All 102 tests pass. All 210 outer pipelines reproduce predictions after loading.
An independent check verified 48 repeat metric rows, 240 fold AUC rows, 7,728
predictions, inner/outer isolation, recorded selections and provenance. No previous
base tables, labels or development/test predictions were changed. Raw data,
participant records and checkpoints remain ignored locally. Stage 10 remains pending.

```powershell
.venv\Scripts\python.exe -m src.dataset.representations
.venv\Scripts\python.exe -m src.experiments.representation_study
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/representations-check --tb=short
```

The extraction requires the preceding cohort/audit artifacts. The next distinct
representation experiment should use a speech-trained encoder rather than treating
these supplied image-network spectrogram features as equivalent to Wav2Vec2.


---

## Record: research_progress_20261102.md

# Current research reference and completed follow-ups

Verified 2026-11-02. **The strongest observed research reference is screened
MPNet text + standardized whitened PCA32 + linear SVM, with 58.1% mean positive-
class F1 and 80.4% mean fold AUC.** It improves on the previous MiniLM text SVM
by 5.23 F1 points in the matched 161-person training comparison, with gains in
all three repeats. It passes the predeclared exploratory improvement screen.

| Matched training-CV reference | F1 | Precision | Recall | Specificity | Accuracy | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original audio + face SVM | 37.1% | 32.1% | 44.1% | 72.0% | 65.6% | 58.1% | 56.9% |
| Previous MiniLM text SVM | 52.8% | 43.9% | 66.7% | 74.2% | 72.5% | 70.4% | 79.4% |
| Current MPNet text SVM | 58.1% | 47.8% | 73.9% | 76.1% | 75.6% | 75.0% | 80.4% |

Four additional studies were completed, retaining all unsuccessful procedures:

1. [Text refinement and late fusion](text_refinement_results.md): 210 verified
   bundles, 7,728 predictions. Retaining all embedding dimensions, variability
   summaries, threshold search and late fusion failed to improve the old baseline.
2. [Screened lexical features](lexical_results.md): 165 verified bundles,
   6,279 predictions. Word/phrase features and semantic combinations also failed.
3. [Frozen MPNet](mpnet_results.md): 60 verified pipelines, 2,898 predictions.
   SVM improved; logistic did not clear the three-point improvement threshold.
4. [MPNet compression follow-up](mpnet_rank_results.md): 165 verified bundles,
   6,279 predictions. Broader rank/threshold searches did not beat the original
   MPNet recipe. The simpler reference is retained.

Across these studies, 600 saved model bundles and 23,184 predictions were checked.
All 119 tests pass. Bundles can share fitted parameters across cutoff conditions;
this is not a claim of 600 distinct training fits. MPNet cache hashes and encoder
replay passed; the replay difference was zero. No labels changed, participant
data uploaded, new development/test scores generated, or model deployed.

## Reusable full-training checkpoint

Built under the separate [refit protocol](research_reference_protocol.md):
`models/research_reference_20261102T172007773516Z/model.joblib`.
The folder is local and ignored by Git. It contains the ordered 768-feature
schema, pinned encoder revision/hashes, fitting IDs, input/source/model hashes,
selection audit, environment and independent verification record.

The refit uses all 161 training participants. Its three-fold training-only
selection chose C=0.1 and balanced class weights, with PCA32 and zero-margin
cutoff. Serialization, input schema, fitted scaling statistics, selection/fold
records and hashes were independently checked. This refit is a convenient
research artifact; it does not supply a new generalization estimate. The 58.1%
figure belongs to the preceding nested-CV recipe, not a test of this one fitted
checkpoint. Input must be screened MPNet mean embeddings using the recorded
encoder/pooling contract, not raw text or MiniLM embeddings.

```powershell
.venv\Scripts\python.exe -m src.experiments.refit_reference
.venv\Scripts\python.exe -m src.experiments.verify_reference models/research_reference_20261102T172007773516Z
```

## Remaining limits

The 65%-70% F1 aspiration is still 6.9-11.9 percentage points above the current
mean CV result. MPNet SVM F1 is 60.9% in WoZ interviews but only 38.9% in the small
AI subgroup (18 people, five positives). The historical test partition is all AI;
pooled training improvements cannot be assumed to transfer. Historical Stage 9
test scores remain separate and unchanged. Reliability-aware fusion has not yet
demonstrated its desired advantage, and the strongest reference is text-only.

The provisional label definition and transcript speaker/disclosure limitations
remain unresolved. The provider inquiry is drafted but unsent. Error diagnostics
did not show false positives concentrated in the retained label-disagreement
group; that observation neither resolves nor dismisses the label issue.

These repeated training studies establish the best observed recipe among the
tested conditions, not the best possible model. Further adaptive searching can
overfit this evaluation. A credible superiority claim requires fresh independent
validation; these numbers do not establish competitiveness with a paper using a
different cohort, label definition and experiment setup. Original Stage 10 remains
pending.


---

## Record: research_summary.md

# Research summary — 2026-11-04

## Objective and scope

Investigate Reliability-Aware Sparse Adaptive Fusion (R-SAF) for depression-label
classification from E-DAIC acoustic and facial behavioral features, with later
text-based comparisons. R-SAF is a provisional project name, not established
algorithmic novelty or a clinical diagnostic system. OpenFace action units are
facial behavioral measurements, not validated micro-expression detection.

## Process

1. Inventory archives, official partitions, supplied targets and feature schemas.
2. Preserve raw data; audit missing measurements, timing and target discrepancies.
3. Fit preprocessing within training folds; keep each participant in one partition.
4. Establish audio, facial and concatenated classical baselines.
5. Compare equal-weight, adaptive and quality-conditioned compact neural fusion.
6. Compare all descriptors, Binary HHO selection and L1 selection.
7. Explain models with Integrated Gradients and test attribution faithfulness.
8. Freeze the original model roster and evaluate official development/test sets.
9. Conduct separately documented exploratory improvements using training data only:
   audio representations, screened text, MPNet, neural pooling, quality fusion,
   thresholds, topic summaries and estimated interviewer-prompt exclusion.

The original data has 163/56/56 train/development/test participants. Later matched
studies retain 161 training participants, including 37 released positives. Five-fold
outer/three-fold inner CV is repeated with seeds42/137/2026. The revised folds also
balance interview domain and label. Settings, preprocessing and cutoffs use inner
training data. Repeated adaptive research on the same cohort still limits claims.

## Methods

- Acoustic eGeMAPS descriptors and OpenFace pose/gaze/action-unit measurements.
- Training-fitted imputation, scaling, constant/duplicate handling.
- Logistic regression, linear/RBF SVM and compact neural modality encoders.
- Adaptive gating with and without measured quality inputs; explicit availability.
- Binary HHO: local sigmoid-based binary adaptation, eight hawks/eight updates;
  fitness 0.95*(1-CV F1)+0.05*(selected/144), compared with L1 selection.
- Integrated Gradients, global and participant-level attribution, modality
  contributions, completeness checks, perturbation and ranking stability.
- Frozen MiniLM/MPNet text encoders. Strong reference: screened MPNet utterance
  mean, training-fitted standardized whitened PCA32, linear SVM.
- Later fusion: nested cross-fitted stacking with observed quality interactions;
  conservative auxiliary residuals with text fallback; missing-input tests.
- Inner-selected precision cutoffs, with a 65% recall floor and limited F1 loss.
- Fixed semantic-topic summaries and provisional transcript-context speaker filters.

## Results: separate evaluation settings

Original official-test results (56 people, mean F1 across three seeds):

| Family | Test F1 |
|---|---:|
| Audio logistic | 56.3% |
| Facial logistic | 44.8% |
| Concatenation logistic | 48.5% |
| All-feature equal-weight fusion | 52.4% |
| All-feature adaptive fusion | 53.8% |
| All-feature quality-aware fusion | 53.8% |
| HHO adaptive/quality fusion | 47.4% |
| L1 adaptive/quality fusion | 46.8% |

Source: `stage9_results.md`. No central fusion/HHO improvement was demonstrated.
Explanation tests nevertheless verified numerical completeness for 1,008 rows;
top-ranked feature perturbations exceeded random perturbations in all 54 aggregate
model/fraction comparisons. Rankings were unstable; these are not clinical biomarkers.

Revised training-CV comparison (161 people; not official-test performance):

| Procedure | Precision | Recall | F1 | Mean fold AUC |
|---|---:|---:|---:|---:|
| Retained MPNet SVM | 47.3% | 73.9% | 57.5% | 0.805 |
| Inner-selected text cutoff | 49.9% | 72.1% | 58.5% | 0.805 |
| Ordinary aligned fusion | 39.3% | 81.1% | 52.9% | 0.782 |
| Quality-aware aligned fusion | 39.5% | 81.1% | 53.1% | 0.770 |
| Conservative quality residual | 49.4% | 68.5% | 57.1% | 0.802 |
| Topic-balanced policy | 44.4% | 69.4% | 53.9% | 0.782 |
| Concatenated-topic policy | 46.2% | 71.2% | 55.9% | 0.794 |
| Regularized attention pooling | 37.6% | 64.0% | 47.2% | 0.749 |
| Context-supported prompt exclusion | 44.7% | 75.7% | 56.2% | 0.804 |
| Broader prompt exclusion | 44.3% | 73.0% | 55.1% | 0.804 |

Latest speaker-filter run: `speaker_filter_v4_20261104T172832492344Z`.
The filters removed 625/629 of 14,724 screened utterances, respectively, without
dropping participants. Neither improved the original reference. Verification
restored all 90 models, checked 2,898 predictions and replayed all 1,080 inner
candidate score vectors. The earlier partial run has no final manifest and is
excluded. Speaker-role estimates remain provisional; this result does not establish
that interviewer contamination is harmless or that either filter is accurate.

Sources: `pipeline_v2_results.md` and `bounded_v3_results.md`. The threshold policy
has 76.2% accuracy and is a modest candidate improvement, below the predeclared
promotion criteria. Its conditional F1-difference interval includes zero. Original
reference accuracy is 74.7%. Historical different-fold point estimates must remain
separate. No fresh independent holdout remains.

## Difference from the benchmark paper

Benchmark: Khan, Khan and Aljubayri, *A Comprehensive Framework for Multi-Modal
Depression Detection: Integrating Adaptive Fusion, Fairness Regularization, and
Explainable AI*, Mathematics 14(4),711 (2026),
https://www.mdpi.com/2227-7390/14/4/711 .

The paper describes dynamic gating, multi-head attention, fairness regularization
and explainability. Our project concentrates on explicit observed input-quality
signals, sparse feature selection via Binary HHO versus L1, and controlled
ablation/faithfulness tests on compact local models. The proposed contribution is
an empirical investigation of this combination, not invention of gating, HHO,
quality-aware fusion or explainability. No exhaustive literature claim or verified
first-of-its-kind claim is supported.

The brief attributes 91.4% F1 and 93.0% accuracy on DAIC-WOZ to the benchmark.
Those claims are not a matched comparison with our E-DAIC released-label setup.
The publisher page was unavailable during this summary's retrieval, so those
numbers remain attributed to the brief rather than newly verified here. A numeric
gap must not be described as a controlled head-to-head experimental difference.

## Limits and present interpretation

Twenty released labels disagree with the score>=10 definition; twelve are in the
retained training cohort. Labels remain unchanged. AI interviews are scarce in
training and constitute the historical test cohort. Speaker attribution is absent
from the supplied transcript schema: human/contextual review supports interviewer
prompts in some rows, but full-corpus roles remain estimated. Quality proxies are
not calibrated reliability probabilities. Reproducibility checks and tests support
implementation correctness, not clinical validity or guaranteed generalization.

The strongest current candidate uses text. The intended reliability-aware sparse
fusion advantage has not been demonstrated. This distinction belongs in the
project's conclusion even while further targeted improvements are investigated.

## Latest matched primary-branch audit (2026-11-04)

The [v6 comparison](branch_audit_v6_results.md) tests original, aligned and richer
audio/facial summaries with training-only selection. Facial F1 improves from
33.7% to about 37.5%; both revised facial branches pass their relative screens.
Audio revisions and fusion changes do not pass. Original concatenation scores
37.2% F1, versus 28.3% for matched quality neural fusion. This does not replace
the retained reference or demonstrate a quality-gating advantage. All 195 models,
7,245 predictions, 540 inner candidate vectors and 45 neural retrainings were
verified. These remain exploratory estimates on reused training participants.


---

## Record: segmentation_study_results.md

# Transcript coverage and facial fusion results

Completed 2026-10-29. Run: `experiments/runs/segmentation_20261029T125733760531Z`.
The [protocol](segmentation_study_protocol.md) was written before extraction and
fitting. This exploratory follow-up is separate from the pending Stage 10 audit.

## Outcome

Transcript-covered audio did not improve this pilot. Full-session audio should
remain the reference. Facial features added to full audio improved mean balanced
accuracy for both classifier families, but did not establish a strong model.
For linear SVM, specificity rose from 0.559 to 0.720 while recall fell from 0.550
to 0.441. Its F1 gain was small, varied by repeat, and its mean fold AUC decreased.
No model was promoted or evaluated again on development/test data.

These are nested training-CV results on 161 participants (37 released positives),
not percentages from a new independent test. Two previously identified timing
anomalies were excluded identically from every condition, before fitting. Both
had released negative labels. Prior experiments and all label values are intact.
The reduced cohort and different folds mean these numbers cannot be compared
directly with the preceding 163-person audio pilot or Stage 9 test results.

## All conditions

Values are means across three repeats. F1 SD is population SD across repeats,
not a confidence interval; repeats share participants. AUC is averaged across
outer folds, avoiding pooled margins from separately fitted SVMs.

| Features / model | F1 +/- SD | Precision | Recall | Specificity | Accuracy | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full audio / logistic | .326 +/- .008 | .246 | .486 | .554 | .538 | .520 | .585 |
| Covered audio / logistic | .306 +/- .054 | .235 | .441 | .581 | .549 | .511 | .482 |
| Facial / logistic | .367 +/- .016 | .281 | .532 | .591 | .578 | .561 | .550 |
| Full audio + facial / logistic | .356 +/- .017 | .289 | .468 | .656 | .613 | .562 | .576 |
| Covered audio + facial / logistic | .320 +/- .012 | .261 | .414 | .651 | .596 | .532 | .540 |
| Full audio / linear SVM | .362 +/- .019 | .271 | .550 | .559 | .557 | .554 | .599 |
| Covered audio / linear SVM | .274 +/- .058 | .210 | .396 | .567 | .528 | .482 | .489 |
| Facial / linear SVM | .351 +/- .005 | .283 | .459 | .653 | .609 | .556 | .571 |
| Full audio + facial / linear SVM | .371 +/- .012 | .321 | .441 | .720 | .656 | .581 | .569 |
| Covered audio + facial / linear SVM | .283 +/- .022 | .248 | .333 | .696 | .613 | .515 | .524 |
| Always negative | .000 +/- .000 | .000 | .000 | 1.000 | .770 | .500 | .500 |
| Always positive | .374 +/- .000 | .230 | 1.000 | .000 | .230 | .500 | .500 |

Every fitted condition's mean F1 is below the always-positive reference. That
reference has no specificity and is not useful screening, but it shows why the
largest F1 here cannot by itself support a strong performance claim. Conversely,
the 77% always-negative accuracy demonstrates why ordinary accuracy is misleading.

## Matched effects

- Transcript coverage reduced mean audio-only F1 by 2.0 percentage points for
  logistic and 8.8 for linear SVM. Mean fold AUC dropped by 10.3 and 11.0 points.
- With facial features included, transcript coverage reduced mean F1 by 3.6
  points for logistic and 8.8 for linear SVM. Each repeat's F1 difference was
  negative for both families.
- Adding facial features to full audio raised logistic F1 by 3.0 points and
  balanced accuracy by 4.2 points, with higher F1 in all three repeats. Mean
  recall fell 1.8 points and mean fold AUC fell 0.9 points.
- Adding facial features to full audio raised linear SVM F1 by 0.9 points and
  balanced accuracy by 2.7 points. F1 increased in two repeats and decreased in
  one; balanced accuracy increased in all three. Recall fell 10.8 points,
  specificity rose 16.1 points, and mean fold AUC fell 2.9 points.

These are descriptive paired differences, not significance tests or independent
confirmation. We did not select an additional threshold after observing them.

## Domain and alignment limitations

The retained cohort has 143 WoZ participants (32 positives) and 18 AI participants
(5 positives). All conditions' domain confusion metrics are saved locally. For
example, facial-only logistic has descriptive mean AI F1 0.700 versus WoZ F1
0.320, but the AI result rests on only five positive people, repeatedly evaluated.
It cannot justify AI-specific tuning or a reliable deployment claim. The official
test's different interview-condition composition remains relevant.

The selected frames follow actual transcript timestamps, including repeated frame
times. Transcript/audio-feature hashes match the earlier audit. No transcript
text or raw audio was exported. Full-session facial summaries retain the original
tracking-confidence filter; audio and facial time sequences are not synchronized.

Speaker identity and detailed timing have not been manually validated. We therefore
tested a *transcript-covered* condition, not participant-only speech. Its poor
result could reflect missing useful information, imperfect alignment or other
factors; this experiment does not determine the cause. The two quarantined source
anomalies remain unresolved. Released-label uncertainty also remains unresolved.

## Verification and next decision

All 97 tests pass. All 150 fitted outer pipelines reproduce predictions after
serialization. A separate verification recomputed all 36 repeat metric rows and
180 fold AUC rows from 5,796 saved predictions, checked fold isolation and every
recorded candidate choice, and verified input hashes. Previous baseline tables
match their existing hashes. All participant records and model files stay in
ignored local outputs. No new development/test predictions were created.

```powershell
.venv\Scripts\python.exe -m src.dataset.segmented_audio
.venv\Scripts\python.exe -m src.experiments.segmentation_study
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/segmentation-check --tb=short
```

The extractor requires the prior label/speech audit cache; stale archive stats or
audit source hashes are rejected. Per-participant extraction is cached separately.
The experiment saves its frozen input/source hashes, environment, inner selection
records, outer checkpoints, predictions, domain metrics and paired differences.

Retain full-session audio as the reference and preserve the modest facial-fusion
tradeoff as a finding. Do not promote transcript filtering based on this run.
Before another costly modeling round, obtain actual timing/speaker validation and
release-specific label clarification. A bounded facial temporal-feature study is
a possible separate experiment, not an improvement established by this one.


---

## Record: speaker_inference_v4.md

# Transcript-context speaker estimates — 2026-11-04

The user authorized inferring speaker roles from transcript context. This pilot
covers the 11 previously flagged utterances in six training participants. Original
listening annotations, labels, trained models and feature caches are preserved.

Source transcripts were checked against existing MPNet transcript hashes, ordered
by actual timestamps, and matched to the full target interval. Each case was read
with up to three preceding and following transcript rows. Source row indices and
timestamps are retained. No depression outcomes or classifier predictions were
used to assign roles. Raw text and participant-level estimates remain in ignored
local `data/processed/transcript_context_v4/` artifacts.

## Estimates

| Estimated role | Cases | Evidence |
|---|---:|---|
| Participant | 7 | Personal accounts, sentence continuations, or answers to surrounding questions |
| Participant, tentative | 1 | Standalone negative response among other brief answers; prompting question absent |
| Interviewer | 2 | Direct second-person questions followed by participant-style answers |
| Interviewer, moderate support | 1 | Second-person question fragment among similar prompts; incomplete context |

All three initial human interviewer flags are supported by the text-based
inference. They concern a family question, a dream-job question, and a question
fragment. This is converging evidence about these particular spans, not measured
speaker-recognition accuracy or an estimate of dataset-wide contamination.

Some neighboring transcript rows also appear to contain a prompt and an answer
within the same row. Row-level filtering therefore cannot guarantee speaker-clean
features. The presence of a question mark, or a second-person pronoun alone, is
not a sufficient universal exclusion rule. Participants can ask questions, quote
other people, or use generic second-person language in personal explanations.

Estimates are saved in `role_estimates.json` with per-case reasons, qualitative
evidence strength, source-row mapping and context hashes. Strength is not a
calibrated probability. The weak standalone response is retained provisionally.
The three estimated interviewer spans are marked as candidates for exclusion in
a separately declared sensitivity study, not automatically removed from historical
training features. A wider contextual labeling pass is needed before applying a
speaker filter throughout the cohort. These 11 pilot labels cannot be extrapolated
to all utterances or used to claim the speaker issue is solved.

Recreate the chronological context with:

```powershell
.venv\Scripts\python.exe -m src.dataset.transcript_context_v4
```

The estimates are assistant-authored contextual annotations; the extractor does
not itself perform speaker classification. They remain separate from human
`annotations.json` and any subsequent `annotations_context.json` review.


---

## Record: speech_encoder_results.md

# Frozen Wav2Vec2 pilot results

Results verified 2026-10-30. Run: `experiments/runs/speech_20261029T201655576434Z`.
The [protocol](speech_encoder_protocol.md) was fixed before extraction and fitting.
This is a separate follow-up to the unsuccessful supplied-DenseNet pilot.

## Decision

None of the six speech-feature candidates passed the predeclared improvement
criterion. Retain the existing reference. Frozen Wav2Vec2 with this sampling,
pooling, PCA and classifier recipe did not establish a stronger foundation for
reliability-aware fusion. No new reliability gate was claimed or promoted.

Audio-only logistic F1 rose from .326 to .338 (1.2 percentage points), but mean
fold AUC fell from .585 to .529. The gain is below the required three-point F1
increase and fails the no-AUC-loss rule. All speech-plus-facial conditions have
lower mean F1 than their respective original audio/facial controls.

This conclusion is specific to the frozen, sampled, pooled representation tested
here. It does not evaluate task-specific fine-tuning, a learned sequence model,
transcript semantics, or the exact competing paper architecture. It does not
establish that Wav2Vec2 generally cannot help depression prediction.

## Complete matched comparison

All results are nested training CV on the same 161 people (37 released positives)
and identical folds as the preceding pilots. Labels and two timing quarantines
are unchanged. F1 SD is across three overlapping repeats, not a confidence
interval. AUC is averaged by outer fold, never pooled across SVM margin scales.

| Features / classifier | F1 +/- SD | Precision | Recall | Specificity | Accuracy | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original audio / logistic | .326 +/- .008 | .246 | .486 | .554 | .538 | .520 | .585 |
| Original audio + face / logistic | .356 +/- .017 | .289 | .468 | .656 | .613 | .562 | .576 |
| Speech / logistic | .338 +/- .025 | .259 | .486 | .583 | .561 | .535 | .529 |
| Speech + original face / logistic | .337 +/- .014 | .266 | .459 | .621 | .584 | .540 | .528 |
| Speech + temporal face / logistic | .320 +/- .012 | .265 | .405 | .664 | .605 | .535 | .557 |
| Original audio / linear SVM | .362 +/- .019 | .271 | .550 | .559 | .557 | .554 | .599 |
| Original audio + face / linear SVM | .371 +/- .012 | .321 | .441 | .720 | .656 | .581 | .569 |
| Speech / linear SVM | .300 +/- .052 | .257 | .360 | .694 | .617 | .527 | .542 |
| Speech + original face / linear SVM | .325 +/- .034 | .287 | .378 | .715 | .638 | .547 | .533 |
| Speech + temporal face / linear SVM | .301 +/- .061 | .305 | .297 | .796 | .681 | .546 | .559 |
| Always negative | .000 +/- .000 | .000 | .000 | 1.000 | .770 | .500 | .500 |
| Always positive | .374 +/- .000 | .230 | 1.000 | .000 | .230 | .500 | .500 |

The speech/temporal-face linear model's 68.1% accuracy accompanies only 29.7%
recall. This is not an improvement hidden by F1; it misses more positive cases.
All fitted conditions remain below the always-positive reference in mean F1,
although that reference has no specificity and is not useful screening.
No thresholds or additional classifier settings were searched after these results.
These CV numbers must not be compared directly with the historical Stage 9 test F1.

## Extraction and verification

The pinned `facebook/wav2vec2-base-960h` encoder ran locally on the RTX 4050 using
float32. No recordings or transcripts were uploaded, and no ASR transcripts were
generated. The model's last hidden states were mean-pooled within windows, then
summarized across windows into 1,536 features per person. Training-only PCA reduced
the speech branch to 32 dimensions; facial preprocessing remained unchanged.

There were 10,282 eight-second windows. Fixed sampling covers a median 55.2% of
each recording (range 26.0%-99.9%), not complete interview sequences or verified
participant-only speech. Recorded extraction time totals 1,011 seconds (16.9
minutes), including per-participant archive reading and inference, excluding model
download/loading, setup and later classification. Peak allocated CUDA memory was
0.780 GiB. These timings are local observations, not hardware superiority claims.

The ASR checkpoint lacks a training-only masking embedding named
`masked_spec_embed`; the loader reports its initialization. All other required
encoder weights loaded. Augmentation is explicitly disabled and the encoder is
in eval mode, so that masking vector does not enter this feature-extraction path.
Model revision, weight hashes, dependency versions and loading diagnostics are
recorded in the local extraction manifest.

Independent extraction verification checked all window-file hashes, window starts,
pooled features and model hashes for 161 participants. Reloading the encoder and
replaying four windows reproduced their embeddings exactly (maximum difference 0).

All 105 tests pass. All 150 outer pipelines reproduce predictions after reload.
Independent checks recomputed all 36 repeat metric rows and 180 fold AUC rows from
5,796 predictions, verified fold isolation and recorded candidate selections, and
checked input provenance. All 1,932 original-control predictions match the previous
pilot, including scores to 1e-8.

No development/test predictions, labels, original feature tables or deployed model
were changed. Participant records and downloaded weights remain ignored locally.
Stage 10 remains pending. The inference dependency is declared in the `speech`
optional dependency group; the run also saves the full installed environment.

```powershell
.venv\Scripts\python.exe -m src.dataset.speech_embeddings
.venv\Scripts\python.exe -m src.experiments.speech_study
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/speech-check --tb=short
```

Extraction expects the pinned local model files in
`models/pretrained/wav2vec2-base-960h` and the preceding cohort artifacts. The
classifier experiment expects the already verified temporal facial table as well.
No network access is needed once weights and source data are present.

The next materially different research direction would be transcript semantics or
learning over sequences, with the label-definition and independent-evaluation
questions resolved explicitly. This completed pilot supplies no evidence to claim
progress toward the 70% F1 milestone or the paper's 90%+ headline performance.


---

## Record: stage8_results.md

# Stage 8 completion: 2026-10-28

Completed run: `experiments/runs/stage8_20261028T180217572773Z`.
Source checkpoints: `experiments/runs/stage7_20261028T142146453803Z`.
All 18 matched Stage 7 models and all 56 development participants were included.
The original checkpoints and their predictions were not modified.

## Findings

- Top-ranked descriptor replacement produced greater mean absolute logit change
  than random replacement in all 54 model/fraction pairs (10%,25%,50%). This is
  aggregate evidence for reference-specific sensitivity, not success on every
  participant or a statistical significance claim.
- Example: all-feature quality model, primary seed 42, replacing the top 25% gave
  mean absolute logit change 0.4036, random 0.1582, bottom 0.0161.
- All 1,008 explanation rows passed the fixed completeness diagnostic. Maximum
  absolute residual was 0.000594 logits. The largest model-level mean L1 difference
  between 256-step and 512-step attribution vectors was 0.002445.
- All-feature global ranking correlations across seeds were about 0.12-0.29;
  top-10 Jaccard overlap was 0.11-0.33. HHO overlap was 0.05-0.11, with negative
  union-ranking correlations because masks and importance changed across seeds.
- L1 top-10 overlap was 0.82-1.00, but each model has only 11 selected features;
  this is not evidence of stable ordering. Its ranking correlations ranged from
  -0.14 to 0.35. Matched quality/adaptive rankings were nearly identical.

## Verification and limitations

78 tests passed without warnings. Tests cover exact linear and quadratic IG,
signed contributions, completeness, zero reference, known influential features,
random repeatability, unavailable-modality attribution, ranking alignment and
provenance mismatch rejection. All 18 restored predictions matched Stage 7.
Repeating the full explanation calculation produced 93 byte-identical output
artifacts (excluding the run metadata whose source hashes changed after an index
tie-handling correction). Details are in the run's verification.json.

An initial test and attempted run failed because reversed NumPy index arrays had
negative strides unsupported by PyTorch. Copying indices fixed that failure.
Bottom-ranking ties now explicitly preserve original feature order. The failed
run remains an incomplete ignored artifact, not a completed result.

Feature attributions are conditional on fixed participant quality. Separate
quality-mean replacement output changes are sensitivity checks, not additive
quality explanations. The reference is the saved training mean, not an actual
person. Perturbing correlated measurements may create unrealistic inputs.
The fixed midpoint integration check is numerical, not proof of meaningful
explanations. Gate weights are not presented as explanations. No feature names
are translated into clinical symptoms, and there is no causal claim.

The test partition remains unevaluated. Label semantics remain provisional.
No model/seed/settings were selected using explanation outcomes. Stage 9
(ablations and final evaluation) and Stage 10 (final reproducibility audit)
require separate continuation.


---

## Record: stage9_results.md

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


---

## Record: text_refinement_results.md

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


---

## Record: text_temporal_results.md

# Text, temporal, and reliability follow-up

Completion review: 2026-10-31. The [protocol](text_temporal_protocol.md) was fixed
before fitting, including the documented facial-timing amendment. These are
exploratory training cross-validation experiments, not new test-set results.


**Outcome:** screened transcript meaning gives a substantial improvement in
matched training CV. Text-only models are the strongest candidates in this
follow-up (52.5%-52.8% F1). The temporal and quality-aware variants do not pass
their predeclared improvement targets. Retain the historical test results as
historical references; do not describe the new candidates as test-validated.

## Inputs and scope

All comparisons use the same 161 official training participants, 37 released
positives, and folds from the previous pilots: five outer folds and seeds
42/137/2026. The two previous whole-participant timing quarantines remain intact.
Targets are unchanged provisional released PHQ_Binary, not a confirmed score
threshold or clinical diagnosis. No development/test predictions were generated.

The frozen local text encoder is `sentence-transformers/all-MiniLM-L6-v2`,
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its
[model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
describes sentence embeddings; it is not a depression-trained classifier.
No participant recordings or transcripts were uploaded.

The audit found 14,961 valid nonempty utterances and 84 direct matches to the
conservative disclosure screen. Removing matches and their immediate neighbors
excluded 237 utterances (1.58%), leaving 14,724. All supplied confidence values
were present and within [0,1]. None of the transcript files has a speaker column.
The heuristic is not a manual leakage or speaker audit; its false positives and
missed disclosures remain limitations. Ordinary symptom discussions remain inputs.

Text-only and text/audio/face classifiers use separate fold-fitted text PCA and
the unchanged acoustic/facial preprocessing. The sequence models use 32
chronological bins of text embeddings, 23 acoustic descriptors and 49 facial
descriptors. Two additional facial timelines exceed audio duration by 5.66 and
110.953 seconds. Their facial sequence branches are masked; the participants
remain included. No timestamp correction was invented. All sequence comparisons
share these masks; original static facial controls remain unchanged.

The pointwise model has 8,324 trainable parameters and the temporal model 9,860.
Consequently, their comparison changes temporal mixing and capacity together;
it does not uniquely isolate ordering from parameter count. Ordinary and
quality-aware temporal models have identical architecture, initialization seeds,
augmentation schedules, and training settings. Quality changes their modality
weights. The blind ablation removes quality weighting from the quality-trained
model at inference.

## Text comparison

Completed run: `experiments/runs/text_20261030T230941483092Z`.
All four text candidates pass the predeclared exploratory screen against the
matched original audio/face classifier: at least three F1 points improvement,
no balanced-accuracy or AUC loss, recall decrease no greater than five points,
and positive F1 differences in at least two repeats. This supports text as a
stronger research input, not a new independently confirmed test result.

Text-only linear SVM improves F1 from 37.1% to 52.8% (+15.7 percentage points),
balanced accuracy from 58.1% to 70.4%, and mean fold AUC from 56.9% to 79.4%.
Adding the original audio/facial summaries to text reduces F1 in both families.
The useful new signal is therefore text; this experiment does not show that
three modalities are better than text alone. Text-only logistic has slightly
higher recall and essentially the same balanced accuracy as text-only SVM.
The small F1 difference between them is not evidence of a decisive winner.

There is also an important domain limitation. Text-only SVM averages 55.1% F1
on the 143 WoZ participants but 37.0% on the 18 AI-interview participants;
text logistic averages 54.3% and 38.7%, respectively. The AI subgroup has only
five positives, so these estimates are unstable. The historical test partition
contains AI interviews only. The pooled training improvement must not be assumed
to transfer to that domain. Subgroup metrics are descriptive, not a tuning target.

All entries below are percentages. F1 SD is in percentage points.

| Features / classifier | F1 +/- SD | Precision | Recall | Specificity | Accuracy | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original audio / logistic | 32.6 +/- 0.8 | 24.6 | 48.6 | 55.4 | 53.8 | 52.0 | 58.5 |
| Original audio / SVM | 36.2 +/- 1.9 | 27.1 | 55.0 | 55.9 | 55.7 | 55.4 | 59.9 |
| Original face / logistic | 36.7 +/- 1.6 | 28.1 | 53.2 | 59.1 | 57.8 | 56.1 | 55.0 |
| Original face / SVM | 35.1 +/- 0.5 | 28.3 | 45.9 | 65.3 | 60.9 | 55.6 | 57.1 |
| Original audio + face / logistic | 35.6 +/- 1.7 | 28.9 | 46.8 | 65.6 | 61.3 | 56.2 | 57.6 |
| Original audio + face / SVM | 37.1 +/- 1.2 | 32.1 | 44.1 | 72.0 | 65.6 | 58.1 | 56.9 |
| Text / logistic | 52.5 +/- 2.5 | 42.1 | 70.3 | 70.7 | 70.6 | 70.5 | 78.2 |
| Text / SVM | 52.8 +/- 3.4 | 43.9 | 66.7 | 74.2 | 72.5 | 70.4 | 79.4 |
| Text + audio + face / logistic | 46.9 +/- 4.0 | 38.7 | 59.5 | 71.8 | 68.9 | 65.6 | 69.3 |
| Text + audio + face / SVM | 42.9 +/- 4.4 | 41.0 | 45.0 | 80.6 | 72.5 | 62.8 | 69.2 |
| Always negative | 0.0 +/- 0.0 | 0.0 | 0.0 | 100.0 | 77.0 | 50.0 | 50.0 |
| Always positive | 37.4 +/- 0.0 | 23.0 | 100.0 | 0.0 | 23.0 | 50.0 | 50.0 |

Independent verification restored all 150 classical pipelines, recomputed all
36 repeat metric rows and 5,796 prediction records, checked fitting/held-out and
inner folds, and reproduced 1,932 prior audio and audio/face control predictions
including decision margins. Every input cache hash passed; replaying the first
participant's local text encoder gave maximum mean-embedding difference zero.

## Temporal and reliability comparison

Completed run: `experiments/runs/temporal_20261030T230941483092Z`.
The temporal model raises clean F1 from 35.1% to 37.7%, a 2.6-point gain versus
the matched pointwise control. It fails the predeclared three-point screen and
remains substantially below the text-only classical models. This bounded model
does not establish an improvement from temporal processing.

Quality weighting reduces clean F1 by 0.87 points versus ordinary temporal
fusion, within the allowed two-point loss, but improves mean stressed F1 by only
0.93 points, below the required five-point gain. The joint reliability target
therefore fails. Against the same quality-trained model with quality blinded,
effects are also small; the observed stress benefit is not sufficient evidence
of a strong reliability contribution. Do not promote this quality model over
the text candidate or claim the paper's reliability objective has been achieved.

For missing text/audio/face respectively, quality fusion differs from the
corresponding available-input training control by +3.18/-1.51/-3.52 F1 points.
Missing-text performance is poor for both models, reinforcing the importance
of text in this cohort. These are descriptive comparisons, not a predeclared
equivalence or noninferiority test.

Every sequence condition is below, including the quality-blind ablation and
models trained with one modality absent. All figures are percentages; SD is
across repeats in percentage points. `mean_plain` is the pointwise/orderless
control, not the original hand-crafted summary baseline.

| Model | Input condition | F1 +/- SD | Precision | Recall | Specificity | Accuracy | Balanced accuracy | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| available_without_audio | missing_audio | 39.7 +/- 2.9 | 31.8 | 53.2 | 66.1 | 63.1 | 59.6 | 69.3 |
| available_without_face | missing_face | 42.8 +/- 4.1 | 33.3 | 60.4 | 64.2 | 63.4 | 62.3 | 71.2 |
| available_without_text | missing_text | 26.5 +/- 2.4 | 20.0 | 39.6 | 53.0 | 49.9 | 46.3 | 46.2 |
| mean_plain | audio_noise | 34.9 +/- 3.4 | 28.3 | 50.5 | 60.5 | 58.2 | 55.5 | 67.0 |
| mean_plain | clean | 35.1 +/- 2.8 | 28.1 | 50.5 | 60.5 | 58.2 | 55.5 | 66.4 |
| mean_plain | face_loss | 34.7 +/- 2.6 | 28.0 | 49.5 | 60.8 | 58.2 | 55.2 | 65.7 |
| mean_plain | missing_audio | 35.0 +/- 4.8 | 29.0 | 48.6 | 64.0 | 60.5 | 56.3 | 67.4 |
| mean_plain | missing_face | 38.3 +/- 1.0 | 31.3 | 53.2 | 63.2 | 60.9 | 58.2 | 67.8 |
| mean_plain | missing_text | 31.3 +/- 2.5 | 22.8 | 53.2 | 45.7 | 47.4 | 49.4 | 47.2 |
| temporal_plain | audio_noise | 36.6 +/- 3.0 | 28.2 | 54.1 | 58.9 | 57.8 | 56.5 | 66.9 |
| temporal_plain | clean | 37.7 +/- 3.0 | 29.1 | 55.0 | 59.9 | 58.8 | 57.5 | 66.5 |
| temporal_plain | face_loss | 38.5 +/- 4.2 | 30.8 | 52.3 | 65.3 | 62.3 | 58.8 | 68.3 |
| temporal_plain | missing_audio | 36.5 +/- 3.3 | 28.0 | 54.1 | 58.6 | 57.6 | 56.3 | 67.3 |
| temporal_plain | missing_face | 39.6 +/- 5.5 | 31.8 | 54.1 | 65.9 | 63.1 | 60.0 | 69.9 |
| temporal_plain | missing_text | 27.2 +/- 3.6 | 20.4 | 41.4 | 52.4 | 49.9 | 46.9 | 45.0 |
| temporal_quality | audio_noise | 36.9 +/- 3.0 | 28.8 | 53.2 | 60.8 | 59.0 | 57.0 | 68.4 |
| temporal_quality | clean | 36.8 +/- 3.1 | 28.9 | 52.3 | 61.6 | 59.4 | 56.9 | 66.6 |
| temporal_quality | face_loss | 39.1 +/- 5.6 | 30.9 | 55.0 | 64.0 | 61.9 | 59.5 | 69.5 |
| temporal_quality | missing_audio | 38.2 +/- 2.3 | 29.8 | 55.0 | 61.0 | 59.6 | 58.0 | 68.1 |
| temporal_quality | missing_face | 39.2 +/- 5.9 | 31.1 | 55.0 | 64.2 | 62.1 | 59.6 | 69.6 |
| temporal_quality | missing_text | 29.7 +/- 2.6 | 22.2 | 45.0 | 52.7 | 50.9 | 48.9 | 45.9 |
| temporal_quality_blind | audio_noise | 36.5 +/- 3.9 | 28.4 | 53.2 | 60.2 | 58.6 | 56.7 | 67.6 |
| temporal_quality_blind | clean | 37.3 +/- 3.1 | 29.2 | 53.2 | 61.6 | 59.6 | 57.4 | 67.1 |
| temporal_quality_blind | face_loss | 38.9 +/- 3.4 | 31.5 | 52.3 | 66.1 | 62.9 | 59.2 | 68.8 |
| temporal_quality_blind | missing_audio | 37.7 +/- 3.1 | 29.5 | 54.1 | 61.3 | 59.6 | 57.7 | 68.2 |
| temporal_quality_blind | missing_face | 39.1 +/- 6.2 | 30.9 | 55.0 | 64.0 | 61.9 | 59.5 | 69.8 |
| temporal_quality_blind | missing_text | 29.7 +/- 2.6 | 22.2 | 45.0 | 52.7 | 50.9 | 48.9 | 45.8 |

Independent verification restored all 90 neural checkpoints, checked their gate
weights and missing-modality masks, and recomputed 81 repeat metric rows and
13,041 predictions. Fold IDs, input/source hashes and checkpoint hashes passed.
Together with the classical study, 240 saved classifiers and 18,837 predictions
were verified. The complete test suite passes 111 tests (one cosmetic pandas
regex capture-group warning). No test or development outcome selected settings.

## Comparison with past studies

These rows use the same 161-participant cohort and training folds. Classifiers
are identified because representation and classifier choices both matter.

| Study / condition | Training-CV F1 | Mean fold AUC |
|---|---:|---:|
| Original audio + face / SVM | 37.1% | 56.9% |
| Supplied DenseNet + temporal facial summaries / logistic | 35.5% | 58.6% |
| Frozen Wav2Vec2 audio / logistic | 33.8% | 52.9% |
| New text / SVM | 52.8% | 79.4% |
| New text / logistic | 52.5% | 78.2% |
| New text + original audio + face / logistic | 46.9% | 69.3% |
| New temporal three-modality fusion | 37.7% | 66.5% |
| New quality-aware temporal fusion | 36.8% | 66.6% |

The original Stage 9 **test-set** F1 values remain 56.3% for audio logistic and
53.8% for adaptive/quality fusion. They were measured on a different partition
and cannot be ranked against the new training-CV figures. No new test score is
available. Our aspirational 65%-70% F1 milestone remains unmet in this follow-up;
the best new mean CV F1 is 12.2-17.2 points below that numerical target.

The practical next research reference is the screened text baseline, retaining
both text classifiers because their small difference is inconclusive. The saved
fold models support further research; no new final-test evaluation, deployment,
automatic production-model replacement, or fresh external validation occurred.
Do not keep adding modalities merely to call the result multimodal: the added
streams need to beat this stronger text reference under matched evaluation.

## Interpretation limits

Noise is injected into standardized acoustic features, and facial loss removes
bins. The quality signal for injected corruption uses known corruption severity.
This controlled experiment does not validate a real-world noise detector, raw
waveform robustness, or clinical reliability. Missing-modality comparisons also
include models trained from the outset with the same modality absent.

F1 refers to the released positive class. AUC is mean outer-fold AUC, not a pooled
ranking of incompatible SVM margin scales. Reported SD across three overlapping
repeats is not a confidence interval or independent replication. Model searches
across these and earlier pilots introduce selection uncertainty. Fresh external
validation and provider clarification remain necessary.

The [provider inquiry](label_provider_inquiry.md) is prepared but has not been
sent. The public manual check did not resolve the 20 label/score disagreements.
No label has been changed. Stage 10 remains pending.

## Reproduction

With the pinned local encoder and existing processed baseline inputs:

```powershell
.venv\Scripts\python.exe -m src.dataset.text_temporal
.venv\Scripts\python.exe -m src.experiments.verify_text_inputs
.venv\Scripts\python.exe -m src.experiments.text_temporal_study text
.venv\Scripts\python.exe -m src.experiments.text_temporal_study temporal
.venv\Scripts\python.exe -m src.experiments.verify_text_temporal <text-run-directory>
.venv\Scripts\python.exe -m src.experiments.verify_text_temporal <temporal-run-directory>
.venv\Scripts\python.exe -m src.experiments.compare_text_temporal <text-run-directory> <temporal-run-directory>
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/text-temporal-final --tb=short
```

The initial fits were interrupted without final results. Their directories,
`text_20261030T111740755918Z` and `temporal_20261030T111740381725Z`, remain under
ignored experiment outputs. The restarted fits use identical frozen settings;
partial runs are not included in comparisons.


---

## Record: undersampling_results.md

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

---

## Matched waveform preprocessing — 2026-11-08

Run: `audio_preprocessing_20261108T154755587738Z`. [Frozen protocol](methods/audio_preprocessing.md).

**Decision: retain the original supplied acoustic features.** Neither offset removal nor gentle high-pass filtering passes the predeclared improvement screen. The best historical audio/facial control is unchanged.

The study processed 43.47 hours of recordings from the same 161 retained training participants (37 positive labels). Every waveform was evaluated untreated, with its constant offset removed, and with offset removal plus a second-order 50 Hz forward/backward Butterworth filter. Duration and pauses were preserved; there was no amplitude normalization, speaker inference, speech trimming or noise-estimation denoiser.

openSMILE Python 2.6.0 extracted eGeMAPSv02 low-level descriptors for every condition. Means and population standard deviations yielded 50 audio features. Original supplied features have 46 dimensions and come from a different extractor release; the re-extracted untreated condition is therefore the proper control for cleanup effects. Facial features were held fixed at the original 98 descriptors with the existing two timing quarantines.

Every model used fold-fitted median imputation, constant/duplicate removal, scaling and balanced logistic regression C=.1 at threshold zero. Five participant folds balanced by domain/label were repeated with seeds 42, 137 and 2026. No hyperparameter or threshold tuning, development/test evaluation, or text predictors were introduced. Audio-only original-feature scores use this fixed classifier, unlike the earlier independently selected audio-only branch.

All classification percentages below are means of repeat-level out-of-fold metrics. AUC/AP are means over 15 folds on a 0–1 scale.

| Condition | Accuracy | Balanced accuracy | Precision | Recall | F1 | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_negative | 77.02 | 50.00 | 0.00 | 0.00 | 0.00 | 0.500 | 0.230 |
| always_positive | 22.98 | 50.00 | 22.98 | 100.00 | 37.37 | 0.500 | 0.230 |
| dc_audio | 56.11 | 50.01 | 22.99 | 38.74 | 28.85 | 0.556 | 0.321 |
| dc_concat | 61.08 | 55.14 | 27.95 | 44.14 | 34.22 | 0.593 | 0.396 |
| face_only | 60.25 | 55.55 | 28.19 | 46.85 | 35.13 | 0.580 | 0.343 |
| highpass_audio | 56.52 | 51.55 | 24.33 | 42.34 | 30.89 | 0.566 | 0.328 |
| highpass_concat | 62.11 | 55.49 | 28.48 | 43.24 | 34.31 | 0.588 | 0.390 |
| original_audio | 57.14 | 52.58 | 25.42 | 44.14 | 32.24 | 0.571 | 0.335 |
| original_concat | 62.94 | 57.61 | 30.42 | 47.75 | 37.16 | 0.600 | 0.401 |
| raw_audio | 56.52 | 50.92 | 23.82 | 40.54 | 30.01 | 0.555 | 0.318 |
| raw_concat | 61.28 | 55.27 | 28.13 | 44.14 | 34.34 | 0.592 | 0.383 |

| Candidate | Matched control | F1 change (percentage points) | Positive repeats | Pass |
|---|---|---:|---:|---|
| dc_audio | raw_audio | -1.16 | 0/3 | False |
| dc_concat | raw_concat | -0.12 | 0/3 | False |
| highpass_audio | raw_audio | +0.88 | 1/3 | False |
| highpass_concat | raw_concat | -0.04 | 1/3 | False |
| raw_audio | original_audio | -2.23 | 0/3 | False |
| raw_concat | original_concat | -2.81 | 0/3 | False |

High-pass filtering increases combined-model accuracy from 61.28% to 62.11% relative to the newly extracted untreated control, but recall drops from 44.14% to 43.24% and F1 changes from 34.34% to 34.31%. This is not a demonstrated screening improvement. The original supplied-feature model remains stronger at 62.94% accuracy, 57.61% balanced accuracy and 37.16% F1.

The median absolute DC offset was already small (about 2.29e-5 in normalized sample units). High-pass processing retained a median 95.5% of original RMS. These measurements do not identify which energy was noise. Filtering can produce transient overshoot: the observed maximum absolute sample was 1.41 versus 1.0 untreated; values were not clipped or renormalized. The findings do not rule out all preprocessing methods or establish a dataset performance ceiling.

All 135 saved models were independently restored and retrained; all 5,313 predictions, folds, preprocessing statistics, input/source hashes, metrics and screens passed verification with zero restored-score error. The historical original-concatenation control was reproduced. All 112 tests passed, with one existing transcript-regex warning. The shared historical verifier also still reproduces the previous 195 models and 7,245 predictions.

The earlier run `audio_preprocessing_20261108T154545839405Z` is superseded by this fully verified run after correcting an index-handling bug in the verification comparison table. Every prediction in the replacement run is exactly identical. No tolerance was widened. Per-participant caches, environment details and saved models remain local and excluded from Git.

This is an exploratory reused-cohort negative result. Keep it in the research record; do not replace the existing audio/facial model with either cleanup variant. Stronger denoising and validated speaker isolation were not tested in this series.

---

## Final sparse and late-fusion comparison — 2026-11-08

Run: `sparse_fusion_20261108T160011416879Z`. Full methods, all results, selection logs and paper-ready interpretation are in the [paper-writer log](paper_logs/README.md). Sparse concatenation reached 37.63% F1 versus 37.16% for original concatenation, but accuracy fell from 62.94% to 59.63%. Weighted fusion reached 35.36% F1 and matched quality adjustment 35.45%. No replacement passed against the original control. Two ancillary comparisons against weaker controls passed; they do not justify promotion. All 180 policy bundles and 6,762 predictions were verified, including 405 inner candidate and 45 weight vectors. All 116 tests passed. The bounded series is finished; retain the original control and report the negative quality-fusion finding.

---

## Corrected voicing aggregation and common-scale fusion — 2026-11-08

Run: `corrected_fusion_20261108T180359546982Z`. [Frozen protocol](methods/corrected_fusion.md).

**Decision: no candidate passes; retain `original_concat`.** This run corrects two defects found by code review and is the valid test of those ideas. It is not a new search.

**Defect 1 (audio features).** `tables.aggregate` averaged openSMILE's unvoiced placeholders (0 for F0/jitter/shimmer/HNR/H1-H2/H1-A3, -201 for F1-F3 amplitude) with real values. For example, participant 302's F0 mean was 4.9 semitones over all frames versus 25.6 over voiced frames. The corrected summaries use voiced frames (F0 > 0) for the 13 `*_sma3nz` descriptors, treat -201 as missing and add `audio__voiced_fraction` (47 features; median voiced fraction 0.267). The same pass recomputed the legacy summaries and reproduced `features.parquet` exactly (maximum difference 0.0). The waveform re-extraction study (`audio_preprocessing`), aligned utterance audio (`audio_summary.summarize`, `contracts.interval_summary`) used the same all-frame averaging and were not rebuilt.

**Defect 2 (late fusion).** `fuse_scores` averaged `expit(margin)` from branches whose margin scales differed up to about 30x in the sparse series, so nominal weights were not applied weights. The corrected rule divides each margin by the RMS of that branch's inner out-of-fold margins before weighting. The historical unscaled rule remains the default so earlier runs still replay.

All branches use the fixed historical recipe (all features, balanced L2 logistic, C=.1). With fixed C, the held-out face/audio margin-scale ratio was only 1.0–2.4 (mean 1.6), so the scale defect mostly arose from per-branch C selection in the sparse series.

| Condition | Accuracy | Balanced accuracy | Precision | Recall | F1 | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_negative | 77.02 | 50.00 | 0.00 | 0.00 | 0.00 | 0.500 | 0.230 |
| always_positive | 22.98 | 50.00 | 22.98 | 100.00 | 37.37 | 0.500 | 0.230 |
| original_audio | 57.14 | 52.58 | 25.42 | 44.14 | 32.24 | 0.571 | 0.335 |
| voiced_audio | 59.42 | 54.06 | 26.69 | 44.14 | 33.23 | 0.576 | 0.329 |
| face | 60.25 | 55.55 | 28.19 | 46.85 | 35.13 | 0.580 | 0.343 |
| original_concat | 62.94 | 57.61 | 30.42 | 47.75 | 37.16 | 0.600 | 0.401 |
| voiced_concat | 63.56 | 57.38 | 30.51 | 45.95 | 36.66 | 0.625 | 0.431 |
| late_equal_raw | 60.46 | 53.47 | 26.48 | 40.54 | 32.00 | 0.595 | 0.400 |
| late_equal_scaled | 61.28 | 53.37 | 26.58 | 38.74 | 31.51 | 0.588 | 0.374 |
| late_weighted_scaled | 62.94 | 56.03 | 29.16 | 43.24 | 34.81 | 0.582 | 0.353 |
| late_quality_scaled | 62.11 | 54.54 | 27.77 | 40.54 | 32.96 | 0.588 | 0.357 |

| Candidate | Control | F1 change (points) | BA change | AUC change | Positive repeats | Pass |
|---|---|---:|---:|---:|---:|---|
| voiced_concat | original_concat | -0.50 | -0.23 | +2.46 | 1/3 | False |
| late_equal_scaled | original_concat | -5.65 | -4.24 | -1.21 | 0/3 | False |
| late_weighted_scaled | original_concat | -2.34 | -1.58 | -1.86 | 1/3 | False |
| late_quality_scaled | original_concat | -4.20 | -3.07 | -1.17 | 1/3 | False |
| voiced_audio | original_audio | +0.99 | +1.48 | +0.42 | 2/3 | False |
| late_equal_scaled | late_equal_raw | -0.49 | -0.09 | -0.73 | 1/3 | False |
| late_weighted_scaled | late_equal_scaled | +3.30 | +2.66 | -0.65 | 2/3 | False |
| late_quality_scaled | late_weighted_scaled | -1.85 | -1.49 | +0.68 | 0/3 | False |
| late_weighted_scaled | face | -0.32 | +0.48 | +0.12 | 1/3 | False |
| late_weighted_scaled | voiced_audio | +1.59 | +1.97 | +0.60 | 2/3 | False |

AUC and AP are means over 15 folds; AUC changes are in points (x100).

Interpretation. Voiced-only audio summaries improved ranking of the combined model (fold AUC 0.600 to 0.625, AP 0.401 to 0.431) but not F1 at the fixed threshold. They are within fold-to-fold noise, and AUC is not the screen's primary measure. Scaled fusion now uses both branches (held-out sign agreement about 0.77 with audio and 0.75 with face), but stays below early concatenation; the fixed quality discount reduced F1 by 1.85 points. Every audio/face policy is at or below the always-positive F1 of 37.37%, so F1 alone does not distinguish them from a trivial rule. The quality-fusion conclusion is now a valid negative result rather than an artefact of unequal scales.

Verification: 135 policy bundles restored and refitted, 5,313 predictions with zero restored-score error, 30 branch scales recomputed from saved inner margins, 15 fusion weights recomputed from saved scores, element-wise fusion recomputation, and the historical original-concatenation scores reproduced. The earlier `sparse_fusion` run still re-verifies under the changed source (180 bundles, 6,762 predictions, zero error). Verifiers now accept source files committed at `feadacb` (`RESULT_COMMITS` in `src/training/provenance.py`), so earlier runs remain verifiable after source changes. All 124 tests passed. Exploratory reused-cohort result; no development/test evaluation.


---

## PHQ-score target, eGeMAPS functionals and domain scaling — 2026-11-08

Run: `score_functionals_domain_20261108T190307688594Z`. [Frozen protocol](methods/score_functionals_domain.md). The user approved this bundled comparison, including use of the released PHQ score as a training target.

**Decision: no candidate passes; retain `original_concat`.** This is the last planned audio/face series. Every audio/face policy remains within about one F1 point of the always-positive rule (37.37%).

Three changes were tested on the same 161 participants and folds:

1. **Official eGeMAPSv02 functionals** (88 features, openSMILE 2.6.0) extracted from 43.47 hours of untreated recordings. As an independent check of the earlier voicing fix, the functionals' voiced F0 mean agrees with the corrected voiced-only F0 summary from the supplied descriptors at r = 0.99998 (median difference 0.007 semitones).
2. **Domain-conditional standardization** (WoZ vs AI, shrinkage 10 toward pooled statistics, fitted within training partitions).
3. **Released PHQ score as training target**: balanced-weighted ridge on PHQ − 9.5, predicted positive at ≥ 9.5, alpha from [1, 10, 100, 1000] by inner squared error on the score. Evaluation used the unchanged binary labels.

| Policy | Accuracy | Balanced accuracy | Precision | Recall | F1 ± SD | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_concat | 62.94 | 57.61 | 30.42 | 47.75 | 37.16 ± 4.02 | 0.600 | 0.401 |
| voiced_concat | 63.56 | 57.38 | 30.51 | 45.95 | 36.66 ± 2.66 | 0.625 | 0.431 |
| func_concat | 63.35 | 57.25 | 30.24 | 45.95 | 36.45 ± 3.44 | 0.586 | 0.352 |
| voiced_concat_domain | 62.94 | 56.98 | 30.00 | 45.95 | 36.30 ± 1.75 | 0.614 | 0.408 |
| func_concat_domain | 63.35 | 56.30 | 29.62 | 43.24 | 35.15 ± 1.35 | 0.578 | 0.343 |
| score_voiced_concat | 57.56 | 57.28 | 28.59 | 56.76 | 38.02 ± 3.96 | 0.613 | 0.396 |
| score_func_concat | 56.31 | 55.52 | 27.24 | 54.05 | 36.22 ± 2.05 | 0.594 | 0.368 |
| score_func_concat_domain | 56.73 | 56.11 | 27.73 | 54.95 | 36.86 ± 0.74 | 0.587 | 0.361 |
| original_audio | 57.14 | 52.58 | 25.42 | 44.14 | 32.24 ± 5.22 | 0.571 | 0.335 |
| voiced_audio | 59.42 | 54.06 | 26.69 | 44.14 | 33.23 ± 3.86 | 0.576 | 0.329 |
| func_audio | 60.87 | 53.11 | 26.15 | 38.74 | 31.21 ± 3.01 | 0.522 | 0.286 |
| face | 60.25 | 55.55 | 28.19 | 46.85 | 35.13 ± 1.45 | 0.580 | 0.343 |
| always_negative | 77.02 | 50.00 | 0.00 | 0.00 | 0.00 ± 0.00 | 0.500 | 0.230 |
| always_positive | 22.98 | 50.00 | 22.98 | 100.00 | 37.37 ± 0.00 | 0.500 | 0.230 |

Classification metrics are means of three repeat-level pooled out-of-fold values (SD across repeats, not a confidence interval). AUC/AP are means over 15 folds.

| Candidate | Control | F1 change | BA change | AUC change | Recall change | Positive repeats | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| func_concat | original_concat | -0.71 | -0.36 | -1.43 | -1.80 | 1/3 | False |
| voiced_concat_domain | original_concat | -0.86 | -0.63 | +1.42 | -1.80 | 1/3 | False |
| func_concat_domain | original_concat | -2.01 | -1.31 | -2.26 | -4.50 | 1/3 | False |
| score_voiced_concat | original_concat | +0.86 | -0.33 | +1.30 | +9.01 | 1/3 | False |
| score_func_concat | original_concat | -0.94 | -2.09 | -0.63 | +6.31 | 1/3 | False |
| score_func_concat_domain | original_concat | -0.30 | -1.50 | -1.33 | +7.21 | 1/3 | False |
| func_audio | original_audio | -1.04 | +0.52 | -4.90 | -5.41 | 1/3 | False |
| func_audio | voiced_audio | -2.02 | -0.96 | -5.32 | -5.41 | 0/3 | False |
| func_concat | voiced_concat | -0.21 | -0.13 | -3.89 | +0.00 | 1/3 | False |
| voiced_concat_domain | voiced_concat | -0.36 | -0.40 | -1.04 | +0.00 | 1/3 | False |
| func_concat_domain | func_concat | -1.30 | -0.95 | -0.82 | -2.70 | 1/3 | False |
| score_voiced_concat | voiced_concat | +1.36 | -0.11 | -1.16 | +10.81 | 1/3 | False |
| score_func_concat | func_concat | -0.23 | -1.73 | +0.80 | +8.11 | 1/3 | False |
| score_func_concat_domain | score_func_concat | +0.63 | +0.58 | -0.70 | +0.90 | 1/3 | False |

Changes are in percentage points (AUC x100). Paired label-stratified participant bootstrap (2,000 resamples, seed 20261108) of the mean-over-repeats difference against `original_concat`:

| Candidate vs original_concat | F1 change 95% interval | Share of resamples > 0 | Pooled AUC change 95% interval | Share > 0 |
|---|---:|---:|---:|---:|
| func_concat | -7.0 to +6.0 | 0.42 | -6.7 to +2.9 | 0.19 |
| voiced_concat_domain | -4.9 to +3.1 | 0.33 | -1.7 to +5.1 | 0.82 |
| func_concat_domain | -8.1 to +4.1 | 0.26 | -7.0 to +1.9 | 0.13 |
| score_voiced_concat | -7.5 to +9.4 | 0.58 | -7.2 to +10.7 | 0.63 |
| score_func_concat | -9.9 to +8.3 | 0.42 | -9.5 to +8.4 | 0.44 |
| score_func_concat_domain | -9.6 to +9.3 | 0.47 | -9.9 to +8.2 | 0.42 |

Pooled AUC here is computed over each repeat's combined held-out folds, unlike the screen's mean fold AUC.

**Interpretation.**
- **Score target.** `score_voiced_concat` had the highest F1 (38.02%, +0.86 points) and recall (56.76%, +9.0), but balanced accuracy fell 0.33 points and F1 improved in only 1/3 repeats; the bootstrap interval is −7.5 to +9.4 points. Ridge chose the strongest regularization (alpha = 1000, the grid edge) in all 45 score fits. Held-out PHQ prediction was weak (RMSE 6.5–6.6, Pearson r 0.17–0.23), so audio/face summaries carry little linear information about symptom severity in this cohort. The grid was fixed in advance and was not extended.
- **Functionals.** The 88 official functionals were worse than the corrected 47 voiced summaries for audio alone (fold AUC 0.522 vs 0.576) and in combination (0.586 vs 0.625). More features without more participants added variance.
- **Domain scaling.** It did not help (−0.4 to −1.3 F1 points against the matching unscaled models).
- **Label/score disagreements.** For the 12 participants with PHQ ≥ 10 but released label 0, the original control already predicts 58% positive, and score models 61–64%. These people look positive to every model, which lowers measured precision for all of them.
- **AI subgroup.** AI-interview F1 varies widely (0.45–0.65) but rests on 18 participants with 5 positives; do not interpret it.

Verification: all 180 policy bundles restored and refitted; 6,762 predictions with zero restored-score error; 45 alpha choices recomputed from saved inner scores; 45 closed-form weighted-ridge checks; 45 domain-statistic checks; independent sklearn cross-check of the vectorized bootstrap on the first 25 resamples; regression and disagreement tables recomputed; historical original-concatenation scores reproduced. A pre-run multi-agent code review found no leakage and one verifier coverage gap, fixed before running. Exploratory reused-cohort result; no development/test evaluation.


---

## Participant speech, pooled cohort, text-assisted fusion and abstention — 2026-11-12

Run: `participant_pooled_20261112T071357784484Z`. Protocol frozen before fitting, with label-free pre-fit observations: `docs/methods/participant_pooled.md`.

**Decision: no candidate passes the screen; `original_concat` stays the retained control.** The best audio/face result so far is `segment_concat`: 65.22% accuracy, 59.40% balanced accuracy, 39.13% F1 and fold AUC 0.631. That is +1.97 F1 points over the control with every metric higher, but below the +3-point screen.

### T161 (same 161 participants and folds as earlier runs)

| Policy | Accuracy | Balanced accuracy | Precision | Recall | F1 ± SD | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_concat | 62.94 | 57.61 | 30.42 | 47.75 | 37.16 ± 4.02 | 0.600 | 0.401 |
| voiced_concat | 63.56 | 57.38 | 30.51 | 45.95 | 36.66 ± 2.66 | 0.625 | 0.431 |
| segment_concat | 65.22 | 59.40 | 32.73 | 48.65 | 39.13 ± 0.35 | 0.631 | 0.405 |
| participant_concat | 64.80 | 58.82 | 32.14 | 47.75 | 38.42 ± 1.33 | 0.619 | 0.383 |
| original_concat_nopose | 61.08 | 54.82 | 27.74 | 43.24 | 33.78 ± 0.87 | 0.562 | 0.369 |
| participant_concat_nopose | 63.98 | 57.65 | 30.91 | 45.95 | 36.96 ± 1.77 | 0.579 | 0.353 |
| original_audio | 57.14 | 52.58 | 25.42 | 44.14 | 32.24 ± 5.22 | 0.571 | 0.335 |
| voiced_audio | 59.42 | 54.06 | 26.69 | 44.14 | 33.23 ± 3.86 | 0.576 | 0.329 |
| segment_audio | 62.11 | 57.07 | 29.68 | 47.75 | 36.57 ± 2.93 | 0.598 | 0.357 |
| participant_audio | 60.87 | 55.63 | 28.18 | 45.95 | 34.87 ± 4.36 | 0.572 | 0.333 |
| face | 60.25 | 55.55 | 28.19 | 46.85 | 35.13 ± 1.45 | 0.580 | 0.343 |
| text | 74.74 | 74.44 | 47.31 | 73.87 | 57.52 ± 5.03 | 0.805 | 0.607 |
| text_av_equal | 73.91 | 69.16 | 45.00 | 60.36 | 51.55 ± 2.16 | 0.784 | 0.569 |
| text_av_weighted | 74.33 | 72.59 | 46.24 | 69.37 | 55.43 ± 4.25 | 0.822 | 0.599 |
| always_positive | 22.98 | 50.00 | 22.98 | 100.00 | 37.37 ± 0.00 | 0.500 | 0.230 |
| always_negative | 77.02 | 50.00 | 0.00 | 0.00 | 0.00 ± 0.00 | 0.500 | 0.230 |

### P217 (161 training + 56 development participants, folds rebuilt)

| Policy | Accuracy | Balanced accuracy | Precision | Recall | F1 ± SD | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_concat_p217 | 59.91 | 52.66 | 25.18 | 39.46 | 30.69 ± 1.73 | 0.529 | 0.339 |
| segment_concat_p217 | 60.52 | 53.78 | 26.34 | 41.50 | 32.20 ± 0.42 | 0.570 | 0.347 |
| participant_concat_p217 | 59.75 | 53.29 | 25.89 | 41.50 | 31.83 ± 1.01 | 0.553 | 0.343 |
| original_audio_p217 | 53.76 | 51.35 | 23.63 | 46.94 | 31.44 ± 1.12 | 0.557 | 0.324 |
| segment_audio_p217 | 58.99 | 55.92 | 27.63 | 50.34 | 35.65 ± 0.58 | 0.583 | 0.337 |
| participant_audio_p217 | 55.30 | 52.58 | 24.66 | 47.62 | 32.48 ± 0.10 | 0.551 | 0.311 |
| face_p217 | 56.84 | 51.88 | 24.34 | 42.86 | 30.97 ± 3.07 | 0.501 | 0.282 |
| always_positive_p217 | 22.58 | 50.00 | 22.58 | 100.00 | 36.84 ± 0.00 | 0.500 | 0.226 |

### Screens

| Candidate | Control | F1 change | BA change | AUC change | Recall change | Positive repeats | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| participant_concat | original_concat | +1.26 | +1.21 | +1.87 | +0.00 | 2/3 | False |
| segment_concat | original_concat | +1.97 | +1.79 | +3.07 | +0.90 | 2/3 | False |
| original_concat_nopose | original_concat | -3.38 | -2.79 | -3.77 | -4.50 | 1/3 | False |
| participant_concat_nopose | original_concat | -0.20 | +0.04 | -2.11 | -1.80 | 1/3 | False |
| participant_audio | segment_audio | -1.70 | -1.44 | -2.56 | -1.80 | 1/3 | False |
| participant_concat | segment_concat | -0.72 | -0.58 | -1.21 | -0.90 | 1/3 | False |
| participant_concat_nopose | participant_concat | -1.46 | -1.17 | -3.97 | -1.80 | 0/3 | False |
| text_av_equal | text | -5.98 | -5.28 | -2.18 | -13.51 | 0/3 | False |
| text_av_weighted | text | -2.10 | -1.85 | +1.63 | -4.50 | 0/3 | False |
| participant_concat_p217 | original_concat_p217 | +1.14 | +0.62 | +2.42 | +2.04 | 2/3 | False |
| segment_concat_p217 | original_concat_p217 | +1.51 | +1.12 | +4.06 | +2.04 | 2/3 | False |

Changes are in percentage points (AUC x100).

### Does more training data help the original 161?

| Policy (trained on 217, scored on the 161) | Accuracy | Balanced accuracy | Recall | F1 | F1 when trained on 161 |
|---|---:|---:|---:|---:|---:|
| original_concat | 61.49 | 54.77 | 42.34 | 33.64 | 37.16 |
| segment_concat | 61.90 | 55.67 | 44.14 | 34.76 | 39.13 |
| participant_concat | 61.49 | 55.41 | 44.14 | 34.58 | 38.42 |
| original_audio | 54.66 | 52.23 | 47.75 | 32.61 | 32.24 |
| segment_audio | 60.25 | 57.44 | 52.25 | 37.65 | 36.57 |
| participant_audio | 55.90 | 54.30 | 51.35 | 34.83 | 34.87 |
| face | 56.73 | 52.00 | 43.24 | 31.45 | 35.13 |

### Abstention (descriptive)

| Model | Subset | People | Positives | Accuracy | Balanced accuracy | F1 |
|---|---|---:|---:|---:|---:|---:|
| original_concat | most confident 100% | 161 | 37.0 | 62.9 | 57.6 | 37.2 |
| original_concat | most confident 80% | 129 | 32.0 | 65.4 | 58.9 | 39.6 |
| original_concat | most confident 60% | 97 | 26.3 | 67.7 | 61.2 | 44.0 |
| original_concat | most confident 40% | 64 | 18.3 | 70.8 | 63.7 | 47.9 |
| participant_concat | most confident 100% | 161 | 37.0 | 64.8 | 58.8 | 38.4 |
| participant_concat | most confident 80% | 129 | 30.0 | 68.2 | 61.5 | 41.7 |
| participant_concat | most confident 60% | 97 | 20.7 | 71.1 | 63.4 | 42.6 |
| participant_concat | most confident 40% | 64 | 12.7 | 74.0 | 63.8 | 41.2 |
| text | most confident 100% | 161 | 37.0 | 74.7 | 74.4 | 57.5 |
| text | most confident 80% | 129 | 28.3 | 78.8 | 78.4 | 61.8 |
| text | most confident 60% | 97 | 20.3 | 79.4 | 76.9 | 59.6 |
| text | most confident 40% | 64 | 12.7 | 84.9 | 81.6 | 66.9 |
| text_av_weighted | most confident 100% | 161 | 37.0 | 74.3 | 72.6 | 55.4 |
| text_av_weighted | most confident 80% | 129 | 26.0 | 80.4 | 78.7 | 61.0 |
| text_av_weighted | most confident 60% | 97 | 17.0 | 84.9 | 83.8 | 65.5 |
| text_av_weighted | most confident 40% | 64 | 10.7 | 90.1 | 89.3 | 74.7 |
| original_concat | not quality-flagged | 82 | 15.0 | 64.2 | 53.1 | 26.5 |
| original_concat | quality-flagged | 79 | 22.0 | 61.6 | 59.9 | 44.9 |

### Findings

- **Interviewer speech is mostly absent from the voiced descriptors.** In 90% of 164 unambiguous scripted-prompt transcript rows, the supplied descriptors contain no voiced segment. Removing estimated interviewer segments (a median 3.6% of segments) did not help: `participant_concat` is 0.72 F1 points below the matched `segment_concat`. Separation accuracy for higher-pitched participants is unverified.
- **Segment-based voiced summaries are the best audio representation so far.** `segment_audio` uses voiced frames inside segments of at least 0.2 s, plus voiced fraction, segment rate and segment length. Alone it reached 36.57% F1 and AUC 0.598, against 32.24% and 0.571 for the original audio.
- **Head-translation features should stay.** Dropping the six `pose_T*` summaries lowered F1 by 1.5 to 3.4 points.
- **Adding development participants hurt.** Every P217 policy scored lower than its T161 counterpart, also when scored only on the original 161 (for example `original_concat` 33.64% vs 37.16% F1). Either the cohorts differ or the T161 estimate is optimistic; in both cases the earlier 37% is not a stable figure. Development facial timing was not anomaly-checked.
- **Text-assisted fusion did not beat text alone on F1** (55.43% vs 57.52%), although mean fold AUC rose from 0.805 to 0.822. The inner selection chose text weight .75 in all 15 folds.
- **Confidence is informative.** On the most confident 40% of participants, `text_av_weighted` reached 90.1% accuracy and 74.7% F1. Text alone reached 84.9% and 66.9%, and `original_concat` 70.8% and 47.9%. The cut-offs use held-out scores, so this is descriptive and not a deployable rule.
- **The fixed quality flags do not mark unreliable predictions.** Flagged participants had higher F1 (44.9%) than unflagged ones (26.5%), with similar accuracy.

### Verification

- 285 branch models restored and refitted; 13,587 predictions with zero restored-score error.
- 30 text-assisted fusions recomputed element by element; metrics, screens and abstention tables recomputed.
- The historical `original_concat` scores and the retained text reference were both reproduced exactly.
- All 135 tests pass. The test split was not read.
- Development participants were pooled into cross-validation, so this run is marked `development_evaluated: true`.

This is an exploratory result on a reused cohort.


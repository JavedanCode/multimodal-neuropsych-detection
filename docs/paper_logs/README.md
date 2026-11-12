# Paper-writer experiment log

## Newest: stacked text + voice + face fusion under three fold schemes (2026-11-12)

Files: `stacked_fusion/`. Read this first; the earlier runs follow below.

Run: `stacked_fusion_20261112T073832688696Z`. Protocol frozen before fitting: `docs/methods/stacked_fusion.md`. Cohort: the 161 training participants; no development or test data.

**Result: `stack2` (text combined with the voice + face model through a learned combiner) is the best fusion model.** Under the existing folds (scheme A) it matches the text model on F1 (57.74% vs 57.52%) with higher recall and balanced accuracy but lower accuracy, and it does not pass the screen. Under 10-fold (scheme C) it reaches 59.51% F1 and passes the screen against text on the same folds. Scheme A stays the primary scheme.

#### All schemes

| Scheme | Model | Accuracy | Balanced accuracy | Precision | Recall | F1 ± SD | AUC | AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A: 5-fold (existing) | Text | 74.74 | 74.44 | 47.31 | 73.87 | 57.52 ± 5.03 | 0.805 | 0.607 |
| A: 5-fold (existing) | stack2 (text + voice/face) | 72.88 | 75.44 | 45.15 | 80.18 | 57.74 ± 6.16 | 0.809 | 0.585 |
| A: 5-fold (existing) | stack3 (text + voice + face) | 71.43 | 72.29 | 43.12 | 73.87 | 54.42 ± 4.96 | 0.797 | 0.575 |
| A: 5-fold (existing) | gate3 (stack3 with confidence gate) | 71.84 | 72.56 | 43.61 | 73.87 | 54.81 ± 4.29 | 0.795 | 0.580 |
| A: 5-fold (existing) | Voice + face | 65.22 | 59.40 | 32.73 | 48.65 | 39.13 ± 0.35 | 0.631 | 0.405 |
| A: 5-fold (existing) | Always positive | 22.98 | 50.00 | 22.98 | 100.00 | 37.37 ± 0.00 | 0.500 | 0.230 |
| B: 5-fold, gender-balanced | Text | 76.40 | 75.51 | 49.14 | 73.87 | 58.95 ± 1.20 | 0.822 | 0.586 |
| B: 5-fold, gender-balanced | stack2 (text + voice/face) | 73.29 | 76.02 | 45.51 | 81.08 | 58.29 ± 5.22 | 0.809 | 0.552 |
| B: 5-fold, gender-balanced | stack3 (text + voice + face) | 70.19 | 71.16 | 41.56 | 72.97 | 52.95 ± 2.88 | 0.783 | 0.525 |
| B: 5-fold, gender-balanced | gate3 (stack3 with confidence gate) | 72.05 | 73.32 | 43.77 | 75.68 | 55.46 ± 1.87 | 0.806 | 0.557 |
| B: 5-fold, gender-balanced | Voice + face | 64.39 | 59.18 | 32.21 | 49.55 | 39.03 ± 2.91 | 0.634 | 0.418 |
| B: 5-fold, gender-balanced | Always positive | 22.98 | 50.00 | 22.98 | 100.00 | 37.37 ± 0.00 | 0.500 | 0.230 |
| C: 10-fold | Text | 74.95 | 73.31 | 47.00 | 70.27 | 56.31 ± 0.55 | 0.811 | 0.633 |
| C: 10-fold | stack2 (text + voice/face) | 75.78 | 76.37 | 48.32 | 77.48 | 59.51 ± 1.43 | 0.817 | 0.613 |
| C: 10-fold | stack3 (text + voice + face) | 73.71 | 74.40 | 45.71 | 75.68 | 56.98 ± 3.30 | 0.796 | 0.592 |
| C: 10-fold | gate3 (stack3 with confidence gate) | 73.71 | 74.08 | 45.70 | 74.77 | 56.70 ± 3.02 | 0.800 | 0.610 |
| C: 10-fold | Voice + face | 67.08 | 61.25 | 35.00 | 50.45 | 41.33 ± 0.96 | 0.650 | 0.463 |
| C: 10-fold | Always positive | 22.98 | 50.00 | 22.98 | 100.00 | 37.37 ± 0.00 | 0.500 | 0.230 |

Classification metrics are means of three repeat-level pooled out-of-fold values (± SD across repeats). AUC and AP are means over held-out folds.

#### Fusion against text on the same folds

| Candidate | Control | F1 change | BA change | AUC change | Recall change | Positive repeats | Pass | Bootstrap F1 change (95%) | Share > 0 |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| stack3_A | text_A | -3.10 | -2.15 | -0.89 | +0.00 | 0/3 | False | -7.2 to +1.2 | 0.07 |
| stack2_A | text_A | +0.22 | +1.00 | +0.37 | +6.31 | 2/3 | False | -3.8 to +5.0 | 0.54 |
| gate3_A | text_A | -2.72 | -1.88 | -1.00 | +0.00 | 0/3 | False | -6.2 to +0.8 | 0.06 |
| stack3_B | text_B | -5.99 | -4.35 | -3.87 | -0.90 | 0/3 | False | -10.3 to -2.3 | 0.00 |
| stack2_B | text_B | -0.65 | +0.51 | -1.27 | +7.21 | 2/3 | False | -5.5 to +4.4 | 0.41 |
| gate3_B | text_B | -3.49 | -2.19 | -1.61 | +1.80 | 0/3 | False | -6.5 to -0.3 | 0.01 |
| stack3_C | text_C | +0.67 | +1.09 | -1.51 | +5.41 | 2/3 | False | -4.3 to +5.9 | 0.61 |
| stack2_C | text_C | +3.20 | +3.07 | +0.58 | +7.21 | 3/3 | True | -1.5 to +8.5 | 0.91 |
| gate3_C | text_C | +0.39 | +0.77 | -1.17 | +4.50 | 2/3 | False | -4.6 to +5.6 | 0.56 |

Changes are in percentage points (AUC x100). The bootstrap is a paired, label-stratified participant bootstrap with 2,000 resamples.

#### Abstention under scheme A (descriptive)

| Model (scheme A) | Most confident | Accuracy | Balanced accuracy | F1 |
|---|---|---:|---:|---:|
| text | 100% | 74.7 | 74.4 | 57.5 |
| text | 80% | 78.8 | 78.4 | 61.8 |
| text | 60% | 79.4 | 76.9 | 59.6 |
| text | 40% | 84.9 | 81.6 | 66.9 |
| stack2 | 100% | 72.9 | 75.4 | 57.7 |
| stack2 | 80% | 76.2 | 78.5 | 59.8 |
| stack2 | 60% | 79.4 | 82.5 | 64.4 |
| stack2 | 40% | 81.8 | 85.0 | 67.9 |
| stack3 | 100% | 71.4 | 72.3 | 54.4 |
| stack3 | 80% | 75.5 | 76.8 | 59.0 |
| stack3 | 60% | 78.7 | 81.8 | 63.0 |
| stack3 | 40% | 80.2 | 83.5 | 62.7 |
| gate3 | 100% | 71.8 | 72.6 | 54.8 |
| gate3 | 80% | 75.2 | 76.0 | 58.5 |
| gate3 | 60% | 76.3 | 79.0 | 62.4 |
| gate3 | 40% | 79.7 | 82.8 | 65.8 |

#### Findings

- **The learned combiner fixed the decision-point problem.** The earlier fixed-weight text + audio/face fusion scored 55.43% F1 under scheme A. `stack2` scores 57.74%, with recall 80.18% (text: 73.87%) and balanced accuracy 75.44% (text: 74.44%). Accuracy is lower (72.88% vs 74.74%) because it predicts more positives.
- **Two inputs beat three.** `stack3` (separate voice and face inputs) and `gate3` were below text under schemes A and B. The combiner gives voice and face small weights (about 0.1–0.3 against 1.3–1.4 for text).
- **Fold scheme matters as much as the model.** Text alone scores 57.52%, 58.95% and 56.31% F1 under schemes A, B and C. That spread (2.6 points) is as large as most model differences in this project.
- **10-fold is the most favourable scheme for fusion.** `stack2` passes the screen only under scheme C (+3.20 F1, 3/3 repeats). Its bootstrap interval is −1.5 to +8.5 points, with 91% of resamples above zero. Because scheme C was picked from three schemes after seeing results, its 59.51% is optimistic; scheme A's 57.74% is the comparable figure.
- **Gender-balanced folds did not help fusion.** Scheme B made folds more uniform in gender (SD of female share 0.023 against 0.069) and gave the highest text score, but fusion was below text.
- **Abstention.** On the most confident 60% of participants, `stack2` reaches 82.5% balanced accuracy and 64.4% F1 (text: 76.9% and 59.6%).

#### Verification

- 60 fold bundles restored and refitted; 10,143 predictions with zero restored-score error.
- Combiners and the confidence gate recomputed element by element from the saved parts.
- Under scheme A, the retained text reference and the earlier `segment_concat` model were both reproduced exactly.
- Every participant is held out exactly once per repeat in every scheme. All tests pass.

This is an exploratory result on a reused cohort.


---

## Latest: participant speech, pooled cohort, text-assisted fusion and abstention (2026-11-12)

Files: `participant_pooled/`. This run followed the three runs described in the next section. Same repeated nested cross-validation, labels unchanged, test split not read.

Run: `participant_pooled_20261112T071357784484Z`. Protocol frozen before fitting, with label-free pre-fit observations: `docs/methods/participant_pooled.md`.

**Decision: no candidate passes the screen; `original_concat` stays the retained control.** The best audio/face result so far is `segment_concat`: 65.22% accuracy, 59.40% balanced accuracy, 39.13% F1 and fold AUC 0.631. That is +1.97 F1 points over the control with every metric higher, but below the +3-point screen.

#### T161 (same 161 participants and folds as earlier runs)

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

#### P217 (161 training + 56 development participants, folds rebuilt)

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

#### Screens

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

#### Does more training data help the original 161?

| Policy (trained on 217, scored on the 161) | Accuracy | Balanced accuracy | Recall | F1 | F1 when trained on 161 |
|---|---:|---:|---:|---:|---:|
| original_concat | 61.49 | 54.77 | 42.34 | 33.64 | 37.16 |
| segment_concat | 61.90 | 55.67 | 44.14 | 34.76 | 39.13 |
| participant_concat | 61.49 | 55.41 | 44.14 | 34.58 | 38.42 |
| original_audio | 54.66 | 52.23 | 47.75 | 32.61 | 32.24 |
| segment_audio | 60.25 | 57.44 | 52.25 | 37.65 | 36.57 |
| participant_audio | 55.90 | 54.30 | 51.35 | 34.83 | 34.87 |
| face | 56.73 | 52.00 | 43.24 | 31.45 | 35.13 |

#### Abstention (descriptive)

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

#### Findings

- **Interviewer speech is mostly absent from the voiced descriptors.** In 90% of 164 unambiguous scripted-prompt transcript rows, the supplied descriptors contain no voiced segment. Removing estimated interviewer segments (a median 3.6% of segments) did not help: `participant_concat` is 0.72 F1 points below the matched `segment_concat`. Separation accuracy for higher-pitched participants is unverified.
- **Segment-based voiced summaries are the best audio representation so far.** `segment_audio` uses voiced frames inside segments of at least 0.2 s, plus voiced fraction, segment rate and segment length. Alone it reached 36.57% F1 and AUC 0.598, against 32.24% and 0.571 for the original audio.
- **Head-translation features should stay.** Dropping the six `pose_T*` summaries lowered F1 by 1.5 to 3.4 points.
- **Adding development participants hurt.** Every P217 policy scored lower than its T161 counterpart, also when scored only on the original 161 (for example `original_concat` 33.64% vs 37.16% F1). Either the cohorts differ or the T161 estimate is optimistic; in both cases the earlier 37% is not a stable figure. Development facial timing was not anomaly-checked.
- **Text-assisted fusion did not beat text alone on F1** (55.43% vs 57.52%), although mean fold AUC rose from 0.805 to 0.822. The inner selection chose text weight .75 in all 15 folds.
- **Confidence is informative.** On the most confident 40% of participants, `text_av_weighted` reached 90.1% accuracy and 74.7% F1. Text alone reached 84.9% and 66.9%, and `original_concat` 70.8% and 47.9%. The cut-offs use held-out scores, so this is descriptive and not a deployable rule.
- **The fixed quality flags do not mark unreliable predictions.** Flagged participants had higher F1 (44.9%) than unflagged ones (26.5%), with similar accuracy.

#### Verification

- 285 branch models restored and refitted; 13,587 predictions with zero restored-score error.
- 30 text-assisted fusions recomputed element by element; metrics, screens and abstention tables recomputed.
- The historical `original_concat` scores and the retained text reference were both reproduced exactly.
- All 135 tests pass. The test split was not read.
- Development participants were pooled into cross-validation, so this run is marked `development_evaluated: true`.

This is an exploratory result on a reused cohort.


---

## Start here: follow-up studies after code review — 2026-11-08

This log covers three runs on the same E-DAIC training cohort: 161 retained participants, 37 positive released labels, 143 WoZ and 18 AI interviews. All three use the same repeated nested cross-validation:
- 5 outer folds, repeated with seeds 42, 137 and 2026, plus 3 inner folds;
- folds stratified jointly by label and interview domain;
- every preprocessing, scaling, selection and setting step fitted on training partitions only.

No development or test data were used, and released labels were never changed.

| Order | Run | What it tested | Files |
|---|---|---|---|
| 1 | `sparse_fusion_20261108T160011416879Z` | L1 sparse selection, late fusion, quality-weighted fusion | this folder's root (section further below) |
| 2 | `corrected_fusion_20261108T180359546982Z` | Fixes for two code defects found by review | `corrected_fusion/` |
| 3 | `score_functionals_domain_20261108T190307688594Z` | PHQ score as training target, official eGeMAPS functionals, interview-domain scaling | `score_functionals_domain/` |

### Overall conclusion

- **No audio/face variant beat the retained control** under the predeclared screen.
  - Control: original supplied audio + facial summaries with balanced logistic regression, C=.1. It scores 62.94% accuracy, 57.61% balanced accuracy, 37.16% positive-class F1 and fold AUC 0.600.
  - Screen: mean F1 +3 points, no loss in balanced accuracy or fold AUC, recall loss ≤ 5 points, and an F1 gain in at least 2 of 3 repeats.
- **The most useful observation is a ranking gain from the corrected audio summaries.** Voiced-only audio summaries improve ranking in the combined model (fold AUC 0.600 → 0.625, AP 0.401 → 0.431). F1 at the fixed threshold does not improve. The gain is within fold-to-fold noise and should be reported as suggestive only.
- **Every audio/face model is within about one F1 point of predicting everyone positive** (37.37% F1). Report AUC, AP and balanced accuracy alongside F1, and always show the always-positive row.
- **The transcript (MPNet) reference remains much stronger**, at 74.74% accuracy and 57.52% F1. It uses a different input modality.
- **These results are exploratory.** The cohort has been reused heavily, so a fresh holdout is needed before calling any variant an improvement.

### 1. Code review: two defects found and corrected

A multi-agent code review used independent reviewers, then adversarially verified each finding. It found no data leakage, but two defects that weakened the audio/face results.

1. **Unvoiced placeholders in the audio summaries.**
   - On frames with no voice, the supplied openSMILE eGeMAPS descriptor files write 0 (pitch, jitter, shimmer, HNR, harmonic differences) or −201 (formant amplitudes).
   - The pipeline averaged these placeholders together with real measurements. As a result, about 18 of the 46 audio features mainly measured how much of the recording was voiced.
   - Example: one participant's mean pitch was 4.9 semitones over all frames, against 25.6 over voiced frames.
   - Fix: voice-dependent descriptors are summarized over voiced frames only (F0 > 0), −201 is treated as missing, and the voiced fraction is added as an explicit feature, giving 47 features.
   - Checks: with the fix switched off, the code reproduces the old summaries exactly. openSMILE's official functionals agree with the corrected pitch mean at r = 0.99998.
2. **Late fusion on unequal scales.**
   - Audio and face branch outputs were averaged without first putting them on a common scale.
   - Their score ranges differed by up to about 30-fold, so the nominal fusion weight was not the weight actually applied. One branch dominated each fold.
   - Fix: each branch's score is divided by the root-mean-square of its inner out-of-fold scores before weighting.

Corrected run (`corrected_fusion/`):

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

After the fixes, fusion does use both modalities: held-out decisions agree with the audio branch about 77% of the time and with the face branch 75%. It still scores below early concatenation, and the fixed quality discount lowered F1 by 1.85 points. "Quality-aware fusion did not help" is therefore now a valid negative result, not an artefact of the scaling defect.

### 2. PHQ score target, official eGeMAPS functionals and domain scaling

Run `score_functionals_domain/`. The protocol was frozen before extraction and fitting: `docs/methods/score_functionals_domain.md`. The user approved using the released PHQ score as a training target.

- **Official eGeMAPSv02 functionals:** 88 features, extracted with openSMILE 2.6.0 from 43.47 hours of untreated recordings. There was no denoising, trimming or speaker isolation.
- **Domain scaling:** features re-standardized within each interview type (WoZ vs AI) and shrunk toward pooled statistics (shrinkage 10). Fitted on training partitions only. Domain is never used as a predictor.
- **Score target:**
  - Model: ridge regression on the released PHQ score minus 9.5, with balanced class weights.
  - Decision: positive when the predicted score is ≥ 9.5. This threshold is fixed, not tuned.
  - Ridge alpha: chosen from [1, 10, 100, 1000] by inner squared error on the score.
  - Evaluation always used the released binary labels. The PHQ score was never an input feature.

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

Classification metrics are percentages. Each is the mean of three repeat-level pooled out-of-fold values; ± is the population SD across repeats, not a confidence interval. AUC and AP are means over the 15 held-out folds, on a 0–1 scale.

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

Paired, label-stratified participant bootstrap (2,000 resamples, fixed seed 20261108). Values are the mean-over-repeats change against `original_concat`, in percentage points. Pooled AUC is computed over each repeat's combined held-out folds, which differs from the screen's mean fold AUC.

| Candidate vs original_concat | F1 change 95% interval | Share of resamples > 0 | Pooled AUC change 95% interval | Share > 0 |
|---|---:|---:|---:|---:|
| func_concat | -7.0 to +6.0 | 0.42 | -6.7 to +2.9 | 0.19 |
| voiced_concat_domain | -4.9 to +3.1 | 0.33 | -1.7 to +5.1 | 0.82 |
| func_concat_domain | -8.1 to +4.1 | 0.26 | -7.0 to +1.9 | 0.13 |
| score_voiced_concat | -7.5 to +9.4 | 0.58 | -7.2 to +10.7 | 0.63 |
| score_func_concat | -9.9 to +8.3 | 0.42 | -9.5 to +8.4 | 0.44 |
| score_func_concat_domain | -9.6 to +9.3 | 0.47 | -9.9 to +8.2 | 0.42 |

Findings:
- **Score target:**
  - `score_voiced_concat` had the highest F1 (38.02%, +0.86 points) and recall (+9.0 points).
  - But its balanced accuracy fell 0.33 points, F1 improved in only one of three repeats, and the bootstrap interval for the F1 change spans −7.5 to +9.4 points.
  - Ridge chose the strongest regularization (alpha 1000, the edge of the fixed grid) in all 45 fits.
  - Held-out PHQ prediction was weak (RMSE about 6.5–6.6 points, Pearson r 0.17–0.23). Audio/face summaries carry little linear information about symptom severity in this cohort.
- **Official functionals:** worse than the corrected 47 voiced summaries, both audio-only (fold AUC 0.522 vs 0.576) and combined (0.586 vs 0.625). More features without more participants added variance.
- **Domain scaling:** no benefit (−0.4 to −1.3 F1 points against the matching unscaled models).
- **Label/score disagreements:**
  - 12 retained participants have PHQ ≥ 10 but released label 0. No participant has label 1 with PHQ < 10.
  - Every model predicts many of these 12 as positive: 58% of held-out predictions for the original control, 61–64% for the score models.
  - This lowers measured precision for all models.
- **AI subgroup:** only 18 participants, 5 of them positive. Subgroup F1 ranges from 0.45 to 0.65 and should not be interpreted.

### Verification

- **`corrected_fusion`:**
  - 135 policy bundles restored and refitted; 5,313 predictions with zero restored-score error.
  - 30 branch scales and 15 fusion weights recomputed from saved inner scores.
  - Element-wise fusion recomputed; historical control reproduced.
- **`score_functionals_domain`:**
  - 180 bundles restored and refitted; 6,762 predictions with zero restored-score error.
  - 45 alpha choices recomputed from saved inner scores; 45 closed-form weighted-ridge checks; 45 domain-statistic checks.
  - Independent sklearn cross-check of the bootstrap; regression and disagreement tables recomputed; historical control reproduced.
  - A multi-agent review before the run found no leakage.
- **Earlier `sparse_fusion` run:** still re-verifies under the changed code (180 bundles, 6,762 predictions, zero error).
- **Tests:** all 130 automated tests pass. Machine-readable checks are in `verification.json` in each subfolder.

### Suggested Results/Discussion wording

"A code review of the audio/visual pipeline identified two defects: voice-dependent acoustic summaries had been averaged together with openSMILE's placeholder values for unvoiced frames, and late fusion combined branch outputs on unequal scales. After correction, voiced-only acoustic summaries combined with facial features achieved a mean fold AUC of 0.625 (control 0.600) and a positive-class F1 of 36.66% (control 37.16%). Scale-corrected weighted late fusion achieved 34.81% F1, and its quality-adjusted variant 32.96%. A further predeclared comparison tested the released PHQ score as a training target, the official eGeMAPSv02 functional set and interview-domain standardization. The best variant (PHQ-score ridge regression on voiced acoustic and facial features) reached 38.02% F1 with higher recall but slightly lower balanced accuracy (57.28% vs 57.61%), and a paired bootstrap interval for its F1 change of −7.5 to +9.4 points. No variant met the predeclared improvement criteria. All audio/visual models remained close to the 37.37% F1 of a trivial all-positive rule, whereas the transcript-based reference reached 57.52%. These exploratory, repeatedly reused training-set results do not establish a predictive advantage for reliability-aware or sparse fusion."

### Claims these follow-ups do not support

- That R-SAF, reliability gating, HHO, L1 sparsity, domain scaling or the score target improves screening performance.
- That the AUC improvement from the corrected audio summaries is confirmed. It is within fold-to-fold noise and was observed on a reused cohort.
- Any multi-dataset, multi-disorder, clinical or independent-generalization evidence. All results are for E-DAIC depression, on the training cohort, cross-validated.
- That these negative results prove a dataset performance ceiling. Only the tested recipes were evaluated.

### Files in the subfolders

Both subfolders contain:
- `summary.csv`: one row per policy; values are fractions, so multiply by 100 for percentages.
- `repeat_metrics.csv`, `domain_metrics.csv` and `fold_average_precision.csv`.
- `comparison.json`: screen decisions with unrounded differences.
- `verification.json`, `provenance.json` (source and input hashes) and `environment.txt`.

`corrected_fusion/` also has `fusion_weights.csv`. `score_functionals_domain/` also has `run.log`, `bootstrap_summary.json`, `score_regression.csv`, `score_label_disagreements.csv` and `feature_extraction.json`.

No recordings, transcripts, participant IDs, per-person predictions or model binaries are included.

To reproduce, run these from this commit with the authorized local data:

```powershell
.venv\Scripts\python.exe -c "from pathlib import Path; from src.dataset.voiced_audio import build; build(Path('.').resolve())"
.venv\Scripts\python.exe -m src run audio-visual --corrected
.venv\Scripts\python.exe -c "from pathlib import Path; from src.dataset.egemaps_functionals import build; build(Path('.').resolve())"
.venv\Scripts\python.exe -m src run audio-visual --score-study
.venv\Scripts\python.exe -m src verify audio-visual --run experiments/runs/<run>
```

---

## Earlier: final bounded sparse and reliability-aware fusion comparison

The files in this folder's root (`results.csv`, `comparisons.json`, `selected_settings.csv` and the rest) belong to this earlier run.

Run ID: `sparse_fusion_20261108T160011416879Z`. Protocol: `docs/methods/sparse_fusion.md`. The protocol was written before model fitting. This package contains aggregate results and methods; participant predictions, recordings and model binaries are excluded.

### Main conclusion

None of the new policies passed the predeclared improvement screen against the retained original audio/facial concatenation model. Sparse concatenation reached 37.63% F1 versus 37.16% for the control, but accuracy fell from 62.94% to 59.63% and balanced accuracy from 57.61% to 57.36%. Reliability adjustment added only about 0.08 percentage points F1 to ordinary weighted fusion and reduced mean fold AUC. Keep the original primary reference and stop this bounded tuning series.

### Dataset and evaluation

- Local E-DAIC official training partition: 161 retained people, 37 positive and 124 negative released labels. There are 143 WoZ and 18 AI interviews. All participants remain in the evaluation, including two with quarantined facial timing.
- Labels remain unchanged. Earlier audits identified label-definition discrepancies and uncertain transcript speaker attribution; this experiment does not resolve them.
- Three repeats (seeds 42, 137, 2026), five outer participant folds per repeat, jointly stratified by label and interview domain. Three inner folds per outer fitting partition use seed + outer-fold number + 1.
- Preprocessing and supervised feature selection are refitted within every inner fitting partition. Inner out-of-fold F1 selects settings; outer held-out people never select features, settings, weights or thresholds.
- No new official development/test evaluation. These training participants have been examined across earlier studies; outer CV within this experiment does not erase that broader adaptive reuse.
- WoZ/AI stratification is interview-domain handling within one depression dataset. It is not evidence of performance across different neurological/psychological disorders.

### Inputs and model recipes

Use original supplied eGeMAPS acoustic summaries (46 features) and OpenFace facial summaries (98 features). The earlier waveform cleanup was rejected. No transcript embeddings or lexical predictors enter these models. Quality proxies are derived from the existing utterance-aligned caches, so timing still depends on transcript intervals.

Fold-fitted preprocessing drops wholly missing features, median-imputes, removes constant/exact duplicate columns, and standardizes. The fixed control concatenates the original features and fits balanced logistic regression with C=.1, liblinear, max_iter=5000 and the repeat seed.

For each input family (audio, face, concatenation), evaluate nine candidates: all features, top5, or top10, each with final logistic C=.001/.01/.1. The smaller C values provide stronger regularization. For top-k selection, fit a balanced L1 logistic selector with fixed C=1 on that fitting partition, retain nonzero absolute coefficients (>1e-8), and select at most k in stable descending order. If no coefficient survives, predict the fitting prevalence. All-feature and sparse candidates are reported separately; the selected unimodal branch can choose either. Candidate ties select the first entry: all/top5/top10, then C ascending.

Late fusion averages the two selected branches' logistic probabilities. These class-weighted outputs are not calibrated clinical risks. Equal fusion uses .5 audio/.5 face. Weighted fusion chooses the audio weight from [.5,.25,.75] by inner OOF F1, using the same inner branch predictions. Missing modalities receive zero weight and the remaining weights are normalized; no available modality falls back to fitting prevalence. Inner fallback uses the relevant inner fitting prevalence. Thresholds remain fixed: zero for margins, .5 for fused probabilities.

The reliability control uses exactly the same selected branches and base weight as ordinary weighted fusion. An available audio input receives half weight if clipping >.001, zero fraction >.01 or frame coverage <.95. An available facial input receives half weight if retained fraction <.8 or confidence <.8. Weights are renormalized. These are fixed engineering thresholds, not learned probabilities of correctness; the aligned quality proxies do not directly validate the reliability of whole-interview summaries.

### All results

Accuracy, balanced accuracy, precision, recall and F1 are percentages. F1 SD is the population SD across the three repeat-level pooled out-of-fold values, not a confidence interval. AUC and AP are unweighted means across 15 held-out folds on a 0–1 scale. Repeats overlap and are not independent samples.

| Policy | Accuracy | Balanced accuracy | Precision | Recall | F1 ± SD | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_concat | 62.94 | 57.61 | 30.42 | 47.75 | 37.16 ± 4.02 | 0.600 | 0.401 |
| concat_all | 55.90 | 53.36 | 25.67 | 48.65 | 33.55 ± 4.56 | 0.563 | 0.365 |
| concat_sparse | 59.63 | 57.36 | 29.13 | 53.15 | 37.63 ± 3.06 | 0.598 | 0.367 |
| audio_all | 53.62 | 52.19 | 24.93 | 49.55 | 33.11 ± 3.25 | 0.582 | 0.348 |
| audio_sparse | 51.35 | 49.77 | 23.30 | 46.85 | 31.02 ± 6.16 | 0.525 | 0.323 |
| selected_audio | 50.31 | 49.41 | 22.80 | 47.75 | 30.81 ± 4.26 | 0.545 | 0.333 |
| face_all | 55.07 | 55.35 | 26.94 | 55.86 | 36.34 ± 4.16 | 0.545 | 0.310 |
| face_sparse | 56.11 | 56.02 | 27.70 | 55.86 | 37.00 ± 2.52 | 0.592 | 0.368 |
| selected_face | 54.45 | 55.58 | 27.12 | 57.66 | 36.86 ± 2.61 | 0.568 | 0.345 |
| late_equal | 53.62 | 54.41 | 26.21 | 55.86 | 35.67 ± 7.12 | 0.569 | 0.369 |
| late_weighted | 54.45 | 54.31 | 26.28 | 54.05 | 35.36 ± 4.34 | 0.579 | 0.368 |
| late_quality | 54.66 | 54.45 | 26.37 | 54.05 | 35.45 ± 4.18 | 0.575 | 0.365 |
| always_negative | 77.02 | 50.00 | 0.00 | 0.00 | 0.00 ± 0.00 | 0.500 | 0.230 |
| always_positive | 22.98 | 50.00 | 22.98 | 100.00 | 37.37 ± 0.00 | 0.500 | 0.230 |

### Predeclared decision screens

Pass requires mean F1 improvement of at least 3 percentage points, no mean balanced-accuracy or mean fold-AUC reduction, recall loss no greater than 5 points, and positive F1 change in at least two repeats. These are engineering selection criteria; no multiplicity-adjusted significance claim is made.

| Candidate | Control | F1 change (points) | Positive repeats | Screen passed |
|---|---|---:|---:|---|
| concat_all | original_concat | -3.61 | 0/3 | False |
| concat_sparse | original_concat | +0.47 | 1/3 | False |
| late_equal | original_concat | -1.49 | 1/3 | False |
| late_weighted | original_concat | -1.80 | 1/3 | False |
| late_quality | original_concat | -1.71 | 1/3 | False |
| audio_sparse | audio_all | -2.08 | 1/3 | False |
| face_sparse | face_all | +0.66 | 2/3 | False |
| concat_sparse | concat_all | +4.07 | 2/3 | True |
| late_weighted | late_equal | -0.31 | 1/3 | False |
| late_quality | late_weighted | +0.08 | 1/3 | False |
| late_weighted | selected_audio | +4.56 | 3/3 | True |
| late_weighted | selected_face | -1.50 | 1/3 | False |

Two ancillary comparisons pass: sparse concatenation versus the weaker inner-selected all-feature concatenation (+4.07 points F1), and weighted fusion versus the weaker selected audio branch (+4.56 points). Neither establishes an improvement over the retained original concatenation. The matched quality-versus-weighted comparison fails.

### Selection and compactness

Sparse concatenation selected 5 features in 7/15 outer fits and 10 features in 8/15, averaging 7.67 compared with 144 for the original concatenation. This is a compression trade-off, not demonstrated performance equivalence: it detected more positives but made more false positives, reducing accuracy and precision.

The selected audio branch kept all features in 7/15 fits and a sparse set in 8/15. The selected face branch kept all features in 4/15 and a sparse set in 11/15. Weighted fusion chose audio weight .25 in 10/15 fits, .75 in 4/15, and .5 in 1/15. Fold-level mean effective audio weights averaged approximately .408 for ordinary weighted fusion and .349 with the quality discount. `selected_settings.csv` records each selected setting and feature list; `fusion_weights.csv` records fold-level effective weights. Fold-level averages are descriptive, not independent stability evidence. The fixed rules flagged 78 available audio inputs and zero available facial inputs as poor; two facial inputs were unavailable.

### Comparison with earlier results

| Earlier/current method | Accuracy | F1 | Interpretation |
|---|---:|---:|---|
| Retained text reference | 74.74% | 57.52% | Stronger reference; different input modality |
| Earlier threshold-adjusted text candidate | 76.19% | 58.51% | Modest gain; failed its promotion screen |
| Original audio + facial concatenation | 62.94% | 37.16% | Matched control reproduced in this run |
| Earlier neural quality fusion | 62.11% | 28.30% | Different architecture; not a matched quality ablation here |
| Earlier high-pass audio + facial combination | 62.11% | 34.31% | Waveform cleanup did not improve the original control |
| Current sparse concatenation | 59.63% | 37.63% | Small F1 gain, lower accuracy; not promoted |
| Current weighted late fusion | 54.45% | 35.36% | Does not beat original concatenation |
| Current quality-adjusted late fusion | 54.66% | 35.45% | Negligible matched quality benefit; not promoted |

These figures come from repeated training CV. Historical Stage 9 test-set findings are a separate experiment and must not be mixed with these estimates. Always predicting the negative label gives 77.02% accuracy on this cohort but zero positive recall/F1, so accuracy alone is not a useful screening criterion.

### Verification and reproducibility

- All 180 serialized policy bundles were restored and final policies refitted; 6,762 predictions were checked with zero restored-score error. Bundles share branch models, so 180 files do not mean 180 independent fits.
- All 405 inner candidate vectors and 45 fusion-weight vectors were replayed, including selection choices and feature identities. Reliability calculations were independently reconstructed.
- Folds, preprocessing statistics, hashes, confusion-derived metrics, domain metrics, AUC/AP, summaries and screens were checked. Original concatenation reproduces the historical held-out scores.
- All 116 tests passed; one pre-existing transcript-regex warning remains. Training was CPU-based. No new dataset or external holdout was used.
- `verification.json` gives machine-readable checks. `source_hashes.json` identifies exact source/configuration/input versions, and `environment.txt` records the environment. The full source/model manifests and participant-level audit trail remain in the local run folder.

Reproduce from the matching source and existing authorized local caches:

```powershell
.venv\Scripts\python.exe -m src run audio-visual --sparse-fusion
.venv\Scripts\python.exe -m src verify audio-visual --run experiments/runs/sparse_fusion_20261108T160011416879Z
```

### Suggested Results/Discussion wording

“Within the repeated nested participant-level evaluation, L1-based sparse concatenation reduced the model input to five or ten features per outer fit and achieved a mean positive-class F1 of 37.63%, compared with 37.16% for the fixed original-feature concatenation control. Accuracy decreased from 62.94% to 59.63%, and balanced accuracy decreased from 57.61% to 57.36%. Weighted late fusion and its matched quality-adjusted variant achieved F1 scores of 35.36% and 35.45%, respectively. No proposed replacement met the predeclared improvement criteria against the retained control. These exploratory results support reporting the sparsity–performance trade-off, but do not establish a predictive advantage for reliability-aware fusion.”

### Claims this experiment does not support

- That R-SAF, quality gating or HHO outperforms the retained reference. This follow-up tests L1 selection; it does not rerun or validate an HHO advantage. Historical Integrated Gradients results concern earlier neural models and must not be presented as explanations of these newly fitted logistic policies.
- That five clinical datasets, multiple disorders, clinical deployment, or generalization to new patients were validated.
- That a higher F1 alone establishes superiority, or that the best observed candidate is an independently confirmed improvement.
- That failure of these bounded recipes proves a dataset ceiling. The appropriate conclusion is that the tested variants did not establish the proposed gain.

### Files and units

`results.csv` contains the 14 policy summaries; `repeat_metrics.csv` contains 42 repeat-level rows; `domain_metrics.csv` contains interview-domain summaries. Numeric metric fields in CSV are fractions (multiply by 100 for percentages), except counts; AUC also uses 0–1. `comparisons.json` stores unrounded differences and screen decisions. `run.log` contains the training progress log without local machine paths. There are no recordings, transcripts, participant IDs, per-person predictions or model binaries in this shareable package.

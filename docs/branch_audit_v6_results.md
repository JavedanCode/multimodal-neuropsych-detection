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

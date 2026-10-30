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

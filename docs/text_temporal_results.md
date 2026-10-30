# Text, temporal, and reliability follow-up

Completion review: 2026-10-31. The [protocol](text_temporal_protocol.md) was fixed
before fitting, including the documented facial-timing amendment. These are
exploratory training cross-validation experiments, not new test-set results.

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

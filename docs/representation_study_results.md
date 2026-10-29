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

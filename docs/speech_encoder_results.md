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

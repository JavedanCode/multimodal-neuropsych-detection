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

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

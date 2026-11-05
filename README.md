# Multimodal depression screening

Research implementation for acoustic and facial behavioral features from E-DAIC.
This is a screening research project, not a clinical diagnostic system. R-SAF is
a provisional method name; novelty and performance have not been established.

## Matched audio/facial branch audit (2026-11-04)

Aligned and richer facial summaries improve F1 from 33.7% to about 37.5%,
passing the relative branch screen. Audio revisions and all fusion changes fail
their screens: original concatenation reaches 37.2% F1, quality neural fusion
28.3%. The retained text reference remains stronger. All 195 models and 7,245
predictions were verified, including replay of inner selection and all 45 neural
fits; all 154 tests pass. See [methods and complete results](docs/branch_audit_v6_results.md).

## Audio/facial primary with text fallback (2026-11-04)

The new bounded comparison keeps text out of primary fusion and invokes it only
when fixed availability/quality rules require fallback. Plain primary F1 is 29.5%,
quality primary 33.3%, and fixed-text fallback recovers 55.4% when both auxiliaries
are absent. Clean quality fallback activates for only 1/161 participants and does
not improve clean F1. The retained text reference remains stronger at 57.5%; none
of these policies qualifies for promotion. All 152 tests pass, and 90 saved bundles,
11,592 predictions and 180 inner candidate vectors were checked. See
[methods, routing rates and complete results](docs/primary_fallback_v5_results.md).

## Bounded existing-data follow-up (2026-11-04)

The frozen series tested precision thresholds, fixed semantic topic summaries and
conservative quality fusion on the same v2 folds. Threshold selection reached
**49.9% precision, 72.1% recall and 58.5% F1**, a modest gain below the predeclared
promotion target. No candidate passed either improvement screen; the original
reference remains retained. All 90 bundles and 11,592 predictions were verified,
including replay of 1,020 inner candidate score vectors. See the
[complete comparison and uncertainty limits](docs/bounded_v3_results.md).

Human review flagged possible interviewer speech in three of 16 covered clips;
seven others were uncertain. Full-utterance/context follow-up clips are ready.
Speaker identity and label provenance remain unresolved. No untouched internal
holdout remains, and this study does not perform a new dev/test evaluation.
An authorized [transcript-context review](docs/speaker_inference_v4.md) estimates
three of the 11 flagged utterances as interviewer speech and eight as participant
speech, including one weak guess. These provisional annotations support the
three human interviewer flags but do not label the full dataset.

## Recommended research workflow: v2 remediation

The current v2 path fixes unknown-confidence handling, balances folds jointly by
label and interview domain, selects neural training duration inside training folds,
and aligns MPNet/audio/facial features to utterance timestamps. It also supplies
measured quality proxies and nested cross-fitted fusion. Historical entrypoints
below remain frozen to reproduce earlier experiments; use the explicit v2
commands in [the remediation report](docs/pipeline_v2_results.md) for new work.

The matched MPNet SVM reference on the revised folds is **57.5% F1**, with
**47.3% precision and 73.9% recall**. Revised neural pooling does not improve it.
The completed aligned quality-aware fusion comparison reaches **53.1% F1** and
only **+0.28 percentage points** mean missing-input F1 over ordinary fusion, so
neither improvement screen passes and the text reference remains retained.
Do not rank this estimate against older folds' 58.1%/58.8% point estimates.
All 140 tests pass. Released labels remain provisional; the supplied manual does
not resolve the label discrepancies or establish participant-only transcript speech.
The local listening worksheet and outstanding validation steps are in the report.

## Current stage

Stages 2-3 implement dataset inventory, strict ID/target validation, split checks,
feature-file/header inspection, chunked participant aggregation, and training-fitted
preprocessing. Stage 4 adds provisional CPU logistic-regression baselines. Stage 5 adds compact neural adaptive fusion without quality inputs.
Stage 6 adds quality-conditioned gating. Stage 7 adds Binary HHO and L1 feature-selection comparisons. Explainability is not implemented yet.

## Setup

Python 3.11+ is required. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/verification --tb=short
```

The current checkout was tested with Python 3.12.10. PyTorch is optional for
Stages 2-4 and required for Stage 5. CPU baselines use scikit-learn. Stage 5 supports CUDA or CPU training.

## Local data

Place the authorized E-DAIC participant `NNN_P.tar.gz` archives and the supplied
`train_split.csv`, `dev_split.csv`, and `test_split.csv` in `data/raw/`.
Optional supplementary label CSVs are inventoried separately. No data is downloaded.
Raw data, manifests, checkpoints, and reports are ignored by Git.

```powershell
.\.venv\Scripts\python.exe -m src.dataset.inventory --config configs/default.yaml
.\.venv\Scripts\python.exe -m src.dataset.features --root data/raw --out data/manifests/feature_inventory.json
```

The first command writes `data/manifests/dataset_manifest.json` and
`reports/data_audit/audit.md`. It returns 1 when structural validation fails
(including missing splits, malformed labels, overlapping participants, or missing
archives), 2 for invalid dataset/output locations, and 0 otherwise. Questionnaire
score discrepancies are recorded for investigation; success does not resolve label
semantics or certify readiness for training.

The second command streams the compressed archives sequentially, so the first
run may take several minutes. It inventories all feature filenames/sizes and
reads bounded CSV headers for named OpenSMILE2/OpenFace2 files. Successful
inspections are cached by dataset root, schema version, archive size, and mtime.
Rerunning resumes/reuses successful inspections; failed inspections are retried.
Do not run multiple writers against the same output. Cache identity is not a
cryptographic integrity check. Outputs must stay outside the raw-data directory.

The feature report includes delimiters, columns, available confidence/success
fields, schema variants, and error counts. It does not verify every row, numeric
validity, temporal alignment, or the usefulness of quality measurements.

## Configuration

Copy `configs/default.yaml` to the ignored `configs/local.yaml`, edit paths, and
explicitly pass `--config configs/local.yaml`. Files are not merged automatically.
YAML paths resolve relative to this repository; explicit CLI paths resolve relative
to the working directory and override YAML paths. Inventory consumes `paths.data_root`,
`paths.manifest`, `paths.audit_report`, `inventory.deep_scan`, and
`inventory.deep_scan_cache`. Label names/schema are currently fixed to the E-DAIC
contract; the labels section documents that contract for future loaders.

## Known dataset limitations

The local release contains 275 archives with 163/56/56 train/dev/test rows.
The official manual describes 219 directories; release provenance needs confirmation.
Local `PHQ_Binary` differs from `PHQ_Score >= 10` in 20 rows, and 2 of 219 matched
questionnaire totals differ from `PHQ_Score`. Preserve supplied labels and investigate
these differences before training. The exact binary-target derivation is unresolved.

The manual is available at https://dcapswoz.ict.usc.edu/wwwedaic/E-DAIC%20Manual.pdf.
Use actual feature headers rather than assuming the manual's dimensionality.
Keep participants together across every future window/recording split. Targets,
questionnaire answers, IDs, and demographics must never enter predictive inputs.
Quality fields are available candidates, not validated reliability measures.

Dependencies have version ranges rather than a lockfile. Final reproducibility
requires recording/pinning the training environment in a later stage.


## Stage 3: participant tables and preprocessing

```powershell
.\.venv\Scripts\python.exe -m src.dataset.tables --config configs/default.yaml
```

Run Stage 2 feature inventory first. The builder verifies inventory root and
archive size/mtime, validates labels/partitions, and reads the selected CSV files
inside each archive without extracting recordings. It uses two workers and
10,000-row chunks by default. Successful participant summaries are cached under
`data/processed/participant_cache/`; settings/schema/archive changes invalidate
those entries. Do not run concurrent builders targeting the same output directory.

The initial representation uses eGeMAPS only for audio (MFCC is deferred) and
OpenFace action units, pose, and gaze for facial behavior. Each descriptor yields
its finite-value mean and population standard deviation. This creates 46 acoustic
and 98 facial features before training-only column removal. Timestamps, frame
numbers, confidence, success, and acoustic name fields are not descriptors.
Numeric coercion failures and infinities become missing measurements. Zero is
retained as a valid numeric value; it is not assumed to mean missing.

Facial rows must have success=1 and finite confidence in [0,1] at or above the
configured cutoff (default 0.8). This is a fixed engineering choice, not a validated
quality threshold, and must not be tuned against test performance. An entirely
unusable modality remains missing and receives availability=0. Missing required
files or corrupt inputs stop the build instead of silently excluding participants.

The quality table contains per-modality availability, invalid descriptor-cell
fraction before filtering, retained-row fraction, and facial mean confidence and
valid-confidence fraction. Confidence means use all valid confidence values,
including rejected rows. Audio retained fraction is 1 for nonempty files because
there is no row filter; invalid cells are handled per descriptor. These are
label-independent quality candidates, not calibrated reliability estimates.

Outputs in `data/processed/`:

- `features.parquet`: raw participant summaries, indexed by Participant_ID.
- `quality.parquet`: separate bounded quality indicators with the same index.
- `labels.parquet`: only PHQ_Binary and official split, indexed by Participant_ID.
- `train_normalized.parquet`, `dev_normalized.parquet`, `test_normalized.parquet`:
  descriptor tables transformed using preprocessing fitted on official train only.
- `preprocessing.json`: training participant IDs, feature order, dropped-column
  reasons, training medians, means, and scales.
- `stage3_report.json`: feature/split counts, quality ranges, missingness, and limits.

All fitted decisions use training participants only: drop entirely missing,
constant, and exactly duplicate columns; fill missing values with training
medians; standardize using the training mean and population standard deviation.
Duplicate detection includes the missing-value pattern. Quality is separate and
is not standardized by this transformer. Index columns are identifiers, never
predictors. Scores, questionnaire answers, and demographics are not exported into
these model inputs.

**For cross-validation and HHO, start from `features.parquet` and place
`ParticipantPreprocessor` inside each fold's sklearn Pipeline. Do not use the
already normalized official-split tables for cross-validation.** Fit only on the
training fold. The transformer supports sklearn cloning and preserves names.

This representation aggregates the whole supplied session. It does not align
acoustic/facial timestamps or isolate interviewee speech from interviewer speech.
Such changes require a documented protocol and a new cache version. There is one
row per participant, with no participant crossing the official splits. Label
semantics remain unresolved. Stage 4 is explicitly provisional reproduction of
the released binary field; confirm the semantics before confirmatory screening claims.


## Stage 4: provisional development baselines

Read [the predeclared protocol](docs/stage4_protocol.md) for the label investigation
and experimental choices. The target is unchanged supplied PHQ_Binary, **not a
verified PHQ-8 >=10 definition**. The discrepancy is documented, not silently fixed.

```powershell
.\.venv\Scripts\python.exe -m src.experiments.baselines --config configs/default.yaml
```

The runner trains audio-only (46 features), facial-only (98), and concatenation
(144) logistic-regression baselines, plus a training-majority reference. It reads
raw `features.parquet`, never the already normalized Stage 3 tables. All fitted
preprocessing and C/class-weight selection occur inside five-fold training-only
CV. Balanced weights are estimated inside each fold. Selected pipelines are
refitted on official train; development data is only predicted/scored afterward.

The threshold is fixed at 0.5. Seeds 42, 137, and 2026 are all reported; 42 is
predeclared as primary. The other seeds vary the selection folds. Their standard
deviation is not a confidence interval or evidence from independent cohorts.
No test prediction or evaluation is performed at this stage.

Each run creates a new ignored `experiments/runs/stage4_<UTC timestamp>/` folder
with protocol, JSON results, complete CV candidate scores and participant folds,
class counts/confusion matrices, development predictions, comparison and seed
variability CSVs, timing, environment versions, input/source hashes, and fitted
pipeline checkpoints. Load joblib checkpoints only from trusted local runs.

The first primary-seed development results were:

| Baseline | Precision | Recall | F1 | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Majority | 0.000 | 0.000 | 0.000 | 0.500 | 0.786 |
| Audio | 0.290 | 0.750 | 0.419 | 0.574 | 0.554 |
| Facial | 0.207 | 0.500 | 0.293 | 0.479 | 0.482 |
| Concatenation | 0.258 | 0.667 | 0.372 | 0.549 | 0.518 |

Development has 44 class-0 and 12 class-1 participants. Majority accuracy is high
because it predicts class 0 for everyone; positive-class recall and F1 are zero.
Across the three CV seeds, F1 mean +/- population SD was 0.362 +/- 0.079 (audio),
0.256 +/- 0.052 (facial), and 0.240 +/- 0.095 (concatenation). These are preliminary
development results and do not show a fusion benefit. No parameters were changed
in response to these scores. They are not comparable to the reference paper's
headline scores without reconciling dataset, target, and evaluation protocol.

Stage 4 does not include adaptive fusion; see Stage 5 below. The final held-out test
remains reserved for a frozen experimental protocol.


## Stage 5: adaptive fusion without explicit quality signals

The [Stage 5 protocol](docs/stage5_protocol.md) fixes the architecture and selection
rules before evaluation. Separate 16-dimensional acoustic and facial encoders feed
a two-weight softmax gate. A weighted representation feeds a binary prediction
head. Quality indicators are not inputs to this baseline. Gate weights are learned
mixing coefficients, not causal feature attributions or calibrated reliability.

Install the pinned neural dependency in the project virtual environment. For the
RTX 4050, use the CUDA 12.8 wheel from the official PyTorch index:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/stage5-reproduce --tb=short
.\.venv\Scripts\python.exe -m src.experiments.adaptive --config configs/default.yaml
```

For a CPU-only machine, install the same version from
`https://download.pytorch.org/whl/cpu`. The `neural` optional dependency records the
version; choose the appropriate wheel index explicitly. These commands install
software only; they do not download data or pretrained weights.

The runner uses training-only five-fold CV to choose epoch count (50 or 100) and
weight decay (0.001 or 0.01), with fixed hidden size, dropout and learning rate.
Every fold fits its own preprocessing and class weight. The official development
set does not select epochs, parameters, seeds, thresholds or checkpoints. The
threshold stays at 0.5. All three predeclared seeds are reported; 42 is primary.
The held-out test is not evaluated. Results remain provisional against the
unreconciled released PHQ_Binary labels.

Outputs are written to a new ignored `experiments/runs/stage5_<UTC timestamp>/`
directory: protocol, metrics, CV candidates and participant membership, losses,
participant-level development predictions and gate weights, timing and peak CUDA
tensor allocation, source/data hashes, environment details, and `.pt` checkpoints.
Checkpoints contain model weights and fitted preprocessing state; restore via
`predict_checkpoint` (loads with `weights_only=True`). The runner verifies restored
predictions before reporting success. Neural tests are skipped when PyTorch is
absent; this must not be reported as successful neural validation.

Deterministic settings and seeds are recorded, but exact numerical equivalence
across hardware or library versions is not guaranteed. Peak CUDA allocation is
not total process RAM, GPU reservation, or total system GPU use. Full-batch training
is small here because there is only one feature row per participant.


First Stage 5 development results (same data hashes and CV folds as Stage 4):

| Seed | Precision | Recall | F1 | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| 42 (primary) | 0.273 | 0.500 | 0.353 | 0.614 | 0.607 |
| 137 | 0.241 | 0.583 | 0.341 | 0.542 | 0.518 |
| 2026 | 0.273 | 0.500 | 0.353 | 0.534 | 0.607 |

Across-seed F1 mean is 0.349 with population SD 0.0054. The primary F1 does not
improve on the audio-only or concatenation logistic baselines. A higher primary
ROC-AUC does not establish general superiority. A matched neural non-gated ablation
is still needed to isolate any benefit from gating. All three checkpoints were
verified; the primary CUDA training run also reproduced weights and predictions
exactly on this environment. No settings were changed after viewing these results.
The network has 2,635 parameters. Stage 6 adds the explicit quality extension described below.


## Stage 6: quality-conditioned fusion

See [the predeclared Stage 6 protocol](docs/stage6_protocol.md). This extends the
Stage 5 gate with a zero-initialized linear projection of eight approved quality
indicators. Each indicator is centered using its fitting fold's mean. The existing
encoders and prediction head receive no quality fields. The quality projection
adds 64 parameters (2,699 total with the current 144 descriptors).

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/stage6-reproduce --tb=short
.\.venv\Scripts\python.exe -m src.experiments.adaptive --quality-aware --config configs/default.yaml
```

Without `--quality-aware`, the CLI continues to run Stage 5. Stage 6 reads the
`reliability` config section and the separate `quality.parquet` table. It rejects
unexpected fields, nonfinite/out-of-range indicators, misaligned participant IDs,
and both modalities being unavailable. A missing modality is masked out of both
the gate and fused representation. This rule is tested synthetically; the current
corpus does not contain naturally missing modalities.

Only facial retention and confidence vary in the current data. Six constant
quality indicators become zero after training-only centering and supply no
observed variation. Thus this experiment primarily tests the two facial-quality
signals, not general acoustic reliability. Learned weights are not calibrated
reliability scores, and increasing confidence is not guaranteed to increase a
modality's gate weight.

Stage 6 preserves Stage 5's selection grid, seeds, folds, fixed 0.5 threshold and
provisional released-label policy. It uses no development labels or quality values
to fit preprocessing, quality means, or select settings. Outputs go to a new ignored
`experiments/runs/stage6_<UTC timestamp>/` directory. Checkpoints include the fixed
quality schema and training quality means. Pass the aligned raw quality DataFrame
as `quality=` to `predict_checkpoint` when restoring a Stage 6 model.

All results are development-only. Do not claim a gate benefit without examining
all seeds and appropriate ablations. HHO, explanation faithfulness, broader
robustness/ablation studies, and final held-out evaluation remain later stages.


First Stage 6 development results:

| Seed | Precision | Recall | F1 | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| 42 (primary) | 0.273 | 0.500 | 0.353 | 0.614 | 0.607 |
| 137 | 0.241 | 0.583 | 0.341 | 0.542 | 0.518 |
| 2026 | 0.273 | 0.500 | 0.353 | 0.532 | 0.607 |

Mean F1 is 0.349 (population SD 0.0054), unchanged from Stage 5. Mean ROC-AUC
is 0.5625 versus Stage 5's 0.5631. Quality weights learned nonzero values and
changed predicted probabilities, but this experiment shows no F1 improvement.
Six constant indicators had zero learned projection weights; the observed quality
signal comes from facial confidence and retention. This negative result should
be retained rather than changing settings to force an improvement.

All 68 tests passed. All three checkpoints reproduced predictions; repeating the
primary CUDA training run reproduced weights, losses and predictions exactly.
Previous Stage 5 checkpoints still reproduce their predictions with the updated
runner. Input hashes, CV folds and candidate grids matched Stage 5. Stage 7 has
not started, and the official test remains unevaluated.


## Stage 7: Binary HHO and L1 feature-selection comparisons

The [Stage 7 protocol](docs/stage7_protocol.md) specifies the HHO equations and
binary adaptation, budget, fitness model, leakage safeguards, and limitations.
The optimizer searches binary subsets of the 144 descriptors, preserving at least
one usable descriptor from each modality. Quality indicators are not selected.

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/stage7-reproduce --tb=short
.\.venv\Scripts\python.exe -m src.experiments.selection --config configs/default.yaml
```

HHO uses eight agents and eight iterations per seed. Its configurable objective is
`alpha * (1 - mean_cv_f1) + beta * selected_fraction`, with alpha=0.95, beta=0.05.
The score comes from five-fold training-only CV of a fixed balanced logistic
regression (C=0.1). Every mask is preprocessed separately within each fitting fold.
Identical masks reuse fitness within the run. Invalid masks receive a penalty.
This small search budget does not establish optimality or algorithmic superiority.

The conventional alternative uses fixed-C L1 logistic regression on training only.
A modality with no nonzero coefficients receives a reported deterministic fallback
feature. L1 is not tuned on development data. Its full-training-selected mask is
not used to claim an unbiased CV score.

For each seed, train adaptive and quality-aware fusion using HHO, L1, and all
features. Downstream neural settings are fixed (16 hidden units, dropout 0.2,
50 epochs, AdamW lr=0.001, weight_decay=0.001). There is no second CV search with
an already selected mask. Future downstream hyperparameter tuning requires
nesting the selector inside each tuning fold. These matched all-feature controls
are the relevant comparison; Stage 5/6 used a different epoch-selection protocol.

Each ignored `experiments/runs/stage7_<UTC timestamp>/` folder contains selected
feature names/modality mappings, binary masks, every unique candidate's fold scores,
convergence CSVs, optimization branch counts, CV participants, checkpoints,
development predictions/metrics, source/data hashes and runtime tables. Selection
is performed once per selector/seed and reused by both fusion types; do not double
count it when summing costs. HHO surrogate fitness need not transfer to neural
models. No test evaluation or new target construction occurs.


First Stage 7 results (matched fixed-training controls, three seeds):

| Selector | Selected descriptors | Mean development F1 | F1 population SD |
|---|---|---:|---:|
| All features | 144 / 144 / 144 | 0.327 | 0.029 |
| L1 | 11 / 11 / 11 | 0.328 | 0.038 |
| Binary HHO | 29 / 20 / 80 | 0.318 | 0.106 |

Adaptive and quality-aware models had the same classification metrics in this
run. L1 retained far fewer descriptors with similar mean F1, but this small result
does not establish superiority. HHO did not improve mean F1 and varied markedly
across seeds. HHO selection took 20.65 / 49.84 / 106.35 seconds, versus approximately
0.14 / 0.45 / 0.47 seconds for L1. Feature reduction did not demonstrate a reduction
in total computation cost. No settings or seeds were changed to improve results.

All 72 tests passed without warnings. All 18 checkpoints were verified. HHO search
trajectories replayed exactly from cached mask fitness, selected-mask CV fitness
was recomputed independently, L1 masks reproduced, and both primary HHO neural
models reproduced weights and predictions on CUDA. Full convergence histories,
selected feature names/modalities and timing are saved in the run artifacts.
Stage 8 explanation and faithfulness work is complete; see below.

## Stage 8: explanations and faithfulness checks

Explain every frozen Stage 7 model on the development participants:

```powershell
.venv\Scripts\python.exe -m src.experiments.explain --source-run experiments/runs/stage7_20261028T142146453803Z
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/stage8-check --tb=short
```

This uses Integrated Gradients on the positive-class logit with training-mean
descriptors as the reference. It supports both fusion models, preserves each
checkpoint's selected feature names and preprocessing, and keeps participant
quality fixed. No retraining or test evaluation occurs. The explicit source-run
argument avoids accidentally explaining whichever run happened to finish last.
See [the protocol](docs/stage8_protocol.md) and [results](docs/stage8_results.md).

Each ignored `experiments/runs/stage8_<UTC timestamp>/` folder includes signed
participant attribution Parquet files, global feature CSVs, individual readable
explanations, audio/facial contributions, numerical completeness diagnostics,
perturbation results, quality-mean replacement sensitivity, ranking stability,
input/checkpoint/source hashes and a report. Exact names preserve descriptor
identity: `audio__`/`facial__` identify modality and `__mean`/`__std` identify the
whole-interview summary. They do not identify moments in the interview or symptoms.

In the completed run, highly ranked features produced greater mean output changes
than random subsets in all 54 model/fraction comparisons (18 models at 10%, 25%,
50%). All 1,008 participant-model explanations met the predeclared numerical
completeness tolerance. However, rankings varied across seeds, particularly with
HHO; this does not establish stable biomarkers, causality, clinical validity or
better prediction accuracy. L1 top-10 overlap is inflated by selecting only 11
features. Quality/adaptive rankings were nearly identical for matched seeds.
The full suite passed 78 tests at Stage 8. Stage 9 is now complete below.

## Stage 9: ablations and provisional final evaluation

The user authorized final evaluation on unchanged released labels. Their meaning
remains unresolved; no clinical or benchmark-superiority claims follow. The test
partition has now been evaluated. Earlier stage statements about an unevaluated
test describe the status at those earlier stages.

New equal-weight controls remove the adaptive gate while preserving the same
encoders/head initialization and training settings. Nine new controls plus all
27 existing Stage 4/7 models and a majority reference form the fixed 37-model
roster. All fit only official training participants. No development+training
refit or test-based model selection occurred. See [protocol](docs/stage9_protocol.md)
and [completion results](docs/stage9_results.md).

Install the optional plot dependency, prepare a new frozen run, then evaluate it:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[plots]"
.venv\Scripts\python.exe -m src.experiments.final prepare --stage4 experiments/runs/stage4_20261027T154348588155Z --stage7 experiments/runs/stage7_20261028T142146453803Z
# Replace <new-run> with the directory printed by prepare.
.venv\Scripts\python.exe -m src.experiments.final evaluate --run experiments/runs/<new-run>
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/stage9-check --tb=short
```

Completed run: `experiments/runs/stage9_20261028T180907085552Z`. Its evaluation
folder contains all metrics, paired ablations, primary-seed bootstrap intervals,
seed summaries, predictions and PNG/SVG figures. The frozen manifest records
provenance and historical runtime metadata. Existing evaluations cannot be
overwritten. Results and restricted participant records remain ignored locally.

Observed test F1 means: audio-only logistic 0.563; all-feature adaptive/quality
0.538; HHO adaptive/quality 0.474; L1 adaptive/quality 0.468. Quality inputs did not
change F1; gate effects varied by seed. All primary paired F1 difference intervals
included zero. The proposed additions did not demonstrate the intended benefit.
See the full report rather than choosing a favorable seed. All 83 tests passed;
all 74 metric rows and confusion matrices were independently verified. Stage 10
is the only remaining stage.

## Follow-up audio improvement study

The user requested a bounded audio-focused improvement study after Stage 9.
This is separate from the pending Stage 10 audit. It uses only the 163 official
training participants; development/test predictions are not regenerated.

```powershell
.venv\Scripts\python.exe -m src.dataset.audio_summary
.venv\Scripts\python.exe -m src.experiments.audio_study
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/audio-study-check --tb=short
```

The extractor adds quantiles and temporal-variation summaries to the original
audio averages, using bounded archive reads and a separate cache. The experiment
compares logistic regression, linear SVM and RBF SVM, with default versus
inner-selected cutoffs, in matched five-fold outer/three-fold inner validation
repeated for three seeds. The exact original baseline recipe is also rerun inside
the outer folds. Every preprocessing fit and threshold choice stays inside its
fitting partition. See [protocol](docs/audio_study_protocol.md) and
[full findings](docs/audio_study_results.md).

Completed run: `experiments/runs/audio_study_20261028T181901228265Z`.
Richer logistic with a selected cutoff achieved mean nested-CV F1 0.371 versus
0.324 for the original recipe on the same folds, but mostly by increasing recall
with many false positives. An always-positive rule has F1 0.370 on this cohort;
the headline increase alone is not convincing evidence of a useful model.
Rich linear SVM at the default cutoff showed a modest improvement in mean F1
(0.342) and mean fold AUC (0.586), with lower recall than the original recipe.
No independent confirmation or model superiority is claimed. These numbers are
not directly comparable to the previous test scores. All 89 tests passed and all
195 outer checkpoints were verified. No final model was selected or deployed.

## Label and speech feasibility investigation

The follow-up investigation on 2026-10-29 checked all 275 audio/transcript pairs
and reconciled the supplied label tables. It trained no models and changed no
labels or previous predictions. See [findings and sources](docs/label_speech_investigation.md).

```powershell
.venv\Scripts\python.exe -m src.dataset.label_speech_audit
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/label-speech-check --tb=short
```

The 20 binary/score disagreements are repeated in the supplied master labels,
while all master item sums match the split scores. There is independent published
corroboration, but no verified official correction notice. Transcripts have
timestamps but no speaker column. Training transcript intervals cover a median
60.0% of feature frames; this is not a verified participant-speech percentage.
Two training files have substantial feature/WAV duration mismatches. Ten training
WAVs are stereo, but their channels are identical or nearly perfectly correlated,
so channel-based speaker separation is not established. The train/test interview
conditions also differ substantially. Validate timing and speaker identity before
a transcript-covered feature experiment; do not silently shift timestamps or
relabel participants. All 93 tests pass. Stage 10 remains pending.

## Transcript coverage and facial fusion pilot

Completed 2026-10-29 after the user requested further improvement. The experiment
quarantines the two known timing anomalies identically in all conditions and uses
only 161 official training participants, retaining all 37 released positives.
It compares full/transcript-covered audio, facial-only, and audio-plus-facial
summaries using logistic regression and linear SVM in matched nested folds.
Transcript coverage is not verified participant-only speech.

```powershell
.venv\Scripts\python.exe -m src.dataset.segmented_audio
.venv\Scripts\python.exe -m src.experiments.segmentation_study
```

Transcript filtering hurt mean F1 and AUC. Adding facial features to full audio
improved balanced accuracy but traded recall for specificity; full-audio/facial
linear SVM reached mean F1 .371 and balanced accuracy .581. This is not a strong
or independently confirmed improvement. All 97 tests pass; 150 saved models and
all 5,796 predictions were checked. No new development/test evaluation, relabeling
or deployment. See [protocol](docs/segmentation_study_protocol.md) and
[all results and limitations](docs/segmentation_study_results.md). Stage 10 remains
pending.

## MPNet undersampling follow-up (2026-11-03)

Training-only random undersampling at 1:1 and 2:1 negative-to-positive ratios,
including seven-model ensembles, did not beat the existing MPNet reference.
The procedure selected within training folds reached 55.1% F1 versus 58.1% for
the reference. Evaluation participants were unchanged. All 122 tests passed;
105 saved bundles and 3,381 predictions were independently checked. See the
[complete results](docs/undersampling_results.md). The current research model
is retained; no new development/test evaluation was performed.

## Precision experiment series (2026-11-03)

Four sequential experiments tested constrained threshold selection, class
penalties, nonlinear SVM and SVM/logistic agreement. Class penalties reached
49.2% precision / 58.8% F1; agreement reached 49.3% precision / 58.2% F1, compared
with the retained reference's 47.8% precision / 58.1% F1. None met the declared
five-point precision improvement target with recall/F1 safeguards. All 124 tests
passed; 120 bundles and 3,864 predictions were independently checked. These are
training-CV selection policies, not newly deployed models or test scores.
See the [experiment list](docs/precision_protocol.md) and
[complete comparison](docs/precision_results.md).

## Learned pooling and encoder ensemble (2026-11-03)

Two structural follow-ups did not improve the reference: inner-selected learned
utterance pooling reached 45.2% F1, and MPNet/MiniLM score blending reached 54.5%,
versus 58.1% for the retained MPNet SVM. The previous class-penalty experiment's
58.8% remains the highest observed mean, but its small gain did not qualify for
promotion. All 129 tests passed; 75 saved bundles and 2,898 predictions were
checked. See [results, protocols and limitations](docs/representation_followup_results.md).
No new development/test evaluation or deployment was performed.

## Pretrained audio and temporal facial pilot

The 2026-10-29 follow-up tested supplied DenseNet audio embeddings and richer
OpenFace temporal summaries on the same 161-person training cohort. All 14 fitted
conditions and two constant references are reported. No candidate passed the
predeclared improvement rule, so the existing reference remains unchanged.
This was not a Wav2Vec2, text-encoder or full sequence-model experiment.
See [protocol](docs/representation_study_protocol.md) and
[results](docs/representation_study_results.md). All 102 tests pass; all 210 saved
pipelines and 7,728 predictions were verified, including reproduction of the prior
controls. No development/test reevaluation or relabeling was performed.

## Frozen Wav2Vec2 pilot

Verified 2026-10-30. A pinned speech encoder extracted 10,282 windows locally on
the RTX 4050, with 0.780 GiB peak allocated GPU memory. The same 161-person nested
training comparison tested speech features alone and with facial summaries.
None passed the predeclared improvement rule. Audio-only logistic F1 increased
from .326 to .338, but AUC fell from .585 to .529; speech/facial variants did not
improve their matched controls. The existing reference remains unchanged.

See [protocol](docs/speech_encoder_protocol.md) and
[complete results](docs/speech_encoder_results.md). All 105 tests pass; all 150
saved models and 5,796 predictions were checked. Encoder replay reproduced sampled
embeddings exactly. No recordings were uploaded, no labels changed, and no new
development/test predictions were generated. Stage 10 remains pending.

## Text, temporal, and reliability follow-up

Verified 2026-10-31 on the same 161-person training cohort and matching folds.
Screened, locally encoded transcripts provide the first substantial improvement
in these follow-ups: text-only SVM reaches 52.8% F1 and 79.4% mean fold AUC,
versus 37.1% F1 and 56.9% AUC for original audio-plus-face SVM. Text-only logistic
is similar (52.5% F1). Adding original audio/facial features reduces performance
relative to text alone, although it improves on the old audio/facial controls.

The small temporal model reaches 37.7% F1 versus 35.1% for its pointwise control,
below the predeclared improvement threshold. Quality-aware temporal fusion
reaches 36.8% clean F1 and only +0.9 points mean stressed F1 versus ordinary
temporal fusion, below its +5-point target. Controlled feature corruption uses
known severity; it is not a validated real-world quality detector.

See the [frozen protocol](docs/text_temporal_protocol.md),
[full comparison and reproduction commands](docs/text_temporal_results.md), and
[unsent provider inquiry](docs/label_provider_inquiry.md). All 111 tests pass;
240 saved classifiers and 18,837 predictions were independently checked. The
text encoder replay matched exactly. The extra facial timing issue is handled
by masking two facial sequences, preserving the participant cohort and labels.
These are training-CV improvements, not new test results. Stage 10 remains pending.

## Current research reference: MPNet text

Verified 2026-11-02. Screened MPNet text with PCA32 and a linear SVM reaches
**58.1% F1, 73.9% recall, 75.6% accuracy and 80.4% AUC** on the same 161-person
training comparison. F1 improves by 5.2 percentage points over the previous
MiniLM reference, with gains in all three repeats. These are training-CV
estimates, not new test scores.

Four follow-up studies checked text refinement, lexical features, MPNet and
compression/threshold choices. Only the original MPNet SVM recipe cleared the
predeclared improvement screen. Across them, 600 saved model bundles and 23,184
predictions were independently checked; all 119 tests pass. A separately verified
checkpoint fitted on all training participants is saved locally for research
reuse; its refit does not provide another performance estimate.

See the [full comparison and checkpoint details](docs/research_progress_20261102.md).
The strongest reference is still text-only: reliability-aware fusion has not
shown its intended advantage. Performance on the small AI-interview subgroup
remains weaker, and the provisional labels remain unresolved. Stage 10 remains
pending.

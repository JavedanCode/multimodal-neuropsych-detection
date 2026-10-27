# Multimodal depression screening

Research implementation for acoustic and facial behavioral features from E-DAIC.
This is a screening research project, not a clinical diagnostic system. R-SAF is
a provisional method name; novelty and performance have not been established.

## Current stage

Stages 2-3 implement dataset inventory, participant aggregation, and training-fitted
preprocessing. Classifiers are not implemented yet.

## Setup

Python 3.11+ is required. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/verification --tb=short
```

The current checkout was tested with Python 3.12.10. PyTorch is not required
for this stage; no GPU training is implemented yet.

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
semantics remain unresolved and need investigation before classifier experiments.

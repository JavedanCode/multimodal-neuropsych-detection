# Multimodal depression screening

Research implementation for acoustic and facial behavioral features from E-DAIC.
This is a screening research project, not a clinical diagnostic system. R-SAF is
a provisional method name; novelty and performance have not been established.

## Current stage

Stage 2 implements dataset inventory, strict label validation, split checks,
and feature-file/header inspection. Preprocessing and classifiers are not implemented yet.

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

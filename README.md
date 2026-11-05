# Multimodal depression screening

Python research pipeline for E-DAIC acoustic, facial and transcript features.
The research question is whether measured input quality improves audio/facial
fusion. Text is a reference and an explicitly tested fallback. This is research,
not a clinical diagnostic system.

## Start here

- [Project guide](docs/project_guide.md): where the code lives and what it does.
- [Experiment history](docs/experiment_history.md): findings and limitations.
- [Aggregate results](docs/results.csv): summary metrics with run identifiers.
- [Cleanup record](docs/cleanup.md): retained studies and historical reproduction.

## Structure

```text
src/
  dataset/       Read, validate, summarize and align inputs
  models/        Prediction models, fusion and preprocessing
  training/      Shared fitting, folds, checkpoints and provenance
  selection/     Binary HHO and L1 feature selection
  evaluation/    Metrics, explanations and independent verification
  experiments/   Four maintained study runners
configs/         Default settings and historical-source revision
tests/           Automated checks
docs/            Guide, experiment log, aggregate results and methods
data/            Local raw data and prepared caches (not committed)
models/          Local pretrained/reference checkpoints (not committed)
experiments/runs/ Local run artifacts and predictions (not committed)
reports/         Generated local reports (not committed)
```

## Maintained studies

| Command | Purpose | Recorded training-CV result |
|---|---|---|
| `reference` | Retained MPNet text classifier | 57.5% F1 |
| `text-candidates` | Threshold, topic and conservative fusion comparisons | Threshold candidate: 58.5% F1; promotion screen failed |
| `audio-visual` | Matched original/aligned/richer features and fusion controls | Original concatenation: 37.2% F1; richer face: 37.5% |
| `fallback` | Acoustic/facial primary with explicit text fallback | Quality primary: 33.3% F1; fallback with both primary inputs absent: 55.4% |

These are exploratory estimates on reused training participants, not fresh test
results. The reliability-aware advantage remains unproven. Controls and unfavorable
comparisons remain within each study so its result can be interpreted honestly.

## Run

Use PowerShell from this folder. The existing `.venv` contains the local setup.
For a new environment, install with `python -m pip install -e '.[dev,neural,speech]'`.
The authorized dataset and cached encoders must be provided locally; E-DAIC is not
downloaded by these commands. See the guide for cache preparation dependencies.

```powershell
.venv\Scripts\python.exe -m src --help
.venv\Scripts\python.exe -m src run reference
.venv\Scripts\python.exe -m src run audio-visual
```

Each study prints its output folder. Reference-dependent studies require a baseline
run folder. Verification accepts the study name and saved run folder:

```powershell
.venv\Scripts\python.exe -m src run fallback --reference-run experiments/runs/baseline_v2_20261103T165509103723Z
.venv\Scripts\python.exe -m src run text-candidates --reference-run experiments/runs/baseline_v2_20261103T165509103723Z
.venv\Scripts\python.exe -m src verify audio-visual --run experiments/runs/branch_audit_v6_20261104T202915295134Z
```

## Check changes

```powershell
.venv\Scripts\python.exe -m pytest -o addopts='' --basetemp=.pytest_cache/verification --tb=short
.venv\Scripts\python.exe -m black --workers 1 --check src tests
```

Keep supplied labels unchanged, participants together across folds, and fitting
statistics confined to each training fold. Historical test results have already
been inspected; they are not an untouched holdout. Dataset contents, participant
predictions, local environments and model binaries stay out of Git.

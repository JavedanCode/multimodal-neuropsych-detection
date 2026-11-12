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

## Audio preprocessing follow-up (2026-11-08)

The matched raw/DC-removal/50 Hz high-pass comparison did not improve the retained
audio/facial control. Original features: **62.9% accuracy, 37.2% F1**; newly
extracted untreated audio + face: **61.3%, 34.3%**; high-pass audio + face:
**62.1%, 34.3%**. All 135 models and 5,313 predictions were verified; all 112 tests
pass. Keep the original supplied acoustic features. Full methods and all results
are recorded at the end of the [experiment history](docs/experiment_history.md).

## Run

The final sparse/late-fusion series is complete. Sparse concatenation reached
37.6% F1 but lower accuracy than the original control; weighted fusion reached
35.4% F1 and the matched quality adjustment added less than 0.1 percentage points.
No replacement passed the predeclared screen against the retained control.
See the [paper-writer log and supporting files](docs/paper_logs/README.md) for all
results, exact methods, selected settings and reproducibility checks.

```powershell
.venv\Scripts\python.exe -m src run audio-visual --sparse-fusion
```

This uses the existing original feature caches and does not repeat waveform
extraction. Its [frozen protocol](docs/methods/sparse_fusion.md) defines the finite
candidate set and stopping rule.

A code review then found two defects: unvoiced openSMILE placeholders were
averaged into the voicing-dependent audio summaries, and late fusion averaged
branch probabilities on unequal scales. The [corrected run](docs/methods/corrected_fusion.md)
fixes both. Voiced-only audio + face reached **63.6% accuracy, 36.7% F1** and fold
AUC 0.625 (original 0.600); scaled weighted fusion reached 34.8% F1. No candidate
passed the screen, so the original control is retained. Build the voiced summaries
once (reads every training archive), then run:

```powershell
.venv\Scripts\python.exe -c "from pathlib import Path; from src.dataset.voiced_audio import build; build(Path('.').resolve())"
.venv\Scripts\python.exe -m src run audio-visual --corrected
```

A final predeclared comparison ([protocol](docs/methods/score_functionals_domain.md))
tested the released PHQ score as a training target, the official 88 eGeMAPSv02
functionals and interview-domain scaling. The best variant, a PHQ-score ridge on
voiced audio + face, reached **57.6% accuracy, 38.0% F1** with higher recall but
slightly lower balanced accuracy; no variant passed the screen. The audio/face
series is finished: keep the original control. The
[paper-writer log](docs/paper_logs/README.md) covers all three runs.

```powershell
.venv\Scripts\python.exe -c "from pathlib import Path; from src.dataset.egemaps_functionals import build; build(Path('.').resolve())"
.venv\Scripts\python.exe -m src run audio-visual --score-study
```

A further run ([protocol](docs/methods/participant_pooled.md)) tested label-free
removal of interviewer speech, pooling the development participants into
cross-validation, text-assisted fusion and abstention. Segment-based voiced audio +
face is the best audio/face result so far (**65.2% accuracy, 39.1% F1**, AUC 0.631)
but does not pass the screen; pooling development participants lowered scores.

```powershell
.venv\Scripts\python.exe -c "from pathlib import Path; from src.dataset.participant_speech import build; build(Path('.').resolve())"
.venv\Scripts\python.exe -m src run audio-visual --pooled-study
```

Use PowerShell from this folder. The existing `.venv` contains the local setup.
For a new environment, install with `python -m pip install -e '.[dev,neural,speech]'`.
The authorized dataset and cached encoders must be provided locally; E-DAIC is not
downloaded by these commands. See the guide for cache preparation dependencies.

```powershell
.venv\Scripts\python.exe -m src --help
.venv\Scripts\python.exe -m src run reference
.venv\Scripts\python.exe -m src run audio-visual
```

The controlled audio-cleanup follow-up is part of the audio/visual runner:

```powershell
.venv\Scripts\python.exe -m pip install -e '.[acoustic]'
.venv\Scripts\python.exe -m src run audio-visual --preprocess-audio
```

It extracts matched untreated, DC-removed and gently high-pass-filtered audio
features, then compares them with fixed facial features and the same folds.
Extraction is resumable and keeps raw recordings unchanged. See the
[fixed comparison protocol](docs/methods/audio_preprocessing.md).

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

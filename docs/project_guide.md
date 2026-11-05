# Project guide

The pipeline reads recordings and supplied features, creates numerical summaries,
fits models, and writes comparisons. `src/__main__.py` provides the single command
entry point: `python -m src`.

## Data preparation: src/dataset

| Files | Responsibility |
|---|---|
| `inventory.py`, `features.py` | Validate archives, labels/splits and feature schemas. |
| `tables.py`, `preprocessing.py`, `quality.py` | Participant summaries, training-fitted cleanup and quality fields. |
| `audio_decode.py`, `audio_summary.py` | Decode recordings and construct richer acoustic summaries. |
| `representations.py` | Read supplied deep audio descriptors and summarize facial dynamics. |
| `lexical.py`, `text_temporal.py`, `mpnet.py` | Screen transcripts and create cached text representations. |
| `contracts.py`, `alignment.py` | Validate time intervals, align modalities and measure quality. |
| `topics.py` | Fixed topic summaries for the text follow-up. |
| `*_inputs.py` | Load caches and check labels, participant order and provenance. |
| `speaker_review.py`, `speaker_context.py`, `review_annotations.py`, `transcript_context.py` | Listening/context worksheets and review annotations. |
| `label_speech_audit.py` | Investigate labels and transcript assumptions. |

Versioned names under `data/processed` identify existing cache formats. They remain
unchanged because saved manifests refer to them. Some caches depend on earlier
extraction steps; original end-to-end recipes remain in the history snapshot
listed in `cleanup.md`. The maintained studies consume the verified existing
caches. This cleanup does not introduce a single-command rebuild from empty data.

Basic preparation commands:

```powershell
.venv\Scripts\python.exe -m src.dataset.inventory --config configs/default.yaml
.venv\Scripts\python.exe -m src.dataset.features --root data/raw --out data/manifests/feature_inventory.json
.venv\Scripts\python.exe -m src.dataset.tables --config configs/default.yaml
```

Read each preparation module's arguments before rebuilding expensive caches. Later
builders may require predecessor caches. Do not run concurrent writers to one cache.

## Prediction models: src/models

`text_preprocessing.py` prepares embeddings for the retained classifier.
`fusion.py` implements matched audio/facial branches and equal, adaptive and
quality-conditioned fusion. `adaptive.py` and `reliability.py` hold neural
components also used by feature selection and explainability.

`stacking.py` builds training predictions without fitting a base model on the same
people being predicted. `fallback.py` decides whether to use primary predictions
or text fallback. `text_candidates.py` supplies threshold policies and conservative
quality adjustments. Other preprocessing classes support the input families.

Root `models/` contains model **files**; `src/models/` contains model **code**.
Only load trusted local checkpoints: Python pickle can execute code.

## Fitting: src/training and src/selection

`training/splits.py` balances folds by label/interview domain while keeping people
separate. `classical.py` owns classifier definitions and bounded inner selection.
`neural.py` and `feature_selection.py` provide shared neural fitting.
`representations.py` fits preprocessing inside representation comparisons.

`validation.py` validates tables and hashes files. `provenance.py` distinguishes
current artifacts from archived source. `checkpoints.py` handles known historical
model-module names after the cleanup.

`selection/hho.py` searches feature subsets. `selection/fitness.py` scores subsets
on training folds and provides L1 selection. Retaining these components does not
claim that they outperform the reference.

## Studies and evaluation

`src/experiments` contains four runners: `reference.py`, `text_candidates.py`,
`audio_visual.py` and `fallback.py`. They assemble shared components, fit folds and
save timestamped outputs. The package marker `__init__.py` is normal Python structure.

`src/evaluation` owns metrics, prediction summaries, paired comparisons, uncertainty,
explainability and independent verifiers. Some verifiers replay inner selection and
neural training. New checks write `verification_current.json`, leaving the original
verification report intact.

Typical outputs include `summary.csv`, `repeat_metrics.csv`, `domain_metrics.csv`,
`predictions.parquet`, fold decisions, `manifest.json` and a `models/` subfolder.
Only aggregate summary metrics are copied to `docs/results.csv` for Git.

## Documentation and generated files

`docs/methods` contains the retained protocols. `experiment_history.md` collects
research notes, including unsuccessful methods. Historical paths/commands in that
log refer to the recorded Git snapshot, not the reorganized working tree.

`.venv`, `.pytest_cache`, `__pycache__`, editor settings and package metadata are
local tooling and are ignored by Git. Empty scaffold folders have been removed.

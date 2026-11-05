# Pipeline remediation results — completed 2026-11-04

This is an exploratory training-cohort comparison, with unchanged released binary
labels. Historical experiments and their source files remain frozen. The v2
entrypoints below are the recommended path for further work. No new official
development/test evaluation, production deployment, or provider message occurred.

## What was corrected

| Finding | Implemented response | Remaining limit |
|---|---|---|
| Unknown ASR confidence previously became perfect confidence | v2 uses zero plus an explicit known flag | Confidence is not calibrated diagnostic reliability |
| Label-only folds sometimes omitted the AI subgroup | Joint domain/label stratification, including inner selection | Only 18 AI participants, five positive, in retained training data |
| Neural fitting performance diverged from validation | Inner-selected epochs and regularization, saved learning curves, matched initialization | Neural models still underperform the text SVM |
| Coarse temporal bins discarded utterance boundaries | Rebuilt all 14,724 screened utterances using actual feature timestamps | Fusion currently pools these aligned summaries; it is not a full sequence model |
| Transcript row order was not always chronological | Stable timestamp sorting moves embeddings and metadata together | Timestamp agreement alone cannot establish speaker identity |
| Most original quality fields were constant; stress tests exposed severity | Measured waveform/tracker/ASR proxies and observable availability, no severity input | These are not validated SNR, diarization, or reliability probabilities |
| Temporal branch used weaker text representations | Aligned cache retains MPNet embeddings; fusion retains the original MPNet mean | Text remains a fixed encoder |
| Stacking could learn from optimistic fitted margins | Base-model cross-fitting nested within inner and outer training partitions | Repeated adaptive work on the same cohort still limits generalization claims |
| New cache metadata key was accidentally overwritten during sorting | Separate field variable and independent cache-key verification | Rebuilt before fusion fitting |

Two facial streams remain quarantined for endpoint timing mismatch. The participant
cohort is unchanged. The original screening policy is retained before sorting so
that transcript-to-embedding correspondence remains reproducible.

## Manual and unresolved data provenance

The supplied four-page `data/raw/E-DAIC Manual.pdf` was inspected. SHA256:
`eff6958ac918492b20cae9506e3f69e2a87f0fd5ce9cf1382ca5da16b634e706`.
It confirms mixed WoZ/AI train and development interviews and AI-only test
interviews. It lists PHQ_Binary and PHQ_Score but does not specify how to reconcile
their disagreements, nor guarantee participant-only transcript speech. Its stated
219 directories differ from this release's 275. It therefore does not resolve the
20 binary-versus-score discrepancies, including 12 in the retained training cohort.
No automatic relabeling or speaker-cleaning claim is justified.

A local listening worksheet contains 24 short clips from eight participants,
sampled across domain and class without model predictions. Review covered
intervals and gaps at `data/processed/aligned_v2/speaker_review/index.html`, then
export annotations locally. No speaker annotations have been invented. This
sample can identify problems; it cannot certify the entire dataset.

## Matched results on the revised folds

Mean of three repeats; percentages except AUC shown on a 0–1 scale. F1 is computed
from pooled participant predictions per repeat, while AUC is averaged across folds.

| Procedure | Precision | Recall | F1 | Mean fold AUC |
|---|---:|---:|---:|---:|
| MPNet SVM reference | 47.3% | 73.9% | 57.5% | 0.805 |
| Regularized mean pooling | 35.8% | 60.4% | 44.9% | 0.716 |
| Regularized attention pooling | 37.6% | 64.0% | 47.2% | 0.749 |
| Inner-selected neural procedure | 36.6% | 62.2% | 46.0% | 0.739 |

Mean fold average precision is 0.607 for the reference, 0.480 for mean pooling,
0.520 for attention, and 0.513 for the selected neural procedure. Average precision
summarizes ranking across thresholds; it is not the precision at cutoff zero.

The historical 58.1% reference and highest observed 58.8% class-penalty result used
different folds. They must not be ranked directly against these new numbers.
The reference recipe remains retained; none of the neural procedures passed the
predeclared improvement screen.

The WoZ-only fitting / AI-only evaluation diagnostic reaches 36.9% F1 and 0.621
AUC. It uses only the official training partition, not a fresh test cohort, and
demonstrates that domain transfer remains a substantial weakness.
Within the joint-fold comparison, the reference achieves 60.2% F1 on WoZ and
42.2% on AI interviews. These subgroup estimates are especially uncertain for AI.

Attention's mean inner fitting F1 rises from 76.3% at epoch 40 to 99.0% at epoch
100, while mean inner-validation F1 falls from 46.4% to 37.0%. These averages
cover all inner runs and regularization candidates, not selected outer scores.
The learning curves support controlling training duration; they do not establish
that learned attention is better than the simpler reference.
Attention selected epoch 40 in nine outer fits, epoch 20 in three, and epoch 10
in three; none selected epoch 100. Mean pooling selected epoch 100 in six fits,
illustrating why training duration is selected rather than universally shortened.

## Reproduction and evidence

### Aligned fusion comparison

Completed and independently verified on 2026-11-04. Run:
`experiments/runs/aligned_fusion_v2_20261104T104203391619Z`.

| Clean-input procedure | Precision | Recall | F1 | Mean fold AUC | Mean fold AP |
|---|---:|---:|---:|---:|---:|
| Retained text SVM | 47.3% | 73.9% | 57.5% | 0.805 | 0.607 |
| Text-margin calibration control | 40.1% | 80.2% | 53.5% | 0.805 | 0.618 |
| Ordinary aligned fusion | 39.3% | 81.1% | 52.9% | 0.782 | 0.550 |
| Measured-quality aligned fusion | 39.5% | 81.1% | 53.1% | 0.770 | 0.526 |
| Inner-selected procedure, including reference | 44.5% | 75.7% | 55.9% | 0.805 | 0.609 |

All four candidate procedures failed the predeclared clean improvement screen.
Measured-quality fusion increases recall but reduces precision, F1, balanced
accuracy, AUC, and AP relative to the reference. No replacement is promoted.

| Missing-input condition | Ordinary F1 | Quality-aware F1 | Text reference F1 |
|---|---:|---:|---:|
| Audio absent | 53.5% | 54.1% | 57.5% |
| Face absent | 52.4% | 52.4% | 57.5% |
| Audio and face absent | 52.9% | 53.0% | 57.5% |

Quality-aware minus ordinary fusion gains **0.28 percentage points** on average
over these three conditions, below the required +5 points. Clean F1 improves by
0.14 points over ordinary fusion. The robustness screen fails. Text-only scores
remain identical when audio/face are removed, as expected.

Quality-aware fusion's clean subgroup F1 is 45.8% for AI and 54.3% for WoZ,
versus the reference's 42.2% and 60.2%. The small AI increase does not compensate
for the overall reduction or establish transfer success. These subgroup results
are descriptive; they were not used to choose a new winner.

Verification restored all 75 saved fusion bundles and reproduced all 9,660
prediction rows, 60 repeat-metric rows, 135 inner candidate scores, domain metrics,
selection rules, comparison screens, and nested cross-fitting membership. The
reference predictions exactly match the separately verified baseline. Across the
three v2 studies, 138 bundles and 12,129 prediction rows were verified.

Alignment verification checked all cache hashes, shapes, confidence flags,
chronological ordering and cache keys, and independently recomputed source
intervals for one interview from each domain. A subsequent build reused all
161 caches without changing artifacts or the manifest. Nine of 11 quality fields
vary in these data; finite-audio fraction and confidence-known are constant, and
audio frame coverage varies very little. This is measured variation, not proof
of a predictive quality signal.

### Commands

Use the repository venv from PowerShell. These commands create local ignored
artifacts and never call the official development/test evaluation entrypoints.

```powershell
.venv\Scripts\python.exe -m src.experiments.baseline_v2
.venv\Scripts\python.exe -m src.experiments.verify_v2 experiments/runs/baseline_v2_20261103T165509103723Z
.venv\Scripts\python.exe -m src.experiments.pooling_v2 experiments/runs/baseline_v2_20261103T165509103723Z
.venv\Scripts\python.exe -m src.experiments.verify_v2 experiments/runs/pooling_v2_20261103T165631446124Z
.venv\Scripts\python.exe -m src.dataset.aligned_v2
.venv\Scripts\python.exe -m src.dataset.verify_aligned_v2
.venv\Scripts\python.exe -m src.dataset.speaker_review_v2
.venv\Scripts\python.exe -m src.experiments.aligned_fusion_v2 experiments/runs/baseline_v2_20261103T165509103723Z
.venv\Scripts\python.exe -m src.experiments.verify_aligned_fusion_v2 experiments/runs/aligned_fusion_v2_20261104T104203391619Z
```

New runs receive new timestamped paths: substitute their printed paths when
reproducing dependent commands. The listed baseline and pooling paths identify
the verified runs reported here. Earlier partial/superseded v2 runs are excluded.
Baseline verification restored 18 models and checked 537 predictions; neural
verification restored 45 bundles and checked 1,932 predictions. Protocols are
`pipeline_v2_protocol.md` and `aligned_fusion_v2_protocol.md`.
The complete suite passes 140 tests. One pre-existing regex capture-group warning
remains in frozen historical screening code; the v2 expression uses noncapturing
groups. A cache regression test builds unsorted synthetic transcript features and
proves that a second call reuses the cache without reopening the archive.

## Outstanding work requiring evidence

Provider clarification of labels, human speaker/timing review, calibrated quality
validation, and fresh independent evaluation remain open. Existing repeated CV
cannot become an untouched validation set by reshuffling it. Missing-input tests
do not prove robustness to realistic microphone noise or mixed speakers. No model
is promoted merely because a point estimate is the largest.

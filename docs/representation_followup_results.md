# Learned pooling and complementary encoders

Completed and independently verified 2026-11-03. Two additional structural
experiments tested alternatives to averaging MPNet embeddings and changing SVM
thresholds. Neither improved the retained reference. Keep all negative results.

| Training-CV procedure | Precision | Recall | F1 +/- repeat SD | Specificity | Accuracy | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Retained MPNet SVM reference | 47.8% | 73.9% | 58.1 +/- 5.1% | 76.1% | 75.6% | 75.0% | 80.4% |
| Previous class-penalty selection | 49.2% | 73.0% | 58.8 +/- 4.3% | 77.7% | 76.6% | 75.3% | 80.1% |
| Learned mean-pooling network | 38.6% | 57.7% | 46.2 +/- 4.6% | 72.3% | 68.9% | 65.0% | 71.9% |
| Learned attention-pooling network | 38.3% | 53.2% | 44.4 +/- 2.4% | 74.5% | 69.6% | 63.8% | 75.4% |
| Inner-selected pooling architecture/epoch | 40.2% | 52.3% | 45.2 +/- 6.0% | 77.2% | 71.4% | 64.7% | 73.0% |
| Inner-selected MPNet/MiniLM blend | 46.1% | 66.7% | 54.5 +/- 4.5% | 76.9% | 74.5% | 71.8% | 80.3% |

All use the same 161-person training cohort, released labels, five outer folds
and three repeats. SD is not a confidence interval, and repeats overlap. Previous
class-penalty results are from the [precision experiment series](precision_results.md).
They remain the highest observed mean F1 (58.8%) but did not clear its precision
promotion screen; their +0.69 F1-point difference is not a demonstrated reliable
improvement. The saved full-training research reference remains unchanged.

## Learned pooling

[Protocol](utterance_pooling_protocol.md), run
`experiments/runs/utterance_pooling_20261103T115821003735Z`.

Preserved all 14,724 screened utterances, with 41-197 per participant. Frozen
MPNet embeddings underwent a label-independent fixed random projection to64
dimensions. A small 16-unit learned representation feeds either masked mean
pooling or learned attention. Both use identical training schedules, with epoch
40/100 chosen on inner validation. One loss per participant avoids treating
utterances as independent labeled people. These networks are order-invariant;
they do not model temporal order. Attention adds 17 parameters to its control.

The inner-selected procedure loses 12.82 F1 points against the reference, with
losses in all three repeats. Attention has higher AUC than its neural mean
control but lower F1. It does not pass the clean or precision improvement screen.
The inner selector chose mean pooling in nine folds and attention in six.

This rules out this particular small projected architecture/training recipe as
an improvement. It does not prove all attention methods fail: the projection,
small hidden layer, training schedule and frozen encoder remain limitations.
No causal claim that the neural model overfit is made without a learning-curve
analysis. No encoder fine-tuning was performed.

## Complementary encoder blend

[Protocol](encoder_blend_protocol.md), run
`experiments/runs/encoder_blend_20261103T120034683682Z`.

Joint inner selection tested SVM settings for both encoders and five MPNet
weights (1/.75/.5/.25/0). Each encoder has independently fold-fitted PCA32.
Combining raw margins is not probability calibration. Selected MPNet weights:
.75 in six folds, .25 in four, 1 in three, .5 in two. The procedure loses 3.57 F1
points and 7.21 recall points against the reference, with F1 losses in all three
repeats. Its AUC remains similar, but its decisions do not improve. Do not choose
a different blend weight retrospectively from outer predictions.

## Domain checks

| Procedure | AI F1 | WoZ F1 |
|---|---:|---:|
| Reference | 38.9% | 60.9% |
| Neural mean pooling | 44.3% | 46.6% |
| Neural attention | 29.9% | 46.7% |
| Inner-selected pooling | 38.1% | 46.5% |
| Encoder blend | 40.9% | 56.4% |

AI has only 18 participants/five positives. Its apparent subgroup gains do not
justify post-hoc model selection; the historical test cohort is AI-only, and no
new test predictions were generated. Label and speaker-attribution uncertainties
remain unchanged.

## Verification

All 129 tests passed (existing pandas regex warning only). Pooling verification
restored 45 bundles, checked 1,932 predictions, 12 repeat metric rows and 60 inner
candidate records. It verified all participant caches and replayed 12 sampled
inner epoch checkpoints by retraining, as well as checking training-only scalers.
Encoder-blend verification restored 30 bundles, checked 966 predictions, six
metric rows and all 1,800 inner combinations, and refitted selected inner SVMs.
Both runs reproduced all 483 historical MPNet reference scores. Total new saved
bundles checked:75; prediction records:2,898. Bundles can duplicate a selected
architecture and are not a count of unique training fits.

```powershell
.venv\Scripts\python.exe -m src.experiments.utterance_pooling
.venv\Scripts\python.exe -m src.experiments.verify_utterance_pooling experiments/runs/utterance_pooling_20261103T115821003735Z
.venv\Scripts\python.exe -m src.experiments.encoder_blend
.venv\Scripts\python.exe -m src.experiments.verify_encoder_blend experiments/runs/encoder_blend_20261103T120034683682Z
```

These experiments establish negative findings for tested recipes, not the best
possible model. Further adaptive search on the same folds risks optimizing the
research process to this cohort. Fresh independent validation, a confirmed label
definition and verified participant speech remain necessary for stronger claims.
No model was deployed and the best-F1 reference checkpoint was not replaced.

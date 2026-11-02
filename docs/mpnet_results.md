# MPNet encoder results

Verified 2026-11-02. Run: `experiments/runs/mpnet_20261102T171156498575Z`.
The [protocol](mpnet_protocol.md) preceded extraction/fitting. Same 161 training
participants, 37 released positives, screened transcripts, pooling and classifier
recipe; no new development/test predictions or label changes.

**MPNet with linear SVM passes the predeclared exploratory improvement screen.**
Mean F1 rises from 52.8% to 58.1% (+5.23 points), with positive F1 differences in
all three repeats. Balanced accuracy, AUC and recall also improve. MPNet logistic
does not clear the three-point F1 requirement. MPNet SVM becomes the stronger
research reference; it is not an independently test-validated replacement.

All entries are percentages; F1 SD is across overlapping repeats, not a CI.

| Encoder / classifier | F1 +/- SD | Precision | Recall | Specificity | Accuracy | Balanced accuracy | Mean fold AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniLM / SVM | 52.8 +/- 3.4 | 43.9 | 66.7 | 74.2 | 72.5 | 70.4 | 79.4 |
| MiniLM / logistic | 52.5 +/- 2.5 | 42.1 | 70.3 | 70.7 | 70.6 | 70.5 | 78.2 |
| MPNet / SVM | 58.1 +/- 5.1 | 47.8 | 73.9 | 76.1 | 75.6 | 75.0 | 80.4 |
| MPNet / logistic | 54.0 +/- 5.9 | 45.4 | 66.7 | 76.3 | 74.1 | 71.5 | 79.2 |
| Always negative | 0.0 +/- 0.0 | 0.0 | 0.0 | 100.0 | 77.0 | 50.0 | 50.0 |
| Always positive | 37.4 +/- 0.0 | 23.0 | 100.0 | 0.0 | 23.0 | 50.0 | 50.0 |

The domain limitation remains: MPNet SVM averages 60.9% F1 in 143 WoZ interviews
but 38.9% in 18 AI interviews (only five positives). The historical test partition
contains AI interviews only; the pooled improvement cannot be assumed to transfer
there. Neither that subgroup nor the previously inspected test was used to tune
this encoder comparison. The 65%-70% pooled F1 aspiration remains unmet.

## Provenance and verification

Frozen local `sentence-transformers/all-mpnet-base-v2` revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`, safetensors, float32, eager attention,
TF32 disabled. The unused learned pooler is deliberately disabled; its two
checkpoint keys are unused, and no required encoder weights are missing.
The same encoding helper as MiniLM uses 254-token content chunks plus special
tokens, without dropping longer utterance remainders. No transcripts were uploaded.
The [model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
describes a general sentence encoder, not a depression-specific model.

Extraction covers all 14,724 screened utterances. Recorded per-participant
extraction time totals 307 seconds (5.1 minutes), excluding download/loading and
classifier fitting. Peak allocated CUDA memory is 0.818 GiB. All 161 caches and
pooled features passed hash/shape/finite checks; replaying the first participant
reproduced utterance embeddings exactly (maximum difference zero).

Independent checks restored 60 pipelines, recomputed 2,898 predictions and 18
repeat metric rows, verified folds/source/model hashes, and matched 966 prior
MiniLM control predictions and scores. The existing 117-test suite passes; the
new encoder is additionally covered by real-data extraction and replay checks.
Complete metrics, repeat/domain summaries and screening decisions are in the run.

An additional descriptive audit of the previous MiniLM errors found no obvious
false-positive concentration in the 12 retained label/score-disagreement cases:
SVM false-positive rates were 25.0% there versus 25.9% among the other 112 released
negatives; logistic rates were 22.2% versus 30.1%. These small-group observations
do not resolve the labels or prove they are harmless. Scores were used only for
this post-hoc diagnostic, never as predictors or replacement targets.

```powershell
.venv\Scripts\python.exe -m src.dataset.mpnet
.venv\Scripts\python.exe -m src.experiments.mpnet_study
.venv\Scripts\python.exe -m src.experiments.verify_mpnet experiments/runs/mpnet_20261102T171156498575Z
```

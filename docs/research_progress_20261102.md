# Current research reference and completed follow-ups

Verified 2026-11-02. **The strongest observed research reference is screened
MPNet text + standardized whitened PCA32 + linear SVM, with 58.1% mean positive-
class F1 and 80.4% mean fold AUC.** It improves on the previous MiniLM text SVM
by 5.23 F1 points in the matched 161-person training comparison, with gains in
all three repeats. It passes the predeclared exploratory improvement screen.

| Matched training-CV reference | F1 | Precision | Recall | Specificity | Accuracy | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original audio + face SVM | 37.1% | 32.1% | 44.1% | 72.0% | 65.6% | 58.1% | 56.9% |
| Previous MiniLM text SVM | 52.8% | 43.9% | 66.7% | 74.2% | 72.5% | 70.4% | 79.4% |
| Current MPNet text SVM | 58.1% | 47.8% | 73.9% | 76.1% | 75.6% | 75.0% | 80.4% |

Four additional studies were completed, retaining all unsuccessful procedures:

1. [Text refinement and late fusion](text_refinement_results.md): 210 verified
   bundles, 7,728 predictions. Retaining all embedding dimensions, variability
   summaries, threshold search and late fusion failed to improve the old baseline.
2. [Screened lexical features](lexical_results.md): 165 verified bundles,
   6,279 predictions. Word/phrase features and semantic combinations also failed.
3. [Frozen MPNet](mpnet_results.md): 60 verified pipelines, 2,898 predictions.
   SVM improved; logistic did not clear the three-point improvement threshold.
4. [MPNet compression follow-up](mpnet_rank_results.md): 165 verified bundles,
   6,279 predictions. Broader rank/threshold searches did not beat the original
   MPNet recipe. The simpler reference is retained.

Across these studies, 600 saved model bundles and 23,184 predictions were checked.
All 119 tests pass. Bundles can share fitted parameters across cutoff conditions;
this is not a claim of 600 distinct training fits. MPNet cache hashes and encoder
replay passed; the replay difference was zero. No labels changed, participant
data uploaded, new development/test scores generated, or model deployed.

## Reusable full-training checkpoint

Built under the separate [refit protocol](research_reference_protocol.md):
`models/research_reference_20261102T172007773516Z/model.joblib`.
The folder is local and ignored by Git. It contains the ordered 768-feature
schema, pinned encoder revision/hashes, fitting IDs, input/source/model hashes,
selection audit, environment and independent verification record.

The refit uses all 161 training participants. Its three-fold training-only
selection chose C=0.1 and balanced class weights, with PCA32 and zero-margin
cutoff. Serialization, input schema, fitted scaling statistics, selection/fold
records and hashes were independently checked. This refit is a convenient
research artifact; it does not supply a new generalization estimate. The 58.1%
figure belongs to the preceding nested-CV recipe, not a test of this one fitted
checkpoint. Input must be screened MPNet mean embeddings using the recorded
encoder/pooling contract, not raw text or MiniLM embeddings.

```powershell
.venv\Scripts\python.exe -m src.experiments.refit_reference
.venv\Scripts\python.exe -m src.experiments.verify_reference models/research_reference_20261102T172007773516Z
```

## Remaining limits

The 65%-70% F1 aspiration is still 6.9-11.9 percentage points above the current
mean CV result. MPNet SVM F1 is 60.9% in WoZ interviews but only 38.9% in the small
AI subgroup (18 people, five positives). The historical test partition is all AI;
pooled training improvements cannot be assumed to transfer. Historical Stage 9
test scores remain separate and unchanged. Reliability-aware fusion has not yet
demonstrated its desired advantage, and the strongest reference is text-only.

The provisional label definition and transcript speaker/disclosure limitations
remain unresolved. The provider inquiry is drafted but unsent. Error diagnostics
did not show false positives concentrated in the retained label-disagreement
group; that observation neither resolves nor dismisses the label issue.

These repeated training studies establish the best observed recipe among the
tested conditions, not the best possible model. Further adaptive searching can
overfit this evaluation. A credible superiority claim requires fresh independent
validation; these numbers do not establish competitiveness with a paper using a
different cohort, label definition and experiment setup. Original Stage 10 remains
pending.

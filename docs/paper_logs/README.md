# Paper-writer experiment log

**Final bounded sparse and reliability-aware fusion comparison — 2026-11-08**

Run ID: `sparse_fusion_20261108T160011416879Z`. Protocol: `docs/methods/sparse_fusion.md`. The protocol was written before model fitting. This package contains aggregate results and methods; participant predictions, recordings and model binaries are excluded.

## Main conclusion

None of the new policies passed the predeclared improvement screen against the retained original audio/facial concatenation model. Sparse concatenation reached 37.63% F1 versus 37.16% for the control, but accuracy fell from 62.94% to 59.63% and balanced accuracy from 57.61% to 57.36%. Reliability adjustment added only about 0.08 percentage points F1 to ordinary weighted fusion and reduced mean fold AUC. Keep the original primary reference and stop this bounded tuning series.

## Dataset and evaluation

- Local E-DAIC official training partition: 161 retained people, 37 positive and 124 negative released labels. There are 143 WoZ and 18 AI interviews. All participants remain in the evaluation, including two with quarantined facial timing.
- Labels remain unchanged. Earlier audits identified label-definition discrepancies and uncertain transcript speaker attribution; this experiment does not resolve them.
- Three repeats (seeds 42, 137, 2026), five outer participant folds per repeat, jointly stratified by label and interview domain. Three inner folds per outer fitting partition use seed + outer-fold number + 1.
- Preprocessing and supervised feature selection are refitted within every inner fitting partition. Inner out-of-fold F1 selects settings; outer held-out people never select features, settings, weights or thresholds.
- No new official development/test evaluation. These training participants have been examined across earlier studies; outer CV within this experiment does not erase that broader adaptive reuse.
- WoZ/AI stratification is interview-domain handling within one depression dataset. It is not evidence of performance across different neurological/psychological disorders.

## Inputs and model recipes

Use original supplied eGeMAPS acoustic summaries (46 features) and OpenFace facial summaries (98 features). The earlier waveform cleanup was rejected. No transcript embeddings or lexical predictors enter these models. Quality proxies are derived from the existing utterance-aligned caches, so timing still depends on transcript intervals.

Fold-fitted preprocessing drops wholly missing features, median-imputes, removes constant/exact duplicate columns, and standardizes. The fixed control concatenates the original features and fits balanced logistic regression with C=.1, liblinear, max_iter=5000 and the repeat seed.

For each input family (audio, face, concatenation), evaluate nine candidates: all features, top5, or top10, each with final logistic C=.001/.01/.1. The smaller C values provide stronger regularization. For top-k selection, fit a balanced L1 logistic selector with fixed C=1 on that fitting partition, retain nonzero absolute coefficients (>1e-8), and select at most k in stable descending order. If no coefficient survives, predict the fitting prevalence. All-feature and sparse candidates are reported separately; the selected unimodal branch can choose either. Candidate ties select the first entry: all/top5/top10, then C ascending.

Late fusion averages the two selected branches' logistic probabilities. These class-weighted outputs are not calibrated clinical risks. Equal fusion uses .5 audio/.5 face. Weighted fusion chooses the audio weight from [.5,.25,.75] by inner OOF F1, using the same inner branch predictions. Missing modalities receive zero weight and the remaining weights are normalized; no available modality falls back to fitting prevalence. Inner fallback uses the relevant inner fitting prevalence. Thresholds remain fixed: zero for margins, .5 for fused probabilities.

The reliability control uses exactly the same selected branches and base weight as ordinary weighted fusion. An available audio input receives half weight if clipping >.001, zero fraction >.01 or frame coverage <.95. An available facial input receives half weight if retained fraction <.8 or confidence <.8. Weights are renormalized. These are fixed engineering thresholds, not learned probabilities of correctness; the aligned quality proxies do not directly validate the reliability of whole-interview summaries.

## All results

Accuracy, balanced accuracy, precision, recall and F1 are percentages. F1 SD is the population SD across the three repeat-level pooled out-of-fold values, not a confidence interval. AUC and AP are unweighted means across 15 held-out folds on a 0–1 scale. Repeats overlap and are not independent samples.

| Policy | Accuracy | Balanced accuracy | Precision | Recall | F1 ± SD | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_concat | 62.94 | 57.61 | 30.42 | 47.75 | 37.16 ± 4.02 | 0.600 | 0.401 |
| concat_all | 55.90 | 53.36 | 25.67 | 48.65 | 33.55 ± 4.56 | 0.563 | 0.365 |
| concat_sparse | 59.63 | 57.36 | 29.13 | 53.15 | 37.63 ± 3.06 | 0.598 | 0.367 |
| audio_all | 53.62 | 52.19 | 24.93 | 49.55 | 33.11 ± 3.25 | 0.582 | 0.348 |
| audio_sparse | 51.35 | 49.77 | 23.30 | 46.85 | 31.02 ± 6.16 | 0.525 | 0.323 |
| selected_audio | 50.31 | 49.41 | 22.80 | 47.75 | 30.81 ± 4.26 | 0.545 | 0.333 |
| face_all | 55.07 | 55.35 | 26.94 | 55.86 | 36.34 ± 4.16 | 0.545 | 0.310 |
| face_sparse | 56.11 | 56.02 | 27.70 | 55.86 | 37.00 ± 2.52 | 0.592 | 0.368 |
| selected_face | 54.45 | 55.58 | 27.12 | 57.66 | 36.86 ± 2.61 | 0.568 | 0.345 |
| late_equal | 53.62 | 54.41 | 26.21 | 55.86 | 35.67 ± 7.12 | 0.569 | 0.369 |
| late_weighted | 54.45 | 54.31 | 26.28 | 54.05 | 35.36 ± 4.34 | 0.579 | 0.368 |
| late_quality | 54.66 | 54.45 | 26.37 | 54.05 | 35.45 ± 4.18 | 0.575 | 0.365 |
| always_negative | 77.02 | 50.00 | 0.00 | 0.00 | 0.00 ± 0.00 | 0.500 | 0.230 |
| always_positive | 22.98 | 50.00 | 22.98 | 100.00 | 37.37 ± 0.00 | 0.500 | 0.230 |

## Predeclared decision screens

Pass requires mean F1 improvement of at least 3 percentage points, no mean balanced-accuracy or mean fold-AUC reduction, recall loss no greater than 5 points, and positive F1 change in at least two repeats. These are engineering selection criteria; no multiplicity-adjusted significance claim is made.

| Candidate | Control | F1 change (points) | Positive repeats | Screen passed |
|---|---|---:|---:|---|
| concat_all | original_concat | -3.61 | 0/3 | False |
| concat_sparse | original_concat | +0.47 | 1/3 | False |
| late_equal | original_concat | -1.49 | 1/3 | False |
| late_weighted | original_concat | -1.80 | 1/3 | False |
| late_quality | original_concat | -1.71 | 1/3 | False |
| audio_sparse | audio_all | -2.08 | 1/3 | False |
| face_sparse | face_all | +0.66 | 2/3 | False |
| concat_sparse | concat_all | +4.07 | 2/3 | True |
| late_weighted | late_equal | -0.31 | 1/3 | False |
| late_quality | late_weighted | +0.08 | 1/3 | False |
| late_weighted | selected_audio | +4.56 | 3/3 | True |
| late_weighted | selected_face | -1.50 | 1/3 | False |

Two ancillary comparisons pass: sparse concatenation versus the weaker inner-selected all-feature concatenation (+4.07 points F1), and weighted fusion versus the weaker selected audio branch (+4.56 points). Neither establishes an improvement over the retained original concatenation. The matched quality-versus-weighted comparison fails.

## Selection and compactness

Sparse concatenation selected 5 features in 7/15 outer fits and 10 features in 8/15, averaging 7.67 compared with 144 for the original concatenation. This is a compression trade-off, not demonstrated performance equivalence: it detected more positives but made more false positives, reducing accuracy and precision.

The selected audio branch kept all features in 7/15 fits and a sparse set in 8/15. The selected face branch kept all features in 4/15 and a sparse set in 11/15. Weighted fusion chose audio weight .25 in 10/15 fits, .75 in 4/15, and .5 in 1/15. Fold-level mean effective audio weights averaged approximately .408 for ordinary weighted fusion and .349 with the quality discount. `selected_settings.csv` records each selected setting and feature list; `fusion_weights.csv` records fold-level effective weights. Fold-level averages are descriptive, not independent stability evidence. The fixed rules flagged 78 available audio inputs and zero available facial inputs as poor; two facial inputs were unavailable.

## Comparison with earlier results

| Earlier/current method | Accuracy | F1 | Interpretation |
|---|---:|---:|---|
| Retained text reference | 74.74% | 57.52% | Stronger reference; different input modality |
| Earlier threshold-adjusted text candidate | 76.19% | 58.51% | Modest gain; failed its promotion screen |
| Original audio + facial concatenation | 62.94% | 37.16% | Matched control reproduced in this run |
| Earlier neural quality fusion | 62.11% | 28.30% | Different architecture; not a matched quality ablation here |
| Earlier high-pass audio + facial combination | 62.11% | 34.31% | Waveform cleanup did not improve the original control |
| Current sparse concatenation | 59.63% | 37.63% | Small F1 gain, lower accuracy; not promoted |
| Current weighted late fusion | 54.45% | 35.36% | Does not beat original concatenation |
| Current quality-adjusted late fusion | 54.66% | 35.45% | Negligible matched quality benefit; not promoted |

These figures come from repeated training CV. Historical Stage 9 test-set findings are a separate experiment and must not be mixed with these estimates. Always predicting the negative label gives 77.02% accuracy on this cohort but zero positive recall/F1, so accuracy alone is not a useful screening criterion.

## Verification and reproducibility

- All 180 serialized policy bundles were restored and final policies refitted; 6,762 predictions were checked with zero restored-score error. Bundles share branch models, so 180 files do not mean 180 independent fits.
- All 405 inner candidate vectors and 45 fusion-weight vectors were replayed, including selection choices and feature identities. Reliability calculations were independently reconstructed.
- Folds, preprocessing statistics, hashes, confusion-derived metrics, domain metrics, AUC/AP, summaries and screens were checked. Original concatenation reproduces the historical held-out scores.
- All 116 tests passed; one pre-existing transcript-regex warning remains. Training was CPU-based. No new dataset or external holdout was used.
- `verification.json` gives machine-readable checks. `source_hashes.json` identifies exact source/configuration/input versions, and `environment.txt` records the environment. The full source/model manifests and participant-level audit trail remain in the local run folder.

Reproduce from the matching source and existing authorized local caches:

```powershell
.venv\Scripts\python.exe -m src run audio-visual --sparse-fusion
.venv\Scripts\python.exe -m src verify audio-visual --run experiments/runs/sparse_fusion_20261108T160011416879Z
```

## Suggested Results/Discussion wording

“Within the repeated nested participant-level evaluation, L1-based sparse concatenation reduced the model input to five or ten features per outer fit and achieved a mean positive-class F1 of 37.63%, compared with 37.16% for the fixed original-feature concatenation control. Accuracy decreased from 62.94% to 59.63%, and balanced accuracy decreased from 57.61% to 57.36%. Weighted late fusion and its matched quality-adjusted variant achieved F1 scores of 35.36% and 35.45%, respectively. No proposed replacement met the predeclared improvement criteria against the retained control. These exploratory results support reporting the sparsity–performance trade-off, but do not establish a predictive advantage for reliability-aware fusion.”

## Claims this experiment does not support

- That R-SAF, quality gating or HHO outperforms the retained reference. This follow-up tests L1 selection; it does not rerun or validate an HHO advantage. Historical Integrated Gradients results concern earlier neural models and must not be presented as explanations of these newly fitted logistic policies.
- That five clinical datasets, multiple disorders, clinical deployment, or generalization to new patients were validated.
- That a higher F1 alone establishes superiority, or that the best observed candidate is an independently confirmed improvement.
- That failure of these bounded recipes proves a dataset ceiling. The appropriate conclusion is that the tested variants did not establish the proposed gain.

## Files and units

`results.csv` contains the 14 policy summaries; `repeat_metrics.csv` contains 42 repeat-level rows; `domain_metrics.csv` contains interview-domain summaries. Numeric metric fields in CSV are fractions (multiply by 100 for percentages), except counts; AUC also uses 0–1. `comparisons.json` stores unrounded differences and screen decisions. `run.log` contains the training progress log without local machine paths. There are no recordings, transcripts, participant IDs, per-person predictions or model binaries in this shareable package.

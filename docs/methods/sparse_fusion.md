# Final bounded sparse/late-fusion comparison — 2026-11-08

Freeze this protocol before fitting. Keep the 161-person training cohort, 37
released positive labels, original 46 acoustic/98 facial summaries and existing
two facial timing quarantines. No waveform cleanup, text predictors, relabeling,
new dataset, new dev/test evaluation or extra search after this series.

Outer five-fold and inner three-fold splits jointly stratify label/interview
domain, repeated with seeds 42, 137, 2026; inner seed = outer seed + fold + 1.
Each participant appears in exactly one held-out fold per repeat. All imputation,
redundancy removal, scaling, supervised selection and parameter choices are fitted
inside the relevant training partition. Preprocessing uses BranchPreprocessor.

For audio, face and their concatenation separately, compare nine fixed candidates:
all features or at most 5/10 L1-ranked features; final balanced L2 logistic C in
[.001, .01, .1]. Rank nonzero absolute coefficients (>1e-8) from a balanced L1
logistic selector with fixed C=1, liblinear, max_iter=5000. Stable coefficient ties
preserve original column order. If no feature is selected, predict the fitting
partition's positive prevalence; do not invent a selected feature. Smaller C means
stronger regularization. Report all-feature and sparse choices separately.

Choose by pooled inner out-of-fold F1 at margin zero, first candidate on ties.
Order: all, top5, top10; C ascending. Select each branch from its nine candidates.
There is no threshold tuning. Keep the original concatenated balanced logistic
C=.1 as a matched historical control.

Late fusion averages the two selected branches' logistic probabilities. These
class-weighted probabilities are not calibrated clinical risks. Equal fusion uses
audio weight .5; weighted fusion selects from [.5, .25, .75] using the same inner
OOF predictions and F1 rule. Normalize over available modalities in every fusion
control. If neither is available, use the outer-training positive prevalence.
Inner OOF fallback instead uses each inner fitting partition's prevalence.

The reliability variant uses the *identical* branches and selected base weight as
ordinary weighted fusion. Multiply the weight by .5 for an available but poor
modality, then renormalize. Audio is poor if clipping fraction >.001, zero fraction
>.01 or frame coverage <.95. Face is poor if retained fraction <.8 or confidence
<.8. Unavailable modalities get weight zero. These fixed thresholds reuse earlier
engineering rules; the proxies come from aligned utterances and do not directly
measure whole-interview predictive reliability. Do not tune the adjustment.

Report 12 fitted policies: original_concat, concat_all, concat_sparse, audio_all,
audio_sparse, face_all, face_sparse, selected_audio, selected_face, late_equal,
late_weighted, late_quality. Add always-negative/positive controls. Save 180 policy
bundles and 6,762 predictions, 405 inner candidate score vectors, 45 inner fusion
weight score vectors, feature identities/counts, choices, weights, quality counts,
timings and source/input/environment hashes. Bundles reuse fitted branch models;
the number of files is not the number of distinct training fits.

Primary screens: concat_all/concat_sparse/late_equal/late_weighted/late_quality
against original_concat; sparse versus all within each input family; weighted
versus equal; quality versus weighted; weighted versus selected_audio/selected_face.
Use the existing screen: mean F1 +3 percentage points, no mean balanced-accuracy or
fold-AUC loss, recall loss <=5 points, F1 positive in at least 2/3 repeats.
Report accuracy, balanced accuracy, precision, recall, F1, fold AUC/AP, domains and
all failures. No multiplicity-adjusted significance or fresh replication claim.

Independently replay all inner choices, fit final models again, restore predictions,
check preprocessing and feature identities, independently check reliability weights,
and reproduce the historical original-concatenation control. Stop after this fixed
series. Preserve the retained reference if no candidate qualifies. Supply a
standalone paper-writer log distinguishing measured results from proposed novelty.

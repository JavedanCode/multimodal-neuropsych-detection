# Score target, eGeMAPS functionals and domain scaling — 2026-11-08

Freeze this protocol before extracting features or fitting. Approved by the user
on 2026-11-08 as one bundled comparison ("go ahead with whatever you think will
yield the best result"), including use of the released PHQ score as a training
target. Reuse the 161-person training cohort, unchanged released `PHQ_Binary`
labels for every evaluation, original 98 facial summaries and two facial timing
quarantines, nine quality fields, and the existing joint label/domain splits
(5 outer folds, seeds 42, 137, 2026; 3 inner folds, seed = outer seed + fold + 1).
No development/test evaluation, relabeling, decision-threshold search, feature
selection, text predictors or extra seeds. Stop after this run.

## Three changes, each motivated by earlier evidence

1. **Official eGeMAPSv02 functionals (88 audio features).** openSMILE Python
   2.6.0, `FeatureLevel.Functionals`, on each untreated whole recording after the
   existing shared mono/16 kHz decode (`audio_preprocessing.read_waveform`,
   `clean_waveform(..., "raw")`). These compute voicing-dependent statistics on
   voiced segments and add pause/rhythm measures (voiced segments per second,
   voiced/unvoiced segment lengths, loudness peaks). Motivation: the corrected
   voiced summaries improved ranking (fold AUC .600 to .625). Recordings are the
   released interview audio; no speaker isolation, trimming or denoising.
2. **Domain-conditional standardization.** Audio features separate WoZ from AI
   interviews strongly (median single-feature domain AUC .74). After the usual
   fold-fitted `BranchPreprocessor`, re-standardize each feature within interview
   domain using fitting-partition statistics shrunk toward the pooled ones:
   `mean_d = n_d*m_d/(n_d+10)`, `var_d = (n_d*v_d + 10)/(n_d+10)` in pooled
   standardized units. Domain comes from the participant ID range (300–492 WoZ,
   600–718 AI) and is used for normalization only, never as a predictor.
3. **Released PHQ score as training target.** Fit ridge regression to
   `PHQ_Score - 9.5` with balanced class weights from `PHQ_Binary`
   (n / (2 n_class)), so the weighted training mean sits near the decision
   point. A participant is predicted positive when the predicted score is at
   least 9.5, a fixed threshold between the questionnaire's 9 and 10. Ridge
   alpha is chosen from [1, 10, 100, 1000] by pooled inner-OOF weighted squared
   error on the score (not on binary F1); first on ties. Evaluation always uses
   the unchanged binary labels. Caveat: 12 retained training participants have
   PHQ score >= 10 with released label 0; no participant with label 1 has score
   < 10. The score target pushes those 12 toward positive, which counts as a
   false positive against the released labels. This is reported, not hidden.

## Fixed policies (12 fitted per fold, plus always-negative/positive)

Classifier recipe ("clf"): BranchPreprocessor and balanced L2 logistic,
liblinear, C=.1, all features, margin threshold zero (historical control).

| Policy | Audio input | Face | Target | Domain scaling |
|---|---|---|---|---|
| original_audio | supplied 46 | – | binary clf | no |
| voiced_audio | voiced 47 | – | binary clf | no |
| func_audio | functionals 88 | – | binary clf | no |
| face | – | 98 | binary clf | no |
| original_concat (control) | supplied 46 | 98 | binary clf | no |
| voiced_concat | voiced 47 | 98 | binary clf | no |
| func_concat | functionals 88 | 98 | binary clf | no |
| voiced_concat_domain | voiced 47 | 98 | binary clf | yes |
| func_concat_domain | functionals 88 | 98 | binary clf | yes |
| score_voiced_concat | voiced 47 | 98 | PHQ score ridge | no |
| score_func_concat | functionals 88 | 98 | PHQ score ridge | no |
| score_func_concat_domain | functionals 88 | 98 | PHQ score ridge | yes |

`original_concat` must reproduce the historical scores exactly.

## Comparisons, screen and reporting

Primary, against `original_concat`: func_concat, voiced_concat_domain,
func_concat_domain, score_voiced_concat, score_func_concat,
score_func_concat_domain. Isolating: func_audio vs original_audio; func_audio vs
voiced_audio; func_concat vs voiced_concat; voiced_concat_domain vs voiced_concat;
func_concat_domain vs func_concat; score_voiced_concat vs voiced_concat;
score_func_concat vs func_concat; score_func_concat_domain vs score_func_concat.

Existing screen unchanged: mean F1 +3 points, no mean balanced-accuracy or
fold-AUC loss, recall loss <= 5 points, F1 gain in >= 2/3 repeats. Engineering
criteria, not multiplicity-adjusted tests. Also report a paired participant
bootstrap (2,000 label-stratified resamples, seed 20261108) of the mean-over-
repeats F1 and AUC difference for every primary comparison, plus always-positive
F1, AP, domain metrics, chosen alphas, regression error and the 12
score/label disagreements' predictions.

If several primary candidates pass, all are reported; the one with the highest
mean F1 is named the exploratory best, with the warning that choosing among six
candidates on the same folds inflates its estimate. If none pass, keep
`original_concat`. The cohort has been reused across many studies, so any pass
needs a fresh holdout before it is called an improvement.

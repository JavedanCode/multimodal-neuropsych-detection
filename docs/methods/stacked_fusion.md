# Stacked text + voice + face fusion and three fold schemes — 2026-11-12

Freeze before fitting. Requested by the user: build a purpose-made fusion of the
three strongest components (text reference, segment-based voice, face) and
evaluate it under three cross-validation fold schemes, one of them the existing
scheme. Cohort: the 161 retained training participants, released `PHQ_Binary`
unchanged. No development or test data. Seeds 42, 137, 2026 in every scheme.

## Model

Branches, each fitted on the outer-training partition only:

- **text**: the retained MPNet reference recipe, unchanged (its own inner
  selection on three inner folds).
- **voice**: 49 segment-based voiced summaries (`participant_speech_v1/voiced`).
- **face**: 98 OpenFace summaries (existing two timing quarantines).
- **av**: voice and face concatenated (the best audio/face model so far).

Voice, face and av use BranchPreprocessor and balanced L2 logistic regression,
liblinear, C=.1. Inner out-of-fold margins for every branch come from three
inner folds (seed = outer seed + fold + 1); the text branch refits the
parameters its own inner selection chose. Each branch margin is divided by the
RMS of its inner out-of-fold margins.

Policies:

- `text`, `av`: single branches (margin threshold zero).
- `stack3`: balanced L2 logistic regression (lbfgs, C=1, with intercept) on the
  scaled inner out-of-fold margins of text, voice and face. The learned
  intercept sets the decision point from training data only; no threshold search.
- `stack2`: the same combiner on text and av.
- `gate3`: `stack3`, except that when the text branch is confident the voice
  and face contributions are set to zero. Confident means the absolute scaled
  text margin is at least the median absolute scaled inner out-of-fold text
  margin of that outer-training partition (fixed rule, no tuning).

## Fold schemes

- **A (existing, kept):** 5 outer folds stratified by label x interview domain.
- **B (gender-balanced):** 5 outer folds stratified by label x domain x gender;
  a stratum with fewer than 5 people falls back to label x domain. Rationale:
  gender shifts pitch and other voice features strongly, so balancing it should
  lower fold-to-fold variance. Gender is used only to assign folds, never as a
  feature.
- **C (10-fold):** 10 outer folds stratified by label x domain; a stratum with
  fewer than 10 people is merged into the same-label stratum. Rationale: each
  model trains on 90% of participants instead of 80%, which reduces the
  pessimistic bias of small-sample cross-validation.

Inner folds are the same in all schemes (three, label x domain).

## Reporting and decision

Report every policy under every scheme: accuracy, balanced accuracy, precision,
recall, F1, fold AUC, AP, always-positive F1. Within each scheme apply the
existing screen to `stack3`, `stack2` and `gate3` against `text` on the same
folds (mean F1 +3 points, no balanced-accuracy or fold-AUC loss, recall loss
<= 5 points, F1 gain in >= 2/3 repeats), plus a paired label-stratified
participant bootstrap (2,000 resamples, seed 20261112) of the F1 and pooled-AUC
change. Descriptive abstention table (most confident 100/80/60/40%) for `text`
and the fusion policies under scheme A.

Scheme A stays the primary scheme because all earlier results use it. As the
user asked, the scheme with the highest mean F1 for the best fusion policy is
identified. Choosing an evaluation scheme by its score makes that score
optimistic, so all three schemes are reported together, model-versus-text
comparisons are always made within one scheme, and a scheme's numbers are never
compared with another scheme's baseline. Exploratory, reused cohort.

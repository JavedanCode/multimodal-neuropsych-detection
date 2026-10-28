# Follow-up audio study v1

Fixed 2026-10-28 before extracting enriched features or running this study.
This is a new exploratory study motivated by Stage 9. It does not revise the
completed test evaluation and is separate from the pending Stage 10 audit.

## Scope and target

Only the 163 official training participants enter extraction, fitting, selection
and scoring. Development/test rows are not passed into the experiment. Released
PHQ_Binary remains unchanged and provisional. No new holdout claim: prior results
informed the research direction. Independent data are needed for confirmation.

Use the same 23 supplied eGeMAPS descriptors. Compare the existing 46 mean/std
features against 230 features: mean, population std, 10th/25th/50th/75th/90th
percentiles, interquartile range, mean absolute adjacent-frame change, and population
std of nonoverlapping 10-second window means for each descriptor. Quantiles are
exact NumPy linear quantiles. Windows use supplied frameTime in seconds, anchored
at zero; partial windows remain included. Window means receive equal weight.
Missing/nonfinite values are excluded per descriptor, zeros remain valid; adjacent
changes require both consecutive values finite (never bridge missing frames).
Entirely missing descriptors produce missing statistics; singleton/window std is
zero. No speaker separation, clinical interpretations or automatic voiced filtering.

Extraction reads only training archives without writing raw data. Bound each CSV
to 256 MiB and 500,000 rows, with two workers, and fail rather than truncate.
Cache per participant using archive size/mtime, extractor source hash, schema and
settings. Record the selected CSV SHA256. Verify enriched mean/std match Stage 3.
No dataset download or existing-table overwrite.

## Matched nested evaluation

Five stratified outer folds for each of seeds 42,137,2026; three stratified inner
folds inside each outer fitting set, seeded with outer seed plus fold index+1.
Use identical splits for every condition. Each participant has one outer-held-out
prediction per repeat. Every imputation, constant/duplicate removal and scaling
fit occurs inside the fitting fold via ParticipantPreprocessor pipelines.

Three model families: logistic regression (liblinear, max_iter 5000), linear SVM,
and RBF SVM (SVC, probability=False). For each, C=[0.01,0.1,1.0], class_weight=
[None,'balanced']; RBF gamma=['scale',0.01]. No resampling/SMOTE, GPU training,
HHO or separate feature selection. Logistic scores are logits and SVM scores are
signed decision margins; SVM margins must not be described as probabilities.

For each feature set/model family, evaluate two predeclared procedures:
1. Default decision cutoff zero (equivalent to probability 0.5 for logistic).
2. Inner-OOF-selected cutoff from [0,-0.5,0.5,-1,1].

For every candidate, concatenate three inner held-out score vectors and select
candidate/cutoff by pooled positive F1. First candidate and first cutoff win ties.
The default and tuned procedures may select different hyperparameters. Refit the
chosen pipeline on outer fitting participants and predict outer holdout once.
Never choose thresholds from outer labels. Reuse identical inner candidate fits
for the two procedures. This is a bounded pilot, not an exhaustive search.

Additionally repeat the exact Stage 4 selection recipe on the same outer folds:
mean/std logistic, five inner folds, C=[0.01,0.1,1], weights None/balanced, default
cutoff, mean inner-fold F1 selection. This anchors comparison to the existing
audio baseline without comparing training-CV scores with the earlier test score.

## Reporting and decision boundary

Save every candidate's inner scores, selected settings/cutoff, inner/outer participant
IDs, fitted pipelines, outer predictions, input/source hashes and environment.
Report pooled outer F1, precision, recall, specificity, accuracy and balanced
accuracy per repeat, with mean and population SD across the three repeats. SD
is not a CI and repeats are not independent cohorts. Report AUC per outer fold
and average it (do not pool margins from differently fitted SVMs). Include all
13 conditions, not just the best one. Compare richer/base features and tuned/default
threshold within model family/seed; compare every condition with the exact Stage 4
recipe. Preserve negative outcomes. Summary plots use fixed 0-1 F1 axes.

No test reevaluation, relabeling, additional searches or deployment. A promising
procedure needs independent confirmation; selecting the largest outer score is
itself exploratory. No guaranteed improvement or clinical-validity claim.

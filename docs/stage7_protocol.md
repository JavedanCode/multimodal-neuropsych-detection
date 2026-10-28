# Stage 7 protocol: Binary HHO and conventional sparse selection

Fixed on 2026-10-28 before real-data fitting. Target stays provisional supplied
PHQ_Binary, with official training/development partitions. Test is not evaluated.

## HHO variant and references

The continuous movement rules follow the exploration/besiege/rapid-dive structure
of Heidari et al., Harris hawks optimization: Algorithm and applications,
Future Generation Computer Systems 97 (2019), 849-872,
https://doi.org/10.1016/j.future.2019.02.028 . The author's reference implementation
was consulted to check the update equations and Levy-flight distribution:
https://github.com/aliasgharheidaricom/Harris-Hawks-Optimization-Algorithm-and-Applications/blob/master/HHO.m

This is an explicitly specified local binary adaptation, not a claim of exact
replication of a particular published BHHO variant or a novel optimizer. Latent
positions are clipped to [-6,6]; independent sigmoid probabilities generate binary
masks. One decoded mask and score stay paired with each position. Rapid-dive
acceptance compares those stored scores, without resampling the incumbent mask.
Levy steps use Mantegna normal draws with beta=1.5 and a numerical denominator floor.
Each iteration freezes the population/mean for proposals while updating the global
best immediately. This differs from the author's in-place population updates.

Use 8 hawks and 8 update iterations, seeds 42/137/2026. Include all features as an
initial reference. Repair an absent modality by enabling the largest-latent-score
feature in that modality. Penalize subsets that lose a modality after fold-level
preprocessing. Cache identical masks within a run. Record every unique mask,
fitness, fold scores, elapsed time, branch counts and best-so-far convergence.
This is a bounded pilot budget, not a search-optimality claim.

## Fitness and conventional comparison

144 named acoustic/facial summary descriptors are eligible; quality signals,
labels, scores, IDs and demographics are not selectable predictor descriptors.
Fitness = 0.95*(1 - mean training-CV positive-class F1) + 0.05*(selected/144).
Weights are configurable, nonnegative, and sum to one. Invalid subsets score 2,
strictly worse than any valid fitness in [0,1].

Fitness estimator: logistic regression, C=0.1, balanced class weights, liblinear,
max_iter=5000, fixed threshold 0.5. Use five stratified folds of official training
participants. Imputation, constant/duplicate detection and scaling fit on each
fitting fold for each mask. No development or test values enter fitness. Scores
are selection statistics and are not unbiased generalization estimates.

Conventional alternative: L1 logistic regression, C=0.1, balanced weights, fitted
on training data only after training-fitted preprocessing. Select coefficients
with absolute value >1e-8. If a modality has none, retain its largest absolute
coefficient (first feature breaks ties), and report this forced fallback explicitly.
C is fixed, not selected on development data. Do not report a CV estimate with an
L1 mask learned outside those folds.

## Downstream evaluation

Train both adaptive and quality-conditioned fusion on each selected subset, with
matched all-feature controls. Use fixed hidden=16, dropout=0.2, AdamW lr=0.001,
weight_decay=0.001, epochs=50, threshold=0.5 for all methods and seeds. Class weights,
descriptor preprocessing and quality means use official train only. Do not perform
a second CV search with a mask already selected using the entire training set.
If downstream tuning is added later, selectors must be nested inside its folds.

This fixed-training protocol differs from Stage 5/6 epoch selection; comparisons
must use the new matched all-feature controls, not cherry-picked older results.
Select once per method/seed and reuse that mask for both fusion types. Count
selection time once; total-per-model timings include it but must not be summed as
if selection ran twice. Report original/selected/retained feature counts, names
and modalities, metrics/confusion matrices, runtime, convergence, checkpoint
verification, source/data hashes and all three seeds. No best-seed selection.

Logistic fitness may not transfer to neural fusion. Small development-set changes
are exploratory; seed SD is not a confidence interval. No guaranteed accuracy or
speed improvement, clinical diagnosis, benchmark-superiority or novelty claim.
Stage 8 explanation/faithfulness work and final evaluation remain separate stages.

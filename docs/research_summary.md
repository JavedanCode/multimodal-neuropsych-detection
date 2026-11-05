# Research summary — 2026-11-04

## Objective and scope

Investigate Reliability-Aware Sparse Adaptive Fusion (R-SAF) for depression-label
classification from E-DAIC acoustic and facial behavioral features, with later
text-based comparisons. R-SAF is a provisional project name, not established
algorithmic novelty or a clinical diagnostic system. OpenFace action units are
facial behavioral measurements, not validated micro-expression detection.

## Process

1. Inventory archives, official partitions, supplied targets and feature schemas.
2. Preserve raw data; audit missing measurements, timing and target discrepancies.
3. Fit preprocessing within training folds; keep each participant in one partition.
4. Establish audio, facial and concatenated classical baselines.
5. Compare equal-weight, adaptive and quality-conditioned compact neural fusion.
6. Compare all descriptors, Binary HHO selection and L1 selection.
7. Explain models with Integrated Gradients and test attribution faithfulness.
8. Freeze the original model roster and evaluate official development/test sets.
9. Conduct separately documented exploratory improvements using training data only:
   audio representations, screened text, MPNet, neural pooling, quality fusion,
   thresholds, topic summaries and estimated interviewer-prompt exclusion.

The original data has 163/56/56 train/development/test participants. Later matched
studies retain 161 training participants, including 37 released positives. Five-fold
outer/three-fold inner CV is repeated with seeds42/137/2026. The revised folds also
balance interview domain and label. Settings, preprocessing and cutoffs use inner
training data. Repeated adaptive research on the same cohort still limits claims.

## Methods

- Acoustic eGeMAPS descriptors and OpenFace pose/gaze/action-unit measurements.
- Training-fitted imputation, scaling, constant/duplicate handling.
- Logistic regression, linear/RBF SVM and compact neural modality encoders.
- Adaptive gating with and without measured quality inputs; explicit availability.
- Binary HHO: local sigmoid-based binary adaptation, eight hawks/eight updates;
  fitness 0.95*(1-CV F1)+0.05*(selected/144), compared with L1 selection.
- Integrated Gradients, global and participant-level attribution, modality
  contributions, completeness checks, perturbation and ranking stability.
- Frozen MiniLM/MPNet text encoders. Strong reference: screened MPNet utterance
  mean, training-fitted standardized whitened PCA32, linear SVM.
- Later fusion: nested cross-fitted stacking with observed quality interactions;
  conservative auxiliary residuals with text fallback; missing-input tests.
- Inner-selected precision cutoffs, with a 65% recall floor and limited F1 loss.
- Fixed semantic-topic summaries and provisional transcript-context speaker filters.

## Results: separate evaluation settings

Original official-test results (56 people, mean F1 across three seeds):

| Family | Test F1 |
|---|---:|
| Audio logistic | 56.3% |
| Facial logistic | 44.8% |
| Concatenation logistic | 48.5% |
| All-feature equal-weight fusion | 52.4% |
| All-feature adaptive fusion | 53.8% |
| All-feature quality-aware fusion | 53.8% |
| HHO adaptive/quality fusion | 47.4% |
| L1 adaptive/quality fusion | 46.8% |

Source: `stage9_results.md`. No central fusion/HHO improvement was demonstrated.
Explanation tests nevertheless verified numerical completeness for 1,008 rows;
top-ranked feature perturbations exceeded random perturbations in all 54 aggregate
model/fraction comparisons. Rankings were unstable; these are not clinical biomarkers.

Revised training-CV comparison (161 people; not official-test performance):

| Procedure | Precision | Recall | F1 | Mean fold AUC |
|---|---:|---:|---:|---:|
| Retained MPNet SVM | 47.3% | 73.9% | 57.5% | 0.805 |
| Inner-selected text cutoff | 49.9% | 72.1% | 58.5% | 0.805 |
| Ordinary aligned fusion | 39.3% | 81.1% | 52.9% | 0.782 |
| Quality-aware aligned fusion | 39.5% | 81.1% | 53.1% | 0.770 |
| Conservative quality residual | 49.4% | 68.5% | 57.1% | 0.802 |
| Topic-balanced policy | 44.4% | 69.4% | 53.9% | 0.782 |
| Concatenated-topic policy | 46.2% | 71.2% | 55.9% | 0.794 |
| Regularized attention pooling | 37.6% | 64.0% | 47.2% | 0.749 |
| Context-supported prompt exclusion | 44.7% | 75.7% | 56.2% | 0.804 |
| Broader prompt exclusion | 44.3% | 73.0% | 55.1% | 0.804 |

Latest speaker-filter run: `speaker_filter_v4_20261104T172832492344Z`.
The filters removed 625/629 of 14,724 screened utterances, respectively, without
dropping participants. Neither improved the original reference. Verification
restored all 90 models, checked 2,898 predictions and replayed all 1,080 inner
candidate score vectors. The earlier partial run has no final manifest and is
excluded. Speaker-role estimates remain provisional; this result does not establish
that interviewer contamination is harmless or that either filter is accurate.

Sources: `pipeline_v2_results.md` and `bounded_v3_results.md`. The threshold policy
has 76.2% accuracy and is a modest candidate improvement, below the predeclared
promotion criteria. Its conditional F1-difference interval includes zero. Original
reference accuracy is 74.7%. Historical different-fold point estimates must remain
separate. No fresh independent holdout remains.

## Difference from the benchmark paper

Benchmark: Khan, Khan and Aljubayri, *A Comprehensive Framework for Multi-Modal
Depression Detection: Integrating Adaptive Fusion, Fairness Regularization, and
Explainable AI*, Mathematics 14(4),711 (2026),
https://www.mdpi.com/2227-7390/14/4/711 .

The paper describes dynamic gating, multi-head attention, fairness regularization
and explainability. Our project concentrates on explicit observed input-quality
signals, sparse feature selection via Binary HHO versus L1, and controlled
ablation/faithfulness tests on compact local models. The proposed contribution is
an empirical investigation of this combination, not invention of gating, HHO,
quality-aware fusion or explainability. No exhaustive literature claim or verified
first-of-its-kind claim is supported.

The brief attributes 91.4% F1 and 93.0% accuracy on DAIC-WOZ to the benchmark.
Those claims are not a matched comparison with our E-DAIC released-label setup.
The publisher page was unavailable during this summary's retrieval, so those
numbers remain attributed to the brief rather than newly verified here. A numeric
gap must not be described as a controlled head-to-head experimental difference.

## Limits and present interpretation

Twenty released labels disagree with the score>=10 definition; twelve are in the
retained training cohort. Labels remain unchanged. AI interviews are scarce in
training and constitute the historical test cohort. Speaker attribution is absent
from the supplied transcript schema: human/contextual review supports interviewer
prompts in some rows, but full-corpus roles remain estimated. Quality proxies are
not calibrated reliability probabilities. Reproducibility checks and tests support
implementation correctness, not clinical validity or guaranteed generalization.

The strongest current candidate uses text. The intended reliability-aware sparse
fusion advantage has not been demonstrated. This distinction belongs in the
project's conclusion even while further targeted improvements are investigated.

## Latest matched primary-branch audit (2026-11-04)

The [v6 comparison](branch_audit_v6_results.md) tests original, aligned and richer
audio/facial summaries with training-only selection. Facial F1 improves from
33.7% to about 37.5%; both revised facial branches pass their relative screens.
Audio revisions and fusion changes do not pass. Original concatenation scores
37.2% F1, versus 28.3% for matched quality neural fusion. This does not replace
the retained reference or demonstrate a quality-gating advantage. All 195 models,
7,245 predictions, 540 inner candidate vectors and 45 neural retrainings were
verified. These remain exploratory estimates on reused training participants.

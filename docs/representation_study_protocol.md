# Pretrained audio and temporal facial pilot

Frozen 2026-10-29 before extraction and model fitting. Exploratory follow-up to
the paper comparison, not a reproduction of its architecture or headline scores.
No new development/test predictions, relabeling, or promotion based on test scores.

Use the same 161 official training participants and unchanged released labels as
the preceding segmentation pilot. Its two timing-quarantined recordings remain
excluded from all conditions, including controls. No transcript filtering.

The supplied DenseNet201 CSV contains 1,920 neuron activations per audio window.
The official E-DAIC manual identifies this as pretrained-network audio features
from four-second mel-spectrogram windows with one-second hops:
https://dcapswoz.ict.usc.edu/wwwedaic/E-DAIC%20Manual.pdf
Use each neuron's whole-session mean and population SD (3,840 inputs), keeping
zeros and ignoring nonfinite activations. Record timestamp diagnostics, member
hashes, and archive stats. These supplied embeddings are not Wav2Vec2, and their
source extraction/alignment has not been independently reproduced.

For each of the 49 OpenFace descriptors retain the existing mean and population
SD and add p10, p25, median, p75, p90, IQR, mean absolute consecutive-frame change,
and SD of ten-second window means: 490 features. Use confidence >=0.8 and success
==1, finite values, actual nondecreasing timestamps. Consecutive change must not
bridge rejected/missing frames or timestamp gaps >0.1 seconds. Ten-second windows
are anchored at zero and equally weighted; partial windows remain. These summarize
temporal variation, but do not preserve a full sequence or audio/face alignment.
Verify original facial mean/SD against existing tables to 1e-8 tolerance.

Seven feature conditions: original audio; original facial; original concatenation;
temporal facial; DenseNet audio; DenseNet + original facial; DenseNet + temporal
facial. Fit logistic and linear SVM per condition, plus constant references.
DenseNet branch preprocessing: fitting-fold median imputation (empty -> zero),
remove constant columns, standardize, PCA to min(32,n_fit-1,n_features) components
with randomized solver/random_state 42 and whitening. Other descriptors use the
existing fitting-fold ParticipantPreprocessor. PCA never sees inner or outer
validation participants. All preprocessing is saved with the classifier.

Reuse the preceding pilot's exact matched five outer folds x seeds 42,137,2026,
three inner folds, C=[.01,.1,1], class_weight=[None,balanced], fixed zero margin
cutoff. Choose by pooled inner F1, first candidate breaks ties. No new thresholds,
architecture search, subgroup optimization or post-result additional searches.
Report all conditions, seed mean/SD (not CI), precision, recall, F1, specificity,
accuracy, balanced accuracy and mean fold AUC. Save all candidate scores, fold IDs,
restored pipelines, predictions, subgroup diagnostics and provenance.

Primary comparisons: temporal facial vs original facial; DenseNet vs original
audio; DenseNet+temporal facial vs original concatenation, matched within classifier.
An exploratory candidate merits a further reliability study only if mean F1 rises
at least 0.03, balanced accuracy and mean fold AUC do not decrease, recall drops
no more than 0.05, and F1 improves in at least two repeats against its matched
control. This is a project screening rule, not statistical significance or an
independent confirmation claim. Preserve negative findings. Do not automatically
replace the reference or claim improvement simply because one score is largest.

First test a stronger representation foundation. If no condition clears that rule,
finish and report this bounded study before implementing a larger reliability model.
The paper's 90%+ results and proposed project milestones are goals, not guarantees.

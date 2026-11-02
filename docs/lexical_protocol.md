# Screened lexical and semantic text comparison

Fixed 2026-11-02 before extraction/fitting. Distinct follow-up to the negative
text-refinement study. Same 161 official training participants, unchanged
released provisional labels and existing disclosure/neighbor screen. No new
development/test predictions, restored disclosures, or label-derived features.

Extract only training transcript members with bounded safe archive reads. Reuse
the exact screen and verify transcript hashes and screen counts against the
previous text study. Hash lowercase word unigrams and bigrams into 65,536 fixed
nonnegative bins, without normalization; sum within-utterance counts per person.
Do not introduce bigrams crossing utterance boundaries. Save only sparse counts
and audit metadata locally under ignored processed data, not transcript text.
Hashing is label/corpus-independent; collisions are possible and these are not
interpretable named word features. Fit sublinear-TF, smoothed IDF and L2
normalization using fitting participants only, including each inner fold.

Compare lexical-only and semantic-plus-lexical logistic/linear SVM, plus exact
previous PCA32 semantic controls. Semantic preprocessing is the previous
fold-fitted standardized whitened PCA32; normalize the resulting participant
vector to L2 before concatenation with lexical TF-IDF, using semantic relative
weights .25/1/4 (inner selected). Lexical-only has no semantic branch. Search
C .01/.1/1/10 and weights none/balanced for new conditions. Controls retain
C .01/.1/1, weights none/balanced and margin cutoff zero. New conditions use
default zero and an inner-selected cutoff among 0/-.5/.5.

Same 5 outer folds x seeds 42/137/2026 and 3 inner folds (seed+fold+1).
Choose by pooled inner F1, first tie. Report each family/representation procedure,
and an inner-selected lexical-or-combined procedure. The latter is the primary
new search procedure; no outer outcome selects its settings. Include constants,
all metrics, repeat SD and AI/WoZ descriptions. Primary improvement screen versus
previous text SVM: +3 F1 points, no balanced-accuracy or AUC loss, recall loss
<=5 points, positive differences in >=2 repeats. Retain all negative results.
This is further exploration of repeatedly used training data, not fresh evidence
of generalization or a claim of the best possible model. Stop this registered
round at its conclusion; do not expand its settings based on outer outcomes.

# Text refinement and conservative late fusion

Frozen 2026-10-31 before fitting. Preserve the 161-person official training
cohort, screened transcripts, released provisional labels, and previous outer
folds (5 folds, seeds 42/137/2026). No development/test scoring, relabeling,
diagnostic-disclosure restoration or outcome-based participant exclusions.

Rationale: the previous text classifier discards dimensions through whitened
PCA32. Test retaining all 384 dimensions, and adding within-interview variability
from the existing screened text bins. Rich text concatenates utterance mean,
population SD and maximum of valid bin embeddings (1152 values); this is a
distribution summary, not a new order-aware model. No new encoder extraction.

Use 3 inner participant folds (seed outer_seed+fold+1) for all selection.
Retain exact old PCA32 logistic and linear-SVM controls (C .01/.1/1,
weight none/balanced, cutoff zero). New text search: mean or rich features,
StandardScaler or participant L2 normalization, logistic/linear SVM/RBF SVM,
C .01/.1/1, class_weight none/balanced; RBF gamma=scale. Thresholds are
0/-.5/.5 decision margin. Preprocessing fits only on inner fitting data.
Report each fixed feature/scaler family's default and inner-threshold-selected
procedure, and a search procedure selecting all choices on inner pooled F1,
first tie in declared order. Never select candidates on outer predictions.

Late fusion uses probabilities from independently trained logistic branches:
full mean text, original audio, original face. Each branch selects C/weight on
inner F1 using fixed .5 probability threshold (equivalent to zero margin).
Text scaler is standard scaling. The fusion candidates assign total weight
1/.9/.75/.5 to text, sharing remaining weight equally between audio and face.
Choose weights and probability cutoff .5/.4/.6 using the same inner predictions.
Their reuse is internal tuning, so only outer outcomes estimate performance.
Report fixed text-only and selected-fusion procedures. This includes weight=1
so fusion can reject weak modalities; it is not quality-aware fusion.

The overall selected strategy chooses between the selected text search and
selected fusion by their inner F1. Save candidate scores, fold IDs, selected
parameters, preprocessing and classifiers, and reload every saved outer model.
Fit no new final model unless its research purpose is clearly recorded.

Primary screen against each previous text control: mean F1 gain >=3 points,
no balanced-accuracy/AUC decrease, recall decrease <=5 points, gain in >=2 of
3 repeats. Also report all metrics, constant references, seed SD, and AI/WoZ
subgroups. Repeated adaptive use of these folds is exploratory, not fresh
confirmation. A broad search may underperform simpler fixed procedures.
Stop this round after the declared comparisons; do not expand it based on its
outer-fold outcomes. Retain failed candidates and the strongest existing reference.

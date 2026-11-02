# Frozen MPNet comparison

Fixed 2026-11-02 before extraction or fitting. Distinct encoder experiment after
negative text-refinement and lexical studies. Same 161 training participants,
unchanged provisional released labels, disclosure screen, utterance weighting,
and original five outer folds x seeds 42/137/2026. No new dev/test predictions.

Download only public sentence-transformers/all-mpnet-base-v2 model/tokenizer
files; pin the resolved revision and hashes. Load local safetensors without
remote code. Its model card documents 768-dimensional sentence embeddings and
attention-mask mean pooling followed by normalization:
https://huggingface.co/sentence-transformers/all-mpnet-base-v2
It is not a depression-trained model; greater size is not evidence of superiority.

Use exactly the existing MiniLM encoding helper: 254-token content chunks plus
special tokens (256 total), batch32, float32 inference, mean pooling over valid
tokens, L2 normalization per chunk, average chunks per utterance, then equal
utterance-mean pooling per participant. Thus MPNet does not use its larger default
384-token limit in this comparison. Tokenization and encoder both differ.
Read only training transcripts, verify source hashes and screening counts against
the earlier study, and save embeddings/metadata locally, not transcript text.
Use encoder eval mode, no gradients, TF32 off, eager attention, and seed42.
Record CUDA memory, versions, model loading information and per-participant cache
hashes. Replay one participant to check saved mean and utterance embeddings.

Compare MiniLM and MPNet separately with the same TextPreprocessor (fit-only
standardization and whitened PCA32), logistic/linear SVM, C .01/.1/1 and class
weights none/balanced. Three inner folds (seed+fold+1), pooled inner F1, first
tie, fixed zero decision margin. No new threshold, PCA or fusion search. Include
constant controls, all metrics and AI/WoZ descriptions. Restore saved pipelines
and reproduce prior MiniLM scores. Primary screen within classifier: >=3 F1
points gain, no balanced-accuracy or AUC loss, recall decrease <=5 points, and
improvement in >=2 repeats. Reused training folds remain exploratory. Stop the
registered comparison after these conditions and retain the stronger reference.

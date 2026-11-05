"""Verify the full-training artifact without reporting fitting-set performance."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.baselines import sha256,TARGET_POLICY


def verify(project,root):
    manifest=json.loads((root/'manifest.json').read_text())
    audit=json.loads((root/'selection.json').read_text())
    assert manifest['target_policy']==TARGET_POLICY and not manifest['test_evaluated'] and not manifest['development_evaluated']
    for name,digest in manifest['source_hashes'].items():assert sha256(project/name)==digest,name
    assert sha256(root/'model.joblib')==manifest['model_sha256']
    x=pd.read_parquet(project/'data/processed/mpnet_v1/features.parquet')
    y=pd.read_parquet(project/'data/processed/mpnet_v1/labels.parquet').PHQ_Binary
    official=pd.read_parquet(project/'data/processed/labels.parquet')
    assert x.index.equals(y.index) and list(y.index)==manifest['participants']
    assert (official.loc[y.index,'split']=='train').all()
    pd.testing.assert_series_equal(y,official.loc[y.index,'PHQ_Binary'])
    assert audit['selected']==max(audit['candidates'],key=lambda r:r['selection_f1'])
    assert audit['selected']['parameters']==manifest['selected_parameters']
    for (a,b),record in zip(StratifiedKFold(3,shuffle=True,random_state=43).split(x,y),audit['folds']):
        assert record=={'fit_ids':list(y.index[a]),'heldout_ids':list(y.index[b])}
    model=joblib.load(root/'model.joblib');prep=model.named_steps['preprocess'];classifier=model.named_steps['classifier']
    assert prep.input_columns_==manifest['input_columns']==list(x)
    assert prep.pca_.n_components_==32 and classifier.n_features_in_==32
    for key,value in manifest['selected_parameters'].items():assert classifier.get_params()[key]==value
    np.testing.assert_array_equal(classifier.classes_,[0,1])
    np.testing.assert_allclose(prep.scale_.mean_,x.to_numpy(dtype=float).mean(0),atol=1e-12)
    np.testing.assert_allclose(prep.scale_.var_,x.to_numpy(dtype=float).var(0),atol=1e-12)
    score=model.decision_function(x)
    assert np.isfinite(score).all()
    np.testing.assert_array_equal(score,joblib.load(root/'model.joblib').decision_function(x))
    result={'training_participants':len(y),'feature_count':x.shape[1],'pca_dimensions':32,
            'selected_parameters':manifest['selected_parameters'],'hashes_folds_preprocessing_and_replay_verified':True,
            'new_performance_estimate':False,'test_evaluated':False,'deployed':False}
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('reference',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):print(verify(Path(__file__).resolve().parents[2],args.reference))

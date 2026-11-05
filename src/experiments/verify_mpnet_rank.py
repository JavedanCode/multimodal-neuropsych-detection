"""Check nested rank/cutoff choices, saved models and the prior MPNet control."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.mpnet_study import inputs
from src.experiments.verification_utils import verify_outputs


def verify(project,out):
    sets,y=inputs(project);x=sets['mpnet']
    protocol,pred,metric_count=verify_outputs(project,out,y)
    restored=0;candidate_count=0
    for seed in protocol['seeds']:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(x,y)):
            audit=json.loads((out/f'fold_{seed}_{fold}.json').read_text())
            assert audit['fit_ids']==list(y.index[a]) and audit['heldout_ids']==list(y.index[b])
            inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(x.iloc[a],y.iloc[a]))
            for (ia,ib),saved in zip(inner,audit['inner_ids']):
                assert saved=={'fit':list(y.index[a][ia]),'heldout':list(y.index[a][ib])}
            candidates=audit['candidates']
            for r in candidates:
                scores=np.array(r['inner_scores'])
                assert scores.shape==(len(a),) and np.isfinite(scores).all()
                np.testing.assert_allclose(r['inner_f1'],f1_score(y.iloc[a],scores>=r['threshold'],zero_division=0),atol=1e-12)
                candidate_count+=1
            for name,selected in audit['selected'].items():
                pool=candidates
                if name=='previous':pool=[r for r in pool if r['rank']==32 and r['parameters']['C']<=1 and r['threshold']==0]
                elif name.startswith('rank'):
                    rank=int(name.split('_')[0][4:]);pool=[r for r in pool if r['rank']==rank]
                if name.endswith('_default'):pool=[r for r in pool if r['threshold']==0]
                assert selected==max(pool,key=lambda r:r['inner_f1'])
                bundle=joblib.load(out/'models'/f'{name}_{seed}_{fold}.joblib')
                assert bundle['threshold']==selected['threshold']
                assert bundle['model'].named_steps['preprocess'].rank==selected['rank']
                raw=bundle['model'].decision_function(x.iloc[b]);scores=raw-bundle['threshold']
                g=pred[(pred.condition==name)&(pred.seed==seed)&(pred.fold==fold)].set_index('participant_id').loc[y.index[b]]
                np.testing.assert_allclose(g.raw_score,raw,atol=1e-10);np.testing.assert_allclose(g.score,scores,atol=1e-10)
                restored+=1
    old=pd.read_parquet(project/'experiments/runs/mpnet_20261102T171156498575Z/predictions.parquet')
    old=old[old.condition=='mpnet_linear'].set_index(['seed','fold','participant_id'])
    current=pred[pred.condition=='previous'].set_index(['seed','fold','participant_id']).loc[old.index]
    np.testing.assert_allclose(current.score,old.score,rtol=1e-8,atol=1e-8)
    np.testing.assert_array_equal(current.prediction,old.prediction)
    result={'restored_bundles':restored,'prediction_rows':len(pred),'metric_rows':metric_count,
            'inner_candidates_verified':candidate_count,'prior_control_predictions_matched':len(old),'folds_and_hashes_verified':True}
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):print(verify(Path(__file__).resolve().parents[2],args.run))

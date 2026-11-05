"""Independently verify every inner blend and saved model prediction."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.audio_study import SEEDS,estimator
from src.experiments.mpnet_study import inputs
from src.experiments.verification_utils import verify_outputs
from src.models.text_preprocessing import TextPreprocessor


def verify(project,out):
    sets,y=inputs(project);_,pred,nmetrics=verify_outputs(project,out,y);count=0
    for seed in SEEDS:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(sets['mpnet'],y)):
            audit=json.loads((out/f'fold_{seed}_{fold}.json').read_text());target=y.iloc[a];held=y.iloc[b]
            assert audit['fit_ids']==list(target.index) and audit['heldout_ids']==list(held.index)
            splits=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(a,target))
            for j,(ia,ib) in enumerate(splits):
                assert audit['inner_ids'][j]=={'fit':list(target.index[ia]),'heldout':list(target.index[ib])}
            for r in audit['candidates']:
                first=np.array(audit['inner_scores']['mpnet'][r['mpnet']]['scores'])
                second=np.array(audit['inner_scores']['minilm'][r['minilm']]['scores'])
                assert first.shape==second.shape==(len(target),) and np.isfinite(first).all() and np.isfinite(second).all()
                blended=r['weight']*first+(1-r['weight'])*second
                np.testing.assert_allclose(r['inner_f1'],f1_score(target,blended>=0),atol=1e-12);count+=1
            assert audit['selected']==max(audit['candidates'],key=lambda r:r['inner_f1'])
            assert audit['reference']==max([r for r in audit['candidates'] if r['weight']==1],key=lambda r:r['inner_f1'])
            checked=set()
            for name,spec in [('reference',audit['reference']),('blend',audit['selected'])]:
                bundle=joblib.load(out/'models'/f'{name}_{seed}_{fold}.joblib');assert bundle['weight']==spec['weight']
                scores=np.zeros(len(held))
                for encoder,w in [('mpnet',spec['weight']),('minilm',1-spec['weight'])]:
                    if not w:assert encoder not in bundle['models'];continue
                    model=bundle['models'][encoder];x=sets[encoder]
                    np.testing.assert_allclose(model['preprocess'].scale_.mean_,x.iloc[a].to_numpy(dtype=np.float64).mean(0),atol=1e-12)
                    scores+=w*model.decision_function(x.iloc[b])
                    key=(encoder,spec[encoder])
                    if key not in checked:
                        r=audit['inner_scores'][encoder][spec[encoder]];replay=np.full(len(target),np.nan)
                        for ia,ib in splits:
                            prep=TextPreprocessor().fit(x.iloc[a[ia]])
                            inner=estimator('linear',r['parameters'],seed).fit(prep.transform(x.iloc[a[ia]]),target.iloc[ia])
                            replay[ib]=inner.decision_function(prep.transform(x.iloc[a[ib]]))
                        np.testing.assert_allclose(replay,r['scores'],atol=1e-12,rtol=1e-12);checked.add(key)
                g=pred[(pred.condition==name)&(pred.seed==seed)&(pred.fold==fold)].set_index('participant_id').loc[held.index]
                np.testing.assert_allclose(scores,g.score,atol=1e-12,rtol=1e-12)
    old=pd.read_parquet(project/'experiments/runs/mpnet_20261102T171156498575Z/predictions.parquet')
    old=old[old.condition=='mpnet_linear'].set_index(['seed','fold','participant_id']).sort_index()
    current=pred[pred.condition=='reference'].set_index(['seed','fold','participant_id']).sort_index()
    assert old.index.equals(current.index)
    np.testing.assert_allclose(old.score,current.score,atol=1e-12,rtol=1e-12)
    result={'restored_bundles':30,'prediction_rows':len(pred),'metric_rows':nmetrics,'inner_combinations_checked':count,
            'prior_reference_scores_matched':len(current),'hashes_folds_scalers_selection_inner_refits_verified':True,'test_evaluated':False}
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    with threadpool_limits(limits=1):print(verify(Path(__file__).resolve().parents[2],Path(sys.argv[1])))

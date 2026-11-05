"""Replay saved models and independently audit sampling and nested selections."""
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.audio_study import SEEDS
from src.experiments.mpnet_study import inputs
from src.experiments.verification_utils import verify_outputs
from src.models.undersampling import decision


def check_samples(samples, fit, forbidden, ratio, members):
    assert len(samples)==members
    positives=set(fit.index[fit==1]); negatives=fit[fit==0]
    for ids in samples:
        assert len(ids)==len(set(ids)) and set(ids)<=set(fit.index)
        assert set(ids).isdisjoint(forbidden) and positives<=set(ids)
        if ratio is None:
            assert set(ids)==set(fit.index)
        else:
            n=min(len(negatives),ratio*len(positives))
            assert len(ids)==len(positives)+n
            selected_negative=fit.loc[ids];selected_negative=selected_negative[selected_negative==0]
            for domain in (False,True):
                expected=n*sum((negatives.index>=600)==domain)/len(negatives)
                assert abs(sum((selected_negative.index>=600)==domain)-expected)<1.000001


def verify(project,out):
    sets,y=inputs(project);x=sets['mpnet']
    _,pred,nmetrics=verify_outputs(project,out,y)
    definitions={'reference':(None,1),'weighted':(None,1),'single_1':(1,1),'single_2':(2,1),
                 'ensemble_1':(1,7),'ensemble_2':(2,7)}
    restored=0;checked=0
    for seed in SEEDS:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(x,y)):
            audit=json.loads((out/f'fold_{seed}_{fold}.json').read_text());fit=y.iloc[a];held=y.iloc[b]
            assert audit['fit_ids']==list(fit.index) and audit['heldout_ids']==list(held.index)
            splits=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(x.iloc[a],fit))
            for j,(ia,ib) in enumerate(splits):
                assert audit['inner_ids'][j]=={'fit':list(fit.index[ia]),'heldout':list(fit.index[ib])}
            for record in audit['candidates']:
                np.testing.assert_allclose(record['inner_f1'],f1_score(fit,np.array(record['inner_scores'])>=0),atol=1e-12)
                ratio,members=definitions[record['procedure']]
                for j,(ia,ib) in enumerate(splits):
                    check_samples(record['inner_sample_ids'][j],fit.iloc[ia],set(fit.index[ib])|set(held.index),ratio,members)
                checked+=1
            for name,spec in audit['selected'].items():
                eligible=[r for r in audit['candidates'] if (r['procedure'] not in ('reference','weighted') if name=='selected_sampling' else r['procedure']==name)]
                assert spec==max(eligible,key=lambda r:r['inner_f1'])
                bundle=joblib.load(out/'models'/f'{name}_{seed}_{fold}.joblib')
                assert bundle['sample_ids']==audit['outer_sample_ids'][name]
                check_samples(bundle['sample_ids'],fit,set(held.index),*definitions[spec['procedure']])
                g=pred[(pred.condition==name)&(pred.seed==seed)&(pred.fold==fold)].set_index('participant_id').loc[held.index]
                np.testing.assert_allclose(decision(bundle,x.loc[held.index]),g.score,atol=1e-12,rtol=1e-12)
                restored+=1
    old=pd.read_parquet(project/'experiments/runs/mpnet_20261102T171156498575Z/predictions.parquet')
    old=old[old.condition=='mpnet_linear'].set_index(['seed','fold','participant_id']).sort_index()
    current=pred[pred.condition=='reference'].set_index(['seed','fold','participant_id']).sort_index()
    assert old.index.equals(current.index)
    np.testing.assert_allclose(old.score,current.score,atol=1e-12,rtol=1e-12)
    result={'restored_bundles':restored,'prediction_rows':len(pred),'metric_rows':nmetrics,
            'inner_candidates_checked':checked,'prior_control_predictions_matched':len(current),
            'sampling_folds_hashes_and_metrics_verified':True,'test_evaluated':False}
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):print(verify(Path(__file__).resolve().parents[2],Path(sys.argv[1])))

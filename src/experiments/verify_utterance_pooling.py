"""Independent saved-prediction and fold audit for learned pooling."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.audio_study import SEEDS
from src.experiments.utterance_pooling import load,configure
from src.experiments.verification_utils import verify_outputs
from src.models.utterance_pooling import predict,scale_fit,scale_transform,train


def verify(project,out):
    configure();features,x,mask,y,_=load(project)
    _,pred,nmetrics=verify_outputs(project,out,y);restored=0;inner_replays=0
    for seed in SEEDS:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(features,y)):
            audit=json.loads((out/f'fold_{seed}_{fold}.json').read_text());target=y.iloc[a];held=y.iloc[b]
            assert audit['fit_ids']==list(target.index) and audit['heldout_ids']==list(held.index)
            splits=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(a,target))
            for j,(ia,ib) in enumerate(splits):
                assert audit['inner_ids'][j]=={'fit':list(target.index[ia]),'heldout':list(target.index[ib])}
            for r in audit['candidates']:
                scores=np.array(r['scores']);assert scores.shape==(len(target),) and np.isfinite(scores).all()
                np.testing.assert_allclose(r['inner_f1'],f1_score(target,scores>=0),atol=1e-12)
            if fold==0:
                ia,ib=splits[0];stats=scale_fit(x[a[ia]],mask[a[ia]])
                z=scale_transform(x[a[ia]],mask[a[ia]],stats);v=scale_transform(x[a[ib]],mask[a[ib]],stats)
                for attention in (False,True):
                    _,replay=train(z,mask[a[ia]],target.iloc[ia],attention,seed+fold,100,(v,mask[a[ib]]))
                    for epoch,scores in replay.items():
                        r=next(r for r in audit['candidates'] if r['attention']==attention and r['epochs']==epoch)
                        np.testing.assert_allclose(scores,np.array(r['scores'])[ib],atol=1e-6,rtol=1e-6)
                        inner_replays+=1
            for name,spec in audit['selected'].items():
                eligible=[r for r in audit['candidates'] if name=='selected' or r['attention']==(name=='attention')]
                assert spec==max(eligible,key=lambda r:r['inner_f1'])
                bundle=joblib.load(out/'models'/f'{name}_{seed}_{fold}.joblib')
                assert bundle['attention']==spec['attention'] and bundle['epochs']==spec['epochs']
                for actual,expected in zip(bundle['stats'],scale_fit(x[a],mask[a])):np.testing.assert_array_equal(actual,expected)
                scores=predict(bundle,x[b],mask[b])
                g=pred[(pred.condition==name)&(pred.seed==seed)&(pred.fold==fold)].set_index('participant_id').loc[held.index]
                np.testing.assert_allclose(scores,g.score,atol=1e-6,rtol=1e-6);restored+=1
    old=pd.read_parquet(project/'experiments/runs/mpnet_20261102T171156498575Z/predictions.parquet')
    old=old[old.condition=='mpnet_linear'].set_index(['seed','fold','participant_id']).sort_index()
    current=pred[pred.condition=='reference'].set_index(['seed','fold','participant_id']).sort_index()
    assert old.index.equals(current.index)
    np.testing.assert_allclose(old.score,current.score,atol=1e-12,rtol=1e-12)
    metrics=pd.read_csv(out/'repeat_metrics.csv').set_index(['condition','seed'])
    screens={}
    for name in ['mean','attention','selected']:
        diff=metrics.loc[name].select_dtypes('number')-metrics.loc['reference'].select_dtypes('number');mean=diff.mean()
        checks={'precision_gain_5pp':bool(mean.precision>=.05),'recall_loss_at_most_5pp':bool(mean.recall>=-.05),
                'recall_at_least_65':bool(metrics.loc[name].recall.mean()>=.65),'f1_loss_at_most_2pp':bool(mean.f1>=-.02),
                'precision_gain_two_repeats':bool((diff.precision>0).sum()>=2)}
        screens[name]={'checks':checks,'passes':all(checks.values())}
    (out/'precision_screens.json').write_text(json.dumps(screens,indent=2))
    result={'restored_bundles':restored,'prediction_rows':len(pred),'metric_rows':nmetrics,
            'inner_candidate_records_checked':60,'sampled_inner_checkpoint_replays':inner_replays,'prior_reference_scores_matched':len(current),
            'cache_hashes_folds_scalers_selection_metrics_verified':True,'test_evaluated':False}
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    with threadpool_limits(limits=1):print(verify(Path(__file__).resolve().parents[2],Path(sys.argv[1])))

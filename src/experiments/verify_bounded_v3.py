"""Replay feature construction, inner selection, saved inference and metrics."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from src.experiments.baselines import sha256
from src.experiments.bounded_v3 import load,inner_candidates,precision_screen
from src.experiments.domain_v2 import splits
from src.experiments.audio_study import SEEDS
from src.models.quality_stack_v2 import take
from src.models.bounded_v3 import predict,SCENARIOS


def verify_topics(project,data,y):
    root=project/'data/processed/topics_v3';anchors=np.load(root/'anchors.npy').astype(float)
    m=json.loads((root/'manifest.json').read_text());prior=json.loads((project/'data/processed/mpnet_v1/manifest.json').read_text())
    for p,h in m['encoder'].items():assert sha256(project/'models/pretrained/all-mpnet-base-v2'/p)==h
    counts=pd.read_csv(root/'counts.csv',index_col=0)
    for pid in y.index:
        path=project/'data/processed/mpnet_v1/participant_cache'/f'{pid}.npz'
        assert sha256(path)==prior['participants'][str(pid)]['hash']
        with np.load(path) as z:x=z['embeddings'].astype(float)
        similarities=x@anchors.T/(np.linalg.norm(x,axis=1)[:,None]*np.linalg.norm(anchors,axis=1)[None,:])
        labels=np.argmax(similarities,axis=1)
        means=[]
        for k in range(len(anchors)):
            members=np.flatnonzero(labels==k)
            means.append(np.sum(x[members],axis=0)/len(members) if len(members) else np.sum(x,axis=0)/len(x))
        np.testing.assert_allclose(data['balanced'].loc[pid],np.mean(means,axis=0),atol=1e-10)
        np.testing.assert_allclose(data['concatenated'].loc[pid],np.concatenate(means),atol=1e-10)
        np.testing.assert_array_equal(counts.loc[pid],np.bincount(labels,minlength=len(anchors)))


def verify(project,out):
    m=json.loads((out/'manifest.json').read_text())
    for p,h in m['hashes'].items():assert sha256(project/p)==h
    for p,h in m['models'].items():assert sha256(out/'models'/p)==h
    data,y=load(project);verify_topics(project,data,y)
    pred=pd.read_parquet(out/'predictions.parquet');assert not pred.duplicated(['condition','scenario','seed','participant_id']).any()
    np.testing.assert_array_equal(pred.target,y.loc[pred.participant_id]);np.testing.assert_array_equal(pred.prediction,(pred.score>=0).astype(int))
    metrics=pd.read_csv(out/'repeat_metrics.csv').set_index(['condition','scenario','seed'])
    aps=pd.read_csv(out/'fold_average_precision.csv').set_index(['condition','scenario','seed','fold'])
    domains=pd.read_csv(out/'domain_metrics.csv').set_index(['condition','scenario','seed','domain'])
    summary=pd.read_csv(out/'summary.csv').set_index(['condition','scenario'])
    for key,g in pred.groupby(['condition','scenario','seed']):
        assert len(g)==len(y) and set(g.participant_id)==set(y.index)
        for suffix,v,row in [('',g,metrics.loc[key])]+[(d,v,domains.loc[(*key,d)]) for d,v in g.groupby(np.where(g.participant_id>=600,'AI','WoZ'))]:
            tn,fp,fn,tp=confusion_matrix(v.target,v.prediction,labels=[0,1]).ravel()
            expected={'tn':tn,'fp':fp,'fn':fn,'tp':tp,'precision':tp/max(1,tp+fp),'recall':tp/(tp+fn),
                      'f1':2*tp/max(1,2*tp+fp+fn),'accuracy':(tp+tn)/len(v),'specificity':tn/(tn+fp),
                      'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp))}
            for name,value in expected.items():np.testing.assert_allclose(row[name],value,atol=1e-12)
        auc=[]
        for fold,v in g.groupby('fold'):
            auc.append(roc_auc_score(v.target,v.score))
            np.testing.assert_allclose(aps.loc[(*key,fold),'average_precision'],average_precision_score(v.target,v.score),atol=1e-12)
        np.testing.assert_allclose(metrics.loc[key,'mean_fold_auc'],np.mean(auc),atol=1e-12)
    for key,g in metrics.reset_index().groupby(['condition','scenario']):
        for name in ['f1','precision','recall','accuracy','specificity','balanced_accuracy','mean_fold_auc']:
            np.testing.assert_allclose(summary.loc[key,name+'_mean'],g[name].mean(),atol=1e-12)
            np.testing.assert_allclose(summary.loc[key,name+'_sd'],g[name].std(ddof=0),atol=1e-12)
    restored=0;replayed=0
    for seed in SEEDS:
        for fold,(a,b) in enumerate(splits(y,5,seed)):
            audit=json.loads((out/f'fold_{seed}_{fold}.json').read_text());target=y.iloc[a];held=y.iloc[b]
            assert audit['fit_ids']==list(target.index) and audit['heldout_ids']==list(held.index)
            train,valid=take(data,a),take(data,b)
            chosen,again=inner_candidates(train,target,seed,fold)
            assert audit['inner_ids']==again['inner_ids']
            for name,records in again['groups'].items():
                assert len(records)==len(audit['groups'][name])
                for old,new in zip(audit['groups'][name],records):
                    np.testing.assert_allclose(old['scores'],new['scores'],atol=1e-10);assert old['metrics']==new['metrics'];replayed+=1
            for name,spec in chosen.items():
                assert spec==audit['selected'][name]
                bundle=joblib.load(out/'models'/f'{name}_{seed}_{fold}.joblib')
                assert (bundle['feature'],bundle['alpha'],bundle['threshold'])==(spec['feature'],spec['alpha'],spec['threshold'])
                assert bundle['text']['model'].C==spec['parameters']['C']
                assert bundle['text']['model'].class_weight==spec['parameters']['class_weight']
                np.testing.assert_allclose(bundle['text']['prep'].scale_.mean_,train[spec['feature']].to_numpy(dtype=float).mean(0),atol=1e-12)
                for scenario in SCENARIOS:
                    g=pred[(pred.condition==name)&(pred.scenario==scenario)&(pred.seed==seed)&(pred.fold==fold)].set_index('participant_id').loc[held.index]
                    np.testing.assert_allclose(predict(bundle,valid,scenario),g.score,atol=1e-12)
                restored+=1
    old=pd.read_parquet(Path(m['baseline'])/'predictions.parquet').query('condition=="joint"').set_index(['seed','fold','participant_id']).sort_index()
    current=pred.query('condition=="reference" and scenario=="clean"').set_index(['seed','fold','participant_id']).sort_index()
    assert old.index.equals(current.index);np.testing.assert_allclose(old.score,current.score,atol=1e-12)
    comparison=json.loads((out/'comparison.json').read_text())
    for name,expected in comparison['precision'].items():
        actual=precision_screen(metrics.reset_index(),name)
        assert expected['checks']==actual['checks'] and expected['passes']==actual['passes']
        for field in ['mean_precision_difference','mean_f1_difference']:
            np.testing.assert_allclose(expected[field],actual[field],atol=1e-12)
    for entry in comparison['f1']:
        a=metrics.loc[(entry['candidate'],'clean')];b=metrics.loc[('reference','clean')]
        d=a-b;means=d.mean(numeric_only=True)
        for name,value in entry['mean_differences'].items():np.testing.assert_allclose(value,means[name],atol=1e-12)
        assert entry['passes_clean_screen']==bool(means.f1>=.03 and means.balanced_accuracy>=0 and means.mean_fold_auc>=0 and means.recall>=-.05 and (d.f1>0).sum()>=2)
    result={'models_restored':restored,'predictions_checked':len(pred),'inner_candidate_score_vectors_replayed':replayed,
            'topic_participants_independently_recomputed':len(y),'metrics_and_selection_verified':True,'test_evaluated':False}
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    with threadpool_limits(limits=1):print(verify(Path(__file__).resolve().parents[2],Path(sys.argv[1])))

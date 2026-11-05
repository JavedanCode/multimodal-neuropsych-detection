"""Independent audit of nested quality stacking and missing-input evaluation."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,balanced_accuracy_score,roc_auc_score,average_precision_score,confusion_matrix
from threadpoolctl import threadpool_limits
from src.experiments.baselines import sha256
from src.experiments.aligned_fusion_v2 import load
from src.experiments.domain_v2 import splits
from src.experiments.audio_study import SEEDS
from src.models.quality_stack_v2 import take,predict,SCENARIOS


def check_crossfit(records,expected,forbidden):
    held=[]
    for r in records:
        a=set(r['fit_ids']);b=set(r['heldout_ids'])
        assert a.isdisjoint(b) and a|b==set(expected) and (a|b).isdisjoint(forbidden)
        held.extend(r['heldout_ids'])
    assert len(held)==len(set(held)) and set(held)==set(expected)


def verify(project,out):
    m=json.loads((out/'manifest.json').read_text())
    for p,h in m['hashes'].items():assert sha256(project/p)==h
    for p,h in m['model_hashes'].items():assert sha256(out/'models'/p)==h
    data,y,_=load(project);pred=pd.read_parquet(out/'predictions.parquet')
    assert not pred.duplicated(['condition','scenario','seed','participant_id']).any()
    np.testing.assert_array_equal(pred.target,y.loc[pred.participant_id]);np.testing.assert_array_equal(pred.prediction,(pred.score>=0).astype(int))
    metrics=pd.read_csv(out/'repeat_metrics.csv').set_index(['condition','scenario','seed'])
    summary=pd.read_csv(out/'summary.csv').set_index(['condition','scenario'])
    aps=pd.read_csv(out/'fold_average_precision.csv').set_index(['condition','scenario','seed','fold'])
    domains=pd.read_csv(out/'domain_metrics.csv').set_index(['condition','scenario','seed','domain'])
    measures=['f1','precision','recall','specificity','accuracy','balanced_accuracy','mean_fold_auc']
    for key,g in pred.groupby(['condition','scenario','seed']):
        assert len(g)==len(y) and set(g.participant_id)==set(y.index)
        tn,fp,fn,tp=confusion_matrix(g.target,g.prediction,labels=[0,1]).ravel()
        values=[f1_score(g.target,g.prediction),precision_score(g.target,g.prediction,zero_division=0),recall_score(g.target,g.prediction),tn/(tn+fp),
                accuracy_score(g.target,g.prediction),balanced_accuracy_score(g.target,g.prediction),np.mean([roc_auc_score(v.target,v.score) for _,v in g.groupby('fold')])]
        for name,value in zip(measures,values):np.testing.assert_allclose(metrics.loc[key,name],value,atol=1e-12)
        for fold,v in g.groupby('fold'):np.testing.assert_allclose(aps.loc[(*key,fold),'average_precision'],average_precision_score(v.target,v.score),atol=1e-12)
        for domain,v in g.groupby(np.where(g.participant_id>=600,'AI','WoZ')):
            row=domains.loc[(*key,domain)]
            assert row['n']==len(v) and row['positives']==int(v.target.sum())
            for metric,fn in [('f1',f1_score),('precision',precision_score),('recall',recall_score),
                              ('accuracy',accuracy_score),('balanced_accuracy',balanced_accuracy_score)]:
                np.testing.assert_allclose(row[metric],fn(v.target,v.prediction),atol=1e-12)
    for key,g in metrics.reset_index().groupby(['condition','scenario']):
        for name in measures:
            np.testing.assert_allclose(summary.loc[key,name+'_mean'],g[name].mean(),atol=1e-12)
            np.testing.assert_allclose(summary.loc[key,name+'_sd'],g[name].std(ddof=0),atol=1e-12)
    restored=0
    for seed in SEEDS:
        for fold,(a,b) in enumerate(splits(y,5,seed)):
            audit=json.loads((out/f'fold_{seed}_{fold}.json').read_text());target=y.iloc[a];held=y.iloc[b];valid=take(data,b)
            assert audit['fit_ids']==list(target.index) and audit['heldout_ids']==list(held.index)
            for j,(ia,ib) in enumerate(splits(target,3,seed+fold+1)):
                assert audit['inner_ids'][j]=={'fit':list(target.index[ia]),'heldout':list(target.index[ib])}
                check_crossfit(audit['inner_crossfit'][j],target.index[ia],set(target.index[ib])|set(held.index))
            check_crossfit(audit['outer_crossfit'],target.index,set(held.index))
            for r in audit['candidates']:np.testing.assert_allclose(r['inner_f1'],f1_score(target,np.array(r['scores'])>=0),atol=1e-12)
            reference=audit['selected']['reference']
            for name,spec in audit['selected'].items():
                if name=='selected':assert spec==max([reference]+audit['candidates'],key=lambda r:r['inner_f1'])
                elif name!='reference':assert spec==max([r for r in audit['candidates'] if r['kind']==name],key=lambda r:r['inner_f1'])
                model=joblib.load(out/'models'/f'{name}_{seed}_{fold}.joblib');assert model['kind']==spec['kind']
                if model['kind']!='reference':assert model['C']==spec['C'] and model['meta']['model'].C==spec['C']
                for scenario in SCENARIOS:
                    score=model['model'].decision_function(valid['text']) if model['kind']=='reference' else predict(model,valid,scenario)
                    g=pred[(pred.condition==name)&(pred.scenario==scenario)&(pred.seed==seed)&(pred.fold==fold)].set_index('participant_id').loc[held.index]
                    np.testing.assert_allclose(score,g.score,atol=1e-12,rtol=1e-12)
                restored+=1
    baseline=pd.read_parquet(Path(m['baseline'])/'predictions.parquet');baseline=baseline[baseline.condition=='joint'].set_index(['seed','fold','participant_id']).sort_index()
    current=pred[(pred.condition=='reference')&(pred.scenario=='clean')].set_index(['seed','fold','participant_id']).sort_index()
    assert baseline.index.equals(current.index);np.testing.assert_array_equal(baseline.score,current.score)
    comparison=json.loads((out/'comparison.json').read_text())
    for entry in comparison['clean']+comparison['quality']:
        a=metrics.loc[(entry['candidate'],entry['scenario'])].sort_index()
        b=metrics.loc[(entry['control'],entry['scenario'])].sort_index()
        delta=a[measures]-b[measures];means=delta.mean()
        for metric in measures:np.testing.assert_allclose(entry['mean_differences'][metric],means[metric],atol=1e-12)
        assert entry['f1_positive_repeats']==int((delta.f1>0).sum())
        passes=means.f1>=.03 and means.balanced_accuracy>=0 and means.mean_fold_auc>=0 and means.recall>=-.05 and (delta.f1>0).sum()>=2
        assert entry['passes_clean_screen']==bool(passes)
    gain=np.mean([r['mean_differences']['f1'] for r in comparison['quality'][1:]])
    np.testing.assert_allclose(comparison['quality_missing_gain'],gain,atol=1e-12)
    assert comparison['passes_robustness_screen']==bool(comparison['quality'][0]['mean_differences']['f1']>=-.02 and gain>=.05)
    result={'models_restored':restored,'prediction_rows':len(pred),'metric_rows':len(metrics),'inner_candidates_checked':135,
            'nested_base_crossfit_membership_verified':True,'hashes_folds_predictions_metrics_verified':True,'test_evaluated':False}
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    with threadpool_limits(limits=1):print(verify(Path(__file__).resolve().parents[2],Path(sys.argv[1])))

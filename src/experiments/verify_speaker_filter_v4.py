"""Verify speaker-filter inputs, inner choices and saved model predictions."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from src.experiments.baselines import sha256
from src.experiments.speaker_filter_v4 import load,inner_candidates,predict
from src.experiments.domain_v2 import splits
from src.experiments.audio_study import SEEDS
from src.experiments.bounded_v3 import precision_screen
from src.dataset.speaker_filter_v4 import masks
from src.dataset.contracts_v2 import screened_rows


def verify_data(project):
    data,y=load(project);root=project/'data/processed/speaker_filter_v4';manifest=json.loads((root/'manifest.json').read_text())
    prior=json.loads((project/'data/processed/mpnet_v1/manifest.json').read_text())
    totals={name:{'removed':0,'affected_participants':0,'fallbacks':0} for name in ['context_filter','prompt_filter']}
    for pid in y.index:
        meta=manifest['participants'][str(pid)];cache=root/'participant_cache'/f'{pid}.npz';framepath=cache.with_suffix('.parquet')
        assert sha256(cache)==meta['cache_hash'] and sha256(framepath)==meta['frame_hash']
        assert meta['transcript_hash']==prior['participants'][str(pid)]['key']['transcript_hash']
        emb=project/'data/processed/mpnet_v1/participant_cache'/f'{pid}.npz';assert sha256(emb)==prior['participants'][str(pid)]['hash']
        duration=json.loads((project/'reports/data_audit/label_speech_cache'/f'{pid}.json').read_text())['result']['audio_seconds']
        frame=pd.read_parquet(framepath);rows=screened_rows(frame,duration);expected=masks(frame,rows)
        with np.load(cache) as z,np.load(emb) as e:
            np.testing.assert_array_equal(z['row_indices'],rows['row_indices'])
            assert (z['context_filter']|~z['prompt_filter']).all()
            for name in totals:
                mask=z[name];np.testing.assert_array_equal(mask,expected[name])
                audit=meta['audit'][name];assert audit['removed']==int((~mask).sum()) and audit['retained']==int(mask.sum())
                assert audit['fallback']==bool(not mask.any())
                value=e['embeddings'][mask].sum(axis=0)/mask.sum() if mask.any() else data['reference'].loc[pid].to_numpy()
                np.testing.assert_allclose(data[name].loc[pid],value,atol=1e-8)
                totals[name]['removed']+=audit['removed'];totals[name]['affected_participants']+=int(audit['removed']>0);totals[name]['fallbacks']+=int(audit['fallback'])
    result={'participants':len(y),'utterances':sum(r['utterances'] for r in manifest['participants'].values()),
            'filters':totals,'source_masks_and_means_verified':True,'speaker_verified':False}
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2));return result


def verify(project,out):
    dataset=verify_data(project);data,y=load(project);m=json.loads((out/'manifest.json').read_text())
    for p,h in m['hashes'].items():assert sha256(project/p)==h
    for p,h in m['models'].items():assert sha256(out/'models'/p)==h
    pred=pd.read_parquet(out/'predictions.parquet');assert not pred.duplicated(['condition','seed','participant_id']).any()
    np.testing.assert_array_equal(pred.target,y.loc[pred.participant_id]);np.testing.assert_array_equal(pred.prediction,(pred.score>=0).astype(int))
    metrics=pd.read_csv(out/'repeat_metrics.csv').set_index(['condition','scenario','seed'])
    summary=pd.read_csv(out/'summary.csv').set_index(['condition','scenario'])
    ap=pd.read_csv(out/'fold_average_precision.csv').set_index(['condition','scenario','seed','fold'])
    domain=pd.read_csv(out/'domain_metrics.csv').set_index(['condition','scenario','seed','domain'])
    for key,g in pred.groupby(['condition','scenario','seed']):
        assert set(g.participant_id)==set(y.index) and len(g)==len(y)
        parts=[(g,metrics.loc[key])]+[(v,domain.loc[(*key,d)]) for d,v in g.groupby(np.where(g.participant_id>=600,'AI','WoZ'))]
        for v,row in parts:
            tn,fp,fn,tp=confusion_matrix(v.target,v.prediction,labels=[0,1]).ravel()
            expected={'tn':tn,'fp':fp,'fn':fn,'tp':tp,'precision':tp/max(1,tp+fp),'recall':tp/(tp+fn),
                      'f1':2*tp/max(1,2*tp+fp+fn),'accuracy':(tp+tn)/len(v),'specificity':tn/(tn+fp),
                      'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp))}
            for name,value in expected.items():np.testing.assert_allclose(row[name],value,atol=1e-12)
        auc=[]
        for fold,v in g.groupby('fold'):
            auc.append(roc_auc_score(v.target,v.score));np.testing.assert_allclose(ap.loc[(*key,fold),'average_precision'],average_precision_score(v.target,v.score),atol=1e-12)
        np.testing.assert_allclose(metrics.loc[key,'mean_fold_auc'],np.mean(auc),atol=1e-12)
    for key,g in metrics.reset_index().groupby(['condition','scenario']):
        for name in ['precision','recall','f1','specificity','accuracy','balanced_accuracy','mean_fold_auc']:
            np.testing.assert_allclose(summary.loc[key,name+'_mean'],g[name].mean(),atol=1e-12)
            np.testing.assert_allclose(summary.loc[key,name+'_sd'],g[name].std(ddof=0),atol=1e-12)
    restored=0;replayed=0
    for seed in SEEDS:
        for fold,(a,b) in enumerate(splits(y,5,seed)):
            target=y.iloc[a];held=y.iloc[b];train={k:x.iloc[a] for k,x in data.items()};valid={k:x.iloc[b] for k,x in data.items()}
            saved=json.loads((out/f'fold_{seed}_{fold}.json').read_text());chosen,again=inner_candidates(train,target,seed,fold)
            assert saved['fit_ids']==list(target.index) and saved['heldout_ids']==list(held.index) and saved['inner_ids']==again['inner_ids']
            assert len(saved['candidates'])==len(again['candidates'])==72
            for old,new in zip(saved['candidates'],again['candidates']):
                np.testing.assert_allclose(old['scores'],new['scores'],atol=1e-12);assert old['metrics']==new['metrics'];replayed+=1
            for name,spec in chosen.items():
                assert saved['selected'][name]==spec
                bundle=joblib.load(out/'models'/f'{name}_{seed}_{fold}.joblib')
                assert bundle['feature']==spec['feature'] and bundle['threshold']==spec['threshold']
                assert bundle['model']['model'].C==spec['parameters']['C'] and bundle['model']['model'].class_weight==spec['parameters']['class_weight']
                np.testing.assert_allclose(bundle['model']['prep'].scale_.mean_,train[spec['feature']].to_numpy(float).mean(0),atol=1e-12)
                g=pred[(pred.condition==name)&(pred.seed==seed)&(pred.fold==fold)].set_index('participant_id').loc[held.index]
                np.testing.assert_allclose(predict(bundle,valid),g.score,atol=1e-12);restored+=1
    baseline=pd.read_parquet(Path(m['baseline'])/'predictions.parquet').query('condition=="joint"').set_index(['seed','fold','participant_id']).sort_index()
    current=pred.query('condition=="reference"').set_index(['seed','fold','participant_id']).sort_index()
    assert baseline.index.equals(current.index);np.testing.assert_allclose(baseline.score,current.score,atol=1e-12)
    comparison=json.loads((out/'comparison.json').read_text())
    for name,expected in comparison['precision'].items():
        actual=precision_screen(metrics.reset_index(),name);assert expected['checks']==actual['checks'] and expected['passes']==actual['passes']
        for field in ['mean_precision_difference','mean_f1_difference']:np.testing.assert_allclose(expected[field],actual[field],atol=1e-12)
    for entry in comparison['f1']:
        d=metrics.loc[(entry['candidate'],'clean')]-metrics.loc[('reference','clean')];means=d.mean(numeric_only=True)
        for name,value in entry['mean_differences'].items():np.testing.assert_allclose(value,means[name],atol=1e-12)
        assert entry['passes_clean_screen']==bool(means.f1>=.03 and means.balanced_accuracy>=0 and means.mean_fold_auc>=0 and means.recall>=-.05 and (d.f1>0).sum()>=2)
    result={'models_restored':restored,'prediction_rows':len(pred),'inner_candidate_vectors_replayed':replayed,
            'feature_validation':dataset,'metrics_folds_selection_hashes_verified':True,'test_evaluated':False}
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    with threadpool_limits(limits=1):
        project=Path(__file__).resolve().parents[2]
        print(verify_data(project) if len(sys.argv)==1 else verify(project,Path(sys.argv[1])))

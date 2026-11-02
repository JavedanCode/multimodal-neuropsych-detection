"""Independent metric/selection, cache, IDF and restored-prediction checks."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,balanced_accuracy_score,confusion_matrix,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.baselines import sha256
from src.experiments.lexical_study import inputs,subset,predict


def verify(project,out):
    protocol=json.loads((out/'protocol.json').read_text())
    completed=json.loads((out/'verification.json').read_text())
    for name,digest in protocol['hashes'].items():assert sha256(project/name)==digest,name
    for name,digest in completed['model_hashes'].items():assert sha256(out/'models'/name)==digest,name
    data,y=inputs(project)
    root=project/'data/processed/lexical_v1';manifest=json.loads((root/'manifest.json').read_text())
    for i,pid in enumerate(y.index):
        metadata=manifest['participants'][str(pid)]
        path=root/'participant_cache'/f'{pid}.npz'
        assert sha256(path)==metadata['hash']
        assert (sparse.load_npz(path)!=data['counts'][i]).nnz==0
    pred=pd.read_parquet(out/'predictions.parquet')
    metrics=pd.read_csv(out/'repeat_metrics.csv').set_index(['condition','seed'])
    summary=pd.read_csv(out/'summary.csv').set_index('condition')
    assert not pred.duplicated(['condition','seed','participant_id']).any()
    np.testing.assert_array_equal(pred.target,y.loc[pred.participant_id])
    np.testing.assert_allclose(pred.score,pred.raw_score-pred.threshold,atol=1e-12)
    np.testing.assert_array_equal(pred.prediction,(pred.score>=0).astype(int))
    measures=['f1','precision','recall','specificity','accuracy','balanced_accuracy','mean_fold_auc']
    for key,g in pred.groupby(['condition','seed']):
        assert len(g)==len(y) and set(g.participant_id)==set(y.index)
        tn,fp,fn,tp=confusion_matrix(g.target,g.prediction,labels=[0,1]).ravel()
        expected=[f1_score(g.target,g.prediction,zero_division=0),precision_score(g.target,g.prediction,zero_division=0),
                  recall_score(g.target,g.prediction,zero_division=0),tn/(tn+fp),accuracy_score(g.target,g.prediction),
                  balanced_accuracy_score(g.target,g.prediction),np.mean([roc_auc_score(d.target,d.score) for _,d in g.groupby('fold')])]
        for name,value in zip(measures,expected):np.testing.assert_allclose(metrics.loc[key,name],value,atol=1e-12)
    for condition,g in metrics.reset_index().groupby('condition'):
        for name in measures:
            np.testing.assert_allclose(summary.loc[condition,name+'_mean'],g[name].mean(),atol=1e-12)
            np.testing.assert_allclose(summary.loc[condition,name+'_sd'],g[name].std(ddof=0),atol=1e-12)
    restored=0;candidate_count=0
    for seed in protocol['seeds']:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):
            audit=json.loads((out/f'fold_{seed}_{fold}.json').read_text())
            assert audit['fit_ids']==list(y.index[a]) and audit['heldout_ids']==list(y.index[b])
            inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(np.zeros(len(a)),y.iloc[a]))
            for (ia,ib),saved in zip(inner,audit['inner_ids']):
                assert saved=={'fit':list(y.index[a][ia]),'heldout':list(y.index[a][ib])}
            new=[]
            for name,candidates in audit['candidates'].items():
                for r in candidates:
                    scores=np.array(r['inner_scores'])
                    assert scores.shape==(len(a),) and np.isfinite(scores).all()
                    np.testing.assert_allclose(r['inner_f1'],f1_score(y.iloc[a],scores>=r['threshold'],zero_division=0),atol=1e-12)
                    candidate_count+=1
                if name.startswith('old_'):
                    assert audit['selected'][name]==max(candidates,key=lambda r:r['inner_f1'])
                else:
                    assert audit['selected'][name+'_tuned']==max(candidates,key=lambda r:r['inner_f1'])
                    assert audit['selected'][name+'_default']==max([r for r in candidates if r['threshold']==0],key=lambda r:r['inner_f1'])
                    new.extend(candidates)
            assert audit['selected']['selected_new']==max(new,key=lambda r:r['inner_f1'])
            expected_idf=np.log((1+len(a))/(1+np.asarray((data['counts'][a]>0).sum(0)).ravel()))+1
            valid=subset(data,b)
            for condition,g in pred[(pred.seed==seed)&(pred.fold==fold)].groupby('condition'):
                if condition.startswith('constant_'):continue
                bundle=joblib.load(out/'models'/f'{condition}_{seed}_{fold}.joblib')
                if bundle['kind']!='old':np.testing.assert_allclose(bundle['preprocess'].tfidf.idf_,expected_idf,atol=1e-12)
                raw,scores=predict(bundle,valid)
                ordered=g.set_index('participant_id').loc[y.index[b]]
                np.testing.assert_allclose(ordered.score,scores,atol=1e-10)
                np.testing.assert_allclose(ordered.raw_score,raw,atol=1e-10)
                restored+=1
    old=pd.read_parquet(project/'experiments/runs/text_20261030T230941483092Z/predictions.parquet')
    old=old[old.condition.isin(['text_logistic','text_linear'])].copy()
    old['condition']=old.condition.str.replace('text_','old_',regex=False)
    keys=['condition','seed','fold','participant_id'];old=old.set_index(keys)
    current=pred[pred.condition.isin(['old_logistic','old_linear'])].set_index(keys).loc[old.index]
    np.testing.assert_allclose(current.score,old.score,rtol=1e-8,atol=1e-8)
    np.testing.assert_array_equal(current.prediction,old.prediction)
    result={'restored_bundles':restored,'prediction_rows':len(pred),'metric_rows':len(metrics),'inner_candidates_verified':candidate_count,
            'prior_control_predictions_matched':len(old),'verified_count_caches':len(y),'outer_idf_verified':True,'folds_and_hashes_verified':True}
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):print(verify(Path(__file__).resolve().parents[2],args.run))

"""Independently recompute metrics, inner choices, and restored outer predictions."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.baselines import sha256
from src.experiments.text_refinement import features, predict


def verify(project,out):
    protocol=json.loads((out/'protocol.json').read_text())
    verification=json.loads((out/'verification.json').read_text())
    for name,digest in protocol['hashes'].items():assert sha256(project/name)==digest,name
    for name,digest in verification['model_hashes'].items():assert sha256(out/'models'/name)==digest,name
    data,y=features(project)
    pred=pd.read_parquet(out/'predictions.parquet')
    metric=pd.read_csv(out/'repeat_metrics.csv').set_index(['condition','seed'])
    summary=pd.read_csv(out/'summary.csv').set_index('condition')
    assert not pred.duplicated(['condition','seed','participant_id']).any()
    np.testing.assert_array_equal(pred.target,y.loc[pred.participant_id])
    np.testing.assert_allclose(pred.score,pred.raw_score-pred.threshold,atol=1e-12)
    np.testing.assert_array_equal(pred.prediction,(pred.score>=0).astype(int))
    for key,g in pred.groupby(['condition','seed']):
        assert set(g.participant_id)==set(y.index) and len(g)==len(y)
        tn,fp,fn,tp=confusion_matrix(g.target,g.prediction,labels=[0,1]).ravel()
        expected={'f1':f1_score(g.target,g.prediction,zero_division=0),
                  'precision':precision_score(g.target,g.prediction,zero_division=0),
                  'recall':recall_score(g.target,g.prediction,zero_division=0),
                  'specificity':tn/(tn+fp),'accuracy':accuracy_score(g.target,g.prediction),
                  'balanced_accuracy':balanced_accuracy_score(g.target,g.prediction),
                  'mean_fold_auc':np.mean([roc_auc_score(d.target,d.score) for _,d in g.groupby('fold')])}
        for name,value in expected.items():np.testing.assert_allclose(metric.loc[key,name],value,atol=1e-12)
    for condition,g in metric.reset_index().groupby('condition'):
        for name in ['f1','precision','recall','specificity','accuracy','balanced_accuracy','mean_fold_auc']:
            np.testing.assert_allclose(summary.loc[condition,name+'_mean'],g[name].mean(),atol=1e-12)
            np.testing.assert_allclose(summary.loc[condition,name+'_sd'],g[name].std(ddof=0),atol=1e-12)
    restored=0;inner_checked=0
    for seed in protocol['seeds']:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):
            audit=json.loads((out/f'fold_{seed}_{fold}.json').read_text())
            assert audit['fit_ids']==list(y.index[a]) and audit['heldout_ids']==list(y.index[b])
            inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(np.zeros(len(a)),y.iloc[a]))
            for (ia,ib),saved in zip(inner,audit['inner_ids']):
                assert saved=={'fit':list(y.index[a][ia]),'heldout':list(y.index[a][ib])}
            for key,records in audit['candidates'].items():
                for r in records:
                    scores=np.array(r['inner_scores'])
                    assert scores.shape==(len(a),) and np.isfinite(scores).all()
                    np.testing.assert_allclose(r['inner_f1'],f1_score(y.iloc[a],scores>=r['threshold'],zero_division=0),atol=1e-12)
                    inner_checked+=1
                if key.startswith('previous_'):
                    assert audit['selected'][key]==max(records,key=lambda r:r['inner_f1'])
                if key in ['mean_standard','mean_l2','rich_standard','rich_l2']:
                    assert audit['selected'][key+'_tuned']==max(records,key=lambda r:r['inner_f1'])
                    assert audit['selected'][key+'_default']==max([r for r in records if r['threshold']==0],key=lambda r:r['inner_f1'])
            all_text=sum([audit['candidates'][name] for name in ['mean_standard','mean_l2','rich_standard','rich_l2']],[])
            assert audit['selected']['selected_text']==max(all_text,key=lambda r:r['inner_f1'])
            assert audit['selected']['late_fusion']==max(audit['candidates']['fusion'],key=lambda r:r['inner_f1'])
            branches=audit['fusion_branches']
            logistic=[r for r in audit['candidates']['mean_standard'] if r['family']=='logistic' and r['threshold']==0]
            assert branches[0]==max(logistic,key=lambda r:r['inner_f1'])
            for j,name in enumerate(['audio','face'],1):
                assert branches[j]==max(audit['candidates'][name],key=lambda r:r['inner_f1'])
            probs=np.stack([1/(1+np.exp(-np.array(r['inner_scores']))) for r in branches],1)
            for r in audit['candidates']['fusion']:
                np.testing.assert_allclose(r['inner_scores'],probs@r['weights'],atol=1e-12)
            expected='selected_text' if audit['selected']['selected_text']['inner_f1']>=audit['selected']['late_fusion']['inner_f1'] else 'late_fusion'
            assert audit['selected_strategy']==expected
            valid={k:x.iloc[b] for k,x in data.items()}
            g=pred[(pred.seed==seed)&(pred.fold==fold)]
            for condition,d in g.groupby('condition'):
                if condition.startswith('constant_'):continue
                bundle=joblib.load(out/'models'/f'{condition}_{seed}_{fold}.joblib')
                raw,scores=predict(bundle,valid)
                np.testing.assert_allclose(d.set_index('participant_id').loc[y.index[b]].score,scores,atol=1e-10)
                np.testing.assert_allclose(d.set_index('participant_id').loc[y.index[b]].raw_score,raw,atol=1e-10)
                restored+=1
            chosen=g[g.condition==expected].set_index('participant_id')
            combined=g[g.condition=='selected_strategy'].set_index('participant_id').loc[chosen.index]
            np.testing.assert_array_equal(chosen.score,combined.score)
    old=pd.read_parquet(project/'experiments/runs/text_20261030T230941483092Z/predictions.parquet')
    old=old[old.condition.isin(['text_logistic','text_linear'])].copy()
    old['condition']=old.condition.str.replace('text_','previous_',regex=False)
    keys=['condition','seed','fold','participant_id']
    old=old.set_index(keys)
    new=pred[pred.condition.isin(['previous_logistic','previous_linear'])].set_index(keys).loc[old.index]
    np.testing.assert_allclose(old.score,new.score,rtol=1e-8,atol=1e-8)
    np.testing.assert_array_equal(old.prediction,new.prediction)
    result={'restored_bundles':restored,'prediction_rows':len(pred),'metric_rows':len(metric),
            'inner_candidate_records_verified':inner_checked,'prior_control_predictions_matched':len(old),
            'folds_and_hashes_verified':True}
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):print(verify(Path(__file__).resolve().parents[2],args.run))

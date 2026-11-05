"""Independent folds, metrics, selections and model replay for precision runs."""
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.audio_study import SEEDS, estimator
from src.experiments.mpnet_study import inputs
from src.experiments.verification_utils import verify_outputs
from src.models.text_preprocessing import TextPreprocessor


def params(record):
    p=dict(record['parameters'])
    if isinstance(p.get('class_weight'),dict):p['class_weight']={int(k):v for k,v in p['class_weight'].items()}
    return p


def verify(project,out):
    sets,y=inputs(project);x=sets['mpnet']
    protocol,pred,nmetrics=verify_outputs(project,out,y);experiment=protocol['experiment']
    count=0;replayed=0
    for seed in SEEDS:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(x,y)):
            audit=json.loads((out/f'fold_{seed}_{fold}.json').read_text());target=y.iloc[a];held=y.iloc[b]
            assert audit['fit_ids']==list(target.index) and audit['heldout_ids']==list(held.index)
            splits=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(x.iloc[a],target))
            for j,(ia,ib) in enumerate(splits):
                assert audit['inner_ids'][j]=={'fit':list(target.index[ia]),'heldout':list(target.index[ib])}
            for r in audit['reference_candidates']+audit['candidates']:
                scores=np.array(r['scores']);assert scores.shape==(len(target),) and np.isfinite(scores).all()
                p=scores>=0
                for name,fn in [('precision',precision_score),('recall',recall_score),('f1',f1_score)]:
                    np.testing.assert_allclose(r['metrics'][name],fn(target,p,zero_division=0),atol=1e-12)
                count+=1
            ref=audit['reference'];selected=audit['selected']
            assert ref==max(audit['reference_candidates'],key=lambda r:r['metrics']['f1'])
            eligible=[r for r in audit['candidates'] if r['metrics']['recall']>=.65 and r['metrics']['f1']>=ref['metrics']['f1']-.02]
            expected=max(eligible,key=lambda r:(r['metrics']['precision'],r['metrics']['f1'],r['metrics']['recall'])) if eligible else ref
            assert selected==expected
            # Refit the two selected procedures on inner fitting folds, replaying
            # their stored validation scores without using outer participants.
            for r in [ref,selected]:
                scores=np.full(len(target),np.nan)
                for ia,ib in splits:
                    prep=TextPreprocessor().fit(x.loc[target.index[ia]])
                    z=prep.transform(x.loc[target.index[ia]]);v=prep.transform(x.loc[target.index[ib]])
                    family='rbf' if r['kind']=='rbf' else ('logistic' if r['kind']=='agreement' else 'linear')
                    model=estimator(family,params(r),seed).fit(z,target.iloc[ia]);raw=model.decision_function(v)
                    if r['kind']=='agreement':
                        base=estimator('linear',params(ref),seed).fit(z,target.iloc[ia]).decision_function(v)
                        raw=np.minimum(base,raw)
                    scores[ib]=raw-r['threshold']
                np.testing.assert_allclose(scores,r['scores'],atol=1e-12,rtol=1e-12);replayed+=1
            for name,r in [('reference',ref),(experiment,selected)]:
                bundle=joblib.load(out/'models'/f'{name}_{seed}_{fold}.joblib')
                np.testing.assert_allclose(bundle['preprocess'].scale_.mean_,x.iloc[a].to_numpy(dtype=np.float64).mean(axis=0),atol=1e-12)
                z=bundle['preprocess'].transform(x.loc[held.index])
                raw=np.min([m.decision_function(z) for m in bundle['models']],axis=0)-bundle['threshold']
                g=pred[(pred.condition==name)&(pred.seed==seed)&(pred.fold==fold)].set_index('participant_id').loc[held.index]
                np.testing.assert_allclose(raw,g.score,atol=1e-12,rtol=1e-12)
    old=pd.read_parquet(project/'experiments/runs/mpnet_20261102T171156498575Z/predictions.parquet')
    old=old[old.condition=='mpnet_linear'].set_index(['seed','fold','participant_id']).sort_index()
    current=pred[pred.condition=='reference'].set_index(['seed','fold','participant_id']).sort_index()
    assert old.index.equals(current.index)
    np.testing.assert_allclose(old.score,current.score,atol=1e-12,rtol=1e-12)
    result={'restored_bundles':30,'prediction_rows':len(pred),'metric_rows':nmetrics,'candidate_records_checked':count,
            'selected_inner_procedures_refit':replayed,'prior_control_predictions_matched':len(current),
            'hashes_folds_preprocessing_selection_and_metrics_verified':True,'test_evaluated':False}
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    with threadpool_limits(limits=1):print(verify(Path(__file__).resolve().parents[2],Path(sys.argv[1])))

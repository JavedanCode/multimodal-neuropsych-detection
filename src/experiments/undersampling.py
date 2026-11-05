"""Nested, matched MPNet resampling comparison."""
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.audio_study import SEEDS, estimator
from src.experiments.baselines import sha256, TARGET_POLICY
from src.experiments.mpnet_study import inputs
from src.experiments.text_temporal_study import summarize, save_predictions
from src.models.text_preprocessing import TextPreprocessor
from src.models.undersampling import sample_positions, decision

PROCEDURES = {'reference': (None, 1), 'weighted': (None, 1),
              'single_1': (1, 1), 'single_2': (2, 1),
              'ensemble_1': (1, 7), 'ensemble_2': (2, 7)}


def fit_models(z, y, name, params, seed):
    ratio, members = PROCEDURES[name]
    samples = [np.arange(len(y)) if ratio is None else sample_positions(y, ratio, seed+i)
               for i in range(members)]
    models = [estimator('linear', params, seed).fit(z.iloc[a], y.iloc[a]) for a in samples]
    return models, [list(map(int, y.index[a])) for a in samples]


def run(project):
    sets, y = inputs(project); x = sets['mpnet']
    out = project/'experiments/runs'/('undersampling_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(); (out/'models').mkdir()
    paths = ['docs/undersampling_protocol.md', 'src/models/undersampling.py',
             'src/experiments/undersampling.py', 'src/models/text_preprocessing.py',
             'src/experiments/audio_study.py', 'src/experiments/mpnet_study.py',
             'src/experiments/text_temporal_study.py', 'src/experiments/metrics.py',
             'data/processed/mpnet_v1/features.parquet', 'data/processed/mpnet_v1/labels.parquet']
    hashes = {p: sha256(project/p) for p in paths}
    (out/'protocol.json').write_text(json.dumps({'hashes': hashes, 'target': TARGET_POLICY,
        'seeds': SEEDS, 'development_evaluated': False, 'test_evaluated': False}, indent=2))
    predictions = []
    for seed in SEEDS:
        for fold, (a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(x,y)):
            logging.info('Undersampling seed %s fold %s', seed, fold)
            fit, target, valid, held = x.iloc[a], y.iloc[a], x.iloc[b], y.iloc[b]
            splits = list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(fit,target))
            prepared = []
            for ia,ib in splits:
                prep = TextPreprocessor().fit(fit.iloc[ia])
                prepared.append((ia,ib,prep.transform(fit.iloc[ia]),prep.transform(fit.iloc[ib])))
            records = []
            for name in PROCEDURES:
                weights = [None,'balanced'] if name=='reference' else (['balanced'] if name=='weighted' else [None])
                for c in [.01,.1,1.]:
                    for weight in weights:
                        params = {'C': c, 'class_weight': weight}; scores = np.full(len(target),np.nan); samples=[]
                        for j,(ia,ib,z,v) in enumerate(prepared):
                            models, ids = fit_models(z,target.iloc[ia],name,params,seed*10000+fold*100+j*10)
                            scores[ib] = np.mean([m.decision_function(v) for m in models],axis=0)
                            samples.append(ids)
                        assert np.isfinite(scores).all()
                        records.append({'procedure': name,'parameters': params,'inner_f1':float(f1_score(target,scores>=0)),
                                        'inner_scores':scores.tolist(),'inner_sample_ids':samples})
            selected = {name:max([r for r in records if r['procedure']==name],key=lambda r:r['inner_f1']) for name in PROCEDURES}
            selected['selected_sampling'] = max([r for r in records if r['procedure'] not in ('reference','weighted')],key=lambda r:r['inner_f1'])
            prep = TextPreprocessor().fit(fit); z=prep.transform(fit)
            outer_samples = {}
            for name,spec in selected.items():
                models, ids = fit_models(z,target,spec['procedure'],spec['parameters'],seed*10000+fold*100+90)
                bundle={'preprocess':prep,'models':models,'sample_ids':ids}
                joblib.dump(bundle,out/'models'/f'{name}_{seed}_{fold}.joblib')
                outer_samples[name]=ids
                save_predictions(predictions,name,'clean',seed,fold,held,decision(bundle,valid))
            audit={'fit_ids':list(map(int,target.index)),'heldout_ids':list(map(int,held.index)),
                   'inner_ids':[{'fit':list(map(int,target.index[ia])),'heldout':list(map(int,target.index[ib]))} for ia,ib in splits],
                   'candidates':records,'selected':selected,'outer_sample_ids':outer_samples}
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    pred=pd.DataFrame(predictions);pred.to_parquet(out/'predictions.parquet',index=False);summarize(pred,out)
    for p,h in hashes.items(): assert sha256(project/p)==h
    (out/'verification.json').write_text(json.dumps({'model_hashes':{p.name:sha256(p) for p in (out/'models').iterdir()},'test_evaluated':False},indent=2))
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1): print(run(Path(__file__).resolve().parents[2]))

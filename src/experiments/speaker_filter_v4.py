"""Bounded nested comparison of estimated interviewer-prompt exclusion."""
from pathlib import Path
from datetime import datetime,timezone
import json
import logging
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from src.experiments.baselines import sha256
from src.experiments.mpnet_study import inputs
from src.experiments.audio_study import SEEDS,candidates,estimator
from src.experiments.domain_v2 import splits
from src.experiments.text_temporal_study import save_predictions,summarize
from src.experiments.compare_text_temporal import paired
from src.experiments.bounded_v3 import precision_screen
from src.models.bounded_v3 import measures,choose,fit_text
from src.models.text_preprocessing import TextPreprocessor

FEATURES=('reference','context_filter','prompt_filter')


def load(project):
    sets,y=inputs(project);data={'reference':sets['mpnet']};root=project/'data/processed/speaker_filter_v4'
    m=json.loads((root/'manifest.json').read_text())
    for p,h in m['hashes'].items():assert sha256(project/p)==h
    for p,h in m['outputs'].items():assert sha256(root/p)==h
    pd.testing.assert_series_equal(y,pd.read_parquet(root/'labels.parquet').PHQ_Binary)
    for name in FEATURES[1:]:
        data[name]=pd.read_parquet(root/f'{name}.parquet');assert data[name].index.equals(y.index)
    return data,y


def inner_candidates(data,y,seed,fold):
    inner=splits(y,3,seed+fold+1);records=[]
    for feature in FEATURES:
        prepared=[]
        for a,b in inner:
            prep=TextPreprocessor().fit(data[feature].iloc[a])
            prepared.append((a,b,prep.transform(data[feature].iloc[a]),prep.transform(data[feature].iloc[b])))
        for params in candidates('linear'):
            raw=np.full(len(y),np.nan)
            for a,b,x,v in prepared:raw[b]=estimator('linear',params,seed).fit(x,y.iloc[a]).decision_function(v)
            for cutoff in [0.,.1,.2,.3]:
                scores=raw-cutoff
                records.append({'feature':feature,'parameters':params,'threshold':cutoff,
                                'scores':scores.tolist(),'metrics':measures(y,scores)})
    zero=[r for r in records if r['threshold']==0]
    selected={name:max([r for r in zero if r['feature']==name],key=lambda r:r['metrics']['f1']) for name in FEATURES}
    reference=selected['reference']
    selected['selected']=max(zero,key=lambda r:r['metrics']['f1'])
    selected['threshold']=choose([reference]+[r for r in records if r['feature']=='reference' and r['parameters']==reference['parameters']],reference)
    selected['precision']=choose([reference]+records,reference)
    return selected,{'candidates':records,'selected':selected,
        'inner_ids':[{'fit':list(map(int,y.index[a])),'heldout':list(map(int,y.index[b]))} for a,b in inner]}


def predict(bundle,data):
    return bundle['model'].decision_function(data[bundle['feature']])-bundle['threshold']


def run(project,baseline):
    baseline=baseline.resolve();data,y=load(project)
    bm=json.loads((baseline/'manifest.json').read_text())
    for p,h in bm['hashes'].items():assert sha256(project/p)==h
    paths=['docs/speaker_filter_v4_protocol.md','src/experiments/speaker_filter_v4.py','src/dataset/speaker_filter_v4.py',
           'src/models/bounded_v3.py','src/experiments/bounded_v3.py','src/models/text_preprocessing.py',
           'src/experiments/audio_study.py','src/experiments/domain_v2.py','src/experiments/text_temporal_study.py',
           'data/processed/speaker_filter_v4/manifest.json']
    hashes={p:sha256(project/p) for p in paths}
    out=project/'experiments/runs'/('speaker_filter_v4_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir();(out/'models').mkdir()
    (out/'protocol.json').write_text(json.dumps({'hashes':hashes,'test_evaluated':False,'development_evaluated':False},indent=2))
    predictions=[];aps=[]
    for seed in SEEDS:
        for fold,(a,b) in enumerate(splits(y,5,seed)):
            logging.info('Speaker filter seed%s fold%s',seed,fold)
            train={k:x.iloc[a] for k,x in data.items()};valid={k:x.iloc[b] for k,x in data.items()}
            target,held=y.iloc[a],y.iloc[b];selected,audit=inner_candidates(train,target,seed,fold)
            for name,spec in selected.items():
                bundle={'feature':spec['feature'],'threshold':spec['threshold'],
                        'model':fit_text(train[spec['feature']],target,spec['parameters'],seed)}
                joblib.dump(bundle,out/'models'/f'{name}_{seed}_{fold}.joblib')
                scores=predict(bundle,valid);save_predictions(predictions,name,'clean',seed,fold,held,scores)
                aps.append({'condition':name,'scenario':'clean','seed':seed,'fold':fold,'average_precision':average_precision_score(held,scores)})
            audit.update(fit_ids=list(map(int,target.index)),heldout_ids=list(map(int,held.index)))
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    pred=pd.DataFrame(predictions);pred.to_parquet(out/'predictions.parquet',index=False);metrics=summarize(pred,out)
    pd.DataFrame(aps).to_csv(out/'fold_average_precision.csv',index=False)
    names=[n for n in selected if n!='reference']
    (out/'comparison.json').write_text(json.dumps({'f1':[paired(metrics,n,'reference') for n in names],
        'precision':{n:precision_screen(metrics,n) for n in names}},indent=2))
    old=pd.read_parquet(baseline/'predictions.parquet').query('condition=="joint"').set_index(['seed','fold','participant_id']).sort_index()
    new=pred.query('condition=="reference"').set_index(['seed','fold','participant_id']).sort_index()
    assert old.index.equals(new.index);np.testing.assert_allclose(old.score,new.score,atol=1e-12)
    for p,h in hashes.items():assert sha256(project/p)==h
    (out/'manifest.json').write_text(json.dumps({'hashes':hashes,'baseline':str(baseline),
        'models':{p.name:sha256(p) for p in (out/'models').iterdir()},'test_evaluated':False,'development_evaluated':False},indent=2))
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2],Path(sys.argv[1])))

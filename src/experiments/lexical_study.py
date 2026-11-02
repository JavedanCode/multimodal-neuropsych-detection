"""Nested evaluation of lexical counts and semantic-lexical combinations."""
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.audio_study import SEEDS,estimator
from src.experiments.baselines import sha256,TARGET_POLICY
from src.experiments.text_temporal_study import load,summarize
from src.models.text_preprocessing import TextPreprocessor
from src.models.lexical import LexicalPreprocessor


def inputs(project):
    sets,_,y=load(project)
    root=project/'data/processed/lexical_v1'
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['partition']!='train' or manifest['contract']['source']!=sha256(project/'src/dataset/lexical.py'):raise ValueError('Invalid lexical provenance')
    if manifest['contract']['screen_source']!=sha256(project/'src/dataset/text_temporal.py'):raise ValueError('Screen changed')
    if manifest['source_manifest']!=sha256(project/'data/processed/text_temporal_v1/manifest.json'):raise ValueError('Previous data changed')
    for name,digest in manifest['output_hashes'].items():
        if sha256(root/name)!=digest:raise ValueError('Lexical data changed')
    pd.testing.assert_series_equal(y,pd.read_parquet(root/'labels.parquet').PHQ_Binary)
    counts=sparse.load_npz(root/'counts.npz')
    if counts.shape!=(len(y),65536) or not np.isfinite(counts.data).all() or (counts.data<0).any():raise ValueError('Invalid count matrix')
    return {'counts':counts,'semantic':sets['text']},y


def subset(data,indices):
    return {'counts':data['counts'][indices],'semantic':data['semantic'].iloc[indices]}


def prepare(data,kind,weight):
    if kind=='old':return TextPreprocessor().fit(data['semantic'])
    return LexicalPreprocessor(weight).fit(data)


def transform(prep,data,kind):
    return prep.transform(data['semantic'] if kind=='old' else data)


def inner_search(data,y,splits,kind,family,seed):
    records=[]
    weights=[.25,1.,4.] if kind=='combined' else [0.]
    for weight in weights:
        prepared=[]
        for a,b in splits:
            fit,valid=subset(data,a),subset(data,b)
            prep=prepare(fit,kind,weight)
            prepared.append((a,b,transform(prep,fit,kind),transform(prep,valid,kind)))
        for c in ([.01,.1,1.] if kind=='old' else [.01,.1,1.,10.]):
            for class_weight in [None,'balanced']:
                params={'C':c,'class_weight':class_weight};scores=np.full(len(y),np.nan)
                for a,b,fit,valid in prepared:
                    scores[b]=estimator(family,params,seed).fit(fit,y.iloc[a]).decision_function(valid)
                if not np.isfinite(scores).all():raise ValueError('Invalid inner scores')
                for cutoff in ([0.] if kind=='old' else [0.,-.5,.5]):
                    records.append({'kind':kind,'family':family,'semantic_weight':weight,'parameters':params,'threshold':cutoff,
                        'inner_f1':float(f1_score(y,scores>=cutoff,zero_division=0)),'inner_scores':scores.tolist()})
    return records


def predict(bundle,data):
    raw=bundle['classifier'].decision_function(transform(bundle['preprocess'],data,bundle['kind']))
    return raw,raw-bundle['threshold']


def run(project):
    data,y=inputs(project)
    out=project/'experiments/runs'/('lexical_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir();(out/'models').mkdir()
    names=['docs/lexical_protocol.md','src/dataset/lexical.py','src/models/lexical.py','src/experiments/lexical_study.py',
           'src/dataset/text_temporal.py','src/models/text_preprocessing.py','src/experiments/text_temporal_study.py',
           'src/experiments/audio_study.py','src/experiments/metrics.py','src/experiments/baselines.py',
           'data/processed/lexical_v1/counts.npz','data/processed/lexical_v1/labels.parquet','data/processed/lexical_v1/manifest.json',
           'data/processed/text_temporal_v1/manifest.json','data/processed/text_temporal_v1/sequences.npz','data/processed/labels.parquet']
    hashes={n:sha256(project/n) for n in names}
    (out/'protocol.json').write_text(json.dumps({'hashes':hashes,'target':TARGET_POLICY,'seeds':SEEDS,
        'participants':list(map(int,y.index)),'test_evaluated':False,'development_evaluated':False},indent=2),encoding='utf-8')
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    records=[]
    for seed in SEEDS:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):
            fit,valid=subset(data,a),subset(data,b);target,held=y.iloc[a],y.iloc[b]
            splits=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(np.zeros(len(a)),target))
            audit={'fit_ids':list(map(int,target.index)),'heldout_ids':list(map(int,held.index)),
                   'inner_ids':[{'fit':list(map(int,target.index[ia])),'heldout':list(map(int,target.index[ib]))} for ia,ib in splits],
                   'candidates':{},'selected':{}}
            selected={};new=[]
            for kind in ['old','lexical','combined']:
                for family in ['logistic','linear']:
                    key=kind+'_'+family
                    logging.info('Lexical seed %s fold %s: %s',seed,fold,key)
                    candidates=inner_search(fit,target,splits,kind,family,seed)
                    audit['candidates'][key]=candidates
                    if kind=='old':selected[key]=max(candidates,key=lambda r:r['inner_f1'])
                    else:
                        selected[key+'_default']=max([r for r in candidates if r['threshold']==0],key=lambda r:r['inner_f1'])
                        selected[key+'_tuned']=max(candidates,key=lambda r:r['inner_f1'])
                        new.extend(candidates)
            selected['selected_new']=max(new,key=lambda r:r['inner_f1'])
            audit['selected']=selected;cache={}
            for condition,spec in selected.items():
                key=json.dumps({k:spec[k] for k in ['kind','family','semantic_weight','parameters']},sort_keys=True)
                if key not in cache:
                    prep=prepare(fit,spec['kind'],spec['semantic_weight'])
                    classifier=estimator(spec['family'],spec['parameters'],seed).fit(transform(prep,fit,spec['kind']),target)
                    cache[key]={'preprocess':prep,'classifier':classifier,'kind':spec['kind']}
                bundle={**cache[key],'threshold':spec['threshold']}
                raw,scores=predict(bundle,valid)
                path=out/'models'/f'{condition}_{seed}_{fold}.joblib';joblib.dump(bundle,path)
                np.testing.assert_array_equal(scores,predict(joblib.load(path),valid)[1])
                if not np.isfinite(scores).all():raise ValueError('Invalid outer scores')
                records.extend({'condition':condition,'scenario':'clean','seed':seed,'fold':fold,'participant_id':int(pid),
                    'target':int(t),'raw_score':float(r),'threshold':spec['threshold'],'score':float(s),'prediction':int(s>=0)}
                    for pid,t,r,s in zip(held.index,held,raw,scores))
            for constant in [0,1]:
                records.extend({'condition':f'constant_{constant}','scenario':'clean','seed':seed,'fold':fold,'participant_id':int(pid),
                    'target':int(t),'raw_score':float(constant),'threshold':.5,'score':constant-.5,'prediction':constant} for pid,t in held.items())
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    pred=pd.DataFrame(records)
    for _,g in pred.groupby(['condition','seed']):
        if len(g)!=len(y) or not g.participant_id.is_unique or set(g.participant_id)!=set(y.index):raise ValueError('Incomplete OOF')
    pred.to_parquet(out/'predictions.parquet',index=False);summarize(pred,out)
    for name,digest in hashes.items():
        if sha256(project/name)!=digest:raise ValueError('Input/source changed during fitting')
    (out/'verification.json').write_text(json.dumps({'prediction_rows':len(pred),'participants':len(y),
        'model_hashes':{p.name:sha256(p) for p in (out/'models').iterdir()},'test_evaluated':False,'hashes_verified':True},indent=2),encoding='utf-8')
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2]))

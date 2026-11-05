"""Build a reusable research checkpoint without a new holdout evaluation."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.experiments.mpnet_study import inputs
from src.experiments.audio_study import select_inner,estimator
from src.experiments.baselines import sha256,TARGET_POLICY
from src.models.text_preprocessing import TextPreprocessor


def run(project):
    sets,y=inputs(project);x=sets['mpnet']
    output=project/'models'/('research_reference_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    paths=['docs/research_reference_protocol.md','src/experiments/refit_reference.py','src/experiments/mpnet_study.py',
           'src/experiments/audio_study.py','src/models/text_preprocessing.py','src/experiments/baselines.py',
           'data/processed/mpnet_v1/features.parquet','data/processed/mpnet_v1/labels.parquet','data/processed/mpnet_v1/manifest.json']
    hashes={p:sha256(project/p) for p in paths}
    splits=list(StratifiedKFold(3,shuffle=True,random_state=43).split(x,y))
    prepared=[]
    for a,b in splits:
        prep=TextPreprocessor().fit(x.iloc[a])
        prepared.append((a,b,prep.transform(x.iloc[a]),prep.transform(x.iloc[b])))
    selected,_,candidates=select_inner(x,y,'linear',prepared,42,[0.])
    model=Pipeline([('preprocess',TextPreprocessor()),('classifier',estimator('linear',selected['parameters'],42))]).fit(x,y)
    path=output/'model.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(model.decision_function(x),joblib.load(path).decision_function(x))
    for name,digest in hashes.items():
        if sha256(project/name)!=digest:raise ValueError('Source/input changed during refit')
    mpnet=json.loads((project/'data/processed/mpnet_v1/manifest.json').read_text())
    manifest={'purpose':'research checkpoint; no new generalization estimate','target_policy':TARGET_POLICY,
              'participants':list(map(int,y.index)),'positives':int(y.sum()),'selected_parameters':selected['parameters'],
              'decision_margin_threshold':0.,'input_columns':list(x),'input_shape':[len(x),x.shape[1]],
              'encoder':mpnet['contract']['revision'],'encoder_hashes':mpnet['contract']['models'],
              'source_hashes':hashes,'model_sha256':sha256(path),'serialization_verified':True,
              'test_evaluated':False,'development_evaluated':False,'deployed':False}
    audit={'selection_only_not_performance':True,'selected':selected,'candidates':candidates,
           'folds':[{'fit_ids':list(map(int,y.index[a])),'heldout_ids':list(map(int,y.index[b]))} for a,b in splits]}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'selection.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    return output


if __name__=='__main__':
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2]))

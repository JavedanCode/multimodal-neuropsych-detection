"""Verify encoder caches/replay and independently audit the matched classifiers."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,balanced_accuracy_score,confusion_matrix,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.baselines import sha256
from src.experiments.mpnet_study import inputs
from src.dataset.lexical import read_transcript
from src.dataset.text_temporal import safe_rows,encode


def verify(project,out):
    import torch
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    sets,y=inputs(project);root=project/'data/processed/mpnet_v1'
    manifest=json.loads((root/'manifest.json').read_text())
    model_dir=project/'models/pretrained/all-mpnet-base-v2'
    for name,digest in manifest['contract']['models'].items():assert sha256(model_dir/name)==digest
    for i,pid in enumerate(y.index):
        path=root/'participant_cache'/f'{pid}.npz';record=manifest['participants'][str(pid)]
        assert sha256(path)==record['hash']
        stat=(project/'data/raw'/f'{pid}_P.tar.gz').stat()
        assert stat.st_size==record['key']['previous']['archive_size'] and stat.st_mtime_ns==record['key']['previous']['mtime']
        with np.load(path) as data:
            assert data['embeddings'].shape==(record['audit']['retained'],768)
            assert np.isfinite(data['embeddings']).all()
            np.testing.assert_array_equal(data['embeddings'].mean(0),data['mean'])
            np.testing.assert_array_equal(data['mean'],sets['mpnet'].iloc[i])
    pid=int(y.index[0]);frame,digest=read_transcript(project/'data/raw'/f'{pid}_P.tar.gz')
    assert digest==manifest['participants'][str(pid)]['key']['transcript_hash']
    duration=json.loads((project/'reports/data_audit/label_speech_cache'/f'{pid}.json').read_text())['result']['audio_seconds']
    texts,_,_,_=safe_rows(frame,duration)
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model=AutoModel.from_pretrained(model_dir,local_files_only=True,use_safetensors=True,add_pooling_layer=False,attn_implementation='eager').eval().to('cuda')
    replay=encode(texts,tokenizer,model)
    with np.load(root/'participant_cache'/f'{pid}.npz') as saved:
        maximum=float(np.abs(replay-saved['embeddings']).max())
        np.testing.assert_allclose(replay,saved['embeddings'],atol=1e-6)
    del model
    protocol=json.loads((out/'protocol.json').read_text());completed=json.loads((out/'verification.json').read_text())
    for name,digest in protocol['hashes'].items():assert sha256(project/name)==digest,name
    for name,digest in completed['model_hashes'].items():assert sha256(out/'models'/name)==digest,name
    pred=pd.read_parquet(out/'predictions.parquet')
    metrics=pd.read_csv(out/'repeat_metrics.csv').set_index(['condition','seed'])
    summary=pd.read_csv(out/'summary.csv').set_index('condition')
    assert not pred.duplicated(['condition','seed','participant_id']).any()
    np.testing.assert_array_equal(pred.target,y.loc[pred.participant_id])
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
    restored=0
    for seed in protocol['seeds']:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):
            audit=json.loads((out/f'fold_{seed}_{fold}.json').read_text())
            assert audit['fit_ids']==list(y.index[a]) and audit['heldout_ids']==list(y.index[b])
            inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(np.zeros(len(a)),y.iloc[a]))
            for (ia,ib),saved in zip(inner,audit['inner_ids']):
                assert saved=={'fit':list(y.index[a][ia]),'heldout':list(y.index[a][ib])}
            for name,x in sets.items():
                for family in ['logistic','linear']:
                    condition=name+'_'+family
                    assert audit['selected'][condition]==max(audit['candidates'][condition],key=lambda r:r['selection_f1'])
                    model=joblib.load(out/'models'/f'{condition}_{seed}_{fold}.joblib')
                    scores=model.decision_function(x.iloc[b])
                    stored=pred[(pred.condition==condition)&(pred.seed==seed)&(pred.fold==fold)].set_index('participant_id').loc[y.index[b]]
                    np.testing.assert_allclose(stored.score,scores,atol=1e-10)
                    restored+=1
    old=pd.read_parquet(project/'experiments/runs/text_20261030T230941483092Z/predictions.parquet')
    old=old[old.condition.isin(['text_logistic','text_linear'])].copy()
    old['condition']=old.condition.str.replace('text_','minilm_',regex=False)
    keys=['condition','seed','fold','participant_id'];old=old.set_index(keys)
    current=pred[pred.condition.isin(['minilm_logistic','minilm_linear'])].set_index(keys).loc[old.index]
    np.testing.assert_allclose(current.score,old.score,rtol=1e-8,atol=1e-8)
    np.testing.assert_array_equal(current.prediction,old.prediction)
    result={'verified_embedding_caches':len(y),'encoder_max_replay_difference':maximum,'restored_pipelines':restored,
            'prediction_rows':len(pred),'metric_rows':len(metrics),'prior_control_predictions_matched':len(old),'folds_and_hashes_verified':True}
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):print(verify(Path(__file__).resolve().parents[2],args.run))

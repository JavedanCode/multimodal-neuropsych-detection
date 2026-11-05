"""Fixed semantic-topic summaries using the existing local frozen encoder."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.experiments.baselines import sha256
from src.experiments.mpnet_study import inputs
from src.dataset.text_temporal import encode

ANCHORS = [
    'I describe my daily activities, hobbies, interests, and how I spend my time.',
    'I describe my relationships with friends, family, and other people.',
    'I describe my work, job, school, education, and career experiences.',
    'I describe my sleep, rest, tiredness, and energy during the day.',
    'I describe my feelings, emotions, stress, and how I cope with difficulties.',
    'I describe my physical health, exercise, and bodily experiences.',
]


def summarize_topics(embeddings, anchors):
    x=np.asarray(embeddings,dtype=float);a=np.asarray(anchors,dtype=float)
    if x.ndim!=2 or a.ndim!=2 or x.shape[1]!=a.shape[1] or not len(x):
        raise ValueError('Invalid topic embedding dimensions')
    if not np.isfinite(x).all() or not np.isfinite(a).all():raise ValueError('Nonfinite embedding')
    similarity=(x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)) @ (a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)).T
    assignments=similarity.argmax(1);global_mean=x.mean(0)
    means=np.stack([x[assignments==i].mean(0) if (assignments==i).any() else global_mean for i in range(len(a))])
    return means.mean(0),means.ravel(),np.bincount(assignments,minlength=len(a))


def build(project):
    import torch
    from transformers import AutoModel,AutoTokenizer
    _,y=inputs(project);root=project/'data/processed';prior=json.loads((root/'mpnet_v1/manifest.json').read_text())
    model_dir=project/'models/pretrained/all-mpnet-base-v2'
    for name,h in prior['contract']['models'].items():assert sha256(model_dir/name)==h
    paths=['src/dataset/topics_v3.py','src/dataset/text_temporal.py','docs/bounded_v3_protocol.md','data/processed/mpnet_v1/manifest.json']
    hashes={p:sha256(project/p) for p in paths}
    torch.manual_seed(42);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model=AutoModel.from_pretrained(model_dir,local_files_only=True,use_safetensors=True,add_pooling_layer=False,attn_implementation='eager').eval().to('cuda')
    model.requires_grad_(False);anchors=encode(ANCHORS,tokenizer,model)
    balanced=[];concatenated=[];counts=[]
    for pid in y.index:
        p=root/'mpnet_v1/participant_cache'/f'{pid}.npz';assert sha256(p)==prior['participants'][str(pid)]['hash']
        with np.load(p) as z:b,c,n=summarize_topics(z['embeddings'],anchors)
        balanced.append(b);concatenated.append(c);counts.append(n)
    out=root/'topics_v3';out.mkdir(exist_ok=True)
    for name,values in [('balanced',balanced),('concatenated',concatenated)]:
        pd.DataFrame(values,index=y.index,columns=[f'text__{i}' for i in range(len(values[0]))]).to_parquet(out/f'{name}.parquet')
    np.save(out/'anchors.npy',anchors);y.to_frame().to_parquet(out/'labels.parquet')
    pd.DataFrame(counts,index=y.index,columns=[f'topic_{i}' for i in range(len(ANCHORS))]).to_csv(out/'counts.csv')
    for p,h in hashes.items():assert sha256(project/p)==h
    m={'hashes':hashes,'anchors':ANCHORS,'encoder':prior['contract']['models'],'speaker_validated':False,
       'partition':'train','outputs':{p.name:sha256(p) for p in out.iterdir() if p.name!='manifest.json'}}
    (out/'manifest.json').write_text(json.dumps(m,indent=2));return out


if __name__=='__main__':print(build(Path(__file__).resolve().parents[2]))

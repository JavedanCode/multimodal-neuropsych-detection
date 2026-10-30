"""Local frozen Wav2Vec2 embeddings from a fixed training-only audio sample."""
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import tarfile
import time
import wave

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from src.dataset.inventory import _is_unsafe_member
from src.experiments.baselines import sha256

REVISION='22aad52d435eb6dbaf354bdad9b0da84ce7d6156'
MODEL_ID='facebook/wav2vec2-base-960h'
RATE=16000
WINDOW=8*RATE
MAX_WINDOWS=64
MAX_BYTES=768*1024*1024


def window_starts(samples):
    count=min(MAX_WINDOWS,samples//WINDOW)
    if count<1:raise ValueError('Recording shorter than eight seconds')
    starts=np.rint(np.linspace(0,samples-WINDOW,count)).astype(np.int64)
    if len(starts)>1 and (np.diff(starts)<WINDOW).any():raise ValueError('Overlapping sampled windows')
    return starts


def decode_audio(raw):
    with wave.open(io.BytesIO(raw)) as wav:
        channels=wav.getnchannels();rate=wav.getframerate();n=wav.getnframes()
        if channels not in [1,2] or rate not in [16000,48000] or wav.getsampwidth()!=2 or n/rate>3600:
            raise ValueError('Unsupported/unbounded WAV')
        pcm=wav.readframes(n)
    if len(pcm)!=n*channels*2:raise ValueError('Truncated WAV')
    values=np.frombuffer(pcm,dtype='<i2').reshape(-1,channels).astype(np.float32).mean(axis=1)/32768.
    if rate!=RATE:values=resample_poly(values,RATE//np.gcd(RATE,rate),rate//np.gcd(RATE,rate)).astype(np.float32)
    if not np.isfinite(values).all():raise ValueError('Nonfinite waveform')
    return values,{'channels':channels,'source_rate':rate,'source_frames':n,'duration_seconds':n/rate}


def read_audio(path):
    pid=int(path.name.split('_')[0]);wanted=f'{pid}_AUDIO.wav'
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            if _is_unsafe_member(member.name) or member.issym() or member.islnk():raise ValueError('Unsafe member')
            if Path(member.name).name!=wanted:continue
            if not member.isfile() or member.size>MAX_BYTES:raise ValueError('Invalid/oversized WAV')
            raw=archive.extractfile(member).read(MAX_BYTES+1)
            if len(raw)!=member.size:raise ValueError('WAV size mismatch')
            audio,meta=decode_audio(raw)
            return audio,{**meta,'wav_sha256':hashlib.sha256(raw).hexdigest()}
    raise ValueError('Missing WAV')


def pool_windows(embeddings):
    values=np.asarray(embeddings,dtype=np.float64)
    if values.ndim!=2 or values.shape[0]<1 or values.shape[1]!=768 or not np.isfinite(values).all():
        raise ValueError('Invalid speech embeddings')
    means=values.mean(axis=0);std=values.std(axis=0)
    return {f'audio__dense_wav2vec_{i}__{stat}':float(v) for i in range(768)
            for stat,v in [('mean',means[i]),('std',std[i])]}


def build(project,limit=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    import transformers
    import scipy
    from transformers import Wav2Vec2Model,Wav2Vec2FeatureExtractor
    if not torch.cuda.is_available():raise RuntimeError('GPU required for this extraction protocol')
    torch.manual_seed(42);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    root=project/'data/processed';labels=pd.read_parquet(root/'labels.parquet')
    previous=pd.read_parquet(root/'segmentation_study_v1/labels.parquet')
    prior=json.loads((root/'segmentation_study_v1/manifest.json').read_text())
    ids=labels.index[(labels.split=='train')&~labels.index.isin(prior['excluded_ids'])]
    if not previous.equals(labels.loc[ids,['PHQ_Binary']]):raise ValueError('Training cohort mismatch')
    model_dir=project/'models/pretrained/wav2vec2-base-960h'
    model_hashes={n:sha256(model_dir/n) for n in ['config.json','preprocessor_config.json','model.safetensors']}
    extractor=Wav2Vec2FeatureExtractor.from_pretrained(model_dir,local_files_only=True)
    model,loading=Wav2Vec2Model.from_pretrained(model_dir,local_files_only=True,use_safetensors=True,output_loading_info=True)
    if any(not k.endswith('masked_spec_embed') for k in loading['missing_keys']) or loading['mismatched_keys'] or loading['error_msgs']:
        raise ValueError('Speech encoder weights did not fully load')
    # The ASR checkpoint omits the training-only masking vector. It is unused
    # in eval mode; explicitly disable augmentation rather than train that vector.
    model.config.apply_spec_augment=False
    model=model.eval().to('cuda')
    model.requires_grad_(False)
    out=root/'speech_study_v1';out.mkdir(exist_ok=True);cache=out/'participant_cache';cache.mkdir(exist_ok=True)
    contract={'source':sha256(Path(__file__)),'model_id':MODEL_ID,'revision':REVISION,'model_hashes':model_hashes,
              'rate':RATE,'window':WINDOW,'max_windows':MAX_WINDOWS,'batch_size':4,'dtype':'float32',
              'torch':torch.__version__,'transformers':transformers.__version__,
              'numpy':np.__version__,'scipy':scipy.__version__,
              'loading_info':loading,'gpu':torch.cuda.get_device_name(0)}
    records={}
    for number,pid in enumerate(ids[:limit] if limit else ids,1):
        path=project/'data/raw'/f'{pid}_P.tar.gz';stat=path.stat()
        key={**contract,'archive_size':stat.st_size,'archive_mtime_ns':stat.st_mtime_ns}
        target=cache/f'{pid}.json';windows_file=cache/f'{pid}.npz'
        saved=json.loads(target.read_text()) if target.exists() else {}
        if saved.get('key')!=key or not windows_file.exists() or saved.get('windows_sha256')!=sha256(windows_file):
            started=time.perf_counter();audio,metadata=read_audio(path);starts=window_starts(len(audio));vectors=[]
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for offset in range(0,len(starts),4):
                    batch=[audio[s:s+WINDOW] for s in starts[offset:offset+4]]
                    inputs=extractor(batch,sampling_rate=RATE,return_tensors='pt',padding=False)
                    hidden=model(input_values=inputs.input_values.to('cuda')).last_hidden_state
                    vectors.append(hidden.mean(dim=1).cpu().numpy())
            embeddings=np.concatenate(vectors)
            features=pool_windows(embeddings)
            np.savez_compressed(windows_file,starts=starts,embeddings=embeddings)
            saved={'key':key,'features':features,'windows_sha256':sha256(windows_file),
                'diagnostics':{**metadata,'windows':len(starts),'sampled_seconds':len(starts)*8,
                    'sampled_fraction':len(starts)*WINDOW/len(audio),'elapsed_seconds':time.perf_counter()-started,
                    'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}}
            tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(saved,allow_nan=False),encoding='utf-8');tmp.replace(target)
        records[int(pid)]=saved
        logging.info('Wav2Vec2 %s/%s: %s windows, %.1fs, %.2f GiB peak',number,len(ids),saved['diagnostics']['windows'],
                     saved['diagnostics']['elapsed_seconds'],saved['diagnostics']['peak_cuda_allocated_bytes']/1024**3)
    if limit:return out
    features=pd.DataFrame.from_dict({pid:r['features'] for pid,r in records.items()},orient='index').reindex(ids)
    features.index.name=labels.index.name;features.to_parquet(out/'features.parquet')
    labels.loc[ids,['PHQ_Binary']].to_parquet(out/'labels.parquet')
    manifest={'contract':contract,'partition':'train','excluded_ids':prior['excluded_ids'],'participant_ids':list(map(int,ids)),
        'participants':{str(pid):r['diagnostics'] for pid,r in records.items()},
        'output_hashes':{n:sha256(out/n) for n in ['features.parquet','labels.parquet']},
        'source_tables':{n:sha256(root/n) for n in ['features.parquet','labels.parquet','segmentation_study_v1/manifest.json']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return out


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(message)s')
    print(build(Path(__file__).resolve().parents[2],args.limit))

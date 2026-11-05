"""Versioned utterance-aligned features, measured quality, and local review clips."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import io
import json
import logging
import tarfile
import wave
import hashlib
import numpy as np
import pandas as pd
from src.dataset.inventory import _is_unsafe_member
from src.dataset.contracts_v2 import screened_rows,interval_summary,domain
from src.dataset.speech_embeddings import decode_audio
from src.experiments.baselines import sha256

QUALITY=['audio_finite_fraction','audio_frame_coverage','audio_clipping_fraction','audio_log_rms',
         'audio_energy_spread_db','audio_zero_fraction','face_retained_fraction','face_confidence',
         'face_frame_coverage','text_confidence','text_confidence_known']


def waveform_quality(audio,starts,ends,rate=16000):
    out=[]
    for start,end in zip(starts,ends):
        v=audio[max(0,int(start*rate)):min(len(audio),int(end*rate))].astype(np.float64)
        if not len(v):out.append([0,0,0,1]);continue
        rms=np.sqrt(np.mean(v*v));chunks=len(v)//320
        energy=np.sqrt((v[:chunks*320].reshape(chunks,320)**2).mean(1)) if chunks else np.array([rms])
        lo,hi=np.quantile(energy,[.1,.9]);spread=20*np.log10((hi+1e-8)/(lo+1e-8))
        out.append([np.mean(np.abs(v)>=.999),np.log10(rms+1e-8),spread,np.mean(v==0)])
    return np.asarray(out,np.float32)


def read(path):
    pid=path.name.split('_')[0]
    wanted={f'{pid}_AUDIO.wav':'wave',f'{pid}_Transcript.csv':'text',
            f'{pid}_OpenSMILE2.3.0_egemaps.csv':'audio',f'{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv':'face'}
    result={};hashes={}
    with tarfile.open(path,'r|gz') as archive:
        for m in archive:
            if _is_unsafe_member(m.name) or m.issym() or m.islnk():raise ValueError('Unsafe archive member')
            key=wanted.get(Path(m.name).name)
            if key is None:continue
            if key in result or not m.isfile() or m.size>768*1024**2:raise ValueError('Invalid member')
            raw=archive.extractfile(m).read(m.size+1)
            if len(raw)!=m.size:raise ValueError('Truncated member')
            hashes[key]=hashlib.sha256(raw).hexdigest()
            if key=='wave':result[key],result['wave_meta']=decode_audio(raw)
            else:
                frame=pd.read_csv(io.BytesIO(raw),sep=';' if key=='audio' else ',',nrows=500001)
                frame.columns=frame.columns.str.strip()
                if len(frame)>500000 or frame.columns.duplicated().any():raise ValueError('Invalid table')
                result[key]=frame
            if all(k in result for k in ['wave','text','audio','face']):break
    if not all(k in result for k in ['wave','text','audio','face']):raise ValueError('Missing member')
    return result,hashes


def coverage(count,times,starts,ends):
    delta=np.diff(times);delta=delta[delta>0]
    step=np.median(delta) if len(delta) else 1
    return np.minimum(1,count/np.maximum(1,(ends-starts)/step)).astype(np.float32)


def write_review(folder,pid,audio,rows):
    folder.mkdir(parents=True,exist_ok=True);items=[]
    # Two transcript-covered utterances and the longest uncovered gap, all chosen
    # without model predictions. Files remain local under ignored data/processed.
    indices=sorted(set([len(rows['starts'])//3,2*len(rows['starts'])//3]))
    for i in indices:
        start=float(rows['starts'][i]);end=min(float(rows['ends'][i]),start+8)
        items.append({'kind':'transcript_covered','start':start,'end':end,'transcript':rows['text'][i]})
    boundaries=sorted(zip(rows['starts'],rows['ends']));cursor=0.;gaps=[]
    for start,end in boundaries:
        if start>cursor:gaps.append((cursor,float(start)))
        cursor=max(cursor,float(end))
    if len(audio)/16000>cursor:gaps.append((cursor,len(audio)/16000))
    if gaps:
        start,end=max(gaps,key=lambda v:v[1]-v[0]);items.append({'kind':'outside_screened_intervals','start':start,'end':min(end,start+8),'transcript':None})
    for i,item in enumerate(items):
        name=f'{pid}_{i}.wav';values=audio[int(item['start']*16000):int(item['end']*16000)]
        with wave.open(str(folder/name),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes((np.clip(values,-1,1)*32767).astype('<i2').tobytes())
        item.update(file=name,speaker=None,timing_correct=None,reviewed=False)
    (folder/f'{pid}.json').write_text(json.dumps(items,indent=2),encoding='utf-8')


def build(project):
    root=project/'data/processed';labels=pd.read_parquet(root/'mpnet_v1/labels.parquet').PHQ_Binary
    official=pd.read_parquet(root/'labels.parquet');assert (official.loc[labels.index,'split']=='train').all()
    pd.testing.assert_series_equal(labels,official.loc[labels.index,'PHQ_Binary'])
    prior=json.loads((root/'mpnet_v1/manifest.json').read_text());schemas=json.loads((root/'stage3_report.json').read_text())['schemas']
    out=root/'aligned_v2';cache=out/'participant_cache';cache.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(project)):sha256(p) for p in [Path(__file__),project/'src/dataset/contracts_v2.py',project/'src/dataset/speech_embeddings.py']}
    review_ids=[];domains=domain(labels.index)
    for d in ['WoZ','AI']:
        for t in [0,1]:review_ids.extend(list(labels.index[(domains==d)&(labels==t)])[:2])
    def one(pid):
        archive=project/'data/raw'/f'{pid}_P.tar.gz';stat=archive.stat()
        embpath=root/'mpnet_v1/participant_cache'/f'{pid}.npz';digest=sha256(embpath)
        assert digest==prior['participants'][str(pid)]['hash']
        key={'sources':sources,'size':stat.st_size,'mtime':stat.st_mtime_ns,'embedding_hash':digest}
        target=cache/f'{pid}.npz';meta=cache/f'{pid}.json'
        saved=json.loads(meta.read_text()) if meta.exists() else {}
        if saved.get('key')==key and target.exists() and saved.get('hash')==sha256(target):return pid,saved
        data,hashes=read(archive);duration=data['wave_meta']['duration_seconds'];rows=screened_rows(data['text'],duration)
        if hashes['text']!=prior['participants'][str(pid)]['key']['transcript_hash']:raise ValueError('Transcript changed')
        with np.load(embpath) as e:text=e['embeddings']
        if len(text)!=len(rows['starts']) or rows['audit']['retained']!=prior['participants'][str(pid)]['audit']['retained']:raise ValueError('Screen alignment changed')
        order=np.argsort(rows['starts'],kind='stable');text=text[order]
        rows['text']=[rows['text'][i] for i in order]
        for field in ['starts','ends','row_indices','confidence','confidence_known']:rows[field]=rows[field][order]
        a=data['audio'];f=data['face'];at=pd.to_numeric(a.frameTime).to_numpy(float);ft=pd.to_numeric(f.timestamp).to_numpy(float)
        if abs(at[-1]-duration)>1:raise ValueError('Audio endpoint mismatch')
        audio,am,aq,ac=interval_summary(a[schemas['audio']['columns']],at,rows['starts'],rows['ends'])
        fvalid=(f.success==1)&f.confidence.between(.8,1)
        face,fm,fq,fc=interval_summary(f[schemas['facial']['columns']],ft,rows['starts'],rows['ends'],fvalid)
        confidence,_,_,_=interval_summary(f[['confidence']],ft,rows['starts'],rows['ends'],f.confidence.between(0,1))
        quarantined=abs(ft[-1]-duration)>1
        if quarantined:face[:]=0;fm[:]=False;fq[:]=0;confidence[:]=0;fc[:]=0
        signal=waveform_quality(data['wave'],rows['starts'],rows['ends'])
        q=np.column_stack([aq,coverage(ac,at,rows['starts'],rows['ends']),signal,fq,confidence[:,0],
                           coverage(fc,ft,rows['starts'],rows['ends']),rows['confidence'],rows['confidence_known']]).astype(np.float32)
        masks=np.column_stack([np.ones(len(text),bool),am,fm])
        if not all(np.isfinite(v).all() for v in [text,audio,face,q]):raise ValueError('Nonfinite aligned features')
        np.savez_compressed(target,text=text,audio=audio,face=face,quality=q,masks=masks,starts=rows['starts'],ends=rows['ends'])
        saved={'key':key,'hash':sha256(target),'member_hashes':hashes,'audit':rows['audit'],
               'speaker_validated':False,'facial_timing_quarantined':bool(quarantined),'utterances':len(text)}
        meta.write_text(json.dumps(saved,indent=2))
        if pid in review_ids:write_review(out/'speaker_review',pid,data['wave'],rows)
        return pid,saved
    records={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(pid,saved) in enumerate(pool.map(one,labels.index),1):
            records[str(pid)]=saved;logging.info('Aligned v2 %s/%s',i,len(labels))
    labels.to_frame().to_parquet(out/'labels.parquet')
    manifest={'sources':sources,'participants':records,'quality_columns':QUALITY,'partition':'train',
              'speaker_validated':False,'quality_is_observed_proxy_not_calibrated_reliability':True,
              'review_participants':list(map(int,review_ids)),'label_policy':'released_binary_provisional',
              'labels_hash':sha256(out/'labels.parquet')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s');print(build(Path(__file__).resolve().parents[2]))

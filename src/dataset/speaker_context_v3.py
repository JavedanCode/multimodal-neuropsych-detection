"""Full utterances and context for human-flagged speaker review cases."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import html
import wave
import numpy as np
from src.dataset.aligned_v2 import read
from src.dataset.review_annotations_v3 import run as validate_review
from src.experiments.baselines import sha256


def build(project):
    review=project/'data/processed/aligned_v2/speaker_review';report=validate_review(project)
    if report.get('status'):raise ValueError('Complete the initial review first')
    annotations=json.loads((review/'annotations.json').read_text(encoding='utf-8-sig'))
    selected=[r for r in annotations if r['kind']=='transcript_covered' and
              (r['speaker']!='Participant' or r['timing']!='Matches' or 'longer' in r.get('notes','').lower())]
    out=review/'context_v3';out.mkdir(exist_ok=True);aligned=review.parent
    manifest=json.loads((aligned/'manifest.json').read_text())
    def one(pid):
        data,hashes=read(project/'data/raw'/f'{pid}_P.tar.gz')
        assert hashes==manifest['participants'][str(pid)]['member_hashes']
        records=[]
        with np.load(aligned/'participant_cache'/f'{pid}.npz') as z:
            for r in selected:
                if int(r['file'].split('_')[0])!=pid:continue
                matches=np.flatnonzero(np.isclose(z['starts'],r['start'],rtol=0,atol=1e-8))
                if len(matches)!=1:raise ValueError('Ambiguous utterance start')
                start=float(z['starts'][matches[0]]);end=float(z['ends'][matches[0]])
                assert r['end']<=end+1e-8
                before=max(0,start-5);after=min(len(data['wave'])/16000,end+5)
                names={}
                for name,a,b in [('target',start,end),('context',before,after)]:
                    filename=Path(r['file']).stem+'_'+name+'.wav';names[name]=filename
                    v=data['wave'][int(a*16000):int(b*16000)]
                    with wave.open(str(out/filename),'wb') as w:
                        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes((np.clip(v,-1,1)*32767).astype('<i2').tobytes())
                records.append({'review_key':names['target'],'file':names['target'],'context_file':names['context'],
                    'kind':'transcript_covered','start':start,'end':end,'transcript':r['transcript'],
                    'context_target_start':start-before,'context_target_end':end-before,
                    'prior_speaker':r['speaker'],'prior_timing':r['timing'],'reviewed':False,'speaker':None,'timing':None})
        return records
    ids=sorted({int(r['file'].split('_')[0]) for r in selected})
    with ThreadPoolExecutor(max_workers=2) as pool:items=[r for group in pool.map(one,ids) for r in group]
    (out/'clips.json').write_text(json.dumps(items,indent=2),encoding='utf-8')
    cards=[]
    for i,r in enumerate(items):
        cards.append(f'''<section><h2>Follow-up {i+1}</h2><p>{html.escape(r['transcript'])}</p>
<p>Full target utterance ({r['end']-r['start']:.1f} seconds):</p><audio controls preload="none" src="{r['file']}"></audio>
<p>Surrounding conversation: target is {r['context_target_start']:.1f}–{r['context_target_end']:.1f} seconds within this clip.</p>
<audio controls preload="none" src="{r['context_file']}"></audio>
<p>Judge the target utterance only, even if context contains both speakers.</p>
<label>Target speaker <select id="s{i}"><option value="">Unreviewed</option><option>Participant</option><option>Interviewer</option><option>Mixed</option><option>Uncertain</option></select></label>
<label>Full target transcript <select id="t{i}"><option value="">Unreviewed</option><option>Matches</option><option>Does not match</option><option>Uncertain</option></select></label>
<label>Notes <input id="n{i}"></label></section>''')
    payload=json.dumps(items).replace('<','\\u003c')
    script='''const items=PAYLOAD;document.getElementById('save').onclick=()=>{const rows=items.map((r,i)=>({...r,speaker:document.getElementById('s'+i).value,timing:document.getElementById('t'+i).value,notes:document.getElementById('n'+i).value,reviewed:!!document.getElementById('s'+i).value&&!!document.getElementById('t'+i).value}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));a.download='annotations_context.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};'''.replace('PAYLOAD',payload)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Speaker review with context</title><style>body{font:17px system-ui;max-width:900px;margin:32px auto;padding:16px;background:#f5f6f8}section{background:white;padding:20px;margin:18px 0;border:1px solid #ccc}audio{width:100%}label{display:block;margin:12px 0}select,input,button{font:inherit;padding:8px}button{position:sticky;bottom:16px}</style><h1>Speaker review with context</h1><p>Each case includes the complete target utterance and a second recording with five seconds of surrounding conversation where available. Reassess the target speaker and transcript; use Uncertain when needed. Nothing is uploaded.</p>'''+''.join(cards)+f'<button id="save">Export follow-up annotations</button><script>{script}</script></html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    evidence={'initial_annotations_hash':sha256(review/'annotations.json'),'source_hash':sha256(Path(__file__)),
              'cases':len(items),'participants':len(ids),'files':{p.name:sha256(p) for p in out.glob('*.wav')}}
    (out/'manifest.json').write_text(json.dumps(evidence,indent=2));return out/'index.html'


if __name__=='__main__':print(build(Path(__file__).resolve().parents[2]))

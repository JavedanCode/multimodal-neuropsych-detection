"""Generate a local-only listening worksheet; never infer reviewer answers."""

from pathlib import Path
import json
import html


def build(project):
    folder = project / "data/processed/aligned_v2/speaker_review"
    items = []
    for p in sorted(folder.glob("*.json")):
        if p.name == "annotations.json":
            continue
        for item in json.loads(p.read_text()):
            items.append({**item, "review_key": item["file"]})
    if not items:
        raise ValueError("No review clips generated")
    cards = []
    for i, r in enumerate(items):
        cards.append(f"""<section><h2>Clip {i+1}: {html.escape(r['kind'])}</h2>
<audio controls preload="none" src="{html.escape(r['file'],quote=True)}"></audio>
<p>{html.escape(r['transcript'] or 'Outside screened transcript intervals; speaker unknown.')}</p>
<label>Speaker <select id="speaker{i}"><option value="">Unreviewed</option><option>Participant</option><option>Interviewer</option><option>Mixed</option><option>Uncertain</option></select></label>
<label>Transcript timing <select id="timing{i}"><option value="">Unreviewed</option><option>Matches</option><option>Does not match</option><option>Not applicable</option><option>Uncertain</option></select></label>
<label>Notes <input id="notes{i}" type="text"></label></section>""")
    payload = json.dumps(items).replace("<", "\\u003c")
    script = """const items=PAYLOAD;
document.getElementById('save').onclick=()=>{
 const rows=items.map((r,i)=>({...r,speaker:document.getElementById('speaker'+i).value,
 timing:document.getElementById('timing'+i).value,notes:document.getElementById('notes'+i).value,
 reviewed:!!document.getElementById('speaker'+i).value&&!!document.getElementById('timing'+i).value}));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));
 a.download='annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};""".replace("PAYLOAD", payload)
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>Local speaker and timing review</title>
<style>body{font:17px system-ui;max-width:900px;margin:32px auto;padding:16px;background:#f5f6f8;color:#18202b}section{background:white;padding:20px;margin:16px 0;border:1px solid #d4d9df;border-radius:10px}label{display:block;margin:12px 0}select,input,button{font:inherit;padding:8px}input{width:75%}audio{width:100%}button{position:sticky;bottom:16px}</style>
<h1>Speaker and timing review</h1><p>Local research recordings. Nothing is uploaded. Review each clip before assigning a speaker. A transcript interval does not prove participant identity. This sample cannot validate every recording.</p>
<p>Two covered intervals and one gap per sampled participant, selected without model predictions. Export annotations when finished; retain them locally in this folder.</p>"""
        + "".join(cards)
        + f'<button id="save">Export annotations</button><script>{script}</script></html>'
    )
    target = folder / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build(Path(__file__).resolve().parents[2]))

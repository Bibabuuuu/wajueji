"""Save reviewable results and viewer manifests without running physics or a UI."""
import json
from pathlib import Path

def export_replay(path,contact,tail,label,summary,prefix=None):
    sources=[]
    segments=[(contact,'Dig / curl')]+([] if prefix is None else [(prefix,'Saved finish-curl prefix')])+[(tail,'Loaded exit / fixed validation')]
    for folder,name in segments:
        directory=Path(folder)/'display'
        reconstruction=json.loads((directory/'reconstruction.json').read_text())
        sources.append(dict(path=str(directory.resolve()),label=name,
                            cache_sha256=reconstruction['cache_sha256']))
    Path(path).write_text(json.dumps(dict(label=label,summary=summary,sources=sources),indent=2)+'\n')

def export_comparison(root,report):
    root=Path(root);best=report.get('best')
    rows=['# Night digging parameter search','',
          f"Batch status: {report['status']}",
          'Scope: digging only; no neural weight updates; no formal fill or three-scoop qualification.','']
    if report.get('failure'):rows.extend([f"Stopped: {report['failure']}",''])
    rows.extend(['| Trial | Working fill % | Digging simulation seconds | Selection |',
                 '| --- | ---: | ---: | --- |'])
    for item in report.get('trials',[]):
        score=item['score']
        rows.append(f"| {item['name']} | {score['working_fill_percent']:.3f} | {score['seconds']:.3f} | {item['selection_status']} |")
    if best:
        rows.extend(['',f"Selected: {best['name']} ({best['selection_status']})",
                     f"Parameters: {json.dumps(best['parameters'],sort_keys=True)}"])
        if best['name']=='baseline':source=root/'baseline-replay.json'
        else:source=root/'confirm-best/replay.json'
        if source.exists():
            (root/'selected-replay.json').write_text(source.read_text())
            rows.extend(['',f'Baseline replay: {root / "baseline-replay.json"}',
                         f'Selected replay: {root / "selected-replay.json"}'])
    if report.get('unconfirmed_candidate'):
        rows.extend(['','An unconfirmed candidate is retained in batch.json; it is not selected for deployment.'])
    (root/'comparison.md').write_text('\n'.join(rows)+'\n')

from pathlib import Path
import tarfile, json, hashlib, datetime, subprocess

repo=Path('/home/qunwang/wajueji')
work=repo/'work/roadshow-three-way-20261014-v1'
dest=repo/'runs/roadshow-handover-20261012'
assert not (dest/'stage-03').exists(), 'Existing Stage 03 must not be overwritten'
with tarfile.open(work/'material-package.tar.gz','r:gz') as archive:
    for member in archive.getmembers():
        target=(dest/member.name).resolve()
        assert target.is_relative_to((dest/'stage-03').resolve()) and member.isfile(),member.name
    archive.extractall(dest,filter='data')
root=dest/'stage-03'
manifest=json.loads((root/'manifest.json').read_text())
for relative,expected in manifest.items():
    path=root/relative
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256'],relative
    assert path.stat().st_size==expected['bytes']
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not gpu.stdout.strip(),gpu.stdout
receipt={'status':'verified','stage':str(root),'manifest_files_verified':len(manifest),
         'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_compute_processes':[],
         'training_updates':0,'archive_sha256':hashlib.sha256((work/'material-package.tar.gz').read_bytes()).hexdigest(),
         'artifact_scope':'Three-way data, charts and evidence package. Geometry fill is invalid, not zero.'}
(work/'desktop-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt))

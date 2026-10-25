"""Build a new immutable archive; preserve every frozen v3 producer byte."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, repository, output):
    subprocess.run(['sha256sum','--quiet','-c','MANIFEST.sha256'], cwd=source, check=True)
    name='a30-layered-47-optimized-motion-v1'
    bundle, archive=output/name, output/(name+'.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    bundle.mkdir(parents=True, exist_ok=False)
    for folder in ('repo','excavator-resources','mujoco','truck'):
        shutil.copytree(source/folder,bundle/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for filename in ('build_sparse_inertia_candidate.py','layered_47_optimized_bridge.py','run_layered_47_optimized_motion.py'):
        shutil.copy2(repository/'src/v4'/filename,bundle/'repo/src/v4'/filename)
    for path in (repository/'scripts/a30_optimized_motion').iterdir():
        if path.is_file():
            shutil.copy2(path,bundle/path.name)
    shutil.copy2(source/'MANIFEST.sha256',bundle/'BASE_MOTION_MANIFEST.sha256')
    (bundle/'outputs').mkdir()
    files=sorted(p for p in bundle.rglob('*') if p.is_file())
    (bundle/'MANIFEST.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(bundle)}\n' for p in files))
    with tarfile.open(archive,'w:gz') as stream:
        stream.add(bundle,arcname=name)
    archive.with_name(archive.name+'.sha256').write_text(f'{digest(archive)}  {archive.name}\n')
    print(archive)
    print(digest(archive))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source-bundle','repository-root','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    build(a.source_bundle,a.repository_root,a.output_dir)

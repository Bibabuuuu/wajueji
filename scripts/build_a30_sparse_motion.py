"""Freeze a motion-only bundle while retaining the exact v3 seed producer code."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, repo, output):
    subprocess.run(['sha256sum', '--quiet', '-c', 'MANIFEST.sha256'], cwd=source, check=True)
    name = 'a30-layered-47-sparse-motion-v1'
    bundle = output / name
    archive = output / (name + '.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    bundle.mkdir(parents=True, exist_ok=False)
    for folder in ('repo', 'excavator-resources', 'mujoco', 'truck'):
        shutil.copytree(source / folder, bundle / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for src in (repo / 'scripts/a30_sparse_motion').iterdir():
        if src.is_file():
            shutil.copy2(src, bundle / src.name)
    shutil.copy2(repo / 'src/v4/run_layered_47_sparse_motion.py', bundle / 'repo/src/v4')
    shutil.copy2(source / 'MANIFEST.sha256', bundle / 'SEED_V3_MANIFEST.sha256')
    (bundle / 'outputs').mkdir()
    files = sorted(p for p in bundle.rglob('*') if p.is_file())
    (bundle / 'MANIFEST.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(bundle)}\n' for p in files))
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(bundle, arcname=name)
    (output / (archive.name + '.sha256')).write_text(f'{digest(archive)}  {archive.name}\n')
    print(archive)
    print(digest(archive))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--repository-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    build(args.source_bundle, args.repository_root, args.output_dir)

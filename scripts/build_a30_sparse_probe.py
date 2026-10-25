"""Build a new immutable offline bundle from the frozen portable approach assets."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, repo, output):
    name = 'a30-layered-sparse-frequency-v3'
    bundle = output / name
    bundle.mkdir(parents=True, exist_ok=False)
    for folder in ('repo', 'excavator-resources', 'mujoco', 'truck'):
        shutil.copytree(source / folder, bundle / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(source / 'preflight_checkpoint_bridge.py', bundle)
    for src in (repo / 'scripts/a30_sparse_probe').iterdir():
        if src.is_file():
            shutil.copy2(src, bundle / src.name)
    for filename in ('run_layered_sparse_a30_probe.py', 'run_layered_47_excavation_preflight.py',
                     'run_layered_47_interactive_proxy_benchmark.py', 'layered_47_interactive_proxy_checkpoint.py',
                     'layered_frequency_bridge.py', 'run_layered_sparse_resume_audit.py'):
        shutil.copy2(repo / 'src/v4' / filename, bundle / 'repo/src/v4' / filename)
    (bundle / 'outputs').mkdir(exist_ok=True)
    files = sorted(p for p in bundle.rglob('*') if p.is_file())
    (bundle / 'MANIFEST.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(bundle)}\n' for p in files))
    archive = output / (name + '.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
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

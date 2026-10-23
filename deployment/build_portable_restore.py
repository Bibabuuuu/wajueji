"""Assemble the frozen project plus its non-environment runtime resources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repository', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    repo = args.repository.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dest = output / 'repo'
    dest.mkdir()
    archive = repo / 'reports/final-archive-20261023'
    for name in ['wajueji-source-final-20261023.tar.gz',
                 'wajueji-model-and-inputs-20261023.tar.gz',
                 'run-summaries-before-cleanup.tar.gz']:
        with tarfile.open(archive / name, 'r:gz') as tar:
            tar.extractall(dest, filter='data')
    frozen = Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
    kin = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
    resources = output / 'resources'
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    for folder in ['repo', 'excavator-resources', 'mujoco', 'truck']:
        shutil.copytree(frozen / folder, resources / 'frozen-runtime' / folder, ignore=ignore)
    backends = resources / 'backends'
    kdest = backends / kin.name
    kdest.mkdir(parents=True)
    for path in kin.glob('*.py'):
        shutil.copy2(path, kdest / path.name)
    shutil.copytree(kin / 'scene/input', kdest / 'scene/input', ignore=ignore)
    for name in ['autonomous-multiscoop-speed-opt-2026-09-14-v1',
                 'compact-bank-2026-09-11', 'manual-comparison-20260908-v1',
                 'soil-reaction-2026-09-12']:
        source = kin.parent / name
        target = backends / name
        target.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.suffix in ('.py', '.obj', '.mtl'):
                shutil.copy2(path, target / path.name)
        for folder in ['terrain', 'planning', 'perception']:
            if (source / folder).is_dir():
                shutil.copytree(source / folder, target / folder, ignore=ignore)
        if (source / 'operator-start.json').exists():
            shutil.copy2(source / 'operator-start.json', target / 'operator-start.json')
    shutil.copy2(Path(__file__).with_name('README.zh-CN.md'), output / 'README.md')
    extra = [
        'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz.json',
        'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/report.json',
        'work/bucket-projection-isolation-20260925-v1/manual-prefix-1800/display/states.json',
        'work/bucket-projection-isolation-20260925-v1/manual-prefix-continuation-1620/display/states.json',
    ]
    for folder, names in [
        ('runs/edge-adjacent-59um-20261015-v1', ['final.npz']),
        ('runs/loaded-closedloop-20261015-v5', ['active120.npz', 'inactive.npz', 'final.npz']),
        ('runs/continuous-single-scoop-20261014-v1', ['final.npz']),
    ]:
        for name in names:
            path = repo / folder / name
            if path.exists():
                extra.append(str(path.relative_to(repo)))
                for suffix in ['.json', '.control.json']:
                    if Path(str(path) + suffix).exists(): extra.append(str(path.relative_to(repo)) + suffix)
    for rel in extra:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, target)
    shutil.copy2(Path(__file__).with_name('restore.py'), output / 'restore.py')
    manifest = dict(schema='wajueji_portable_restore_v1', source_commit='de813c431a42e4977c9238327decec3ee842a168',
        frozen_policy='repo/work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip',
        original_resource_roots=[str(repo), str(frozen), str(kin)] +
            [str(kin.parent / p.name) for p in backends.iterdir() if p.name != kin.name],
        scope='Frozen runtime, policy and critical state recovery; no environment or historic particle replays.', files=[])
    for path in sorted(output.rglob('*')):
        if path.is_symlink(): raise ValueError('symlink_not_allowed:' + str(path))
        if path.is_file():
            manifest['files'].append(dict(path=str(path.relative_to(output)), bytes=path.stat().st_size, sha256=sha(path)))
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), files=len(manifest['files']),
                         MiB=round(sum(x['bytes'] for x in manifest['files']) / 2**20, 2))), flush=True)


if __name__ == '__main__':
    main()

"""Delete only inventoried obsolete replay arrays, after verified cloud upload."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

R = Path('/home/qunwang/wajueji').resolve()
O = R / 'reports/final-archive-20261023'


def main():
    proof = json.loads((O / 'upload-proof.json').read_text())
    assert proof['repo'] == 'Bibabuuuu/wajueji' and proof['private'] is True
    names = {x['name'] for x in proof['assets']}
    assert {'wajueji-final-materials-20261023.zip', 'wajueji-source-final-20261023.tar.gz',
            'run-summaries-before-cleanup.tar.gz', 'wajueji-model-and-inputs-20261023.tar.gz'} <= names
    freeze = json.loads((O / 'freeze-result.json').read_text())
    assert proof['source_commit'] == freeze['commit']
    remote = subprocess.check_output(['git', 'ls-remote', '--heads', 'origin', 'archive/final-20261023'], cwd=R, text=True)
    assert remote.split()[0] == freeze['commit'], 'Remote source snapshot is not verified'
    plan = json.loads((O / 'cleanup-plan.json').read_text())
    keep = set(plan['keep_run_roots'])
    # Stop if another project Python task has started since inventory.
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cwd = os.readlink(proc / 'cwd')
            cmd = (proc / 'cmdline').read_bytes().split(b'\0')
            if cwd.startswith(str(R)) and any(arg.endswith(b'.py') for arg in cmd):
                active.append(proc.name)
        except OSError:
            pass
    assert not active, 'Project processes active: ' + ','.join(active)
    target_names = {str(R / row['path']) for row in plan['files']}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            for line in (proc / 'maps').read_text().splitlines():
                if any(line.endswith(' ' + name) for name in target_names):
                    raise RuntimeError('Replay cache is memory-mapped by process ' + proc.name)
            for fd in (proc / 'fd').iterdir():
                if os.readlink(fd) in target_names:
                    raise RuntimeError('Replay cache is open in process ' + proc.name)
        except (OSError, PermissionError):
            pass
    checked = []
    for row in plan['files']:
        p = R / row['path']
        assert p.resolve().is_relative_to(R / 'runs')
        assert p.parent.name == 'display' and p.name == 'particles.npy'
        assert p.relative_to(R).parts[1] not in keep
        assert not p.is_symlink()
        st = p.stat()
        assert (st.st_size, st.st_dev, st.st_ino, st.st_mtime_ns) == (row['bytes'], row['device'], row['inode'], row['mtime_ns']), str(p)
        with p.open('rb') as f:
            assert hashlib.sha256(f.read(256)).hexdigest() == row['header_sha256']
        checked.append((p, row))
    # Retained final replay caches must exist before and after cleanup.
    retained = [p for root in keep for p in (R / 'runs' / root).rglob('particles.npy')]
    retained_stat = {str(p): (p.stat().st_ino, p.stat().st_size) for p in retained}
    inputs = json.loads((O / 'model-and-inputs-manifest.json').read_text())['files']
    def verify_inputs():
        for row in inputs:
            p = R / row['path']
            assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256'], row['path']
    verify_inputs()
    before = shutil.disk_usage(R)
    journal = O / 'cleanup-deletions.jsonl'
    assert not journal.exists(), 'Cleanup already started; reconcile journal before rerunning'
    with journal.open('x') as log:
        for p, row in checked:
            p.unlink()
            log.write(json.dumps({'path': row['path'], 'bytes': row['bytes'], 'deleted': True}) + '\n')
            log.flush()
    os.sync()
    for name, identity in retained_stat.items():
        p = Path(name)
        assert p.exists() and (p.stat().st_ino, p.stat().st_size) == identity, name
    verify_inputs()
    after = shutil.disk_usage(R)
    result = {'deleted_files': len(checked), 'deleted_logical_GiB': round(sum(row['bytes'] for _,row in checked)/2**30, 2),
              'reclaimed_GiB': round((after.free-before.free)/2**30, 2), 'free_before_GiB': round(before.free/2**30,2),
              'free_after_GiB': round(after.free/2**30,2), 'used_percent_after': round(after.used/after.total*100,1),
              'retained_final_cache_files_verified': len(retained_stat), 'source_commit': freeze['commit'],
              'retained_model_and_input_files_sha256_verified': len(inputs),
              'preserved': ['All sources and environments', 'All model weights', 'All input and endpoint NPZ files',
                            'All states, command traces, reports and failed-run summaries', 'All declared final replay roots'],
              'scope': 'Only the 50 explicitly inventoried superseded runs/**/display/particles.npy files.'}
    (O / 'cleanup-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

"""Run the frozen audit with absent optional reset metadata normalized to None.

The initial audit raised KeyError after command replay because first-scoop
checkpoints omit training_reset. Preserve its hash (captured incidentally by
the runtime's *.py binding), the original exception log and source sidecars.
This wrapper only normalizes the in-memory read of checkpoint metadata.
"""
import audit_resume

original_read = audit_resume.read

def read_optional_reset(path):
    value = original_read(path)
    if str(path).endswith('.npz.json'):
        value.setdefault('training_reset', None)
    return value

if __name__ == '__main__':
    audit_resume.read = read_optional_reset
    audit_resume.main()

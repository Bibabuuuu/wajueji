"""Wait for the already-running contact segment, then continue once, no audit pause."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
source_report=ROOT/'runs/contact-next-scoop-20260925-v1/candidate/report.json'
log=ROOT/'runs/contact-second-finish-unload-20260925-v3-physical.log'
def main():
    deadline=time.monotonic()+1000
    while True:
        if time.monotonic()>deadline:raise TimeoutError('contact_result_wait_budget')
        try:
            report=json.loads(source_report.read_text())
            break
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(5)
    if report['status']!='bounded_probe_completed':raise RuntimeError('contact_stopped:'+str(report.get('failure',report['status'])))
    subprocess.run([sys.executable,str(W/'prepare_contract.py')],cwd=ROOT,check=True)
    with log.open('x') as output:
        result=subprocess.run([sys.executable,str(W/'run.py')],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
    print('SECOND_TAIL_EXIT',result.returncode,'RESULT',ROOT/'runs/contact-second-finish-unload-20260925-v3/physical/report.json',flush=True)
    raise SystemExit(result.returncode)
if __name__=='__main__':main()

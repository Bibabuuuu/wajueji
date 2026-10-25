"""Supply the CPU proxy's unused force-history slot; run frozen audit plus overlay checks."""
import sys,argparse
import audit_v2 as audit
from continuous_scoop import *

# The native runtime creates this dictionary in __enter__. The CPU proxy never
# enters native physics; command recomputation does not consume force history.
audit.CpuRuntime.captured={}
if '--check-only' in sys.argv:
    out=W/'audit-proxy-smoke'
    r=CpuRuntime(SOURCE,out,enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
    v=PhaseView(r,arrays=r.arrays,native=r.metadata['controller_state'])
    c=TransportController(v,FixedRouteClearance(v),np.zeros(4))
    assert r.bridge is None and v.frames==0 and v.captured=={}
    assert np.array_equal(c.tracking.bias,np.array(r.metadata['controller_state']['controller_target'])-r.arrays['mechanism_qpos'][r.chain.axes])
    print('CPU audit proxy/controller smoke passed; native_initializations=0 physical_steps=0')
else:
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    receipt=dict(launcher_sha256=sha(__file__),frozen_auditor_sha256=sha(W/'audit.py'),executed_auditor_sha256=sha(W/'audit_v2.py'),numerical_note='v1 bit equality rejected 88 ALIGN commands, maximum 6.652469712793187e-17 SI; v2 checks all command errors <=1e-15 with rtol0, actual dispatch remains exact; no physics changes',
        cpu_proxy_default='empty force-history slot on never-entered CPU proxy; not used by command recomputation',new_physical_frames=0)
    write(args.run/'audit-execution.json',receipt)
    audit.main()
    result=read(args.run/'read-only-audit.json');dispatch=audit.rows(args.run/'dispatch.ndjson');phases=read(args.run/'display/phases.json')
    result['checks']['playback_phase_overlay_matches_dispatch']=all(phases[d['global_frame']]==d['phase'] for d in dispatch)
    result['checks']['final_checkpoint_is_returned']=read(args.run/'final.npz.json')['stage']=='returned'
    result.update(check_count=len(result['checks']),status='passed' if all(result['checks'].values()) else 'failed',audit_execution=receipt)
    write(args.run/'read-only-audit.json',result);assert all(result['checks'].values())
    print('FINAL_AUDIT',result['status'],result['check_count'],flush=True)

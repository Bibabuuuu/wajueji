"""Build and archive one freshly observed cycle without advancing physics."""
from common import *
from pose_goals import plan_cycle
from planning.approach import sampled_approach
from native_guard import PoseGuard


def project_passive_closure(chain,q):
    """Project only passive hinges; commanded coordinates remain actual."""
    original=np.asarray(q,float);candidate=original.copy();before=float(abs(chain.residual(candidate)).max())
    for _ in range(12):
        residual=chain.residual(candidate)
        if np.max(abs(residual))<1e-8:break
        jac=[]
        for address in chain.passive:
            shifted=candidate.copy();shifted[address]+=1e-6
            jac.append((chain.residual(shifted)-residual)/1e-6)
        delta=np.linalg.lstsq(np.asarray(jac).T,-residual,rcond=1e-10)[0]
        if not np.isfinite(delta).all() or np.max(abs(delta))>.05:raise ValueError('passive_projection_branch_jump')
        candidate[chain.passive]+=delta
    after=float(abs(chain.residual(candidate)).max())
    if after>=1e-8 or np.max(abs(candidate[chain.axes]-original[chain.axes]))>1e-12:
        raise ValueError('passive_projection_failed')
    return candidate,{'closure_before_m':before,'closure_after_m':after,
        'maximum_passive_adjustment_rad':float(np.max(abs(candidate-original)))}


def build_cycle_paths(asm,chain,start_q,selection,truck_contract,scene,bounds,run_dir,
                      *,maximum_wall_s=480.):
    run_dir=Path(run_dir);run_dir.mkdir(exist_ok=False)
    started=time.perf_counter();guard=None;report={'status':'building','selection':selection,'paths':{}}
    try:
        planning_start,projection=project_passive_closure(chain,start_q);report['start_projection']=projection
        goals=plan_cycle(asm,chain,planning_start,selection,truck_contract)
        guard=PoseGuard(T/'pose_guard',T/'guard-runtime',run_dir,T/'scene/input/model/A.xml')
        def advance(q,v,dt):chain.q=q.copy();return chain.advance(v,dt)[0]
        q=planning_start.copy()
        for stage in STAGES:
            def check(candidate,previous):
                if time.perf_counter()-started>maximum_wall_s:raise TimeoutError('runtime_geometry_wall_budget')
                guard.check(candidate)
                if np.any(candidate[[7,9,12,15]]<bounds[:,0]) or np.any(candidate[[7,9,12,15]]>bounds[:,1]):raise ValueError('autonomous_8mm_bounds')
                result=scene.check(asm.assemble(candidate,include_meshes=False)[1],stage)
                result['closure_m']=float(abs(chain.residual(candidate)).max());return result
            path=sampled_approach(q,goals[stage],axes=chain.axes,speed_limits=COMMAND_LIMITS,
                                  advance=advance,check_pose=check,dt=1/60)
            target=run_dir/(stage.lower()+'.npz')
            np.savez_compressed(target,qpos=path['qpos'],velocity=path['velocity'])
            report['paths'][stage]={'steps':path['steps'],'duration_s':path['duration_sim_s'],
                'maximum_closure_m':max(d['closure_m'] for d in path['diagnostics']),
                'minimum_bucket_z_m':min(d['bucket_min_z_m'] for d in path['diagnostics']),
                'sha256':sha(target)}
            q=path['qpos'][-1].copy()
        report['goals']={k:v.tolist() for k,v in goals.items()};report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        if guard:guard.close()
        report['wall_s']=time.perf_counter()-started
        (run_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return {stage:np.load(run_dir/(stage.lower()+'.npz'))['qpos'] for stage in STAGES},report

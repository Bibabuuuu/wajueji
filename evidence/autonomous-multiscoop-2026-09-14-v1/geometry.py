"""Observe initial soil, select a cut, solve and audit five autonomous stages."""
from common import *
import argparse
from pose_goals import solve_pose,TIP,MOUTH,plan_cycle
from planning.approach import sampled_approach
from native_guard import PoseGuard

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();a.run.mkdir(exist_ok=False)
    report={'status':'building','physics_steps':0,'sources_before':sources(),'paths':{}};guard=None;start=time.perf_counter()
    try:
        import warp as wp
        wp.init();wp.set_device('cuda:0')
        asm,chain,initial,geom,poses,scene,bounds,parts=setup()
        particles=np.load(B/'manual-20260911-090834/initial-bank.npz')['q']
        sensor=Sensor(geom,parts);terrain,arrays,meta=sensor.capture(particles,poses)
        selection=select_cut(terrain);report.update(selection=selection,sensor=meta,initial_particle_sha256=sha(B/'manual-20260911-090834/initial-bank.npz'),target_bounds=bounds.tolist())
        np.savez_compressed(a.run/'observation.npz',**arrays)
        goals=plan_cycle(asm,chain,initial,selection,read(B/'inventory-contract.json')['truck'])
        guard=PoseGuard(T/'pose_guard',T/'guard-runtime',a.run,T/'scene/input/model/A.xml')
        def advance(q,v,dt):chain.q=q.copy();return chain.advance(v,dt)[0]
        q=initial.copy()
        for stage in STAGES:
            def check(q,previous):
                if time.perf_counter()-start>480:raise TimeoutError('geometry_wall_budget')
                guard.check(q)
                if np.any(q[[7,9,12,15]]<bounds[:,0]) or np.any(q[[7,9,12,15]]>bounds[:,1]):raise ValueError('autonomous_8mm_bounds')
                result=scene.check(asm.assemble(q,include_meshes=False)[1],stage)
                result['closure_m']=float(abs(chain.residual(q)).max());return result
            path=sampled_approach(q,goals[stage],axes=chain.axes,speed_limits=COMMAND_LIMITS,advance=advance,check_pose=check,dt=1/60)
            np.savez_compressed(a.run/(stage.lower()+'.npz'),qpos=path['qpos'],velocity=path['velocity'])
            report['paths'][stage]={'steps':path['steps'],'duration_s':path['duration_sim_s'],'maximum_closure_m':max(d['closure_m'] for d in path['diagnostics']),'minimum_bucket_z_m':min(d['bucket_min_z_m'] for d in path['diagnostics'])}
            q=path['qpos'][-1].copy();print(json.dumps({'stage':stage,**report['paths'][stage]}),flush=True)
        report['goals']={k:v.tolist() for k,v in goals.items()};report['path_sha256']={s:sha(a.run/(s.lower()+'.npz')) for s in STAGES};report['status']='passed'
    except Exception as error:report.update(status='failed',error=repr(error))
    finally:
        if guard:guard.close()
        report.update(sources_after=sources(),wall_s=time.perf_counter()-start);report['sources_unchanged']=report['sources_before']==report['sources_after']
        (a.run/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('sources_before','sources_after','goals')},indent=2))
    return 0 if report['status']=='passed' and report['sources_unchanged'] else 1
if __name__=='__main__':raise SystemExit(main())

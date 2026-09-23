from pathlib import Path
import ast,json,hashlib
root=Path('/home/qunwang/wajueji');out=Path(__file__).resolve().parent
s=(root/'work/loaded-control-stop-retention-20260923-v2/replay.py').read_text()
s=s.replace("trace += [{'applied_command_si':[0.,0.,0.,0.]} for _ in range(120)] + [{'applied_command_si':[0.,0.,0.,-.02]} for _ in range(30)] + [{'applied_command_si':[0.,0.,0.,0.]} for _ in range(60)]\nassert len(trace)==1580", "trace=trace[:600] + [{'applied_command_si':[0.,0.,0.,0.]} for _ in range(60)] + [{'applied_command_si':[0.,0.,0.,.02]} for _ in range(360)] + [{'applied_command_si':[0.,0.,0.,0.]} for _ in range(120)] + [{'applied_command_si':[0.,0.,0.,-.02]} for _ in range(30)] + [{'applied_command_si':[0.,0.,0.,0.]} for _ in range(120)]\nassert len(trace)==1290")
a=s.index(' original_render=runtime.render_capacity_bridge');b=s.index(' fixture,held,restored=',a)
wrapper=r'''
 original_render=runtime.render_capacity_bridge
 def render_candidate(*arguments):
  source,manifest=original_render(*arguments)
  def replace_once(old,new):
   nonlocal source
   assert source.count(old)==1,(old,source.count(old))
   source=source.replace(old,new)
  if mode=='candidate':source='#include '+json.dumps(str(parent/'bucket_integral.h'))+'\n'+source
  decl='    std::array<double, 4> unloaded_feedforward{};'
  extra='\n    std::array<double,4> bucket_integral_effort{}, rock_equivalent_drive_load{}, total_drive_peak{}, reduced_mass_diagonal{};\n    std::vector<double> rock_bucket_contact_count(rock_count,0), rock_bucket_max_normal_force(rock_count,0);'
  if mode=='candidate':extra+='\n    BucketIntegral bucket_integrator;'
  replace_once(decl,decl+extra)
  emit='      array("unloaded_feedforward", unloaded_feedforward.data(), 4);'
  replace_once(emit,emit+'\n      array("bucket_integral_effort", bucket_integral_effort.data(), 4);\n      array("rock_equivalent_drive_load", rock_equivalent_drive_load.data(), 4);\n      array("total_drive_peak", total_drive_peak.data(), 4);\n      array("reduced_mass_diagonal", reduced_mass_diagonal.data(), 4);\n      array("rock_bucket_contact_count", rock_bucket_contact_count.data(), rock_count);\n      array("rock_bucket_max_normal_force", rock_bucket_max_normal_force.data(), rock_count);')
  reset='      bucket_rock_impulse.fill(0);'
  replace_once(reset,reset+'\n      std::fill(rock_bucket_contact_count.begin(),rock_bucket_contact_count.end(),0);\n      std::fill(rock_bucket_max_normal_force.begin(),rock_bucket_max_normal_force.end(),0);')
  before='        const double before = data->time;\n        mj_step(model.get(), data.get());'
  added=''
  if mode=='candidate':
   added='''+'"""'+r'''
        if(data->time >= 11.0-1e-9){
          const double error=controller.target[3]-data->qpos[controller.qpos_address[3]];
          bucket_integral_effort[3]=bucket_integrator.Step(command[3],error,model->opt.timestep);
          const int actuator=controller.hold_actuator[3];
          data->ctrl[actuator]+=bucket_integral_effort[3]/model->actuator_gainprm[mjNGAIN*actuator];
        }
'''+ '"""'+r'''
  replace_once(before,added+before)
  marker='        last_bucket_rock_force = substep_bucket_force;'
  observation='''+'"""'+r'''
        std::vector<mjtNum> rock_generalized(model->nv,0);
        mj_applyFT(model.get(),data.get(),substep_bucket_force.data(),substep_bucket_torque.data(),
                   bucket_interval_reference_point.data(),bucket_body,rock_generalized.data());
        for(int axis=0;axis<4;++axis){
          rock_equivalent_drive_load[axis]=0;
          for(int dof=0;dof<model->nv;++dof)
            rock_equivalent_drive_load[axis]+=result.tangent[dof*4+axis]*rock_generalized[dof];
          const double effort=data->actuator_force[controller.hold_actuator[axis]]+data->actuator_force[controller.drive_actuator[axis]];
          total_drive_peak[axis]=std::max(total_drive_peak[axis],std::abs(effort));
          reduced_mass_diagonal[axis]=result.mass[axis*4+axis];
        }
'''+ '"""'+r'''
  replace_once(marker,observation+marker)
  marker='            mj_contactForce(model.get(), data.get(), contact_index, local);'
  replace_once(marker,marker+'\n            const int index=rock_is_first?first_rock:second_rock;\n            rock_bucket_contact_count[index]+=1;\n            rock_bucket_max_normal_force[index]=std::max(rock_bucket_max_normal_force[index],std::abs(local[0]));')
  (out/'instrumented-bridge.cc').write_text(source)
  audit={'integral_candidate':mode=='candidate','default_enabled':False,'native_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'observation_only_fields':['rock_equivalent_drive_load','total_drive_peak','reduced_mass_diagonal','rock_bucket_contact_count','rock_bucket_max_normal_force']}
  (out/'bridge-policy.json').write_text(json.dumps(audit,indent=2))
  return source,{**manifest,'isolated_integral':audit}
 runtime.render_capacity_bridge=render_candidate
'''
s=s[:a]+wrapper+s[b:]
s=s.replace("root/'src/v4/soil_surface.py']","root/'src/v4/soil_surface.py',parent/'bucket_integral.h',parent/'protocol.json']")
s=s.replace('if i>1370:','if i>600:')
s=s.replace("assert np.array_equal(bridge.command,requested),projector.reason","if not np.array_equal(bridge.command,requested):\n      assert projector.reason=='bucket_internal_upper_target' and requested[3]>0,projector.reason")
s=s.replace("'external_load_compensation':bridge.state.get('external_load_compensation',[0.]*4)","'external_load_compensation':[0.]*4,'bucket_integral_effort':bridge.state['bucket_integral_effort'],'rock_equivalent_drive_load':bridge.state['rock_equivalent_drive_load'],'native_total_drive_peak':bridge.state['total_drive_peak'],'reduced_mass_diagonal':bridge.state['reduced_mass_diagonal'],'rock_bucket_contact_count':bridge.state['rock_bucket_contact_count'],'rock_bucket_max_normal_force':bridge.state['rock_bucket_max_normal_force'],'requested_command':requested.tolist(),'projection_reason':projector.reason if i>600 else None")
s=s.replace("('replay' if i<=1370 else 'hold' if i<=1490 else 'reverse' if i<=1520 else 'stop')","('replay' if i<=600 else 'matched_hold' if i<=660 else 'curl' if i<=1020 else 'hold' if i<=1140 else 'reverse' if i<=1170 else 'stop')")
s=s.replace('if i>=1010:positions.append(current.copy());poses.append(pose.copy())','if i>=660:positions.append(current.copy());poses.append(pose.copy())')
s=s.replace('if i in (480,600,1370,1490,1520,1580):','if i in (480,600,660,1020,1140,1170,1290):')
s=s.replace("if i<=480: assert bridge.state.get('external_load_compensation',[0.]*4)==[0.0]*4","if i<=660: assert bridge.state['bucket_integral_effort']==[0.0]*4")
s=s.replace("assert abs(bridge.state.get('external_load_compensation',[0.]*4)[3])<=60000.0001","assert abs(bridge.state['bucket_integral_effort'][3])<=120000.0001")
s=s.replace('final-001580.diagnostic.npz','final-001290.diagnostic.npz').replace('first_frame=1010','first_frame=660')
ast.parse(s);(out/'replay.py').write_text(s)
protocol={'status':'frozen_before_physics','kind':'bucket_conditional_integral_isolated_candidate','initial':'D115 saved complete rear-lower handoff, no APPROACH rerun','matched_prefix':'recorded600frames plus60zero-commandframes; integral exactly0 through660; both branches must match23 full numeric arrays at600 and660','integral':{'ki_N_per_m_s':1500000,'cap_N':120000,'slew_N_per_s':30000,'error_deadband_m':.0005,'nominal_kp_over_ki_s':2,'activation_s':11,'neutral':'no growth; supporting effort kept; opposing error may unwind','reverse':'no growth in old direction; opposing error unwinds','antiwindup':'bounded accumulator itself; no hidden sum','real_hydraulic_limit':False,'reason_for_cap':'initial ~35mm P error corresponds~105kN;120kN gives a finite diagnostic envelope, not a physical force cap'},'phases':[{'name':'curl','frames':[661,1020],'command':[0,0,0,.02]},{'name':'hold','frames':[1021,1140],'command':[0,0,0,0]},{'name':'reverse','frames':[1141,1170],'command':[0,0,0,-.02]},{'name':'stop','frames':[1171,1290],'command':[0,0,0,0]}],'guards':'all original manual hard guards; first failure save and stop; original upper-target projection preserved','acceptance':{'curl_end_last30_mean_error_m':.005,'curl_last30_mean_error_relative_baseline_max':.5,'actual_bucket_soft_upper_m':.852,'stop':'both neutral phases last15 frames original motion.stopped','reverse':'negative net actual bucket displacement and <=0.2s latency to velocity below -1mm/s','qualification':'all criteria together; otherwise report partial improvement or reject, never relax after run'},'scope':'no lift/truck or payload-success qualification; diagnostics not training; per-rock contact decomposition observational only','runtime':'serial local RTX4070 cuda:0,1290frames each, about4-6min per process','runner_sha256':hashlib.sha256(s.encode()).hexdigest(),'header_sha256':hashlib.sha256((out/'bucket_integral.h').read_bytes()).hexdigest()}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False));print(out)

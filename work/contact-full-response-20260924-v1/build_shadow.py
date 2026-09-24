"""Create a diagnostic-only persistent native copy; real bridge stays unchanged."""
import json,hashlib,subprocess
from pathlib import Path
from predict_response import W,OUT,BASE,OLD

def main():
    source=BASE/'physics/cycle/capacity-bridge.cc';s=source.read_text()
    def replace(a,b):
        nonlocal s
        if s.count(a)!=1:raise ValueError('shadow_anchor:'+a)
        s=s.replace(a,b)
    replace('        if (ready) throw std::runtime_error("init once only");','''        // Shadow process only: discard the entire preceding prediction.
        mj_resetData(model.get(),data.get());
        ready = false;
        closure = 0;
        rock_ground_contact_substeps=rock_ramp_contact_substeps=rock_bucket_contact_substeps=0;
        rock_rock_contact_substeps=rock_truck_contact_substeps=forbidden_contact_count=0;
        bucket_rock_impulse.fill(0); bucket_rock_angular_impulse.fill(0);
        bucket_rock_contact_torque_impulse.fill(0);bucket_rock_moment_arm_impulse.fill(0);
        last_bucket_rock_force.fill(0);last_bucket_rock_torque.fill(0);
        equivalent_drive_load.fill(0);
''')
    replace('    bool ready = false;', '    std::array<double,3> prediction_origin{};\n    bool ready = false;')
    replace('        ready = true;', '''        std::copy(data->xpos+3*bucket_body,data->xpos+3*bucket_body+3,prediction_origin.begin());
        ready = true;''')
    # Exact endpoint velocity about the fixed initial wrench reference.
    anchor='      array("cycle_native_warmstart", data->qacc_warmstart, model->nv);'
    replace(anchor,'''      std::vector<mjtNum> jp(3*model->nv),jr(3*model->nv);
      mj_jac(model.get(),data.get(),jp.data(),jr.data(),prediction_origin.data(),bucket_body);
      std::array<double,6> exact_twist{};
      for(int r=0;r<6;++r)for(int k=0;k<model->nv;++k)
        exact_twist[r]+=(r<3?jp[r*model->nv+k]:jr[(r-3)*model->nv+k])*data->qvel[k];
      array("prediction_endpoint_twist",exact_twist.data(),6);
'''+anchor)
    replace('    std::string line;','    std::cout << std::unitbuf;\n    std::string line;')
    target=OUT/'shadow-bridge.cc';target.write_text(s)
    cmd=json.loads((OLD/'compile-review.json').read_text())['command'];cmd[cmd.index(str(OLD.relative_to(W.parents[1])/'endpoint_velocity.cc'))]=str(target);cmd[-1]=str(OUT/'shadow-bridge')
    p=subprocess.run(cmd,cwd=W.parents[1],capture_output=True,text=True)
    (W/'shadow-build-review.json').write_text(json.dumps(dict(command=cmd,exit_code=p.returncode,stderr=p.stderr,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2)+'\n');p.check_returncode()
if __name__=='__main__':main()

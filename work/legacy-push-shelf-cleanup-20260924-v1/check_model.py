"""Read-only frozen failure contact regression; only a new generated XML is edited."""
from pathlib import Path
import json,shutil,subprocess,xml.etree.ElementTree as ET
import numpy as np
from legacy_push_shelf_cleanup import clean_generated_model,sha256,NAME
w=Path(__file__).resolve().parent;root=w.parents[1]
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
original=root/'work/active-curl-assist-20260924-v1/recorded-baseline/runtime-build/audit.xml'
snapshot=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-235749-1040071/failure-001498.diagnostic.npz'
source_sha=sha256(original);snapshot_sha=sha256(snapshot)
new=w/'clean-generated-model.xml';shutil.copyfile(original,new);policy=clean_generated_model(new)
a=ET.parse(original).getroot();world=a.find('worldbody');world.remove(world.find(f"geom[@name='{NAME}']"));b=ET.parse(new).getroot()
def signature(node):return node.tag,node.attrib,(node.text or '').strip(),[signature(c) for c in node]
assert signature(a)==signature(b)
assert policy['removed_count']==1
with np.load(snapshot) as arrays:
 state=np.r_[arrays['mechanism_qpos'],arrays['rigid_qpos'].reshape(-1),arrays['mechanism_qvel'],arrays['rigid_qvel'].reshape(-1)]
np.savetxt(w/'failure-state.txt',state,fmt='%.17g')
binary=w/'contact-probe';lib=bundle/'mujoco/lib'
subprocess.run(['g++','-O2','-std=c++17','-I'+str(bundle/'mujoco/include'),str(w/'contact_probe.cc'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-l:libmujoco.so.2.3.8','-o',str(binary)],check=True)
reports=[json.loads(subprocess.check_output([str(binary),str(model),str(w/'failure-state.txt')],text=True)) for model in [original,new]]
old_pairs=reports[0]['pairs'];new_pairs=reports[1]['pairs'];removed_pairs={k:v for k,v in old_pairs.items() if NAME in k}
checks={'removed_exactly_identified_shelf':policy['removed_count']==1,'other_xml_structure_and_attributes_exact':signature(a)==signature(b),'mechanism_and_rock_dimensions_unchanged':all(reports[0][k]==reports[1][k] for k in ['nq','nv','nbody']),'exactly_one_geom_removed':reports[0]['ngeom']-reports[1]['ngeom']==1,'original_failure_contact_reproduced':bool(removed_pairs),'shelf_absent_in_compiled_model':reports[1]['shelf_geom_id']==-1,'all_other_contact_pairs_unchanged':{k:v for k,v in old_pairs.items() if NAME not in k}==new_pairs,'frozen_source_model_unchanged':sha256(original)==source_sha,'failure_snapshot_unchanged':sha256(snapshot)==snapshot_sha}
result={'checks':checks,'passed':all(checks.values()),'source_model':str(original),'source_model_sha256':source_sha,'failure_snapshot':str(snapshot),'failure_snapshot_sha256':snapshot_sha,'cleanup':policy,'original':reports[0],'clean':reports[1],'removed_contact_pairs':removed_pairs,'physics_steps':0,'scope':'native MuJoCo forward collision check only; no failure snapshot continuation'}
(w/'model-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']

"""Bind the collision regression to the exact offline manual model variant."""
import json, os, subprocess, xml.etree.ElementTree as ET
from pathlib import Path
from legacy_push_shelf_cleanup import clean_generated_model, sha256, NAME
from run_layered_mixed_scene_local_dynamics_smoke import _make_model_resources_portable
w=Path(__file__).resolve().parent; root=w.parents[1]
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
run=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-005017-1076447'
r=json.loads((run/'report.json').read_text())
source=root/'work/active-curl-assist-20260924-v1/recorded-baseline/runtime-build/audit.xml'
original=w/'manual-original-model.xml';clean=w/'manual-clean-model.xml'
os.environ['EXCAVATOR_MUJOCO_MESH_DIR']=str(bundle/'excavator-resources/mesh')
os.environ['EXCAVATOR_MUJOCO_TEXTURE_DIR']=str(bundle/'excavator-resources/texture')
tree=ET.parse(source);resources=_make_model_resources_portable(tree.getroot(),bundle/'repo')
ET.indent(tree,space='  ');tree.write(original,encoding='unicode',xml_declaration=False)
with original.open('a') as f:f.write('\n')
clean.write_bytes(original.read_bytes());policy=clean_generated_model(clean)
assert policy==r['legacy_push_shelf_cleanup'],(policy,r['legacy_push_shelf_cleanup'])
reports=[json.loads(subprocess.check_output([str(w/'contact-probe'),str(p),str(w/'failure-state.txt')],text=True)) for p in [original,clean]]
checks={'manual_pre_cleanup_sha_exact':sha256(original)==policy['generated_model_before_sha256'],'manual_post_cleanup_sha_exact':sha256(clean)==r['restore']['runtime_build']['model']['model_sha256'],'original_failure_reproduced':any(NAME in k for k in reports[0]['pairs']),'shelf_absent':reports[1]['shelf_geom_id']==-1,'other_contact_pairs_exact':{k:v for k,v in reports[0]['pairs'].items() if NAME not in k}==reports[1]['pairs']}
result={'passed':all(checks.values()),'checks':checks,'cleanup':policy,'resource_rebinding':resources,'original':reports[0],'clean':reports[1],'physics_steps':0,'manual_report':str(run/'report.json'),'manual_report_sha256':sha256(run/'report.json')}
(w/'manual-model-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']

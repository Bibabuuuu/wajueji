"""Prove the read-only receiver uses the actual runtime truck triangles."""
from common import *
import unittest

asm,chain,initial,geom,poses,scene,bounds,parts=setup()
contract=read(A/'receiver-contract.json');t=np.asarray(contract['local_to_world'])
old=read(B/'inventory-contract.json')['truck']
assert np.array_equal(t,np.asarray(old['local_to_world']))
mesh=np.load(A/'receiver-mesh.npz');assert sha(A/'receiver-mesh.npz')==contract['source_mesh_sha256']
tri=np.concatenate([v[f] for _,v,f,_ in parts]);local=(tri-t[:3,3])@t[:3,:3]
saved=mesh['local_vertices'][mesh['faces']]
assert local.shape==saved.shape
assert np.max(abs(local-saved))<1e-10
for name,ids in contract['triangle_ids'].items():assert np.max(abs(local[ids]-contract['triangles'][name]))<1e-10
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(A),pattern='test_*.py'))
record={'passed':result.wasSuccessful(),'tests':result.testsRun,'maximum_truck_triangle_error_m':float(np.max(abs(local-saved))),
        'receiver_contract_sha256':sha(A/'receiver-contract.json'),'sources':sources()}
(A/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='sources'},indent=2))
assert record['passed']

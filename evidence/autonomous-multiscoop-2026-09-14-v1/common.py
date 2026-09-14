"""Frozen project adapters shared by geometry and true-dynamics validation."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent
B=R/'compact-bank-2026-09-11';T=R/'small-soil-20260910-v17-performance';C=A;H=R/'soil-isolation-2026-09-13';M=A/'mechanism-auto8'
sys.path[:0]=[str(A),str(M),str(B),str(T),str(C),str(R/'soil-reaction-2026-09-12')]
from scene_assembly import Assembler
from fast_kinematics import install
from closed_chain import ClosedChain, COMMAND_LIMITS
from truck_layout import truck_parts
from terrain.solid_bank import bank_candidate
from scene_clearance import SceneClearance
from manual_scene import bucket_clearance
from planning.cycle import select_cut
STAGES=('APPROACH','CUT','DEEPEN','CURL','LIFT','TURN','ALIGN','DUMP','CLEAR_TRUCK','RETURN_SWING','RETURN')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def setup():
    asm=Assembler(T/'scene/input/model/A.xml',T/'scene/input/mesh');install(asm)
    initial=np.array(read(B/'operator-start.json')['qpos']);chain=ClosedChain(asm)
    geom,poses=asm.assemble(initial);bank=bank_candidate()
    parts=list(truck_parts(R/'manual-comparison-20260908-v1'))
    tri=np.concatenate([v[f] for _,v,f,_ in parts])
    scene=SceneClearance(geom,tri,[bank.origin,bank.origin+bank.cell*np.array(bank.solid.shape)],bucket_narrow_phase=bucket_clearance)
    bounds=np.array([[-2.992,2.992]]+[chain.limits[i]+[.008,-.008] for i in [9,12,15]])
    return asm,chain,initial,geom,poses,scene,bounds,parts

class Sensor:
    """Two virtual depth views; planner sees reconstructed cells, no particle IDs."""
    def __init__(self,geom,parts):
        from perception.virtual_depth import VirtualDepth
        from perception.core import Camera,Grid,look_at
        self.camera=Camera();self.grid=Grid();self.names=[g['body'] for g in geom]
        meshes=[(g['body_local_vertices'],g['faces']) for g in geom]+[(v,f) for _,v,f,_ in parts]
        self.sensor=VirtualDepth(self.camera,meshes,robot_count=len(geom))
        self.static=[np.eye(4)]*len(parts)
        self.views=[look_at([8,-8,7],[3.5,1,.7]),look_at([7,6,6],[5,0,.5])]
    def capture(self,q,poses,sequence=0):
        from perception.core import noisy_depth,terrain_from_points
        start=time.perf_counter();points=[];arrays={}
        for i,view in enumerate(self.views):
            depth,robot=self.sensor.capture(q,view,[poses[n] for n in self.names]+self.static)
            depth=noisy_depth(depth,self.camera,seed=sequence*10+i)
            mask=np.isfinite(depth)&np.isfinite(robot)&(abs(depth-robot)<.03)
            depth[mask]=np.nan;points.append(self.camera.points(depth,view)[0]);arrays[f'depth_{i}']=depth
        h,n=terrain_from_points(np.concatenate(points),self.grid)
        terrain={'valid':bool(n.any()),'origin_xy_m':[self.grid.xmin,self.grid.ymin],'resolution_m':self.grid.resolution,'height_m':h,'point_count':n}
        arrays.update(height_m=h,point_count=n)
        return terrain,arrays,{'capture_wall_s':start,'processed_wall_s':time.perf_counter(),'source':'two_virtual_depth_views_with_noise_self_filter_and_cell_max','sequence':sequence}

def sources():
    paths=list(A.glob('*.py'))+[A/n for n in ('retention-front-rim.npz','objectivity-source.json','objectivity-test.json','receiver-contract.json','receiver-mesh.npz')]
    paths+=list(M.glob('*.h'))+list(M.glob('*.cc'))+list(M.glob('*.xml'))+list(M.glob('*.py'))+[M/'mpm_bridge']+list((M/'excavator_simulator').glob('*.h'))
    paths += [B/n for n in read(B/'manual-20260911-090834-launch.json')['source_sha256']]
    paths += [C/'planning'/n for n in ('cycle.py','approach.py')]+list((C/'perception').glob('*.py'))
    paths += [T/n for n in ('cohesive-fine01/soil-config.json','swept_contact.py','gpu_diagnostics.py')]
    paths += [Path(read(A/'objectivity-source.json')['installed_source'])]
    return {str(p):sha(p) for p in paths}

"""Read-only lift sensitivity and final-snapshot detachment; no physics steps."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from layered_excavation_approach_planner import _load_kinematics, rock_surface_points, terrain_from_points
from layered_47_cycle_geometry import stationary_terrain, bucket_mask
from planning.rear_lower_prepoint import proxy_surface_samples

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
KIN=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
BUNDLE=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_grid_contributors(runtime, terrain, world):
    """Reconstruct the existing heuristic with provenance, not a collision test."""
    q=runtime.arrays['particle_q']
    pose=runtime.assembly.assemble(runtime.arrays['mechanism_qpos'],include_meshes=False)[1]['bucket']
    cell=.04;origin=q.min(0)-2*cell;ijk=np.floor((q-origin)/cell).astype(int)
    shape=ijk.max(0)+3
    assert np.prod(shape)<=12000000
    grid=np.zeros(tuple(shape),bool);grid[tuple(ijk.T)]=True
    components,_=label(grid);labels=components[tuple(ijk.T)]
    anchors=np.unique(labels[bucket_mask(q,pose)]);anchors=anchors[anchors>0]
    stationary=np.isin(labels,np.unique(labels[q[:,2]<.06]))
    sources={'ground_connected_soil':q[stationary]}
    _,rocks=rock_surface_points(runtime.root,runtime.records,runtime.arrays['rigid_qpos'])
    for name,points in rocks.items():
        indices=np.floor((points-origin)/cell).astype(int)
        inside=((indices>=0)&(indices<shape)).all(1)
        linked=np.zeros(len(points),bool)
        linked[inside]=np.isin(components[tuple(indices[inside].T)],anchors)
        if not (linked.any() or bucket_mask(points,pose).any()):sources[name]=points
    combined=np.concatenate(list(sources.values()))
    rebuilt=terrain_from_points(combined)
    assert np.array_equal(rebuilt['height_m'],terrain['height_m'],equal_nan=True)
    idx=np.floor((world[:,:2]-terrain['origin_xy_m'])/terrain['resolution_m']).astype(int)
    h=np.asarray(terrain['height_m'])
    valid=(idx[:,0]>=0)&(idx[:,0]<h.shape[1])&(idx[:,1]>=0)&(idx[:,1]<h.shape[0])
    offsets=np.full(len(world),np.inf)
    offsets[valid]=world[valid,2]-h[idx[valid,1],idx[valid,0]]
    offsets[~np.isfinite(offsets)]=np.inf
    assert np.isfinite(offsets).any()
    worst=int(np.argmin(offsets));target_cell=idx[worst]
    contributors=[]
    for name,points in sources.items():
        cells=np.floor((points[:,:2]-terrain['origin_xy_m'])/terrain['resolution_m']).astype(int)
        selected=points[(cells==target_cell).all(1)]
        if len(selected):
            top=selected[np.argmax(selected[:,2])]
            contributors.append(dict(source=name,point_count=len(selected),highest_point_m=top.tolist(),
                offset_to_cell_top_m=float(world[worst,2]-top[2])))
    distances,nearest=cKDTree(combined).query(world,workers=1)
    closest=int(np.argmin(distances));point_index=int(nearest[closest])
    cursor=0;closest_source=None
    for name,points in sources.items():
        if cursor<=point_index<cursor+len(points):closest_source=name
        cursor+=len(points)
    return dict(reconstructed_grid_equal=True,worst_bucket_sample_m=world[worst].tolist(),
        worst_cell_xy=target_cell.tolist(),cell_contributors=contributors,
        minimum_sample_to_stock_point_distance_m=float(distances.min()),
        nearest_pair=dict(source=closest_source,bucket_sample_m=world[closest].tolist(),
            stock_point_m=combined[point_index].tolist()),
        scope='Unsigned distances between sampled points only; not triangle distance, particle support, or collision clearance')


def main():
    assembly,_=_load_kinematics(KIN,KIN/'scene/input/model/A.xml',KIN/'scene/input/mesh')
    manifest_path=BUNDLE/'results/layered-mound-physical-preflight-v1/scene-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    records=[r for r in manifest['rocks'] if r['layer_id'] in ('lower','lower_middle','middle','upper')]
    assert len(records)==47
    cfg=json.loads((BUNDLE/'configs/v4/layered-47-excavation-preflight-v1.json').read_text())
    assert sha(manifest_path)==cfg['physical_geometry']['manifest_sha256']
    proxy_path=BUNDLE/cfg['open_bucket_proxy']['path']
    assert sha(proxy_path)==cfg['open_bucket_proxy']['sha256']
    samples=proxy_surface_samples(json.loads(proxy_path.read_text()))
    prior=json.loads((OUT/'phase-candidate-replay-v1.json').read_text())
    results=[]
    for episode in prior['episodes']:
        run=ROOT/'runs/layered-47-sparse-manual-contact-v1'/episode['run']
        for src in episode['source_inputs']:assert sha(Path(src['path']))==src['sha256']
        feature_path=Path(episode['feature_file']['path'])
        assert sha(feature_path)==episode['feature_file']['sha256']
        d=json.loads(feature_path.read_text());trace=d['trace']
        assert all(r['frame']==i for i,r in enumerate(trace))
        start=episode['events'][1]['confirmation_frame_end']
        booms=np.array([r['q_boom'] for r in trace]);origins=np.array([r['origin_z_m'] for r in trace])
        gates=[]
        for width in (6,12,30,60):
            for rise in (.01,.03,.05,.08,.10):
                hits=[]
                for i in range(max(start+width,width),len(trace)):
                    if booms[i]-booms[i-width]>.0005 and origins[i]-origins[i-width]>rise:
                        hits.append(i)
                gates.append(dict(window_frames=width, minimum_origin_rise_m=rise,
                                  minimum_actual_boom_cylinder_extension_m=.0005,
                                  first_confirmation_frame_end=hits[0] if hits else None,
                                  first_available_decision_frame=hits[0]+1 if hits else None))
        # Describe all actual rise windows, without interpreting the original key
        # as truth. This helps distinguish brief loading corrections from exit.
        segments=[];active=None
        for i in range(start+12,len(trace)):
            rising=booms[i]-booms[i-12]>.0005 and origins[i]-origins[i-12]>.01
            if rising and active is None:active=i
            if active is not None and (not rising or i==len(trace)-1):
                end=i if rising else i-1
                after=min(end+120,len(trace)-1)
                segments.append(dict(first_confirmation_frame_end=active,last_matching_frame_end=end,
                    origin_rise_since_window_start_m=float(origins[end]-origins[active-12]),
                    minimum_following_120_frame_origin_change_m=float(origins[end:after+1].min()-origins[end]),
                    actual_bucket_cylinder_m=trace[active]['q_bucket']))
                active=None
        # Inspect only the stored endpoint. Never back-project this geometry into
        # earlier observations or claim an off-soil start frame from it.
        n=len(trace)-1
        candidates=[run/f'final-{n:06d}.cycle.npz',run/f'final-{n:06d}.diagnostic.npz']
        path=next(p for p in candidates if p.exists())
        side=Path(str(path)+'.json');meta=json.loads(side.read_text())
        digest=sha(path)
        assert digest==meta.get('checkpoint_sha256',meta.get('sha256'))
        with np.load(path,allow_pickle=False) as data:
            arrays={k:data[k] for k in ('particle_q','rigid_qpos','mechanism_qpos')}
        # Cross-check snapshot pose against final recorded rigid state.
        physics=[json.loads(s) for s in (run/'physics.ndjson').read_text().splitlines()]
        assert np.array_equal(arrays['mechanism_qpos'],physics[-1]['after_state']['mechanism_qpos'])
        assert np.array_equal(arrays['rigid_qpos'],[r['qpos'] for r in physics[-1]['after_state']['rocks']])
        runtime=SimpleNamespace(arrays=arrays,assembly=assembly,root=BUNDLE,records=records)
        endpoint=dict(frame_end=n,snapshot=dict(path=str(path),sha256=digest),
                      metadata=dict(path=str(side),sha256=sha(side)),
                      continuous_off_soil_confirmation=False,earliest_off_soil_frame=None)
        try:
            terrain,observation=stationary_terrain(runtime)
            pose=assembly.assemble(arrays['mechanism_qpos'],include_meshes=False)[1]['bucket']
            world=samples@pose[:3,:3].T+pose[:3,3]
            h=np.asarray(terrain['height_m'])
            idx=np.floor((world[:,:2]-terrain['origin_xy_m'])/terrain['resolution_m']).astype(int)
            inside=(idx[:,0]>=0)&(idx[:,0]<h.shape[1])&(idx[:,1]>=0)&(idx[:,1]<h.shape[0])
            heights=h[idx[inside,1],idx[inside,0]];known=np.isfinite(heights)
            offset=world[inside,2][known]-heights[known]
            endpoint.update(status='existing_detached_component_gate_passed',observation=observation,
                known_surface_sample_count=int(known.sum()),unknown_or_outside_sample_count=int(len(samples)-known.sum()),
                minimum_known_stock_vertical_offset_m=float(offset.min()) if offset.size else None,
                scope='Existing 4cm connectivity heuristic plus remaining-stock sampled height grid; final snapshot only, not exact collision or full transport qualification')
            if offset.size and offset.min()<0:
                endpoint['negative_grid_offset_audit']=inspect_grid_contributors(runtime,terrain,world)
        except ValueError as error:
            endpoint.update(status='existing_detachment_gate_rejected',reason=str(error))
        assert sha(path)==digest
        results.append(dict(run=run.name,source_inputs=episode['source_inputs'],feature_file=episode['feature_file'],
                            curl_confirmation_frame_end=start, lift_parameter_sensitivity=gates,
                            actual_rise_segments=segments,
                            original_keys=[e for e in d['annotations'] if e.get('kind')=='phase' and e['label'] in ('lift_clear','off_soil_confirm')],
                            lift_boundary_confirmed=False,endpoint=endpoint))
        print(run.name,'rise segments',[(s['first_confirmation_frame_end'],s['last_matching_frame_end']) for s in segments],
              'endpoint',endpoint['status'],flush=True)
    result=dict(schema='lift_and_endpoint_readonly_audit_v1',status='candidates_only_no_labels',
                physics_advanced=False,parameter_grid_is_not_confidence_interval=True,
                script_sha256=sha(Path(__file__)),
                geometry_inputs=[dict(path=str(p),sha256=sha(p)) for p in (manifest_path,proxy_path,KIN/'scene/input/model/A.xml',ROOT/'src/v4/layered_47_cycle_geometry.py',ROOT/'src/v4/layered_excavation_approach_planner.py',Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json'),Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz'))],
                limitations=['Unsegmented per-frame terrain contains carried soil/rocks; it is not remaining-stock geometry.',
                             'Only endpoint full particles inspected; exact off-soil onset and persistence cannot be recovered from endpoint.',
                             'Lift signals show actual boom extension and bucket origin translation; no accepted semantic boundary yet.'],episodes=results)
    (OUT/'lift-endpoint-audit-v1.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()

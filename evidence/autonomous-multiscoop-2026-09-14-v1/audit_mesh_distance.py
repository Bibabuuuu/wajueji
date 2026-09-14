"""Independent NumPy triangle geometry; no simulator or model modifications.

Unsigned surface distance is NOT signed solid clearance. Tests include piercing,
coplanar crossing, separated and degenerate triangles. All coordinates are world m.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def dot(a,b):
    return np.sum(a*b,axis=-1)

def point_tri(p,t):
    a,b,c=t[...,0,:],t[...,1,:],t[...,2,:]
    e,f=b-a,c-a;g=p-a
    ee,ef,ff=dot(e,e),dot(e,f),dot(f,f)
    den=ee*ff-ef*ef
    valid=den>1e-28
    u=np.divide(dot(g,e)*ff-dot(g,f)*ef,den,out=np.zeros_like(den),where=valid)
    v=np.divide(dot(g,f)*ee-dot(g,e)*ef,den,out=np.zeros_like(den),where=valid)
    projection=a+u[...,None]*e+v[...,None]*f
    inside=valid&(u>=0)&(v>=0)&(u+v<=1)
    choices=[projection]
    distances=[np.where(inside,dot(p-projection,p-projection),np.inf)]
    for x,y in ((a,b),(b,c),(c,a)):
        edge=y-x;den=dot(edge,edge)
        s=np.clip(np.divide(dot(p-x,edge),den,out=np.zeros_like(den),where=den>0),0,1)
        near=x+s[...,None]*edge;choices.append(near);distances.append(dot(p-near,p-near))
    ds=np.stack(distances,axis=-1);pts=np.stack(choices,axis=-2);which=np.argmin(ds,axis=-1)
    return np.min(ds,axis=-1),np.take_along_axis(pts,which[...,None,None],axis=-2)[...,0,:]

def nearest(p,t):
    ds,points=point_tri(np.broadcast_to(p,t[:,0,:].shape),t);i=int(np.argmin(ds))
    return dict(distance_m=float(np.sqrt(ds[i])),triangle=i,point=points[i].tolist())

def segment_pair(p,q,a,b):
    u,v=q-p,b-a;w=p-a
    uu,vv,uv,uw,vw=dot(u,u),dot(v,v),dot(u,v),dot(u,w),dot(v,w)
    den=uu*vv-uv*uv
    s=np.divide(uv*vw-vv*uw,den,out=np.zeros_like(den),where=den>1e-28)
    t=np.divide(uu*vw-uv*uw,den,out=np.zeros_like(den),where=den>1e-28)
    valid=(den>1e-28)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    delta=w+s[...,None]*u-t[...,None]*v
    best=np.where(valid,dot(delta,delta),np.inf)
    for fixed in (0.,1.):
        t=np.clip(np.divide(vw+fixed*uv,vv,out=np.zeros_like(vv),where=vv>0),0,1)
        delta=w+fixed*u-t[...,None]*v;best=np.minimum(best,dot(delta,delta))
        s=np.clip(np.divide(fixed*uv-uw,uu,out=np.zeros_like(uu),where=uu>0),0,1)
        delta=w+s[...,None]*u-fixed*v;best=np.minimum(best,dot(delta,delta))
    return best

def piercing(p,q,t):
    e=t[:,1]-t[:,0];f=t[:,2]-t[:,0];n=np.cross(e,f);direction=q-p
    den=dot(n,direction);valid=np.abs(den)>1e-22
    fraction=np.divide(dot(n,t[:,0]-p),den,out=np.zeros_like(den),where=valid)
    hit=p+fraction[:,None]*direction
    ds,_=point_tri(hit,t)
    return valid&(fraction>=0)&(fraction<=1)&(ds<1e-24)

def paired_tri_distance(a,b):
    best=np.full(len(a),np.inf);intersect=np.zeros(len(a),dtype=bool)
    for k in range(3):
        best=np.minimum(best,point_tri(a[:,k],b)[0]);best=np.minimum(best,point_tri(b[:,k],a)[0])
        intersect|=piercing(a[:,k],a[:,(k+1)%3],b)|piercing(b[:,k],b[:,(k+1)%3],a)
        for j in range(3):best=np.minimum(best,segment_pair(a[:,k],a[:,(k+1)%3],b[:,j],b[:,(j+1)%3]))
    best[intersect]=0
    return best

def mesh_pair_distance(a,b,upper):
    # Triangle AABB lower bounds prune only pairs that cannot improve the known
    # surface-to-surface upper bound. No sampling or mesh decimation is used.
    alo,ahi=a.min(axis=1),a.max(axis=1);blo,bhi=b.min(axis=1),b.max(axis=1)
    first=[];second=[]
    for i in range(len(a)):
        gap=np.maximum(np.maximum(alo[i]-bhi,blo-ahi[i]),0)
        ids=np.flatnonzero(dot(gap,gap)<=(upper+1e-12)**2)
        first.extend([i]*len(ids));second.extend(ids.tolist())
    assert first
    first=np.array(first);second=np.array(second);ds=[]
    for k in range(0,len(first),1024):
        ds.extend(paired_tri_distance(a[first[k:k+1024]],b[second[k:k+1024]]).tolist())
    ds=np.array(ds);i=int(np.argmin(ds));hits=np.flatnonzero(ds<1e-20)
    return dict(unsigned_surface_distance_m=float(np.sqrt(ds[i])),closest_triangles=[int(first[i]),int(second[i])],
                candidate_triangle_pairs=len(first),all_triangle_pairs=len(a)*len(b),
                intersecting_or_within_1e_10m_pairs=len(hits),
                near_zero_triangle_pairs=[[int(first[k]),int(second[k])] for k in hits[:30]])

def tests():
    a=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    assert abs(nearest([.2,.2,2.],a)['distance_m']-2)<1e-12
    assert abs(nearest([2.,0.,0.],a)['distance_m']-1)<1e-12
    assert np.allclose(paired_tri_distance(a,a+[0,0,2]),[4])
    pierce=np.array([[[.25,.25,-1.],[.25,.25,1.],[2.,2.,1.]]])
    assert paired_tri_distance(a,pierce)[0]==0
    crossing=np.array([[[-1.,.5,0.],[1.,.5,0.],[0.,2.,0.]]])
    assert paired_tri_distance(a,crossing)[0]<1e-20
    degenerate=np.zeros((1,3,3));assert nearest([0,0,3],degenerate)['distance_m']==3
    assert np.allclose(segment_pair(np.array([[0.,0.,0.]]),np.array([[1.,0.,0.]]),np.array([[.5,-1.,2.]]),np.array([[.5,1.,2.]])),[4])
    return 7

if __name__=='__main__':
    from scipy.spatial import ConvexHull
    count=tests();meta=json.loads((ROOT/'export01/contact.json').read_text())
    assert meta['physical_time']==meta['soil_updates']==meta['bucket_m3']==0
    meshes=[np.loadtxt(ROOT/'export01'/g['triangles_file'],delimiter=',',skiprows=1,usecols=(2,3,4)).reshape(-1,3,3) for g in meta['geoms']]
    p=np.array(meta['position']);near=[nearest(p,t) for t in meshes]
    hulls=[]
    for t in meshes[2:]:
        hull=ConvexHull(np.unique(t.reshape(-1,3),axis=0))
        hulls.append(dict(max_signed_halfspace_at_reported_point_m=float(np.max(hull.equations[:,:3]@p+hull.equations[:,3]))))
    upper=np.linalg.norm(np.array(near[0]['point'])-np.array(near[1]['point']))
    pair=mesh_pair_distance(meshes[0],meshes[1],upper)
    blue_upper=np.linalg.norm(np.array(near[0]['point'])-np.array(near[3]['point']))
    blue_to_source_rod=mesh_pair_distance(meshes[0],meshes[3],blue_upper)
    result=dict(algorithm_checks_passed=count,coordinate_frame='world',units='m',native_contact=meta,
                nearest_surface_to_native_contact={g['name']:n for g,n in zip(meta['geoms'],near)},proxy_hull_checks=hulls,
                complete_visual_mesh_pair=pair,
                original_rod_vs_entire_blue_proxy=blue_to_source_rod,
                qualification='unsigned triangle-surface evidence only; no model edit or dynamic safety claim',
                files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'export01').rglob('*')) if p.is_file()})
    print(json.dumps(result,indent=2))

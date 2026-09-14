"""Mass-conserving, one-way solid-cell activation for isolated MPM prototypes.

The remaining solid is a rigid approximation. There is no automatic failure-zone
model or particle-to-solid merge. Those limitations must remain visible to users.
"""
import numpy as np


class SolidBank:
    def __init__(self, occupied, origin, cell_m=.1, spacing_m=.025, density=1670.):
        self._solid=np.asarray(occupied,bool).copy();self.origin=np.asarray(origin,float).copy()
        self.cell=float(cell_m);self.spacing=float(spacing_m);self.density=float(density)
        if self._solid.ndim!=3 or not all(self._solid.shape) or self.origin.shape!=(3,) or not np.isfinite(self.origin).all():raise ValueError('invalid_bank_geometry')
        if not np.isfinite([self.cell,self.spacing,self.density]).all() or min(self.cell,self.spacing,self.density)<=0:raise ValueError('invalid_bank_units')
        self.n=round(self.cell/self.spacing)
        if self.n<1 or abs(self.n*self.spacing-self.cell)>1e-10:raise ValueError('nonintegral_particle_subdivision')
        self.initial_cells=int(self._solid.sum());self.active_cells=0;self.version=0

    @property
    def solid(self):return self._solid.copy()

    def _points(self, cells):
        ijk=np.indices((self.n,)*3).reshape(3,-1).T
        return (self.origin+(cells[:,None,:]+(ijk[None,:,:]+.5)/self.n)*self.cell).reshape(-1,3).astype(np.float32)

    def activate(self, low, high, prepare):
        """Commit the solid removal only after prepare(new points, remaining) succeeds.

        prepare owns building/validating a replacement physics state. It must not
        advance a live instance. All previously activated particles remain owned
        by that physics state; no particles are deleted to meet a budget.
        """
        lo=np.asarray(low,float);hi=np.asarray(high,float)
        if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or np.any(lo>=hi):raise ValueError('invalid_activation_bounds')
        cells=np.argwhere(self._solid);centers=self.origin+(cells+.5)*self.cell
        # Whole cells intersecting the requested region, expanded outward at the boundary.
        take=np.all(centers+self.cell/2>lo+1e-9,axis=1)&np.all(centers-self.cell/2<hi-1e-9,axis=1)
        selected=cells[take];remaining=self._solid.copy()
        if not len(selected):return {'new_particles':0,'new_mass_kg':0.,'prepared':None,'version':self.version}
        remaining[tuple(selected.T)]=False;points=self._points(selected)
        prepared=prepare(points.copy(),remaining.copy())
        self._solid=remaining;self.active_cells+=len(selected);self.version+=1
        return {'new_particles':len(points),'new_mass_kg':len(points)*self.spacing**3*self.density,
                'prepared':prepared,'version':self.version}

    def ledger(self, physical_mass_kg):
        initial=self.initial_cells*self.cell**3*self.density
        solid=int(self._solid.sum())*self.cell**3*self.density
        expected=self.active_cells*self.cell**3*self.density
        return {'initial_mass_kg':initial,'solid_mass_kg':solid,'expected_active_mass_kg':expected,
                'physical_mass_kg':float(physical_mass_kg),'residual_kg':solid+float(physical_mass_kg)-initial}

    def surface_mesh(self, occupied=None):
        """Closed exposed voxel faces, outward winding; includes activation cavities."""
        solid=self._solid if occupied is None else np.asarray(occupied,bool)
        if solid.shape!=self._solid.shape:raise ValueError('solid_shape_mismatch')
        facespec=[((-1,0,0),[(0,0,0),(0,0,1),(0,1,1),(0,1,0)]),
                  ((1,0,0),[(1,0,0),(1,1,0),(1,1,1),(1,0,1)]),
                  ((0,-1,0),[(0,0,0),(1,0,0),(1,0,1),(0,0,1)]),
                  ((0,1,0),[(0,1,0),(0,1,1),(1,1,1),(1,1,0)]),
                  ((0,0,-1),[(0,0,0),(0,1,0),(1,1,0),(1,0,0)]),
                  ((0,0,1),[(0,0,1),(1,0,1),(1,1,1),(0,1,1)])]
        cells=np.argwhere(solid);quads=[]
        for direction,corners in facespec:
            adjacent=cells+direction;inside=np.all((adjacent>=0)&(adjacent<solid.shape),axis=1)
            hidden=np.zeros(len(cells),bool);hidden[inside]=solid[tuple(adjacent[inside].T)]
            quads.append(cells[~hidden,None,:]+np.asarray(corners)[None,:,:])
        vertices=self.origin+np.concatenate(quads).reshape(-1,3)*self.cell
        ids=np.arange(len(vertices)).reshape(-1,4)
        triangles=np.concatenate([ids[:,[0,1,2]],ids[:,[0,2,3]]])
        return vertices.astype(np.float32),triangles.astype(np.int32)


def bank_candidate(height_m=1.2):
    """Proposal in world coordinates: closer front, broad supported top, no placement approval."""
    origin=np.array([3.6,-1.2,0.]);cell=.1;shape=(26,24,round(height_m/cell))
    xyz=origin+(np.indices(shape).reshape(3,-1).T+.5)*cell
    # A front ramp and small side shoulders bound the full-scale candidate.
    height=height_m*np.minimum(np.clip((xyz[:,0]-3.6)/.6,0,1),np.clip((1.2-np.abs(xyz[:,1]))/.2,0,1))
    occupied=(xyz[:,2]<=height).reshape(shape)
    return SolidBank(occupied,origin,cell)

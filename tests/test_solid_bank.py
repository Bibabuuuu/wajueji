import unittest
import numpy as np
from terrain.solid_bank import SolidBank,bank_candidate


class SolidBankTests(unittest.TestCase):
    def setUp(self):self.bank=SolidBank(np.ones((8,8,6),bool),[0,0,0]);self.particles=[]
    def prepare(self,p,remaining):self.particles.append(p);return len(p)
    def test_overlap_does_not_create_more_soil_and_arbitrary_second_region(self):
        a=self.bank.activate([.1,.1,.2],[.4,.4,.6],self.prepare)
        self.assertEqual(a['new_particles'],36*64)
        b=self.bank.activate([.1,.1,.2],[.4,.4,.6],self.prepare);self.assertEqual(b['new_particles'],0)
        self.bank.activate([.5,.5,.3],[.8,.8,.6],self.prepare)
        q=np.concatenate(self.particles);self.assertEqual(len(np.unique(q,axis=0)),len(q))
        self.assertAlmostEqual(self.bank.ledger(len(q)*.025**3*1670)['residual_kg'],0,places=9)
    def test_failed_physics_preparation_leaves_solid_intact(self):
        before=self.bank.solid
        def fail(p,s):raise ValueError('physics_build_failed')
        with self.assertRaises(ValueError):self.bank.activate([0,0,0],[.4,.4,.4],fail)
        np.testing.assert_array_equal(self.bank.solid,before);self.assertEqual(self.bank.version,0)
    def test_surface_volume_tracks_remaining_mass_with_cavity(self):
        self.bank.activate([.2,.2,.2],[.6,.6,.6],self.prepare)
        v,f=self.bank.surface_mesh();tri=v[f].astype(float)
        volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
        self.assertAlmostEqual(volume,self.bank.solid.sum()*.1**3,places=7)
        # Every geometric edge is shared by two faces, including newly exposed walls.
        _,index=np.unique(v,axis=0,return_inverse=True);f=index[f]
        edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
        _,counts=np.unique(edges,axis=0,return_counts=True);self.assertTrue((counts==2).all())
    def test_particle_volume_exactly_equals_removed_cells(self):
        self.bank.activate([-.1,-.1,-.1],[1,1,1],self.prepare)
        self.assertEqual(len(self.particles[0]),8*8*6*64);self.assertEqual(self.bank.solid.sum(),0)
        self.assertAlmostEqual(len(self.particles[0])*.025**3,.8*.8*.6)
    def test_invalid_input_never_changes_bank(self):
        for low,high in [([0,0,0],[0,1,1]),([np.nan,0,0],[1,1,1])]:
            with self.assertRaises(ValueError):self.bank.activate(low,high,self.prepare)
        with self.assertRaises(ValueError):SolidBank(np.ones((2,2,2)),[0,0,0],spacing_m=.03)
        self.assertEqual(self.bank.version,0)
    def test_candidate_geometry_and_mass_budget(self):
        b=bank_candidate();v,f=b.surface_mesh()
        self.assertGreater(b.ledger(0)['initial_mass_kg']/1670,3*.277762)
        self.assertAlmostEqual(float(v[:,0].min()),3.6,places=5)
        self.assertAlmostEqual(float(v[:,2].max()),1.2,places=5)

if __name__=='__main__':unittest.main()

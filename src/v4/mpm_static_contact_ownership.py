"""Read-only shape attribution within Newton's merged static collider.

Collider IDs identify bodies, not shapes. Re-query the same mesh/material at
each active static contact node; never split the solver's physical collider.
This adapter deliberately rejects topology layouts outside default discovery.
"""
import numpy as np
import warp as wp
from newton._src.solvers.implicit_mpm.implicit_mpm_model import _get_body_collision_shapes
from newton._src.solvers.implicit_mpm.rasterized_collisions import Collider, _query_collider_sdf


@wp.kernel
def _static_materials(points: wp.array[wp.vec3], collider: Collider,
                      body_q: wp.array[wp.transform], collider_id: int,
                      materials: wp.array[int]):
    i = wp.tid()
    found, sdf, normal, velocity, closest, material = _query_collider_sdf(
        points[i], collider, body_q, collider_id)
    materials[i] = wp.where(found, material, -1)


class StaticContactOwnership:
    def __init__(self, solver, model, named_shapes):
        self.collider = solver._mpm_model.collider
        bodies = solver.collider_body_index.numpy()
        static = np.flatnonzero(bodies == -1)
        if static.size != 1:
            raise ValueError("static_contact_requires_single_merged_collider")
        self.collider_id = int(static[0])
        offsets = self.collider.collider_face_offset.numpy()
        face_materials = self.collider.face_material_index.numpy()
        next_material = 1
        shape_materials = {}
        for cid, body in enumerate(bodies):
            shapes = _get_body_collision_shapes(model, int(body))
            expected = np.arange(next_material, next_material + len(shapes))
            end = int(offsets[cid + 1]) if cid + 1 < len(offsets) else len(face_materials)
            actual = np.unique(face_materials[int(offsets[cid]):end])
            if not np.array_equal(actual, expected):
                raise ValueError("static_contact_material_layout_changed")
            if int(body) == -1:
                shape_materials = dict(zip(map(int, shapes), map(int, expected), strict=True))
            next_material += len(shapes)
        if (len(set(named_shapes.values())) != len(named_shapes)
                or set(named_shapes.values()) != set(shape_materials)):
            raise ValueError("static_contact_shape_ownership_incomplete_or_duplicate")
        self.materials = {name: shape_materials[int(shape)]
                          for name, shape in named_shapes.items()}
        self.device = self.collider.collider_mesh.device
        self.manifest = {
            "method": "merged_static_mesh_material_query_v1",
            "collider_count": int(len(bodies)),
            "merged_static_collider_id": self.collider_id,
            "shape_indices": {name: int(shape) for name, shape in named_shapes.items()},
            "material_ids": self.materials,
            "physical_topology_unchanged": True,
        }

    def count(self, positions, ids, active, body_q):
        selected = np.asarray(positions)[active & (ids == self.collider_id)]
        counts = dict.fromkeys(self.materials, 0)
        if not len(selected):
            return counts, 0
        points = wp.array(selected, dtype=wp.vec3, device=self.device)
        output = wp.empty(len(selected), dtype=int, device=self.device)
        wp.launch(_static_materials, dim=len(selected), inputs=[
            points, self.collider, body_q, self.collider_id, output], device=self.device)
        materials = output.numpy()
        counts = {name: int(np.count_nonzero(materials == material))
                  for name, material in self.materials.items()}
        return counts, int(len(selected) - sum(counts.values()))

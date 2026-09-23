"""GPU density surface used by the current interactive soil windows.

This module is deliberately display-only.  It reads the existing MPM particle
positions on their Warp device, builds a density field and draws its
isosurface; it never writes particle position, velocity, stress or material
state.  Sparse loose particles remain visible around the continuous surface.
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp


@wp.kernel
def _splat_density(
        particle_q: wp.array(dtype=wp.vec3), field: wp.array(dtype=float, ndim=3),
        origin: wp.vec3, voxel_size: float, volume_ratio: float):
    point = (particle_q[wp.tid()] - origin) / voxel_size
    x = int(wp.floor(point[0]))
    y = int(wp.floor(point[1]))
    z = int(wp.floor(point[2]))
    fraction = point - wp.vec3(float(x), float(y), float(z))
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                ix = x + dx
                iy = y + dy
                iz = z + dz
                if (ix >= 0 and iy >= 0 and iz >= 0
                        and ix < field.shape[0] and iy < field.shape[1]
                        and iz < field.shape[2]):
                    weight = (
                        wp.where(dx == 0, 1.0 - fraction[0], fraction[0])
                        * wp.where(dy == 0, 1.0 - fraction[1], fraction[1])
                        * wp.where(dz == 0, 1.0 - fraction[2], fraction[2]))
                    wp.atomic_add(field, ix, iy, iz, weight * volume_ratio)


@wp.kernel
def _blur(
        source: wp.array(dtype=float, ndim=3), target: wp.array(dtype=float, ndim=3),
        axis: int):
    x, y, z = wp.tid()
    value = float(0.0)
    for offset in range(-1, 2):
        ix = x + wp.where(axis == 0, offset, 0)
        iy = y + wp.where(axis == 1, offset, 0)
        iz = z + wp.where(axis == 2, offset, 0)
        if (ix >= 0 and iy >= 0 and iz >= 0
                and ix < source.shape[0] and iy < source.shape[1]
                and iz < source.shape[2]):
            value += source[ix, iy, iz] * wp.where(offset == 0, 0.5, 0.25)
    target[x, y, z] = value


@wp.kernel
def _surface_detail(
        vertices: wp.array(dtype=wp.vec3), uv: wp.array(dtype=wp.vec2)):
    index = wp.tid()
    point = vertices[index]
    # At most 2 mm visual roughness.  This is not a collision or physics edit.
    first = wp.sin(point[0] * 143.0 + point[1] * 81.0 + point[2] * 107.0)
    second = wp.sin(point[0] * 57.0 - point[1] * 111.0 + point[2] * 73.0)
    vertices[index] = point + wp.vec3(
        0.001 * first, 0.001 * second, 0.001 * first * second)
    uv[index] = wp.vec2(
        (point[0] + 0.77 * point[2]) * 3.0,
        (point[1] + 0.53 * point[2]) * 3.0)


@wp.kernel
def _loose_particle_radii(
        particle_q: wp.array(dtype=wp.vec3), field: wp.array(dtype=float, ndim=3),
        origin: wp.vec3, voxel_size: float, radius: float,
        radii: wp.array(dtype=float)):
    index = wp.tid()
    point = (particle_q[index] - origin) / voxel_size
    x = int(wp.floor(point[0] + 0.5))
    y = int(wp.floor(point[1] + 0.5))
    z = int(wp.floor(point[2] + 0.5))
    density = float(0.0)
    if (x >= 0 and y >= 0 and z >= 0
            and x < field.shape[0] and y < field.shape[1] and z < field.shape[2]):
        density = field[x, y, z]
    radii[index] = wp.where(density < 0.40, radius, 0.0)


class SoilSurface:
    """A GPU-resident continuous visual surface for the fixed 47-rock view."""

    def __init__(self, particle_q, *, particle_spacing_m: float,
                 voxel_size_m: float):
        self.voxel_size_m = float(voxel_size_m)
        self.origin = wp.vec3(2.8, -2.0, -0.12)
        self.upper = wp.vec3(7.8, 2.0, 3.6)
        extent = np.asarray([5.0, 4.0, 3.72], dtype=np.float64)
        self.shape = tuple(
            int(np.ceil(value / self.voxel_size_m)) + 1 for value in extent)
        self.particle_spacing_m = float(particle_spacing_m)
        self.field = wp.zeros(self.shape, dtype=float, device=particle_q.device)
        self.temp = wp.zeros_like(self.field)
        self.marching_cubes = wp.MarchingCubes(
            *self.shape, domain_bounds_lower_corner=self.origin,
            domain_bounds_upper_corner=self.upper)
        self.radii = wp.empty(len(particle_q), dtype=float, device=particle_q.device)
        self.colors = wp.full(
            len(particle_q), wp.vec3(0.36, 0.28, 0.19), dtype=wp.vec3,
            device=particle_q.device)
        self.uv = None
        self.last_update_token = None
        self.drawn_update_token = None
        self.update_seconds: list[float] = []
        rng = np.random.default_rng(613)
        noise = rng.random((128, 128))
        coarse = np.repeat(np.repeat(rng.random((16, 16)), 8, axis=0), 8, axis=1)
        self.texture = np.uint8(np.clip(
            np.array([96, 75, 52])[None, None, :]
            * (0.65 + 0.45 * noise[:, :, None] + 0.25 * coarse[:, :, None]),
            0, 255))

    def update(self, particle_q, update_token: int) -> None:
        """Rebuild once per configured token without CPU particle readback."""
        if update_token == self.last_update_token:
            return
        started = time.perf_counter()
        self.field.zero_()
        wp.launch(
            _splat_density, dim=len(particle_q),
            inputs=[particle_q, self.field, self.origin, self.voxel_size_m,
                    (self.particle_spacing_m / self.voxel_size_m) ** 3],
            device=particle_q.device)
        wp.launch(_blur, dim=self.shape, inputs=[self.field, self.temp, 0],
                  device=particle_q.device)
        wp.launch(_blur, dim=self.shape, inputs=[self.temp, self.field, 1],
                  device=particle_q.device)
        wp.launch(_blur, dim=self.shape, inputs=[self.field, self.temp, 2],
                  device=particle_q.device)
        self.marching_cubes.surface(self.temp, 0.45)
        self.uv = wp.empty(
            len(self.marching_cubes.verts), dtype=wp.vec2, device=particle_q.device)
        wp.launch(_surface_detail, dim=len(self.marching_cubes.verts),
                  inputs=[self.marching_cubes.verts, self.uv],
                  device=particle_q.device)
        wp.launch(
            _loose_particle_radii, dim=len(particle_q),
            inputs=[particle_q, self.temp, self.origin, self.voxel_size_m,
                    self.particle_spacing_m * 0.38, self.radii],
            device=particle_q.device)
        wp.synchronize()
        self.update_seconds.append(time.perf_counter() - started)
        self.last_update_token = update_token

    def draw(self, viewer, particle_q) -> None:
        """Draw the current mesh and the loose particles not represented by it."""
        if self.drawn_update_token != self.last_update_token:
            # Newton 1.5.1 caches MeshGL topology.  Recreate only this display
            # object after a density update, leaving the physical model intact.
            object_name = viewer._qualify("/soil_surface")
            old = viewer.objects.pop(object_name, None)
            if old is not None:
                old.destroy()
            viewer.log_mesh(
                "/soil_surface", self.marching_cubes.verts,
                self.marching_cubes.indices, uvs=self.uv, texture=self.texture,
                color=(1.0, 1.0, 1.0), roughness=1.0, metallic=0.0,
                backface_culling=False)
            viewer.objects[object_name].material = (1.0, 0.0, 0.0, 1.0)
            self.drawn_update_token = self.last_update_token
        viewer.log_points("/soil_loose", particle_q, self.radii, self.colors)

"""Runtime observation for the opt-in Hencky particle update candidate.

The observer runs immediately before each candidate particle update and records
whether ``hencky_proposal`` rejected any active particle. It does not alter the
constitutive update or any simulation state.
"""

from __future__ import annotations

from contextlib import contextmanager

import newton
import numpy as np
import warp as wp
import warp.fem as fem

from soil_hencky_update_candidate import hencky_proposal


MAXIMUM_F_FROBENIUS = 1000.0


@fem.integrand
def observe_hencky_rejection(
    sample: fem.Sample,
    domain: fem.Domain,
    elastic_delta: fem.Field,
    velocity: fem.Field,
    dt: float,
    flags: wp.array[wp.int32],
    density: wp.array[float],
    previous: wp.array[wp.mat33],
    rejection: wp.array[wp.int32],
):
    index = sample.qp_index
    if ~flags[index] & newton.ParticleFlags.ACTIVE:
        return
    if density[index] == 0.0:
        return
    gradient = fem.grad(velocity, sample)
    spin = 0.5 * dt * (gradient - wp.transpose(gradient))
    _proposal, code = hencky_proposal(
        previous[index], elastic_delta(sample), spin)
    wp.atomic_max(rejection, index, code)


def summarize_material_state(state, rejection: np.ndarray) -> dict:
    """Return the bounded manual-pilot material guard record."""
    state = getattr(state, "mpm", state)
    elastic = np.asarray(
        state.particle_elastic_strain.numpy(), dtype=np.float64)
    jp = np.asarray(state.particle_Jp.numpy(), dtype=np.float64)
    stress = np.asarray(state.particle_stress.numpy(), dtype=np.float64)
    codes = np.asarray(rejection, dtype=np.int32)
    elastic_norm = np.linalg.norm(elastic, axis=(1, 2))
    nonfinite = {
        "particle_elastic_strain": int(np.count_nonzero(~np.isfinite(elastic))),
        "particle_Jp": int(np.count_nonzero(~np.isfinite(jp))),
        "particle_stress": int(np.count_nonzero(~np.isfinite(stress))),
    }
    jp_finite = bool(np.isfinite(jp).all())
    maximum_f = float(np.max(elastic_norm, initial=0.0)) \
        if np.isfinite(elastic_norm).all() else None
    checks = {
        "all_material_arrays_finite": not any(nonfinite.values()),
        "candidate_update_rejection_zero": not bool(np.count_nonzero(codes)),
        "elastic_F_frobenius_bounded": bool(
            maximum_f is not None and maximum_f <= MAXIMUM_F_FROBENIUS),
        "Jp_positive": bool(jp_finite and np.all(jp > 0.0)),
    }
    return {
        "enabled": True,
        "checks": checks,
        "safe": all(checks.values()),
        "nonfinite_values": nonfinite,
        "rejection_count": int(np.count_nonzero(codes)),
        "rejection_codes": {
            str(code): int(np.count_nonzero(codes == code))
            for code in (1, 2, 3)
        },
        "maximum_F_frobenius": maximum_f,
        "F_frobenius_over_1000": int(np.count_nonzero(
            elastic_norm > MAXIMUM_F_FROBENIUS)),
        "minimum_Jp": float(np.min(jp, initial=np.inf))
            if jp_finite else None,
        "maximum_Jp": float(np.max(jp, initial=-np.inf))
            if jp_finite else None,
        "Jp_nonpositive": int(np.count_nonzero(jp <= 0.0)),
    }


class HenckyMaterialGuard:
    def __init__(self, solver):
        self.solver = solver
        model = solver._mpm_model
        self.rejection = wp.zeros(
            model.material_particle_flags.shape[0], dtype=wp.int32,
            device=model.material_particle_flags.device)

    def prepare_frame(self) -> None:
        self.rejection.zero_()

    def observe_before_update(self, state_in, dt, pic, scratch) -> None:
        model = self.solver._mpm_model
        fem.interpolate(
            observe_hencky_rejection,
            at=pic,
            fields={
                "elastic_delta": scratch.elastic_strain_delta_field,
                "velocity": scratch.velocity_field,
            },
            values={
                "dt": dt,
                "flags": model.material_particle_flags,
                "density": model.particle_density,
                "previous": state_in.mpm.particle_elastic_strain,
                "rejection": self.rejection,
            },
            temporary_store=self.solver.temporary_store,
        )

    def summarize(self, state) -> dict:
        return summarize_material_state(state, self.rejection.numpy())


@contextmanager
def installed_material_guard(solver):
    """Observe one solver process-locally and restore its method on exit."""
    guard = HenckyMaterialGuard(solver)
    original = solver._update_particles

    def observed_update(state_in, state_out, dt, pic, scratch):
        guard.observe_before_update(state_in, dt, pic, scratch)
        return original(state_in, state_out, dt, pic, scratch)

    solver._update_particles = observed_update
    try:
        yield guard
    finally:
        solver._update_particles = original

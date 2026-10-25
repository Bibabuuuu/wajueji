"""Correct the mixed bridge's bucket velocity reference point, opt-in.

MuJoCo mjOBJ_BODY returns velocity at the inertial-frame origin (COM).
The MPM bucket transform and sweep use the regular body origin instead.
Keep the old COM velocity as a separate diagnostic when applying this fix.
"""
from contextlib import contextmanager
import hashlib

OLD = '''      mj_objectVelocity(model.get(), data.get(), mjOBJ_BODY, bucket_body,
                        bucket_velocity, 0);'''
NEW = '''      mj_objectVelocity(model.get(), data.get(), mjOBJ_BODY, bucket_body,
                        bucket_velocity, 0);
      array("bucket_com_linear_velocity_world_m_s", bucket_velocity + 3, 3);
      // MPM body_com=0 and moving sweeps use the bucket's regular body origin.
      mj_objectVelocity(model.get(), data.get(), mjOBJ_XBODY, bucket_body,
                        bucket_velocity, 0);'''


def correct_source(source):
    if source.count(OLD) != 1:
        raise ValueError('bucket_origin_velocity_source_contract')
    return source.replace(OLD, NEW, 1)


@contextmanager
def installed_candidate():
    import layered_mound_checkpoint_runtime as runtime
    original = runtime.render_capacity_bridge
    def render(*args, **kwargs):
        source, manifest = original(*args, **kwargs)
        corrected = correct_source(source)
        manifest = dict(manifest)
        manifest['bucket_velocity_origin_fix'] = {
            'linear_velocity_reference': 'bucket_regular_body_origin',
            'previous_reference': 'bucket_inertial_origin_COM',
            'mujoco_object_type': 'mjOBJ_XBODY',
            'physical_state_modified': False,
            'generated_source_sha256': hashlib.sha256(corrected.encode()).hexdigest(),
        }
        return corrected, manifest
    runtime.render_capacity_bridge = render
    try:
        yield
    finally:
        runtime.render_capacity_bridge = original

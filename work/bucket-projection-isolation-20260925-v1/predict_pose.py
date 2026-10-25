import numpy as np
import warp as wp

def predict(pose, linear, angular, dt):
    w=np.asarray(angular,dtype=float);angle=float(np.linalg.norm(w)*dt)
    spin=wp.quat_identity() if angle==0 else wp.quat_from_axis_angle(wp.vec3(*(w/np.linalg.norm(w))),angle)
    return wp.transform(wp.transform_get_translation(pose)+wp.vec3(*(np.asarray(linear)*dt)),wp.mul(spin,wp.transform_get_rotation(pose)))

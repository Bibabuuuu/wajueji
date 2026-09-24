"""Same preview limits, continuous at measured nonzero closure residual."""
import numpy as np

def advance(chain,velocity,dt,residual_reference):
    velocity=np.asarray(velocity,float)
    if velocity.shape!=(4,) or not np.isfinite(velocity).all() or not 0<dt<=.05:raise ValueError('invalid_command')
    if np.any(abs(velocity)>np.array([.12,.0225,.0225,.0225])+1e-12):raise ValueError('command_budget')
    if not np.any(velocity):return chain.q.copy(),0.
    candidate=chain.q.copy();candidate[chain.axes]+=velocity*dt
    for address,(lo,hi) in chain.limits.items():
        if not lo+.005<=candidate[address]<=hi-.005:raise ValueError('joint_margin')
    for _ in range(12):
        r=chain.residual(candidate)-residual_reference;error=float(np.max(abs(r)))
        if error<1e-8:break
        jac=[]
        for address in chain.passive:
            shifted=candidate.copy();shifted[address]+=1e-6;jac.append((chain.residual(shifted)-residual_reference-r)/1e-6)
        delta=np.linalg.lstsq(np.array(jac).T,-r,rcond=1e-10)[0]
        if not np.isfinite(delta).all() or max(abs(delta))>.05:raise ValueError('passive_branch_jump')
        candidate[chain.passive]+=delta
    else:raise ValueError('closure_unresolved')
    if max(abs(candidate[chain.passive]-chain.q[chain.passive]))>.05:raise ValueError('passive_step_budget')
    if not np.isfinite(candidate).all():raise ValueError('nonfinite_preview')
    chain.q=candidate
    return candidate.copy(),error

def preview(projector,command,dt,residual_reference):
    chain=projector.chain;original=chain.advance
    try:
        chain.advance=lambda velocity,dt:advance(chain,velocity,dt,residual_reference)
        return projector._advance_preview(command,dt)
    finally:chain.advance=original

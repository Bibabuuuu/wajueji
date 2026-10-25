"""Keep measured closure offsets in the rotating chassis frame during preview."""
import importlib.util
import numpy as np
from runtime import ROOT
spec=importlib.util.spec_from_file_location('_frozen_d153_preview',ROOT/'work/contact-preview-lift-20260924-v1/continuous_preview.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def rotation(chain,q):
    return chain.assembly.assemble(q,include_meshes=False)[1]['chassis'][:3,:3]

def advance(chain,velocity,dt,residual_reference,reference_rotation):
    future=chain.q.copy();future[chain.axes]+=np.asarray(velocity)*dt
    target=(np.asarray(residual_reference).reshape(-1,3) @ reference_rotation
            @ rotation(chain,future).T).reshape(-1)
    return old.advance(chain,velocity,dt,target)

def preview(projector,command,dt,residual_reference,reference_rotation):
    chain=projector.chain;original=chain.advance
    try:
        chain.advance=lambda velocity,dt:advance(chain,velocity,dt,residual_reference,reference_rotation)
        return projector._advance_preview(command,dt)
    finally:chain.advance=original

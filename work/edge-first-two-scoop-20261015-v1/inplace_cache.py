"""Trim this run's own display cache in place, preserving every used byte."""
from pathlib import Path
import io,json,os,tempfile,hashlib
import numpy as np
from view_contact_bc_probe import ProbeDisplayCache,cache_digest

def trim_file(path,used):
 path=Path(path)
 with path.open('rb') as f:
  version=np.lib.format.read_magic(f)
  if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f);writer=np.lib.format.write_array_header_1_0
  elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f);writer=np.lib.format.write_array_header_2_0
  else:raise ValueError('unsupported_cache_format')
  offset=f.tell()
 if not 1<=used<=shape[0] or order:raise ValueError('cache_trim_contract')
 buffer=io.BytesIO();writer(buffer,dict(descr=np.lib.format.dtype_to_descr(dtype),fortran_order=False,shape=(used,*shape[1:])))
 header=buffer.getvalue()
 if len(header)!=offset:raise ValueError('cache_header_length_changed')
 with path.open('r+b') as f:
  f.write(header);f.truncate(offset+used*int(np.prod(shape[1:]))*dtype.itemsize);f.flush();os.fsync(f.fileno())
 return dict(original_reserved_frames=shape[0],used_frames=used,data_offset=offset,in_place=True,particle_values_rewritten=False)

class InPlaceDisplayCache(ProbeDisplayCache):
 def finish(self,status):
  self.positions.flush();used=len(self.states);del self.positions
  detail=trim_file(self.output/'particles.npy',used)
  (self.output/'states.json').write_text(json.dumps(self.states,allow_nan=False))
  result=dict(status='captured_probe',capture_origin='live_policy_rollout',probe_status=status,frames=used-1,playback_physics_steps=0,live_inference=False,label='D190 frozen edge-adjacent pair',cache_trim=detail,
   cache_sha256={name:cache_digest(self.output/name) for name in ('particles.npy','states.json','scene.json')})
  (self.output/'reconstruction.json').write_text(json.dumps(result,indent=2)+'\n');return result

def self_check(root):
 root=Path(root);root.mkdir(exist_ok=False);checks=[]
 for capacity,used in [(30001,4655),(30001,21001),(30,1),(30,30)]:
  file=root/f'{capacity}-{used}.npy';a=np.lib.format.open_memmap(file,mode='w+',dtype=np.float32,shape=(capacity,4,3));a[:used]=np.arange(used*12,dtype=np.float32).reshape(used,4,3);expected=a[:used].copy();a.flush();del a
  report=trim_file(file,used);actual=np.load(file);checks.append(np.array_equal(actual,expected) and actual.shape==expected.shape and file.stat().st_size==report['data_offset']+expected.nbytes)
 assert all(checks);return checks

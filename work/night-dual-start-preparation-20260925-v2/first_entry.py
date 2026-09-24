"""Exact first-source entry qualification only; no full candidate or training launch."""
import argparse,copy,json,os,runpy,sys
from pathlib import Path
import numpy as np
from sources import ROOT,W,RUN,FIRST,FIRST_SHA,FIRST_CODE,first_source,sha,read

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args,forward=p.parse_known_args()
 config=args.config.resolve()
 if not config.is_relative_to(RUN):raise ValueError('first_check_config_path')
 spec=read(config)
 if spec['status']!='authorized_first_entry_check' or spec['frames'] not in (0,120):raise ValueError('first_check_only')
 if spec['parameters']!={'boom_scale':1.,'arm_scale':1.,'curl_scale':1.}:raise ValueError('first_check_baseline_only')
 first_source()
 # This is the unchanged first-scoop control/physics family, not the second-scoop loader.
 sys.path.insert(0,str(FIRST_CODE))
 import candidate_multisource as base
 Original=base.MultisourceRuntime
 class BoundFirstRuntime(Original):
  def __init__(self,source,output,**kw):
   if Path(source).resolve()!=FIRST or sha(source)!=FIRST_SHA:raise ValueError('first_exact_source_required')
   if kw.get('enable_truck') is not None or kw.get('freeze_passive_jp') or kw.get('allow_training_reset'):
    raise ValueError('first_source_policy_must_remain_unchanged')
   super().__init__(source,output,**kw)
  def step(self,command):
   if self.frames>=spec['frames']:raise ValueError('first_entry_frame_budget')
   return super().step(command)
  def save(self,path,**kw):
   identity=super().save(path,**kw);side=Path(str(path)+'.json');meta=read(side)
   meta.update(dual_start_entry_check=dict(source_kind='first',maximum_frames=spec['frames'],
      repeated_complete_prepoint=True,not_continuous_scoop_evidence=True,parameters=spec['parameters']),
      entry_check_bindings={str(p):sha(p) for p in [*W.glob('*.py'),config]})
   side.write_text(json.dumps(meta,indent=2)+'\n');return identity
 base.MultisourceRuntime=BoundFirstRuntime
 sys.argv=[str(W/'first_loop.py'),*forward]
 runpy.run_path(str(W/'first_loop.py'),run_name='__main__')
if __name__=='__main__':main()

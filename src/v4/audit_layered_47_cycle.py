#!/usr/bin/env python3
"""Bounded headless holds; each invocation restores in a new process."""
import argparse
import json
from pathlib import Path
from layered_47_cycle_runtime import CycleRuntime


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--frames',type=int,default=30);p.add_argument('--split',type=int,default=0)
    p.add_argument('--truck',action='store_true')
    p.add_argument('--freeze-passive-jp',action='store_true',help='Explicitly freeze unused Jp history; requires exactly zero hardening')
    p.add_argument('--require-no-bucket-contact',action='store_true');args=p.parse_args()
    if not 0<=args.frames<=120 or not 0<=args.split<args.frames:raise ValueError('bounded_audit_frames')
    out=Path(args.output).resolve()
    with CycleRuntime(args.source,out,enable_truck=True if args.truck else None,freeze_passive_jp=args.freeze_passive_jp) as r:
        report=dict(schema='layered_47_cycle_hold_audit_v1',status='running',source=r.metadata['source_identity'],
                    historical_native_history_missing=r.metadata.get('historical_native_history_missing',False),
                    source_metadata=r.metadata['source_metadata_identity'],
                    zero_step=r.restored, frames_requested=args.frames,training_eligible=False,
                    all_hold_frames_zero_bucket_contact=True)
        report['material_history_policy']=r.metadata.get('material_history_policy')
        try:
            for frame in range(args.frames):
                row=r.step([0]*4)
                no_contact=row['bucket_mpm_contact_samples']==0 and row['rock_bucket_contact_substeps']==0
                report['all_hold_frames_zero_bucket_contact'] &= no_contact
                if args.require_no_bucket_contact and not no_contact:
                    raise RuntimeError('cycle_audit_nonzero_bucket_contact')
                if frame+1==args.split:report['split_checkpoint']=r.save(out/'split.npz')
                if (frame+1)%10==0:print('HOLD',frame+1,row['hard_guard'],flush=True)
            report.update(status='hold_passed',final_checkpoint=r.save(out/'final.npz'),frames_completed=r.frames)
        except BaseException as error:
            report.update(status='hold_failed',failure=repr(error),frames_completed=r.frames)
            try:report['failure_checkpoint']=r.save(out/'failure.npz',stage='failure')
            except Exception as nested:report['snapshot_error']=repr(nested)
            raise
        finally:
            (out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='zero_step'},indent=2))


if __name__=='__main__':main()

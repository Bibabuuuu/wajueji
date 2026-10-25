"""Align causal candidate phases to existing action tensors without relabeling keys.

An end-state event at n affects action n+1. This sidecar does not normalize,
change actions/splits, choose loss masks, or grant a candidate training admission.
"""
import numpy as np

PHASES=('approach_contact','penetrate','curl_load','lift','off_soil')


def align_phase_labels(action_frames, phase_trace):
    frames=np.asarray(action_frames)
    if frames.ndim!=1 or not len(frames) or frames.dtype.kind not in 'iu' or np.any(frames<1) or np.any(np.diff(frames.astype(np.int64))<=0):
        raise ValueError('invalid_action_frame_order')
    previous=PHASES[0];before=[]
    for i,row in enumerate(phase_trace,start=1):
        if type(row['frame']) is not int or row['frame']!=i:
            raise ValueError('phase_trace_not_contiguous')
        value=row['phase'];old=value['phase_before_observation'];new=value['phase_for_next_decision']
        if old!=previous or new not in PHASES or value['first_available_decision_frame']!=i+1:
            raise ValueError('noncausal_or_inconsistent_phase_trace')
        event=value['event']
        if new!=old:
            if PHASES.index(new)!=PHASES.index(old)+1 or event is None:
                raise ValueError('phase_transition_skips_or_rewinds')
            if (event['transition']!=old+'_to_'+new or event['confirmation_frame_end']!=i
                    or event['first_available_decision_frame']!=i+1):
                raise ValueError('event_not_next_decision')
        elif event is not None:
            raise ValueError('event_without_transition')
        before.append(PHASES.index(old));previous=new
    if not len(before) or frames[-1]>len(before):
        raise ValueError('phase_trace_does_not_cover_actions')
    ids=np.asarray(before,dtype=np.int8)[frames.astype(np.int64)-1]
    return dict(frame=frames.copy(),phase_id=ids,
                phase_one_hot=np.eye(len(PHASES),dtype=np.float32)[ids],
                latest_allowed_evidence_frame=frames.astype(np.int64)-1,
                terminal_phase=ids==PHASES.index('off_soil'))

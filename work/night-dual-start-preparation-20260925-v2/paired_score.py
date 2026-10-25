"""Paired development selection; every proposed setting must serve both sources."""
import math
SOURCES=('first','second')

def validate(pair):
 if set(pair)!=set(SOURCES):raise ValueError('pair_requires_first_and_second')
 for name,s in pair.items():
  if not s.get('eligible'):return False
  for k in ('working_fill_percent','seconds','command_total_variation'):
   v=s[k]
   if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
    raise ValueError('invalid_pair_metric:'+name+':'+k)
 return True

def assess(candidate,baseline):
 if not validate(baseline):raise ValueError('invalid_paired_baseline')
 if not validate(candidate):return dict(eligible=False,reason='one_or_both_sources_invalid')
 gains={n:candidate[n]['working_fill_percent']-baseline[n]['working_fill_percent'] for n in SOURCES}
 # Same two-percentage-point tolerance as the original development comparison.
 if min(gains.values()) < -2.:return dict(eligible=False,reason='source_fill_regression',gains_pp=gains)
 for n in SOURCES:
  c,b=candidate[n].get('receipt_fraction'),baseline[n].get('receipt_fraction')
  if c is not None and (not math.isfinite(c) or not 0<=c<=1):raise ValueError('invalid_receipt_fraction')
  if c is not None and b is not None and c<b-.02:return dict(eligible=False,reason='source_receipt_regression',gains_pp=gains)
 return dict(eligible=True,gains_pp=gains,minimum_gain_pp=min(gains.values()),mean_gain_pp=sum(gains.values())/2,
   seconds=sum(candidate[n]['seconds'] for n in SOURCES),variation=sum(candidate[n]['command_total_variation'] for n in SOURCES))

def better(candidate,current,baseline):
 c,b=assess(candidate,baseline),assess(current,baseline)
 if not c['eligible']:return False
 if not b['eligible']:return True
 # Protect the worse-served source first, then average improvement.
 for key in ('minimum_gain_pp','mean_gain_pp'):
  delta=c[key]-b[key]
  if abs(delta)>2.:return delta>0
 if c['seconds']<b['seconds']*.98:return True
 if abs(c['seconds']-b['seconds'])<=b['seconds']*.02:return c['variation']<b['variation']
 return False

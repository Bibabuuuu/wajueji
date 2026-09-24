"""Diagnostic comparison only. Never used as a physical guard or state filter."""
import math


def tolerance(path):
    # Absolute tolerances, with SI units named in each field.
    if 'reaction_angular_impulse_nms' in path:return 1e-12  # N m s
    if 'reaction_impulse_ns' in path or 'balance_error_ns' in path:return 1e-12  # N s
    if 'bucket_integral_effort' in path or 'bucket_active_curl_effort' in path:return 1e-9  # N
    if 'mechanism_qvel' in path or '.qvel' in path or 'velocity_world' in path:return 1e-11  # m/s or rad/s
    return 1e-12  # positions m, angles rad, unit quaternion, target speeds, times


def compare(actual, expected, path='', differences=None):
    if differences is None:differences=[]
    if type(actual) is not type(expected):raise ValueError('diagnostic_type:'+path)
    if isinstance(actual,dict):
        if actual.keys()!=expected.keys():raise ValueError('diagnostic_keys:'+path)
        for key in actual:compare(actual[key],expected[key],path+'.'+key,differences)
    elif isinstance(actual,list):
        if len(actual)!=len(expected):raise ValueError('diagnostic_length:'+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+f'[{i}]',differences)
    elif isinstance(actual,float):
        if not math.isfinite(actual) or not math.isfinite(expected):raise ValueError('diagnostic_nonfinite:'+path)
        if actual!=expected:
            error=abs(actual-expected);limit=tolerance(path)
            differences.append(dict(path=path,actual=actual,expected=expected,error=error,absolute_tolerance=limit))
            if error>limit:raise ValueError('diagnostic_numeric_difference:'+path)
    elif actual!=expected:raise ValueError('diagnostic_exact_difference:'+path)
    return differences

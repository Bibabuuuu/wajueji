"""Reuse the exact D162 static-route checks; no soil observation map."""
import importlib.util
from runtime import ROOT
PATH=ROOT/'work/contact-second-fixed-route-20260925-v1/clearance.py'
spec=importlib.util.spec_from_file_location('_night_fixed_route_clearance',PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
FixedRouteClearance=module.FixedRouteClearance

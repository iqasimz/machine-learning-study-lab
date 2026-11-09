"""Locate bundled release lessons, with the original workspace layout as fallback."""
from .artifacts import ROOT

MATERIALS = ROOT / "learning_materials"
if not MATERIALS.is_dir():
    MATERIALS = ROOT.parents[1] / "learning_materials"

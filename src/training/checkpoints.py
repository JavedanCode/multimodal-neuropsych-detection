"""Load trusted local checkpoints after source-module renames."""

import importlib
import sys

import joblib

HISTORICAL_MODULES = {
    "src.models.branch_audit_v6": "src.models.fusion",
    "src.models.primary_fallback_v5": "src.models.fallback",
    "src.models.bounded_v3": "src.models.text_candidates",
    "src.models.quality_stack_v2": "src.models.stacking",
}


def load_checkpoint(path):
    """Load a trusted pickle, resolving only known historical module names.

    Imports stay lazy so the text baseline does not require neural dependencies.
    Pickle is executable; callers must only supply trusted local artifacts.
    """
    while True:
        try:
            return joblib.load(path)
        except ModuleNotFoundError as error:
            replacement = HISTORICAL_MODULES.get(error.name)
            if replacement is None or error.name in sys.modules:
                raise
            sys.modules[error.name] = importlib.import_module(replacement)

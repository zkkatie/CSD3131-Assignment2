"""
Registry of seam finders.

Each method lives in its own file in this folder and defines:

    def find_seam(energy: np.ndarray) -> np.ndarray

To add a method, create the file (e.g. dp.py). It is picked up automatically
if its name is listed in PLANNED_METHODS; no other changes are needed.
"""

import importlib

from . import baseline

# Every method the project will have. Files that don't exist yet are skipped.
PLANNED_METHODS = ("baseline", "dp", "greedy", "graphcut")

METHODS = {"baseline": baseline.find_seam}

for _name in PLANNED_METHODS[1:]:
    try:
        _module = importlib.import_module(f".{_name}", __name__)
    except ModuleNotFoundError as _exc:
        if _exc.name != f"{__name__}.{_name}":
            raise          # the file exists but imports something missing: show the real error
        continue           # the file hasn't been added yet
    METHODS[_name] = _module.find_seam

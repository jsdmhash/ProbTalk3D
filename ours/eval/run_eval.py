"""Runs the authors' evaluation.py unchanged, after the same Python 3.13 / NumPy 2 shims that
render_preview.py uses for chumpy (FLAME loader).
  python ours/eval/run_eval.py folder=model_weights/ProbTalk3D/stage_2 number_of_samples=10
Prints the paper's metrics at the end: MVE, LVE, MEE, CE, FDD, Diversity."""
import inspect, os, runpy, sys
from pathlib import Path
import numpy as np
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for n, t in dict(bool=bool, int=int, float=float, complex=complex, object=object, unicode=str, str=str).items():
    if not hasattr(np, n):
        setattr(np, n, t)
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
sys.argv = [str(ROOT / "evaluation.py")] + sys.argv[1:]
runpy.run_path(str(ROOT / "evaluation.py"), run_name="__main__")

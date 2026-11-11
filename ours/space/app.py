"""
Hugging Face Space entry point. On startup it
  1. clones the project code from GitHub,
  2. downloads the pretrained weights and FLAME file from a PRIVATE HF model repo,
  3. starts the Gradio app (ours/app/app.py).

Space settings -> Variables and secrets:
  secret   HF_TOKEN      a read token that can see the private weights repo
  variable WEIGHTS_REPO  e.g. your-username/probtalk3d-weights
"""
import os
import runpy
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_URL = os.environ.get("REPO_URL", "https://github.com/jsdmhash/ProbTalk3D.git")
BRANCH = os.environ.get("REPO_BRANCH", "colab-inference-fixes")
CODE = Path.home() / "ProbTalk3D"

if not (CODE / "framework").exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(CODE)], check=True)

snapshot_download(os.environ["WEIGHTS_REPO"], repo_type="model", local_dir=str(CODE),
                  token=os.environ.get("HF_TOKEN"))

os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "7860")
runpy.run_path(str(CODE / "ours/app/app.py"), run_name="__main__")

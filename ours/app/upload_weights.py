"""
Upload the pretrained weights (and the FLAME model file) to a PRIVATE Hugging Face
model repo, so the Space can download them at startup without them being public.

  pip install huggingface_hub
  huggingface-cli login            # paste a token with WRITE access
  python ours/app/upload_weights.py <your-hf-username>/probtalk3d-weights
"""
import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
api = HfApi()
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_folder(repo_id=repo_id, folder_path="model_weights/ProbTalk3D",
                  path_in_repo="model_weights/ProbTalk3D")
api.upload_file(repo_id=repo_id, path_or_fileobj="deps/flame/model/generic_model.pkl",
                path_in_repo="deps/flame/model/generic_model.pkl")
print("uploaded to https://huggingface.co/" + repo_id, "(private)")

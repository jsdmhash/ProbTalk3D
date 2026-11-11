"""Create/update the Hugging Face Space and give it access to the private weights.
Usage: python ours/space/deploy_space.py jyomasina/probtalk3d jyomasina/probtalk3d-weights"""
import sys, getpass
from huggingface_hub import HfApi

space_id, weights_repo = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
api.add_space_variable(space_id, "WEIGHTS_REPO", weights_repo)
token = getpass.getpass("Paste a Hugging Face token that can READ the weights repo (hidden): ").strip()
api.add_space_secret(space_id, "HF_TOKEN", token)
api.upload_folder(repo_id=space_id, repo_type="space", folder_path="ours/space",
                  ignore_patterns=["deploy_space.py", "__pycache__/*"])
print("Space created: https://huggingface.co/spaces/" + space_id)
print("Build takes ~10 minutes. Watch the Logs tab on that page.")

"""
Client for the ProbTalk3D service. Calls the running app (local or Hugging Face
Space) over the network and saves the returned video and FLAME parameters.

  python ours/app/client.py --server http://127.0.0.1:7860 --wav results/generation/test_audio/happy.wav
  python ours/app/client.py --server <hf-user>/<space-name> --emotion angry --intensity high --takes 3

For a private Space, set HF_TOKEN in the environment.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from gradio_client import Client, handle_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="URL or Hugging Face Space id (user/space)")
    ap.add_argument("--wav", default="results/generation/test_audio/happy.wav")
    ap.add_argument("--speaker", default="M009")
    ap.add_argument("--emotion", default="happy")
    ap.add_argument("--intensity", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--takes", type=int, default=1)
    ap.add_argument("--out", default="client_outputs")
    a = ap.parse_args()

    client = Client(a.server, hf_token=os.environ.get("HF_TOKEN"))
    t0 = time.time()
    video, npy, info = client.predict(handle_file(a.wav), a.speaker, a.emotion, a.intensity,
                                      a.temperature, not a.deterministic, a.takes, api_name="/generate")
    elapsed = time.time() - t0

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    video_src = video["video"] if isinstance(video, dict) else video
    shutil.copy(video_src, out / Path(video_src).name)
    shutil.copy(npy, out / Path(npy).name)
    info = info if isinstance(info, dict) else json.loads(info)
    info["round_trip_seconds"] = round(elapsed, 2)
    print(json.dumps(info, indent=2))
    print("saved to", out.resolve())


if __name__ == "__main__":
    main()

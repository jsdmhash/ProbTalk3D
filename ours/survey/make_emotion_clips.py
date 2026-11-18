"""Silent emotion-recognition clips for the survey (Part 2). Same speaker (M009), same sentence,
high intensity, deterministic; only the conditioned emotion differs. Order is shuffled; the key is
written to answer_key.txt (do not share it with respondents)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ours"))
from face_render import FaceRenderer
order = ["surprised", "sad", "happy", "disgusted", "angry", "fear"]
r = FaceRenderer(size=540)
key = []
for i, emo in enumerate(order, 1):
    hits = sorted((ROOT / "ours/runs").rglob(f"neutral_M009_{emo}_high*.npy"))
    if not hits:
        print("MISSING", emo); continue
    r.video(np.load(hits[0]).reshape(-1, 403), ROOT / f"ours/survey/clip_B{i}.mp4")
    key.append(f"B{i} = {emo}   (from {hits[0].relative_to(ROOT)})")
    print("made", f"clip_B{i}.mp4", flush=True)
key += ["C1 = sampling ON, two different takes (T=0.2)", "C2 = sampling OFF, identical takes (control)"]
(ROOT / "ours/survey/answer_key.txt").write_text("\n".join(key) + "\n")
print("done")

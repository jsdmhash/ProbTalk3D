"""
Checks that the service engine produces exactly what the authors' generation.py
produces. Deterministic mode (sample=false) has no randomness, so the two must match.

1) make the reference with the authors' script:
   python generation.py folder=model_weights/ProbTalk3D/stage_2 input_path=ours/logs/smoke_audio \
       id=[\"M009\"] emotion=[\"happy\"] intensity=[2] sample=false
2) run this check:
   python ours/app/smoke_test.py results/generation/vqvae_pred/stage_2/none/happy_M009_happy_high.npy
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import ProbTalkEngine, ROOT  # noqa: E402

ref_path = sys.argv[1]
wav = ROOT / "ours/logs/smoke_audio/happy.wav"

eng = ProbTalkEngine()
print(f"model load: {eng.load_seconds:.2f} s on {eng.device}")
seqs, key, timing = eng.generate(str(wav), "M009", "happy", "high", temperature=0.2, sample=False)
ours = seqs[0][None]
ref = np.load(ref_path)
print("style key:", key, "| timing:", timing)
print("engine shape:", ours.shape, "| generation.py shape:", ref.shape)
if ours.shape == ref.shape:
    d = np.abs(ours - ref).max()
    print(f"max abs difference vs generation.py: {d:.3e}")
    print("MATCH" if d < 1e-5 else "MISMATCH")
else:
    print("MISMATCH (shape)")

t0 = time.time()
s, _, t = eng.generate(str(wav), "M009", "happy", "medium", temperature=0.2, sample=True, takes=2)
print("sampled 2 takes:", t, "| takes differ:", not np.allclose(s[0], s[1]))

"""Proves each downloaded MEAD audio file belongs to the speaker it was named after.
Every clip has its own length. Audio labelled speaker A is compared with the 3D ground truth
of every speaker B (clip seconds x 25 fps vs 3D frame count, same sentence/emotion/intensity).
The right pairing matches within ~1 frame; wrong pairings are off by many frames."""
from collections import defaultdict
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[2]
WAV, PARAM = ROOT / "datasets/mead/wav", ROOT / "datasets/mead/param"
def keyset(folder, ext):
    d = defaultdict(dict)
    for f in folder.glob(f"*{ext}"):
        spk, rest = f.stem.split("_", 1); d[spk][rest] = f
    return d
wavs, params = keyset(WAV, ".wav"), keyset(PARAM, ".npy")
speakers = sorted(set(wavs) & set(params))
audio = {s: {k: sf.info(str(f)).duration * 25 for k, f in wavs[s].items()} for s in speakers}
gt = {s: {k: np.load(f, mmap_mode="r").shape[1] for k, f in params[s].items()} for s in speakers}
table = np.array([[np.median([abs(audio[a][k] - gt[b][k]) for k in set(audio[a]) & set(gt[b])]) for b in speakers] for a in speakers])
print("median |audio frames - 3D frames|  (rows = audio file label, cols = 3D speaker)")
print("        " + " ".join(f"{s:>6}" for s in speakers))
ok = True
for i, a in enumerate(speakers):
    best = speakers[int(np.argmin(table[i]))]; ok &= best == a
    print(f"{a:>6}  " + " ".join(f"{v:6.1f}" for v in table[i]) + ("   OK" if best == a else f"   MISMATCH -> looks like {best}"))
print("\nALL LABELS CORRECT" if ok else "\nSOME LABELS WRONG")

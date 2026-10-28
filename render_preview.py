import sys, os, inspect, subprocess
import numpy as np
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
import warnings; warnings.filterwarnings("ignore")
for n, t in dict(bool=bool, int=int, float=float, complex=complex, object=object, unicode=str, str=str).items():
    if not hasattr(np, n):
        setattr(np, n, t)
root, npy, out = sys.argv[1], sys.argv[2], sys.argv[3]
audio = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
max_frames = int(sys.argv[5]) if len(sys.argv) > 5 else None
sys.path.insert(0, root)
import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import cv2
from deps.flame.flame_pytorch import FLAME, get_config

seq = np.squeeze(np.load(npy))                      # (T, 403)
if max_frames:
    seq = seq[:max_frames]
T = seq.shape[0]
m = torch.tensor(seq, dtype=torch.float32)
flame = FLAME(get_config(batch_size=T))
with torch.no_grad():
    verts, _ = flame(m[:, :300], m[:, 300:400],
                     torch.cat([torch.zeros(T, 3), m[:, 400:]], dim=1))
verts = verts.numpy()
faces = flame.faces_tensor.numpy()

lo = verts[:, :, :2].reshape(-1, 2).min(0); hi = verts[:, :, :2].reshape(-1, 2).max(0)
c = (lo + hi) / 2; r = (hi - lo).max() / 2 * 1.05
W, H = 540, 540
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_aspect("equal")
fig.patch.set_facecolor("#F4F1EC")
light = np.array([0.3, 0.4, 1.0]); light /= np.linalg.norm(light)
tmp = out.replace(".mp4", "_noaudio.mp4")
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
for t in range(T):
    v = verts[t]; tri = v[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    vis = n[:, 2] > 0
    shade = np.clip(n @ light, 0, 1) * 0.75 + 0.25
    order = np.argsort(tri[vis][:, :, 2].mean(1))
    polys = tri[vis][order][:, :, :2]
    cols = np.stack([shade[vis][order] * 0.86, shade[vis][order] * 0.74, shade[vis][order] * 0.66], 1)
    pc = PolyCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.2)
    ax.add_collection(pc)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    pc.remove()
    if t == 0 or t == T // 2:
        plt.imsave(out.replace(".mp4", f"_frame{t}.png"), img)
vw.release()
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp] + (["-i", audio] if audio else []) + \
      ["-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac", "-shortest"] if audio else []) + [out]
subprocess.run(cmd, check=True)
print("wrote", out, "frames:", T)

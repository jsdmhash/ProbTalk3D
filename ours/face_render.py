"""
Clear face renderer for ProbTalk3D outputs (no Blender, no torch).

What it adds over render_preview.py, so expression changes are easy to see:
  * eyes drawn as eyes (white sclera, dark iris) instead of skin-coloured balls
  * lips in their own colour
  * the inside of the mouth drawn dark, instead of a see-through hole
  * one fixed camera for every clip, so frames from different runs line up

Usage:
  python ours/face_render.py <file.npy> <out.mp4> [--wav audio.wav] [--size 512]
  python ours/face_render.py <file.npy> <out.png> --frame 60     # single frame
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flame_np import FlameNP, LEFT_EYE, RIGHT_EYE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SKIN = np.array([0.87, 0.74, 0.65])
LIPS = np.array([0.72, 0.36, 0.36])
SCLERA = np.array([0.96, 0.96, 0.94])
IRIS = np.array([0.20, 0.13, 0.09])
INSIDE = np.array([0.22, 0.05, 0.06])
BACKGROUND = "#F4F1EC"
LIGHT = np.array([0.25, 0.35, 1.0]) / np.linalg.norm([0.25, 0.35, 1.0])

# fixed framing (metres, FLAME template coordinates): forehead to chin
VIEW_X = (-0.105, 0.105)
VIEW_Y = (-0.135, 0.085)


def _read_ids(path):
    return np.array([int(x) for x in Path(path).read_text().split(",") if x.strip()])


class FaceRenderer:
    def __init__(self, flame_model_path=None, size=512):
        self.flame = FlameNP(flame_model_path)
        self.faces = self.flame.faces
        self.size = size
        V = self.flame.v_template.shape[0]
        tmpl = self.flame.v_template

        # soft lip weight per vertex (0 = skin, 1 = lip), blended per face for smooth edges
        lip_w = np.zeros(V)
        mouth, w = self._lip_weights(tmpl)
        lip_w[mouth] = w
        self.face_lip = lip_w[self.faces].mean(1)

        # eyes: 0 none, 1 sclera, 2 iris
        eye = np.zeros(V, dtype=np.int64)
        for ids in (LEFT_EYE, RIGHT_EYE):
            d = tmpl[ids] - tmpl[ids].mean(0)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            eye[ids] = 1
            eye[ids[d[:, 2] > 0.86]] = 2          # front cap of each eyeball = iris
        self.face_eye = np.array([np.bincount(r, minlength=3).argmax() for r in eye[self.faces]])

    @staticmethod
    def _lip_weights(tmpl):
        """Lips = the part of the authors' mouth region (datasets/regions/lve.txt)
        inside an ellipse around the mouth line, with a soft edge."""
        mouth = _read_ids(ROOT / "datasets/regions/lve.txt")
        p = tmpl[mouth]
        front = p[:, 2] > np.percentile(p[:, 2], 85)
        y0 = np.median(p[front, 1])                       # mouth line height
        x_r, up_r, low_r = 0.0255, 0.0085, 0.0105         # half-width, upper/lower lip height
        dy = p[:, 1] - y0
        r = np.sqrt((p[:, 0] / x_r) ** 2 + (dy / np.where(dy > 0, up_r, low_r)) ** 2)
        w = np.clip((1.15 - r) / 0.35, 0, 1)              # 1 inside, fades out at the edge
        w[p[:, 2] < np.percentile(p[:, 2], 40)] = 0       # ignore the inner mouth surface
        return mouth, w

    def vertices(self, seq):
        return self.flame.forward(seq)

    def _draw(self, ax, v):
        tri = v[self.faces]                                   # (F, 3, 3)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        front = n[:, 2] > 0
        shade = np.clip(n @ LIGHT, 0, 1) * 0.7 + 0.3
        base = SKIN[None] * (1 - self.face_lip[:, None]) + LIPS[None] * self.face_lip[:, None]
        base[self.face_eye == 1] = SCLERA
        cols = base * shade[:, None]
        iris = self.face_eye == 2
        cols[iris] = IRIS * (0.6 + 0.4 * shade[iris, None])
        # back faces seen through the open mouth -> dark mouth interior
        cols[~front] = INSIDE * (0.6 + 0.4 * np.abs(n[~front, 2:3]))
        order = np.argsort(tri[:, :, 2].mean(1))              # painter's algorithm
        pc = PolyCollection(tri[order][:, :, :2], facecolors=np.clip(cols[order], 0, 1),
                            edgecolors=np.clip(cols[order], 0, 1), linewidths=0.25)
        ax.add_collection(pc)
        return pc

    def _figure(self):
        s = self.size
        fig = plt.figure(figsize=(s / 100, s / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(*VIEW_X)
        ax.set_ylim(*VIEW_Y)
        ax.set_aspect("equal")
        fig.patch.set_facecolor(BACKGROUND)
        return fig, ax

    def frames(self, seq):
        """Yield RGB uint8 frames for a (T, 403) sequence."""
        verts = self.vertices(seq)
        fig, ax = self._figure()
        try:
            for v in verts:
                pc = self._draw(ax, v)
                fig.canvas.draw()
                yield np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                pc.remove()
        finally:
            plt.close(fig)

    def frame(self, seq, index):
        seq = np.asarray(seq).reshape(-1, 403)
        index = min(index, len(seq) - 1)
        return next(self.frames(seq[index:index + 1]))

    def video(self, seq, out_mp4, wav=None, fps=25):
        """Render one sequence, or a list of sequences side by side (same audio)."""
        seqs = seq if isinstance(seq, (list, tuple)) else [seq]
        seqs = [np.asarray(x).reshape(-1, 403) for x in seqs]
        T = min(len(x) for x in seqs)
        out_mp4 = str(out_mp4)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.rgb"
            n = 0
            with open(raw, "wb") as fh:
                for imgs in zip(*[self.frames(x[:T]) for x in seqs]):
                    fh.write(np.concatenate(imgs, axis=1).tobytes())
                    n += 1
            h, w = self.size, self.size * len(seqs)
            cmd = ["ffmpeg", "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                   "-i", str(raw)]
            if wav:
                cmd += ["-i", str(wav)]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
            if wav:
                cmd += ["-c:a", "aac", "-shortest"]
            cmd += [out_mp4]
            subprocess.run(cmd, check=True)
        return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npy")
    ap.add_argument("out")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--frame", type=int, default=None, help="save one PNG frame instead of a video")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--flame", default=None, help="path to generic_model.pkl (default: deps/flame/model)")
    a = ap.parse_args()

    r = FaceRenderer(a.flame, a.size)
    seq = np.load(a.npy).reshape(-1, 403)
    if a.frame is not None:
        plt.imsave(a.out, r.frame(seq, a.frame))
        print("wrote", a.out)
    else:
        n = r.video(seq, a.out, a.wav)
        print("wrote", a.out, "frames:", n)


if __name__ == "__main__":
    main()

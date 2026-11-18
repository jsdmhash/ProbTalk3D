"""
Clear, HD face renderer for ProbTalk3D outputs (no Blender, no torch).

Everything that moves comes from the model's FLAME output. On top of the bare FLAME
head we draw a stylised avatar so expressions are easy to read:
  * smooth (per-vertex) skin shading with soft key light and a little specular
  * eyes with white sclera, brown iris and dark pupil
  * lips in their own colour, dark mouth interior instead of a see-through hole
  * eyebrows and upper-lid lines drawn through FLAME's own facial landmarks, so they
    rise, fall and squint exactly as the model moves the face
  * a stylised hair cap (fixed to the scalp; FLAME has no hair geometry)
  * one fixed camera for every clip, so frames from different runs line up

Usage:
  python ours/face_render.py <file.npy> <out.mp4> [--wav audio.wav] [--size 720]
  python ours/face_render.py <file.npy> <out.png> --frame 60     # single frame
"""
import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

try:                       # optional: edge-preserving smoothing hides the mesh triangles
    import cv2
except ImportError:        # renderer still works without OpenCV, just less smooth
    cv2 = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flame_np import FlameNP, LEFT_EYE, RIGHT_EYE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SKIN = np.array([0.93, 0.76, 0.66])
LIPS = np.array([0.76, 0.40, 0.40])
HAIR = np.array([0.17, 0.11, 0.08])
SCLERA = np.array([0.97, 0.96, 0.94])
IRIS = np.array([0.36, 0.22, 0.12])
PUPIL = np.array([0.05, 0.03, 0.03])
INSIDE = np.array([0.24, 0.06, 0.07])
BROW = (0.20, 0.13, 0.09)
LASH = (0.10, 0.07, 0.06)

KEY = np.array([0.35, 0.45, 1.0]); KEY /= np.linalg.norm(KEY)
FILL = np.array([-0.6, 0.1, 0.8]); FILL /= np.linalg.norm(FILL)
VIEW = np.array([0.0, 0.0, 1.0])

# fixed framing (metres, FLAME template coordinates): top of hair to chin
VIEW_X = (-0.140, 0.140)   # square view: background fills the whole frame
VIEW_Y = (-0.140, 0.140)
BACKGROUNDS = {"light": ([0.93, 0.94, 0.96], [0.80, 0.83, 0.88]),
               "dark": ([0.20, 0.19, 0.31], [0.07, 0.08, 0.14])}


def _read_ids(path):
    return np.array([int(x) for x in Path(path).read_text().split(",") if x.strip()])


_ENCODER = None


def _video_encoder_args():
    """Pick an H.264 encoder this ffmpeg has (Fedora's ffmpeg-free lacks libx264)."""
    global _ENCODER
    if _ENCODER is None:
        try:
            enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        except OSError:
            enc = ""
        if " libx264 " in enc:
            _ENCODER = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        elif " libopenh264 " in enc:
            _ENCODER = ["-c:v", "libopenh264", "-b:v", "4M"]
        else:
            _ENCODER = ["-c:v", "mpeg4", "-q:v", "3"]
    return _ENCODER


def _smooth_curve(p, n=3):
    """Chaikin corner cutting: a smooth line through a few landmark points."""
    for _ in range(n):
        q = np.empty((2 * (len(p) - 1), p.shape[1]))
        q[0::2] = 0.75 * p[:-1] + 0.25 * p[1:]
        q[1::2] = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], q, p[-1:]])
    return p


def _stroke(p, w0, w1):
    """Polygon for a tapered stroke along 2D polyline p (width w0 -> w1)."""
    d = np.gradient(p, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    w = np.linspace(w0, w1, len(p))[:, None]
    taper = np.sin(np.linspace(0.15, np.pi - 0.15, len(p)))[:, None] ** 0.5
    return np.vstack([p + nrm * w * taper, (p - nrm * w * taper)[::-1]])


class FaceRenderer:
    def __init__(self, flame_model_path=None, size=720, background="light"):
        self.flame = FlameNP(flame_model_path)
        self.faces = self.flame.faces
        self.size = size
        tmpl = self.flame.v_template
        V = tmpl.shape[0]

        # landmarks (FLAME static embedding: 0-9 brows, 19-30 eyes)
        with open(ROOT / "deps/flame/model/flame_static_embedding.pkl", "rb") as f:
            emb = pickle.load(f, encoding="latin1")
        self.lmk_faces = self.faces[np.asarray(emb["lmk_face_idx"]).astype(np.int64)]
        self.lmk_bary = np.asarray(emb["lmk_b_coords"], dtype=np.float64)

        # per-vertex base colour: skin, blended lips and hair
        col = np.tile(SKIN, (V, 1))
        mouth, lw = self._lip_weights(tmpl)
        col[mouth] = col[mouth] * (1 - lw[:, None]) + LIPS * lw[:, None]
        hw = self._hair_weights(tmpl)
        col = col * (1 - hw[:, None]) + HAIR * hw[:, None]
        self.vert_col = col
        self.vert_spec = 0.10 * (1 - hw) + 0.04 * hw

        # eyes: 0 none, 1 sclera, 2 iris, 3 pupil
        eye = np.zeros(V, dtype=np.int64)
        for ids in (LEFT_EYE, RIGHT_EYE):
            d = tmpl[ids] - tmpl[ids].mean(0)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            eye[ids] = 1
            eye[ids[d[:, 2] > 0.86]] = 2
            eye[ids[d[:, 2] > 0.965]] = 3
        self.face_eye = np.array([np.bincount(r, minlength=4).argmax() for r in eye[self.faces]])

        # background: soft vertical gradient
        g = np.linspace(0, 1, 256)[:, None, None]
        top, bottom = BACKGROUNDS.get(background, BACKGROUNDS["light"])
        self.bg = (np.array(top) * (1 - g) + np.array(bottom) * g)

    @staticmethod
    def _lip_weights(tmpl):
        mouth = _read_ids(ROOT / "datasets/regions/lve.txt")
        p = tmpl[mouth]
        front = p[:, 2] > np.percentile(p[:, 2], 85)
        y0 = np.median(p[front, 1])
        x_r, up_r, low_r = 0.0255, 0.0085, 0.0105
        dy = p[:, 1] - y0
        r = np.sqrt((p[:, 0] / x_r) ** 2 + (dy / np.where(dy > 0, up_r, low_r)) ** 2)
        w = np.clip((1.15 - r) / 0.35, 0, 1)
        w[p[:, 2] < np.percentile(p[:, 2], 40)] = 0
        return mouth, w

    @staticmethod
    def _hair_weights(tmpl):
        x, y, z = tmpl[:, 0], tmpl[:, 1], tmpl[:, 2]
        hairline = 0.093 - 5.5 * x ** 2                    # arched front hairline
        side = (np.abs(x) > 0.072) & (z < 0.0)             # above and behind the ears
        hairline = np.where(side, np.maximum(0.047, hairline), hairline)
        hairline = np.where(z < -0.035, -0.02, hairline)   # back of the head
        w = np.clip((y - hairline) / 0.016, 0, 1) ** 0.8
        # ears stay skin: the whole ear (front and back of the outer ear) is masked out
        ear = (np.abs(x) > 0.066) & (y > -0.05) & (y < 0.072) & (z > -0.09)
        w[ear] = 0
        return w

    def vertices(self, seq):
        return self.flame.forward(seq)

    def _landmarks(self, v):
        return np.einsum("lk,lkc->lc", self.lmk_bary, v[self.lmk_faces])

    def _draw(self, ax, v):
        tri = v[self.faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, self.faces[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12

        diff = 0.62 * np.clip(vn @ KEY, 0, 1) + 0.18 * np.clip(vn @ FILL, 0, 1) + 0.28
        half = (KEY + VIEW) / np.linalg.norm(KEY + VIEW)
        spec = self.vert_spec * np.clip(vn @ half, 0, 1) ** 24
        vc = np.clip(self.vert_col * diff[:, None] + spec[:, None], 0, 1)
        cols = vc[self.faces].mean(1)                      # Gouraud-style smooth colour

        shade = 0.55 + 0.45 * np.clip(fn @ KEY, 0, 1)
        e = self.face_eye
        cols[e == 1] = SCLERA * (0.75 + 0.25 * shade[e == 1, None])
        cols[e == 2] = IRIS * (0.7 + 0.3 * shade[e == 2, None])
        cols[e == 3] = PUPIL
        front = fn[:, 2] > 0
        cols[~front] = INSIDE * (0.6 + 0.4 * np.abs(fn[~front, 2:3]))

        order = np.argsort(tri[:, :, 2].mean(1))
        cols = np.clip(cols[order], 0, 1)
        pc = PolyCollection(tri[order][:, :, :2], facecolors=cols, edgecolors=cols,
                            linewidths=0.35, antialiased=True)
        ax.add_collection(pc)
        artists = [pc]

        # eyebrows and upper-lid lines from FLAME landmarks (move with the face)
        L = self._landmarks(v)[:, :2]
        s = (VIEW_X[1] - VIEW_X[0])
        for ids, w0, w1 in (([4, 3, 2, 1, 0], 0.0050, 0.0026), ([5, 6, 7, 8, 9], 0.0050, 0.0026)):
            p = _smooth_curve(L[ids] + np.array([0, 0.0012]))
            poly = PolyCollection([_stroke(p, w0, w1)], facecolors=[BROW], edgecolors=[BROW],
                                  linewidths=0.6, alpha=0.93)
            ax.add_collection(poly); artists.append(poly)
        for ids in ([19, 20, 21, 22], [25, 26, 27, 28]):
            p = _smooth_curve(L[ids] + np.array([0, 0.0006]), 2)
            line, = ax.plot(p[:, 0], p[:, 1], color=LASH, lw=self.size / s * 0.0011,
                            solid_capstyle="round", alpha=0.9)
            artists.append(line)
        return artists

    def _figure(self):
        s = self.size
        fig = plt.figure(figsize=(s / 100, s / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.imshow(self.bg, extent=[VIEW_X[0], VIEW_X[1], VIEW_Y[0], VIEW_Y[1]],
                  origin="upper", aspect="auto", zorder=-10)
        ax.set_xlim(*VIEW_X)
        ax.set_ylim(*VIEW_Y)
        ax.set_aspect("equal")
        return fig, ax

    def frames(self, seq):
        """Yield RGB uint8 frames for a (T, 403) sequence."""
        verts = self.vertices(seq)
        fig, ax = self._figure()
        try:
            for v in verts:
                arts = self._draw(ax, v)
                fig.canvas.draw()
                img = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
                if cv2 is not None:
                    img = cv2.bilateralFilter(cv2.bilateralFilter(img, 9, 22, 7), 9, 18, 7)
                yield img.copy()
                for a in arts:
                    a.remove()
        finally:
            plt.close(fig)

    def frame(self, seq, index):
        seq = np.asarray(seq).reshape(-1, 403)
        index = min(index, len(seq) - 1)
        return next(self.frames(seq[index:index + 1]))

    def _render_raw(self, seqs, raw_path, start, stop):
        """Write frames start..stop (side by side for several takes) as raw RGB."""
        with open(raw_path, "wb") as fh:
            for imgs in zip(*[self.frames(x[start:stop]) for x in seqs]):
                fh.write(np.concatenate(imgs, axis=1).tobytes())

    def video(self, seq, out_mp4, wav=None, fps=25, workers=None):
        """Render one sequence, or a list of sequences side by side (same audio).

        workers > 1 splits the frames across CPU cores (Linux fork); falls back to 1 on error."""
        seqs = seq if isinstance(seq, (list, tuple)) else [seq]
        seqs = [np.asarray(x).reshape(-1, 403) for x in seqs]
        T = min(len(x) for x in seqs)
        if workers is None:
            workers = int(os.environ.get("RENDER_WORKERS", "0")) or (os.cpu_count() or 1)
        workers = max(1, min(workers, T // 20 or 1))
        out_mp4 = str(out_mp4)
        with tempfile.TemporaryDirectory() as tmp:
            bounds = np.linspace(0, T, workers + 1).astype(int)
            parts = [Path(tmp) / f"part{i}.rgb" for i in range(workers)]
            done = False
            if workers > 1:
                try:
                    import multiprocessing as mp
                    ctx = mp.get_context("fork")
                    procs = [ctx.Process(target=self._render_raw, args=(seqs, str(parts[i]), bounds[i], bounds[i + 1]))
                             for i in range(workers)]
                    for pr in procs:
                        pr.start()
                    for pr in procs:
                        pr.join()
                    done = all(pr.exitcode == 0 for pr in procs)
                except Exception:
                    done = False
            if not done:
                parts = [Path(tmp) / "raw.rgb"]
                self._render_raw(seqs, str(parts[0]), 0, T)
            raw = Path(tmp) / "all.rgb"
            with open(raw, "wb") as out:
                for part in parts:
                    with open(part, "rb") as fh:
                        while True:
                            buf = fh.read(1 << 24)
                            if not buf:
                                break
                            out.write(buf)
            h, w = self.size, self.size * len(seqs)
            cmd = ["ffmpeg", "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                   "-i", str(raw)]
            if wav:
                cmd += ["-i", str(wav)]
            cmd += _video_encoder_args() + ["-pix_fmt", "yuv420p"]
            if wav:
                cmd += ["-c:a", "aac", "-shortest"]
            cmd += [out_mp4]
            subprocess.run(cmd, check=True)
        return T

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npy")
    ap.add_argument("out")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--frame", type=int, default=None, help="save one PNG frame instead of a video")
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--background", choices=["light", "dark"], default="light")
    ap.add_argument("--flame", default=None, help="path to generic_model.pkl (default: deps/flame/model)")
    a = ap.parse_args()

    r = FaceRenderer(a.flame, a.size, a.background)
    seq = np.load(a.npy).reshape(-1, 403)
    if a.frame is not None:
        plt.imsave(a.out, r.frame(seq, a.frame))
        print("wrote", a.out)
    else:
        n = r.video(seq, a.out, a.wav)
        print("wrote", a.out, "frames:", n)


if __name__ == "__main__":
    main()

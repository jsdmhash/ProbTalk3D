"""
Controlled comparisons for ProbTalk3D (Task 2).

Generates isolated runs with explicit id/emotion/intensity overrides via
generation.py, moves each run's .npy into its own folder under ours/runs/,
and renders with ours/face_render.py:
  a) emotion_grid.png          - 8 conditioned emotions on neutral.wav (M009, high)
  b) intensity_row.png         - happy/angry/surprised x low/medium/high
  c) displacement_heatmaps.png - mean mm displacement from neutral + shared colorbar
     emotion_displacement.csv  - mouth (lve.txt) & upper face (fdd.txt) averages
  d) diversity_row.png + diversity_3takes.mp4 - 3 stochastic takes at T=0.2
"""
import argparse
import csv
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from face_render import BACKGROUND, INSIDE, LIGHT, VIEW_X, VIEW_Y, FaceRenderer, _read_ids  # noqa: E402

AUDIO_SRC = ROOT / "results/generation/test_audio/neutral.wav"
NEUTRAL_AUDIO_DIR = ROOT / "ours/runs/neutral_audio"
GEN_STAGE2_BASE = ROOT / "results/generation/vqvae_pred/stage_2"
RUNS_DIR = ROOT / "ours/runs"
RESULTS_DIR = ROOT / "ours/results"

SPEAKER = "M009"
EMOTIONS = ["neutral", "happy", "sad", "surprised", "fear", "disgusted", "angry", "contempt"]
INTENSITIES = [(0, "low"), (1, "medium"), (2, "high")]
INTENSITY_EMOTIONS = ["happy", "angry", "surprised"]


def ensure_isolated_audio() -> Path:
    NEUTRAL_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for existing in NEUTRAL_AUDIO_DIR.glob("*.wav"):
        if existing.name != AUDIO_SRC.name:
            existing.unlink()
    dst = NEUTRAL_AUDIO_DIR / AUDIO_SRC.name
    if not dst.exists():
        shutil.copy2(AUDIO_SRC, dst)
    return dst


def run_generation(
    run_dir: Path,
    emotion: str,
    intensity: int,
    sample: bool,
    temperature: float = 0.2,
    num_samples: int = 1,
    speaker: str = SPEAKER,
    skip_existing: bool = True,
) -> list[Path]:
    """Run generation.py for one isolated configuration and move .npy outputs into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    int_label = INTENSITIES[intensity][1]
    stem = f"neutral_{speaker}_{emotion}_{int_label}"
    names = [f"{stem}.npy"] if num_samples == 1 else [f"{stem}({k}).npy" for k in range(1, num_samples + 1)]
    dst_paths = [run_dir / name for name in names]

    if skip_existing and all(p.exists() for p in dst_paths):
        print(f"[skip] {run_dir.relative_to(ROOT)} already has {len(dst_paths)} npy file(s)", flush=True)
        return dst_paths

    subdir = "none" if not sample else f"{temperature}"
    if num_samples > 1:
        subdir += "_multi"
    gen_dir = GEN_STAGE2_BASE / subdir
    for name in names:
        stale = gen_dir / name
        if stale.exists():
            stale.unlink()

    cmd = [
        sys.executable,
        str(ROOT / "generation.py"),
        "folder=model_weights/ProbTalk3D/stage_2",
        f"input_path={NEUTRAL_AUDIO_DIR.relative_to(ROOT)}",
        f'id=["{speaker}"]',
        f'emotion=["{emotion}"]',
        f"intensity=[{intensity}]",
        f"sample={'true' if sample else 'false'}",
        f"temperature={temperature}",
        f"number_of_samples={num_samples}",
    ]
    print(f"[run] {shlex.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
        print(proc.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"generation.py failed (exit {proc.returncode}) for {stem}")

    for name, dst in zip(names, dst_paths):
        src = gen_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Expected output {src} not produced by generation.py")
        shutil.move(str(src), str(dst))
    print(f"[ok] {run_dir.relative_to(ROOT)} ({elapsed:.2f} s) -> {[p.name for p in dst_paths]}", flush=True)
    return dst_paths


def choose_open_mouth_frame(seqs_by_emotion: dict[str, np.ndarray], margin: int = 15) -> int:
    """Pick the frame index with largest mean jaw-opening rotation across emotions."""
    T = min(s.shape[0] for s in seqs_by_emotion.values())
    lo = min(margin, max(0, T // 4))
    hi = max(lo + 1, T - lo)
    jaw_x = np.mean([s[:T, 400] for s in seqs_by_emotion.values()], axis=0)
    idx = int(lo + np.argmax(jaw_x[lo:hi]))
    print(
        f"[frame] auto-selected open-mouth frame {idx}/{T - 1} "
        f"(mean jaw_x={jaw_x[idx]:.4f} rad; happy jaw_x={seqs_by_emotion['happy'][idx, 400]:.4f}, "
        f"surprised jaw_x={seqs_by_emotion['surprised'][idx, 400]:.4f})",
        flush=True,
    )
    return idx


def save_grid(
    tiles: list[list[np.ndarray]],
    titles: list[list[str]],
    out_path: Path,
    suptitle: str,
    row_labels: list[str] | None = None,
    dpi: int = 160,
) -> None:
    rows, cols = len(tiles), len(tiles[0])
    fig_w = cols * 3.1
    header_in = 0.95 if "\n" in suptitle else 0.72
    fig_h = rows * 3.15 + header_in
    top_frac = 1.0 - header_in / fig_h
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), facecolor=BACKGROUND)
    axes_2d = np.atleast_2d(axes).reshape(rows, cols)
    for r in range(rows):
        for c in range(cols):
            ax = axes_2d[r, c]
            ax.imshow(tiles[r][c])
            ax.set_title(titles[r][c], fontsize=11, fontweight="bold", pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        if row_labels:
            axes_2d[r, 0].set_ylabel(row_labels[r], fontsize=12, fontweight="bold", labelpad=10)
    fig.suptitle(suptitle, fontsize=12.5, fontweight="bold", y=1.0 - 0.14 / fig_h, va="top")
    fig.subplots_adjust(left=0.06 if row_labels else 0.02, right=0.98, top=top_frac, bottom=0.02, wspace=0.06, hspace=0.18)
    fig.savefig(out_path, dpi=dpi, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[wrote] {out_path.relative_to(ROOT)}", flush=True)


def render_displacement_face(
    renderer: FaceRenderer,
    verts_frame: np.ndarray,
    mean_dist_mm: np.ndarray,
    norm: Normalize,
    cmap,
) -> np.ndarray:
    """Render neutral-condition mesh colored by per-vertex mm displacement."""
    tri = verts_frame[renderer.faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    front = n[:, 2] > 0
    shade = np.clip(n @ LIGHT, 0, 1) * 0.35 + 0.65

    face_val = mean_dist_mm[renderer.faces].mean(axis=1)
    cols = cmap(norm(face_val))[:, :3] * shade[:, None]
    cols[~front] = INSIDE * (0.6 + 0.4 * np.abs(n[~front, 2:3]))

    order = np.argsort(tri[:, :, 2].mean(1))
    fig, ax = renderer._figure()
    try:
        pc = PolyCollection(
            tri[order][:, :, :2],
            facecolors=np.clip(cols[order], 0, 1),
            edgecolors=np.clip(cols[order], 0, 1),
            linewidths=0.25,
        )
        ax.add_collection(pc)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    finally:
        plt.close(fig)


def build_displacement_heatmaps(
    renderer: FaceRenderer,
    seqs_by_emotion: dict[str, np.ndarray],
    frame_idx: int,
    csv_path: Path,
    png_path: Path,
) -> None:
    """Compute per-vertex mm displacement vs neutral, write CSV, and render shared heatmap grid."""
    lve_ids = _read_ids(ROOT / "datasets/regions/lve.txt")
    fdd_ids = _read_ids(ROOT / "datasets/regions/fdd.txt")
    print(f"[regions] lve_ids={len(lve_ids)} fdd_ids={len(fdd_ids)}", flush=True)

    verts_by_emo = {emo: renderer.vertices(seq) for emo, seq in seqs_by_emotion.items()}
    verts_neu = verts_by_emo["neutral"]

    mean_dist_by_emo = {}
    csv_rows = []
    for emo in EMOTIONS:
        verts_e = verts_by_emo[emo]
        T = min(verts_e.shape[0], verts_neu.shape[0])
        dist_tv = np.linalg.norm(verts_e[:T] - verts_neu[:T], axis=-1) * 1000.0
        mean_v = dist_tv.mean(axis=0)
        mean_dist_by_emo[emo] = mean_v
        csv_rows.append({
            "emotion": emo,
            "speaker": SPEAKER,
            "intensity": "high",
            "mouth_lve_mm": f"{mean_v[lve_ids].mean():.4f}",
            "upper_face_fdd_mm": f"{mean_v[fdd_ids].mean():.4f}",
            "full_face_mm": f"{mean_v.mean():.4f}",
            "max_vertex_mm": f"{mean_v.max():.4f}",
        })

    with open(csv_path, "w", newline="") as fh:
        fields = ["emotion", "speaker", "intensity", "mouth_lve_mm", "upper_face_fdd_mm", "full_face_mm", "max_vertex_mm"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[wrote] {csv_path.relative_to(ROOT)}", flush=True)
    for row in csv_rows:
        print(
            f"  {row['emotion']:10s} | mouth(lve)={row['mouth_lve_mm']} mm | "
            f"upper(fdd)={row['upper_face_fdd_mm']} mm | full={row['full_face_mm']} mm | max={row['max_vertex_mm']} mm",
            flush=True,
        )

    vmax = float(max(v.max() for v in mean_dist_by_emo.values()))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = matplotlib.colormaps["viridis"]
    neu_frame_verts = verts_neu[min(frame_idx, len(verts_neu) - 1)]

    heat_tiles = []
    heat_titles = []
    for r in range(2):
        row_imgs, row_titles = [], []
        for c in range(4):
            emo = EMOTIONS[r * 4 + c]
            row_imgs.append(render_displacement_face(renderer, neu_frame_verts, mean_dist_by_emo[emo], norm, cmap))
            row_titles.append(f"{emo}\n(mouth {mean_dist_by_emo[emo][lve_ids].mean():.2f} / full {mean_dist_by_emo[emo].mean():.2f} mm)")
        heat_tiles.append(row_imgs)
        heat_titles.append(row_titles)

    fig, axes = plt.subplots(2, 4, figsize=(12.6, 7.6), facecolor=BACKGROUND)
    for r in range(2):
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(heat_tiles[r][c])
            ax.set_title(heat_titles[r][c], fontsize=10, fontweight="bold", pad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.89, top=0.87, bottom=0.04, wspace=0.06, hspace=0.20)
    cbar_ax = fig.add_axes([0.91, 0.14, 0.022, 0.66])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Mean 3D displacement vs neutral (mm)", fontsize=10, fontweight="bold")
    fig.suptitle(
        f"Per-Vertex Displacement from Neutral Condition (all-frame mean, neutral mesh @ frame {frame_idx}; scale 0–{vmax:.2f} mm)",
        fontsize=12.5,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(png_path, dpi=160, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[wrote] {png_path.relative_to(ROOT)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", type=int, default=None, help="Explicit frame index (default: auto-pick open mouth)")
    ap.add_argument("--size", type=int, default=448, help="Per-tile render resolution")
    ap.add_argument("--no-skip", action="store_true", help="Force re-running generation.py even if .npy exists in ours/runs/")
    a = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_isolated_audio()
    skip = not a.no_skip

    # a) Emotion grid: all 8 conditioned emotions on neutral.wav, M009, high, deterministic
    emo_seqs = {}
    for emo in EMOTIONS:
        npy = run_generation(RUNS_DIR / "emotion_grid" / emo, emo, 2, sample=False, skip_existing=skip)[0]
        emo_seqs[emo] = np.load(npy).reshape(-1, 403)

    frame_idx = a.frame if a.frame is not None else choose_open_mouth_frame(emo_seqs)
    renderer = FaceRenderer(size=a.size)

    emo_tiles = [[renderer.frame(emo_seqs[EMOTIONS[r * 4 + c]], frame_idx) for c in range(4)] for r in range(2)]
    emo_titles = [[f"Conditioned: {EMOTIONS[r * 4 + c]}" for c in range(4)] for r in range(2)]
    save_grid(
        emo_tiles,
        emo_titles,
        RESULTS_DIR / "emotion_grid.png",
        f"Controlled Emotion Grid — neutral.wav, {SPEAKER}, high (2), sample=false (frame {frame_idx})",
    )

    # b) Intensity row: happy, angry, surprised at intensity 0, 1, 2
    int_tiles, int_titles = [], []
    for emo in INTENSITY_EMOTIONS:
        row_imgs, row_titles = [], []
        for int_idx, int_name in INTENSITIES:
            npy = run_generation(
                RUNS_DIR / "intensity_row" / f"{emo}_{int_name}",
                emo,
                int_idx,
                sample=False,
                skip_existing=skip,
            )[0]
            seq = np.load(npy).reshape(-1, 403)
            row_imgs.append(renderer.frame(seq, frame_idx))
            row_titles.append(f"{emo} — {int_name} ({int_idx})")
        int_tiles.append(row_imgs)
        int_titles.append(row_titles)
    save_grid(
        int_tiles,
        int_titles,
        RESULTS_DIR / "intensity_row.png",
        f"Controlled Intensity Comparison — neutral.wav, {SPEAKER}, sample=false (frame {frame_idx})",
        row_labels=[e.capitalize() for e in INTENSITY_EMOTIONS],
    )

    # c) Displacement heatmaps & region CSV
    build_displacement_heatmaps(
        renderer,
        emo_seqs,
        frame_idx,
        RESULTS_DIR / "emotion_displacement.csv",
        RESULTS_DIR / "displacement_heatmaps.png",
    )

    # d) Diversity row: happy high, T=0.2, 3 takes + 3-panel side-by-side video
    div_npys = run_generation(
        RUNS_DIR / "diversity_row" / "happy_high",
        "happy",
        2,
        sample=True,
        temperature=0.2,
        num_samples=3,
        skip_existing=skip,
    )
    div_seqs = [np.load(p).reshape(-1, 403) for p in div_npys]
    div_verts = [renderer.vertices(s) for s in div_seqs]
    pairs = [(0, 1), (0, 2), (1, 2)]
    mean_param_mad = float(np.mean([np.abs(div_seqs[i][:, 300:] - div_seqs[j][:, 300:]).mean() for i, j in pairs]))
    mean_vtx_diff_mm = float(
        np.mean([np.linalg.norm(div_verts[i] - div_verts[j], axis=-1).mean() * 1000.0 for i, j in pairs])
    )
    print(
        f"[diversity] T=0.2 3 takes | mean exp+jaw MAD={mean_param_mad:.5f} | "
        f"mean 3D vertex pairwise diff={mean_vtx_diff_mm:.4f} mm",
        flush=True,
    )

    div_tiles = [[renderer.frame(s, frame_idx) for s in div_seqs]]
    div_titles = [[f"Take {k + 1}" for k in range(3)]]
    save_grid(
        div_tiles,
        div_titles,
        RESULTS_DIR / "diversity_row.png",
        f"Stochastic Diversity — neutral.wav, {SPEAKER}, happy, high, T=0.2\n"
        f"(frame {frame_idx}; mean pairwise 3D diff {mean_vtx_diff_mm:.2f} mm, exp+jaw MAD {mean_param_mad:.4f})",
    )

    div_mp4 = RESULTS_DIR / "diversity_3takes.mp4"
    if skip and div_mp4.exists():
        print(f"[skip] {div_mp4.relative_to(ROOT)} already exists", flush=True)
    else:
        t0 = time.time()
        n_frames = renderer.video(div_seqs, div_mp4, wav=AUDIO_SRC)
        print(
            f"[wrote] {div_mp4.relative_to(ROOT)} "
            f"frames={n_frames} elapsed={time.time() - t0:.2f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()

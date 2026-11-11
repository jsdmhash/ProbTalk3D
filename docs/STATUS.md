# ProbTalk3D Project Status Log

Branch: `colab-inference-fixes`

---

## 2026-11-10 23:22:25 EST — Step A: Commit `ours/` folder (NumPy FLAME renderer and M3 service)

- **Files added or changed:**
  - Added (commit `472d85ee2019ac89839f61d0024425cfafa349e8`): `ours/README.md`, `ours/app/app.py`, `ours/app/client.py`, `ours/app/engine.py`, `ours/app/smoke_test.py`, `ours/app/upload_weights.py`, `ours/face_render.py`, `ours/flame_np.py`, `ours/space/Dockerfile`, `ours/space/README.md`, `ours/space/app.py`, `ours/space/requirements.txt` (12 files, 723 insertions).
- **Exact commands run:**
  ```bash
  git add ours/ && git commit -m "Add ours/: NumPy FLAME renderer and M3 service (Claude)"
  ```
- **Key output lines:**
  ```
  commit 472d85ee2019ac89839f61d0024425cfafa349e8 (HEAD -> colab-inference-fixes, origin/colab-inference-fixes, origin/HEAD)
  12 files changed, 723 insertions(+)
  ```
- **What worked:**
  - Committed all 12 `ours/` files and pushed to `origin/colab-inference-fixes` without touching any authors' code outside the 3 existing `# PATCH` edits.
- **What failed:**
  - None.
- **Open questions:**
  - None.

---

## 2026-11-10 23:25:29 EST — Step B: Test `ours/face_render.py` on `happy_M009_happy_medium.npy`

- **Files added or changed:**
  - Added `ours/checks/frame25.png` (`208K`)
  - Added `ours/logs/step_b_render.txt`
  - Added `docs/STATUS.md`
  - Generated locally (ignored via `.gitignore:25:*.mp4`, not committed): `ours/checks/happy_v2.mp4` (`591K`)
- **Exact commands run:**
  ```bash
  /mnt/SHARED/ProbTalk3D/.venv/bin/python ours/face_render.py results/generation/vqvae_pred/stage_2/0.2/happy_M009_happy_medium.npy ours/checks/happy_v2.mp4 --wav results/generation/test_audio/happy.wav
  /mnt/SHARED/ProbTalk3D/.venv/bin/python ours/face_render.py results/generation/vqvae_pred/stage_2/0.2/happy_M009_happy_medium.npy ours/checks/frame25.png --frame 25
  ffprobe -v error -show_entries stream=codec_type,nb_frames,duration -of default=nw=1 ours/checks/happy_v2.mp4
  ```
- **Key output lines:**
  From `ours/logs/step_b_render.txt`:
  ```
  $ /mnt/SHARED/ProbTalk3D/.venv/bin/python ours/face_render.py results/generation/vqvae_pred/stage_2/0.2/happy_M009_happy_medium.npy ours/checks/happy_v2.mp4 --wav results/generation/test_audio/happy.wav
  wrote ours/checks/happy_v2.mp4 frames: 144
  exit=0 elapsed_seconds=28.42

  $ /mnt/SHARED/ProbTalk3D/.venv/bin/python ours/face_render.py results/generation/vqvae_pred/stage_2/0.2/happy_M009_happy_medium.npy ours/checks/frame25.png --frame 25
  wrote ours/checks/frame25.png
  exit=0 elapsed_seconds=1.52
  ```
  From `ffprobe`:
  ```
  codec_type=video
  duration=5.760000
  nb_frames=144
  codec_type=audio
  duration=5.760000
  nb_frames=91
  ```
- **What worked:**
  - `ours/face_render.py` rendered the 144-frame MP4 with muxed audio (`duration=5.760000`) in `28.42` seconds and extracted frame 25 (`ours/checks/frame25.png`) in `1.52` seconds.
  - Visual check of `ours/checks/frame25.png` confirms white sclera with dark irises, muted red lips, and a dark mouth interior on back faces with no see-through holes or silhouette artifacts.
  - `git check-ignore -v ours/checks/happy_v2.mp4` confirmed `.mp4` output is excluded by `.gitignore`.
- **What failed:**
  - None.
- **Open questions:**
  - None.

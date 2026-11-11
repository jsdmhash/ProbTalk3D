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

---

## 2026-11-10 23:28:14 EST — Step C: Verify `ProbTalkEngine` matches `generation.py` (`smoke_test.py`)

- **Files added or changed:**
  - Modified `.gitignore` (added `ours/logs/smoke_audio/` under Runtime Inference I/O Directories)
  - Modified `docs/STATUS.md` (this entry)
  - Added `ours/logs/step_c_generation.txt`
  - Added `ours/logs/smoke_test.txt`
  - Generated locally (ignored by `.gitignore`): `ours/logs/smoke_audio/happy.wav`, `results/generation/vqvae_pred/stage_2/none/happy_M009_happy_high.npy` (`454K`)
- **Exact commands run:**
  ```bash
  mkdir -p ours/logs/smoke_audio && cp results/generation/test_audio/happy.wav ours/logs/smoke_audio/
  /mnt/SHARED/ProbTalk3D/.venv/bin/python generation.py folder=model_weights/ProbTalk3D/stage_2 input_path=ours/logs/smoke_audio id=[\"M009\"] emotion=[\"happy\"] intensity=[2] sample=false > ours/logs/step_c_generation.txt 2>&1
  /mnt/SHARED/ProbTalk3D/.venv/bin/python ours/app/smoke_test.py results/generation/vqvae_pred/stage_2/none/happy_M009_happy_high.npy > ours/logs/smoke_test.txt 2>&1
  ```
- **Key output lines:**
  Key lines from `ours/logs/step_c_generation.txt`:
  ```
  Audio: happy Style: ['M009_x_1_2'] Sample number: 1
  INFO     All the sampling are done. You can find them here: /mnt/SHARED/ProbTalk3D/results/generation/vqvae_pred/stage_2/none
  ```
  Full output of `ours/logs/smoke_test.txt`:
  ```
  Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  Loading weights: 100%|██████████| 211/211 [00:00<00:00, 12118.45it/s]
  /mnt/SHARED/ProbTalk3D/.venv/lib/python3.13/site-packages/sklearn/base.py:380: InconsistentVersionWarning: Trying to unpickle estimator MinMaxScaler from version 1.1.3 when using version 1.6.1. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
  https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
    warnings.warn(
  model load: 13.27 s on cpu
  style key: M009_x_1_2 | timing: {'audio_seconds': 5.78, 'generate_seconds': 1.76}
  engine shape: (1, 144, 403) | generation.py shape: (1, 144, 403)
  max abs difference vs generation.py: 0.000e+00
  MATCH
  sampled 2 takes: {'audio_seconds': 5.78, 'generate_seconds': 1.18} | takes differ: True
  ```
- **What worked:**
  - `generation.py` produced `results/generation/vqvae_pred/stage_2/none/happy_M009_happy_high.npy` (`454K`).
  - `ProbTalkEngine` loaded `stage_2` in `13.27 s` on `cpu` and generated `(1, 144, 403)` in `1.76 s` with `max abs difference vs generation.py: 0.000e+00` (`MATCH`).
  - Sampling `takes=2` (`sample=True`, `temperature=0.2`, `happy`, `medium`) ran in `1.18 s` and reported `takes differ: True`.
- **What failed:**
  - None.
- **Open questions:**
  - Both `generation.py` and `smoke_test.py` hit `huggingface.co` API endpoints on load even when weights are cached, and `MinMaxScaler` logs `InconsistentVersionWarning` (`1.1.3` vs `1.6.1`) without affecting numerical parity (`0.000e+00`).

---

## 2026-11-11 00:03:06 EST — Step D: Install Gradio and run local service tests (`takes=1`, `takes=3`, `client.py`)

- **Files added or changed:**
  - Modified `.gitignore` (added `client_outputs/` under Runtime Inference I/O Directories)
  - Modified `ours/app/client.py` (selected `token=` vs `hf_token=` via `inspect.signature(Client.__init__)` for `gradio_client` `2.7.2` compatibility)
  - Modified `docs/STATUS.md` (this entry)
  - Added `ours/logs/step_d_pip.txt`, `ours/logs/app_run.txt`, `ours/logs/client_takes1.txt`, `ours/logs/client_takes3.txt`, `ours/logs/client_run.txt`
  - Generated locally (ignored by `.gitignore`): `client_outputs/takes1/M009_happy_medium_T0.2.{mp4,npy}`, `client_outputs/takes3/M009_happy_medium_T0.2.{mp4,npy}`, `client_outputs/M009_angry_high_T0.2.{mp4,npy}`
- **Exact commands run:**
  ```bash
  /mnt/SHARED/ProbTalk3D/.venv/bin/pip install gradio gradio_client > ours/logs/step_d_pip.txt 2>&1
  /mnt/SHARED/ProbTalk3D/.venv/bin/python -u ours/app/app.py > ours/logs/app_run.txt 2>&1
  /mnt/SHARED/ProbTalk3D/.venv/bin/python ours/app/client.py --server http://127.0.0.1:7860 --takes 1 --out client_outputs/takes1 > ours/logs/client_takes1.txt 2>&1
  /mnt/SHARED/ProbTalk3D/.venv/bin/python ours/app/client.py --server http://127.0.0.1:7860 --takes 3 --out client_outputs/takes3 > ours/logs/client_takes3.txt 2>&1
  /mnt/SHARED/ProbTalk3D/.venv/bin/python ours/app/client.py --server http://127.0.0.1:7860 --emotion angry --intensity high > ours/logs/client_run.txt 2>&1
  ```
- **Key output lines:**
  From `ours/logs/step_d_pip.txt`:
  ```
  Successfully installed ... fastapi-0.162.4 gradio-6.31.0 gradio_client-2.7.2 ...
  ```
  From `ours/logs/app_run.txt`:
  ```
  Model ready on cpu in 9.7 s
  * Running on local URL:  http://127.0.0.1:7860
  {"style_key": "M009_x_1_1", "frames": 144, "fps": 25, "takes": 1, "temperature": 0.2, "sampling": true, "device": "cpu", "mean_pairwise_difference": null, "audio_seconds": 5.78, "generate_seconds": 2.18, "render_seconds": 28.05}
  {"style_key": "M009_x_1_1", "frames": 144, "fps": 25, "takes": 1, "temperature": 0.2, "sampling": true, "device": "cpu", "mean_pairwise_difference": null, "audio_seconds": 5.78, "generate_seconds": 0.63, "render_seconds": 27.88}
  {"style_key": "M009_x_1_1", "frames": 144, "fps": 25, "takes": 3, "temperature": 0.2, "sampling": true, "device": "cpu", "mean_pairwise_difference": 0.06603, "audio_seconds": 5.78, "generate_seconds": 2.18, "render_seconds": 72.97}
  {"style_key": "M009_x_6_2", "frames": 144, "fps": 25, "takes": 1, "temperature": 0.2, "sampling": true, "device": "cpu", "mean_pairwise_difference": null, "audio_seconds": 5.78, "generate_seconds": 0.58, "render_seconds": 26.15}
  ```
  From `ours/logs/client_takes1.txt`:
  ```
  Loaded as API: http://127.0.0.1:7860/
  {
    "style_key": "M009_x_1_1",
    "frames": 144,
    "fps": 25,
    "takes": 1,
    "temperature": 0.2,
    "sampling": true,
    "device": "cpu",
    "mean_pairwise_difference": null,
    "audio_seconds": 5.78,
    "generate_seconds": 0.63,
    "render_seconds": 27.88,
    "round_trip_seconds": 28.66
  }
  saved to /mnt/SHARED/ProbTalk3D/client_outputs/takes1
  ```
  From `ours/logs/client_takes3.txt`:
  ```
  Loaded as API: http://127.0.0.1:7860/
  {
    "style_key": "M009_x_1_1",
    "frames": 144,
    "fps": 25,
    "takes": 3,
    "temperature": 0.2,
    "sampling": true,
    "device": "cpu",
    "mean_pairwise_difference": 0.06603,
    "audio_seconds": 5.78,
    "generate_seconds": 2.18,
    "render_seconds": 72.97,
    "round_trip_seconds": 75.3
  }
  saved to /mnt/SHARED/ProbTalk3D/client_outputs/takes3
  ```
  From `ours/logs/client_run.txt`:
  ```
  Loaded as API: http://127.0.0.1:7860/
  {
    "style_key": "M009_x_6_2",
    "frames": 144,
    "fps": 25,
    "takes": 1,
    "temperature": 0.2,
    "sampling": true,
    "device": "cpu",
    "mean_pairwise_difference": null,
    "audio_seconds": 5.78,
    "generate_seconds": 0.58,
    "render_seconds": 26.15,
    "round_trip_seconds": 26.86
  }
  saved to /mnt/SHARED/ProbTalk3D/client_outputs
  ```
- **What worked:**
  - `gradio-6.31.0` and `gradio_client-2.7.2` installed into `.venv`.
  - `ours/app/app.py` loaded `ProbTalkEngine` on `cpu` in `9.7 s` and served `http://127.0.0.1:7860`.
  - After patching `client.py` for `token` vs `hf_token`, all three API calls succeeded:
    - `takes=1` (`happy`, `medium`): `generate_seconds=0.63`, `render_seconds=27.88`, `round_trip_seconds=28.66` (initial cold warmup `takes=1` verification call before `client.py` patch had `generate_seconds=2.18`, `render_seconds=28.05`).
    - `takes=3` (`happy`, `medium`): `generate_seconds=2.18`, `render_seconds=72.97`, `round_trip_seconds=75.3`, `mean_pairwise_difference=0.06603` with 3-panel side-by-side MP4 (`1.3M`).
    - `client.py --emotion angry --intensity high`: `generate_seconds=0.58`, `render_seconds=26.15`, `round_trip_seconds=26.86`.
- **What failed (fixed in `ours/`):**
  - First run of unedited `ours/app/client.py` with `gradio_client-2.7.2` failed with:
    ```
    File "/mnt/SHARED/ProbTalk3D/ours/app/client.py", line 33, in main
      client = Client(a.server, hf_token=os.environ.get("HF_TOKEN"))
    TypeError: Client.__init__() got an unexpected keyword argument 'hf_token'
    ```
  - Fixed in `ours/app/client.py` by inspecting `Client.__init__` parameters (`token` in `2.x`, `hf_token` in `1.x`). Note also `out.mkdir(exist_ok=True)` in `client.py` requires parent directories to already exist when passing nested `--out` paths (`client_outputs` was created beforehand).
- **Open questions:**
  - Note `pip install gradio` downgraded `pillow` from `13.0.0` to `12.3.0` (`ours/space/requirements.txt` does not pin `gradio` or `gradio_client`, so `HF Space` will also install `gradio_client>=2.x` where the `client.py` patch prevents the `TypeError`).

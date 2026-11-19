# ProbTalk3D Studio: our replication and deployment

**Jyothi Masina and Shruti Deulgaonkar** · UNC Charlotte · Fall 2026

We took ProbTalk3D (Wu, Haque and Yumak, ACM SIGGRAPH MIG 2024), got it running on today's software,
checked the paper's claims on real ground-truth data, and turned it into a cloud service with a web
studio, an API, a Python client and an installable phone app.

- **Showcase page:** https://jsdmhash.github.io/ProbTalk3D/
- **Demo recording:** [`docs/final/demo_recording.mp4`](../docs/final/demo_recording.mp4)
- **Academic report (PDF):** [`docs/final/ProbTalk3D_Report.pdf`](../docs/final/ProbTalk3D_Report.pdf) (LaTeX source in [`docs/report/`](../docs/report/))
- **Final presentation, speaker script and study guide:** [`docs/final/`](../docs/final/)

## Results at a glance

**Ground truth** (478 real 3DMEAD test sequences, 8 of 32 speakers, 10 samples each, T = 0.2; paper scales):

| Metric | Paper | Ours |
|---|---|---|
| MVE ↓ | 0.7243 | 0.7056 |
| LVE ↓ | 0.6040 | 0.6418 |
| MEE ↓ | 0.5549 | 0.5918 |
| CE ↓ | 0.5227 | 0.5688 |
| FDD ↓ | 0.0415 | 0.0537 |
| Diversity ↑ | 0.3274 | 0.3116 |

**Temperature matrix** (same clips): sampling off gives the lowest error (MVE 0.6955) and zero diversity;
T = 0.2 costs about 1.5% error for real variety; T = 1.0 gives 2.5× the variety but nearly doubles the error.
See [`results/matrix/`](results/matrix/).

**User survey** (25 people, anonymous Google Form, 12 clips): lip-sync 3.91/5, naturalness 3.50/5,
emotion recognised on silent clips 39% (chance 12.5%), 15 of 25 noticed that sampled takes differ.
See [`results/survey/`](results/survey/). Raw responses are not published.

**Emotion heatmaps:** happy moves the mouth corners (4.6 mm on average); angry and disgusted move the
brows, disgusted most (up to 8.8 mm). See `results/displacement_heatmaps.png`.

## Code map

The authors' code is unchanged apart from the three `# PATCH` fixes in
`framework/`. Everything new lives here.

| File | What it does |
|---|---|
| `flame_np.py` | FLAME head model in pure NumPy (no torch, smplx or chumpy) |
| `face_render.py` | Clear renderer: real-looking eyes, coloured lips, dark mouth interior, fixed camera. Video (with audio), single frames, or several takes side by side |
| `app/engine.py` | Loads ProbTalk3D once and serves requests (same steps as `generation.py`) |
| `app/app.py` | Gradio web app (local or Hugging Face Space), API name `/generate` |
| `app/client.py` | Command-line client that calls the app over the network |
| `app/smoke_test.py` | Checks the engine's output equals `generation.py`'s (deterministic mode) |
| `app/upload_weights.py` | Puts the weights + FLAME file in a private HF model repo |
| `space/` | Files for the Hugging Face Space (Docker) |
| `survey/analyze_survey.py` | Turns the Google Form export into the survey charts and summary |
| `survey/make_emotion_clips.py` | Renders the clips used in the survey |
| `eval/eval_stream.py` | The authors' metrics, streamed so they fit in laptop memory |
| `tools/make_matrix.py` | Builds the temperature matrix and trade-off chart |
| `tools/draw_icon.py` | Draws the app icon |

## Run locally

```
source .venv/bin/activate
pip install gradio gradio_client
python ours/face_render.py results/generation/vqvae_pred/stage_2/0.2/<file>.npy out.mp4 --wav results/generation/test_audio/<clip>.wav
python ours/app/app.py                       # open http://127.0.0.1:7860
python ours/app/client.py --server http://127.0.0.1:7860 --takes 3
```

## Licences and credits

- Model, paper and original code: Sichun Wu, Kazi Injamamul Haque and Zerrin Yumak (CC-BY-NC-4.0; see the top-level README).
- FLAME is licensed separately and is **not** included in this repository. Register at flame.is.tue.mpg.de and
  place `generic_model.pkl` and the landmark embeddings in `deps/flame/model/`.
- We used Claude (Anthropic) and the Antigravity coding agent as assistants for planning, code and writing.
  We ran every step and checked every number ourselves.

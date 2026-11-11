# ours/ : everything we added on top of the authors' code

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

## Run locally

```
source .venv/bin/activate
pip install gradio gradio_client
python ours/face_render.py results/generation/vqvae_pred/stage_2/0.2/<file>.npy out.mp4 --wav results/generation/test_audio/<clip>.wav
python ours/app/app.py                       # open http://127.0.0.1:7860
python ours/app/client.py --server http://127.0.0.1:7860 --takes 3
```

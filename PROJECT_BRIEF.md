# ProbTalk3D: Project Brief for Antigravity

Put this file at the root of the GitHub repo. Give it to Antigravity as the first thing to read.

## 1. Goal

Team project for ITCS 6166 (Generative AI as a Service), UNC Charlotte, Fall 2026. Team of two. Our paper is **ProbTalk3D** (speech-driven 3D facial animation, SIGGRAPH MIG 2024, arXiv 2409.07966). The model turns a speech `.wav` into a face animation (FLAME parameters), and gives a different result on each run.

The course has three milestones:

- **M0** Paper selection and presentation (done)
- **M1** Code review and understanding (done)
- **M2** Replicate the inference code in Google Colab (in progress)
- **M3** Host the model as a GenAI service with a GUI client. Use a free host such as Hugging Face Spaces.
- **Bonus** (not committed): phone app (+10), training and evaluation on HPC (+10)

Graded deliverables include the GitHub repo "with codes and detailed instructions," so the repo has to be runnable by someone who has never seen it.

## 2. What Antigravity should do

In priority order:

1. **Set up a working local environment** from `requirements_colab.txt` (the package versions we actually ran). Verify it by running one inference clip. (Section 6.)
2. **Finish Milestone 2.** Turn our Colab work into a clean, reproducible pipeline:
   - A setup script that downloads the weights and applies the compatibility fixes.
   - A single inference script or notebook that reproduces the 8 test-clip run.
   - A check that every output is shaped `(1, T, 403)` and that its length matches the audio at 25 fps.
   - The temperature and sampling experiments (Section 3), as a script that rewrites the numbers and figure.
   - A preview-video step that uses `render_preview.py`, after confirming it works on real outputs.
3. **Write the README** so a stranger can reproduce M2 from scratch: environment, weights, commands, expected output, known issues.
4. **Build Milestone 3.** A Gradio app that takes audio, speaker ID, emotion, intensity, temperature and number of samples, and returns the animation data and a preview video. Deploy it to Hugging Face Spaces. Build a small client that calls the Space. Plan the weight hosting (Hugging Face model repo or a download at startup). Test whether CPU-only inference on a Space is fast enough.
5. **Stretch goals** only after 2 to 4 are solid: a mobile-friendly client, and HPC training.

## 3. What is done (verified on real weights in Colab, T4 GPU)

The Colab run used the authors' pretrained weights from the README link (2.5 GB, stage_1 and stage_2).

- **Inference runs end to end** on all 8 bundled test clips (`results/generation/test_audio`). Each run writes one `.npy` per clip to `results/generation/vqvae_pred/stage_2/0.2/`.
- **Output shape is `(1, T, 403)`.** Output length matches audio length at 25 fps (within 0.05 s). The 403 values are 300 shape (zeros), 100 expression (50 used), and 3 jaw.
- **Sampling is non-deterministic.** Three samples of one clip (style M009, happy, medium) gave different outputs. Mean absolute difference between pairs: 0.0168 to 0.0171.
- **`sample=false` is deterministic.** Two runs gave identical output.
- **Temperature sweep** (one clip, one style, 3 samples each): mean pairwise difference rises with temperature.

  | Temperature | Mean pairwise difference |
  |---|---|
  | 0.05 | 0.0081 |
  | 0.2 | 0.0169 |
  | 0.5 | 0.0297 |
  | 1.0 | 0.0374 |

  Caveat: one clip, one style, 3 samples. Report it that way.

- **Preview renderer** (`render_preview.py`): tested in the sandbox on synthetic motion only. It draws a FLAME head with the jaw moving. **Not yet confirmed on a real output.**

### Fixes that were needed (all applied in Colab)

The authors' code breaks on current libraries. These fixes are required:

1. `framework/model/feature_extractor/hubert.py`: replace `from deps.hubert.modeling_hubert import HubertModel` with `from transformers import HubertModel`. The vendored copy imports `transformers.deepspeed`, which modern `transformers` lacks.
2. `framework/model/utils/tools.py` in `load_checkpoint`: remap old HuBERT weight-norm keys (`weight_g`/`weight_v` to `parametrizations.weight.original0/1`) before `load_state_dict`.
3. `framework/model/utils/tools.py`: `torch.load(..., weights_only=False)`. PyTorch 2.6 changed the default and the checkpoint stores an OmegaConf object. **Only fix 3 is missing from the sandbox copy.** Make sure it is in the repo.

Check all three with `git diff` after the Colab push (see SETUP_GUIDE.md, Step 3).

## 4. What is left (Milestone 2 and 3)

**M2 (remaining):**
- Confirm the video render on a real output (`render_preview.py` on one `.npy` file with its `.wav`). Record the result honestly.
- Turn the Colab cells into scripts (setup, generate, sweep) and a README.
- Freeze the working environment (`requirements_colab.txt`).
- Make a short demo video or GIF for the milestone presentation.

**M3 (not started):**
- Gradio app for Hugging Face Spaces.
- Client (local script or simple web UI) that calls the Space with `gradio_client`.
- Weight hosting plan.
- CPU timing test. Pass or fail: is one clip under a reasonable wait time?
- Deployment instructions in the README.

**Presentations:**
- M1 deck (12 slides, code review). The current version is an artifact, and its speaker script is `ProbTalk3D_M1_Speaker_Script.txt`. Both are in `docs/`.
- M2 deck: not started. It should show the real results in Section 3, labeled honestly.
- Older files in `docs/` are outdated. `ProbTalk3D_M1_Code_Review.pptx` says the code was not run. Do not use it for the M2 story.

## 5. Things we have NOT verified (do not claim them)

- Python 3.9 (the README version). We ran Python 3.13 in Colab. Treat the Colab versions as the reference.
- CPU-only inference with real weights. Untested. Only the GPU path has run.
- The Gradio app, Hugging Face Spaces, and the client. Nothing built.
- Blender rendering. Not used.
- The mobile app and HPC training. Not started.
- Any statement about output quality. We have not judged the animations visually.

## 6. Environment and commands

Expected layout after setup:

```
ProbTalk3D/
  framework/  configs/  datasets/  deps/  ...
  model_weights/ProbTalk3D/stage_1/   (downloaded, not committed)
  model_weights/ProbTalk3D/stage_2/   (downloaded, not committed)
  results/generation/test_audio/      (8 .wav files, in repo)
  requirements_colab.txt
  render_preview.py
```

Reproduce one inference run:

```
python generation.py folder=model_weights/ProbTalk3D/stage_2 input_path=results/generation/test_audio
```

Outputs go to `results/generation/vqvae_pred/stage_2/0.2/`.

Other settings from `configs/generation.yaml`:
- `temperature=0.2` (default; README suggests 0.1 to 0.5)
- `sample=false` (deterministic)
- `number_of_samples=3`
- `id=[...]`, `emotion=[...]`, `intensity=[...]` (style; random if omitted)

Render one clip:

```
python render_preview.py . <path/to/file.npy> out.mp4 <path/to/audio.wav>
```

Set `PYOPENGL_PLATFORM` as needed on Linux. The renderer uses matplotlib, not GL.

## 7. Rules for Antigravity

- **Never claim something ran** unless a log or output file shows it. Say "untested" otherwise.
- **Keep the authors' code changes minimal** and mark each one with `# PATCH` and a reason.
- **Do not commit** model weights, `.npy` outputs, `.mp4` files, or logs. Use `.gitignore`.
- **FLAME licence:** `deps/flame/model/generic_model.pkl` comes from the authors' repo. The FLAME model has licence restrictions. Keep our repo private, or ask before making it public.
- **Keep the paper's licence** (`LICENSE.txt`) and credit the authors in the README.
- **Work on a branch.** Ask before large refactors.
- **Report what changed and what was verified** at the end of each task.

## 8. Known issues

| Issue | Where | Status |
|---|---|---|
| Vendored HuBERT imports `transformers.deepspeed` | `deps/hubert/` | Fixed by fix 1 |
| Old HuBERT weight-norm key names | `load_checkpoint` | Fixed by fix 2 |
| `torch.load` default in PyTorch 2.6+ | `load_checkpoint` | Fixed by fix 3 |
| Stage 2 log says "crossEntropy" but computes L1 | training logs | Cosmetic |
| README says hubert-xlarge; config loads hubert-base | `configs/model/feature_extractor/hubert.yaml` | Config is what runs. Note in README. |
| Batch size is 1 | model `forward` asserts | Limits throughput for the service |
| Blender needed for the authors' renders | `render_*.py` | Use `render_preview.py` instead |

## 9. First message to give Antigravity

> Read PROJECT_BRIEF.md in full. Then:
> 1. Check the environment in section 6 and tell me what is missing.
> 2. Confirm the four fixes in section 3 are present with `git diff`.
> 3. Run one inference clip and show me the shape of the output.
> Do not start Milestone 3 until those three are done and reported.

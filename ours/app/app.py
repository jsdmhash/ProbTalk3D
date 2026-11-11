"""
ProbTalk3D as a service: Gradio web app (runs locally or on Hugging Face Spaces).

  python ours/app/app.py            # then open http://127.0.0.1:7860

API name for programmatic clients: /generate  (see ours/app/client.py)
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from engine import ProbTalkEngine, EMOTIONS, INTENSITIES, SPEAKERS, MAX_SECONDS  # noqa: E402
from face_render import FaceRenderer  # noqa: E402

print("Loading ProbTalk3D ...", flush=True)
ENGINE = ProbTalkEngine()
RENDERER = FaceRenderer(os.environ.get("FLAME_MODEL_PATH"), size=int(os.environ.get("RENDER_SIZE", "448")))
print(f"Model ready on {ENGINE.device} in {ENGINE.load_seconds:.1f} s", flush=True)

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "results/generation/test_audio"


def generate(audio_path, speaker, emotion, intensity, temperature, sample, takes):
    if not audio_path:
        raise gr.Error("Upload or record a speech clip first.")
    takes = int(takes)
    seqs, keyid, timing = ENGINE.generate(audio_path, speaker, emotion, intensity,
                                          temperature, sample, takes)
    out_dir = Path(tempfile.mkdtemp(prefix="probtalk_"))
    stem = f"{speaker}_{emotion}_{intensity}_T{temperature}"
    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, np.stack(seqs) if takes > 1 else seqs[0][None])   # (takes, T, 403) or (1, T, 403)

    t0 = time.time()
    video_path = out_dir / f"{stem}.mp4"
    RENDERER.video(seqs, video_path, wav=audio_path)
    timing["render_seconds"] = round(time.time() - t0, 2)

    diff = None
    if takes > 1:
        pairs = [(i, j) for i in range(takes) for j in range(i + 1, takes)]
        diff = round(float(np.mean([np.abs(seqs[i][:, 300:] - seqs[j][:, 300:]).mean() for i, j in pairs])), 5)
    info = {"style_key": keyid, "frames": int(seqs[0].shape[0]), "fps": 25, "takes": takes,
            "temperature": temperature, "sampling": bool(sample), "device": ENGINE.device,
            "mean_pairwise_difference": diff, **timing}
    print(json.dumps(info), flush=True)
    return str(video_path), str(npy_path), info


with gr.Blocks(title="ProbTalk3D") as demo:
    gr.Markdown(
        "# ProbTalk3D: speech to 3D talking face\n"
        "Upload or record speech, choose a speaker style, emotion and intensity, and the model "
        "generates FLAME facial animation at 25 fps. With sampling on, every run is different; "
        "temperature controls how different. Set takes to 2 or 3 to see that side by side.\n\n"
        f"Clips longer than {MAX_SECONDS:.0f} s are trimmed. Model: ProbTalk3D (Wu et al., MIG 2024), "
        "pretrained weights from the authors."
    )
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Speech (.wav)")
            speaker = gr.Dropdown(SPEAKERS, value="M009", label="Speaker style")
            emotion = gr.Dropdown(EMOTIONS, value="happy", label="Emotion")
            intensity = gr.Radio(INTENSITIES, value="medium", label="Intensity (ignored for neutral)")
            temperature = gr.Slider(0.05, 1.0, value=0.2, step=0.05, label="Temperature (diversity)")
            sample = gr.Checkbox(True, label="Sampling on (off = deterministic)")
            takes = gr.Slider(1, 3, value=1, step=1, label="Takes (side by side)")
            go = gr.Button("Generate", variant="primary")
        with gr.Column():
            video = gr.Video(label="Animation")
            npy = gr.File(label="FLAME parameters (.npy)")
            info = gr.JSON(label="Run details")
    go.click(generate, [audio, speaker, emotion, intensity, temperature, sample, takes],
             [video, npy, info], api_name="generate")
    if EXAMPLE_DIR.exists():
        gr.Examples([[str(EXAMPLE_DIR / f"{e}.wav"), "M009", e if e in EMOTIONS else "happy", "high", 0.2, True, 1]
                     for e in ["happy", "angry", "sad", "surprised"] if (EXAMPLE_DIR / f"{e}.wav").exists()],
                    [audio, speaker, emotion, intensity, temperature, sample, takes])

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name=os.environ.get("HOST", "127.0.0.1"),
                                  server_port=int(os.environ.get("PORT", "7860")))

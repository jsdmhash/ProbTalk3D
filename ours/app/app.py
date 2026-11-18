"""
ProbTalk3D Studio: speech-driven 3D facial animation as a service.
Gradio web app (runs locally or on Hugging Face Spaces).

  python ours/app/app.py            # then open http://127.0.0.1:7860

API for programmatic clients: /generate  (see ours/app/client.py)
"""
import inspect
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
RENDERER = FaceRenderer(os.environ.get("FLAME_MODEL_PATH"), size=int(os.environ.get("RENDER_SIZE", "720")), background="dark")
print(f"Model ready on {ENGINE.device} in {ENGINE.load_seconds:.1f} s", flush=True)

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "results/generation/test_audio"
REPO_URL = "https://github.com/jsdmhash/ProbTalk3D"
PAPER_URL = "https://arxiv.org/abs/2409.07966"


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


def summary(info):
    if not info:
        return ""
    chips = [
        ("Frames", f"{info['frames']} @ 25 fps"),
        ("Audio", f"{info['audio_seconds']} s"),
        ("Model time", f"{info['generate_seconds']} s"),
        ("Render time", f"{info['render_seconds']} s"),
        ("Takes", str(info["takes"])),
        ("Sampling", f"on, T = {info['temperature']}" if info["sampling"] else "off (deterministic)"),
    ]
    if info.get("mean_pairwise_difference") is not None:
        chips.append(("Take-to-take difference", str(info["mean_pairwise_difference"])))
    return '<div class="chips">' + "".join(
        f'<div class="chip"><span>{k}</span><b>{v}</b></div>' for k, v in chips) + "</div>"


def make_showcase():
    """Render a short looping clip of the avatar at startup for the landing page."""
    wav = EXAMPLE_DIR / "happy.wav"
    if not wav.exists():
        return None
    try:
        seqs, _, _ = ENGINE.generate(str(wav), "M009", "happy", "high", 0.2, True, 1, seed=7)
        out = Path(tempfile.gettempdir()) / "probtalk_showcase.mp4"
        FaceRenderer(os.environ.get("FLAME_MODEL_PATH"), size=480, background="dark").video(seqs, out, wav=str(wav))
        return str(out)
    except Exception as exc:          # the page still works without the showcase
        print("showcase skipped:", exc, flush=True)
        return None


print("Rendering landing-page showcase ...", flush=True)
SHOWCASE = make_showcase()

CSS = """
:root, .dark {--pt-bg:#070a13; --pt-card:rgba(255,255,255,.035); --pt-line:rgba(255,255,255,.09);
              --pt-text:#e6e8f0; --pt-mute:#9aa3b8; --pt-a:#7c6cff; --pt-b:#e45cf0; --pt-c:#36d6c3;}
body, .gradio-container {background: var(--pt-bg) !important;}
.gradio-container {max-width: 1220px !important; margin: auto; color: var(--pt-text);}
#bg {position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden;}
#bg span {position: absolute; border-radius: 50%; filter: blur(90px); opacity: .35; animation: drift 18s ease-in-out infinite alternate;}
#bg .o1 {width: 520px; height: 520px; background: var(--pt-a); top: -140px; left: -120px;}
#bg .o2 {width: 460px; height: 460px; background: var(--pt-b); top: 30%; right: -160px; animation-delay: -6s;}
#bg .o3 {width: 380px; height: 380px; background: var(--pt-c); bottom: -160px; left: 30%; animation-delay: -12s; opacity: .22;}
@keyframes drift {to {transform: translate(60px, 40px) scale(1.12);}}
.gradio-container > * {position: relative; z-index: 1;}
#hero-row {align-items: center; padding: 34px 6px 10px; gap: 28px;}
.kicker {display: inline-flex; gap: 8px; align-items: center; font-size: .78rem; letter-spacing: .14em; text-transform: uppercase;
         color: var(--pt-c); border: 1px solid rgba(54,214,195,.35); padding: 5px 12px; border-radius: 999px; background: rgba(54,214,195,.06);}
.kicker i {width: 7px; height: 7px; border-radius: 50%; background: var(--pt-c); box-shadow: 0 0 12px var(--pt-c); animation: pulse 1.6s infinite;}
@keyframes pulse {50% {opacity: .3;}}
.headline {font-size: 3.1rem; line-height: 1.05; font-weight: 800; margin: 18px 0 14px; letter-spacing: -.02em; color: var(--pt-text);}
.headline em {font-style: normal; color: #b3a6ff; text-shadow: 0 0 28px rgba(124,108,255,.55);}
.lede {font-size: 1.08rem; color: var(--pt-mute); max-width: 560px; line-height: 1.6;}
.cta {display: flex; gap: 12px; margin-top: 24px; flex-wrap: wrap;}
.cta a {text-decoration: none; padding: 12px 20px; border-radius: 12px; font-weight: 700; font-size: .98rem;}
.cta .primary {color: #fff; background: linear-gradient(90deg, var(--pt-a), var(--pt-b)); box-shadow: 0 10px 30px -10px var(--pt-b);}
.cta .ghost {color: var(--pt-text); border: 1px solid var(--pt-line); background: var(--pt-card);}
.wave {display: flex; gap: 4px; align-items: flex-end; height: 34px; margin-top: 26px;}
.wave b {width: 5px; border-radius: 3px; background: linear-gradient(var(--pt-b), var(--pt-a)); animation: eq 1.1s ease-in-out infinite;}
@keyframes eq {0%,100% {height: 18%;} 50% {height: 100%;}}
#showcase {border-radius: 22px !important; overflow: hidden; border: 1px solid var(--pt-line) !important;
           box-shadow: 0 30px 80px -30px rgba(124,108,255,.55), 0 0 0 1px rgba(255,255,255,.03) inset; background: #0d1222 !important;}
.stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 26px 0 8px;}
.stat {border: 1px solid var(--pt-line); background: var(--pt-card); border-radius: 16px; padding: 14px 16px; backdrop-filter: blur(8px);}
.stat b {display: block; font-size: 1.5rem; color: var(--pt-text);}
.stat span {color: var(--pt-mute); font-size: .82rem;}
.features {display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 18px 0 30px;}
.feat {border: 1px solid var(--pt-line); background: var(--pt-card); border-radius: 18px; padding: 18px; transition: transform .2s, border-color .2s;}
.feat:hover {transform: translateY(-3px); border-color: rgba(124,108,255,.5);}
.feat .ic {font-size: 1.4rem; width: 40px; height: 40px; display: grid; place-items: center; border-radius: 12px;
           background: linear-gradient(135deg, rgba(124,108,255,.25), rgba(228,92,240,.18)); margin-bottom: 10px;}
.feat h4 {margin: 0 0 6px; color: var(--pt-text); font-size: 1.02rem;}
.feat p {margin: 0; color: var(--pt-mute); font-size: .9rem; line-height: 1.5;}
.section-title {font-size: 1.5rem; font-weight: 800; color: var(--pt-text); margin: 10px 0 4px;}
.section-sub {color: var(--pt-mute); margin-bottom: 12px;}
.card {border-radius: 18px !important; border: 1px solid var(--pt-line) !important; background: var(--pt-card) !important;}
.step {font-weight: 800; font-size: .74rem; letter-spacing: .14em; text-transform: uppercase;
       color: var(--pt-c); padding: 4px 2px 6px;}
#go {font-size: 1.05rem; padding: 14px; border-radius: 14px; border: 0;
     background: linear-gradient(90deg, var(--pt-a), var(--pt-b)) !important; color: #fff !important; box-shadow: 0 12px 30px -12px var(--pt-b);}
.chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px;}
.chip {background: var(--pt-card); border: 1px solid var(--pt-line); border-radius: 12px; padding: 7px 11px;
       display: flex; flex-direction: column; min-width: 112px;}
.chip span {font-size: .72rem; color: var(--pt-mute);}
.chip b {font-size: .95rem; color: var(--pt-text);}
.pipeline {display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;}
.pipe {border: 1px solid var(--pt-line); border-radius: 16px; padding: 16px; background: var(--pt-card); color: var(--pt-mute);}
.pipe b {display: block; margin-bottom: 6px; color: var(--pt-text);}
#footer {text-align: center; color: var(--pt-mute); font-size: .85rem; padding: 26px 0 10px;}
#footer a {color: var(--pt-text);}
@media (max-width: 760px) {.headline {font-size: 2.2rem;} #hero-row {padding-top: 18px;}}
"""

FORCE_DARK = """
() => {
  document.documentElement.classList.add('dark'); document.body.classList.add('dark');
  if (!document.getElementById('bg')) {
    const bg = document.createElement('div'); bg.id = 'bg';
    bg.innerHTML = '<span class="o1"></span><span class="o2"></span><span class="o3"></span>';
    document.body.prepend(bg);
  }
}
"""

WAVE = '<div class="wave">' + "".join(
    f'<b style="animation-delay:-{(i * 0.13) % 1.1:.2f}s"></b>' for i in range(28)) + "</div>"

HERO_TEXT = f"""
<div class="kicker"><i></i> Live generative AI service</div>
<div class="headline">Give any voice a <em>3D face</em> that feels it.</div>
<div class="lede">ProbTalk3D Studio turns a speech recording into an emotional, lip-synced 3D talking head.
Choose the emotion and its intensity, and every take comes out a little different, just like a real actor.</div>
<div class="cta"><a class="primary" href="#studio">Start creating</a><a class="ghost" href="{PAPER_URL}" target="_blank">Read the paper</a></div>
{WAVE}
"""

STATS = """
<div class="stats">
  <div class="stat"><b>8</b><span>emotions</span></div>
  <div class="stat"><b>3</b><span>intensity levels</span></div>
  <div class="stat"><b>32</b><span>speaker styles</span></div>
  <div class="stat"><b>25 fps</b><span>FLAME 3D motion</span></div>
  <div class="stat"><b>&lt;2 s</b><span>model time per clip (CPU)</span></div>
</div>
"""

FEATURES = """
<div class="features">
  <div class="feat"><div class="ic">🎭</div><h4>Emotion you can dial</h4><p>Happy, sad, angry, surprised and more, each at low, medium or high intensity.</p></div>
  <div class="feat"><div class="ic">🎲</div><h4>No two takes alike</h4><p>The model samples its motions, so the same line comes out fresh every time. Temperature sets how much.</p></div>
  <div class="feat"><div class="ic">🎯</div><h4>Checked against real faces</h4><p>On recorded 3D ground truth, our deployment reproduces the paper's accuracy and diversity metrics.</p></div>
  <div class="feat"><div class="ic">☁️</div><h4>Runs as a service</h4><p>Hosted in the cloud with a public API, a Python client and a phone-installable web app.</p></div>
</div>
"""

HOW = """
<div class="pipeline">
  <div class="pipe"><b>1 · Listen</b>HuBERT turns 16 kHz speech into 768 sound features per frame.</div>
  <div class="pipe"><b>2 · Add style</b>Speaker, emotion and intensity are mixed into those features.</div>
  <div class="pipe"><b>3 · Pick motions</b>A transformer predicts which of 256 learned facial motions fits each frame.</div>
  <div class="pipe"><b>4 · Sample</b>Instead of always the closest motion, it samples one weighted by closeness. Temperature controls how adventurous.</div>
  <div class="pipe"><b>5 · Decode</b>The decoder outputs 50 expression + 3 jaw FLAME values per frame.</div>
  <div class="pipe"><b>6 · Render</b>Our renderer turns those parameters into the video you see.</div>
</div>
"""

ABOUT = f"""
**Model.** ProbTalk3D (Wu, Haque & Yumak, ACM SIGGRAPH MIG 2024, [paper]({PAPER_URL})), running the authors'
pretrained weights unchanged. Our service output is verified to match the authors' `generation.py` exactly.

**What you see.** The model predicts FLAME face parameters: expression and jaw motion. The eyebrows and eyelid
lines are drawn through FLAME's own facial landmarks, so they move exactly as the model moves the face.
The hair cap and colours are a stylised overlay: FLAME has no hair or skin texture.

**Limits.** The model drives the face, not the head pose or eye gaze. Inputs longer than {MAX_SECONDS:.0f} s are trimmed.
It runs on CPU, so a 6-second clip takes about 15 seconds to render on two CPU cores.

**Team.** Jyothi Masina and Shruti Deulgaonkar · ITCS 6166 Generative AI as a Service · UNC Charlotte · Fall 2026 ·
[code]({REPO_URL})
"""

blocks_kw, launch_kw = {"title": "ProbTalk3D Studio"}, {}
theme = gr.themes.Soft(primary_hue="violet", secondary_hue="fuchsia", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]).set(
    body_background_fill_dark="#070a13", block_background_fill_dark="rgba(255,255,255,0.035)",
    block_border_color_dark="rgba(255,255,255,0.09)", input_background_fill_dark="rgba(255,255,255,0.05)",
    body_text_color_dark="#e6e8f0", block_label_text_color_dark="#9aa3b8")
_launch_params = inspect.signature(gr.Blocks.launch).parameters
for key, value in (("theme", theme), ("css", CSS), ("js", FORCE_DARK),
                   ("head", '<meta name="theme-color" content="#070a13">')):
    (launch_kw if key in _launch_params else blocks_kw)[key] = value
ICON = Path(__file__).resolve().parent / "assets" / "icon.png"
if ICON.exists() and "favicon_path" in _launch_params:   # browser tab + home-screen icon
    launch_kw["favicon_path"] = str(ICON)
if "pwa" in _launch_params:          # lets phones install the app to the home screen
    launch_kw["pwa"] = True

with gr.Blocks(**blocks_kw) as demo:
    with gr.Row(elem_id="hero-row", equal_height=False):
        with gr.Column(scale=6):
            gr.HTML(HERO_TEXT)
        with gr.Column(scale=5):
            if SHOWCASE:
                gr.Video(value=SHOWCASE, autoplay=True, loop=True, show_label=False, elem_id="showcase",
                         height=440, interactive=False)
    gr.HTML(STATS + FEATURES)
    gr.HTML('<div id="studio" class="section-title">Studio</div>'
            '<div class="section-sub">Three steps: give it a voice, pick a style, choose how much it varies.</div>')
    with gr.Tabs():
        with gr.Tab("Generate"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="step">Step 1 · Voice</div>')
                        audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                         label="Upload a .wav or record with your microphone")
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="step">Step 2 · Style</div>')
                        with gr.Row():
                            speaker = gr.Dropdown([(f"{s} · {'male' if s.startswith('M') else 'female'} actor", s) for s in SPEAKERS],
                                                  value="M009", label="Speaking style",
                                                  info="How the face moves. The voice comes from your audio.")
                            emotion = gr.Dropdown(EMOTIONS, value="happy", label="Emotion")
                        intensity = gr.Radio(INTENSITIES, value="medium", label="Intensity (neutral has one level)")
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="step">Step 3 · Variation</div>')
                        temperature = gr.Slider(0.05, 1.0, value=0.2, step=0.05,
                                                label="Temperature: higher = more varied takes")
                        with gr.Row():
                            sample = gr.Checkbox(True, label="Sampling on (off = identical every run)")
                            takes = gr.Slider(1, 3, value=1, step=1, label="Takes side by side")
                    go = gr.Button("Generate animation", variant="primary", elem_id="go")
                with gr.Column(scale=6):
                    video = gr.Video(label="Your animation", autoplay=True, height=480)
                    stats = gr.HTML()
                    npy = gr.File(label="Download the FLAME parameters (.npy)")
                    with gr.Accordion("Raw run details (JSON)", open=False):
                        info = gr.JSON(label="Run details")
            if EXAMPLE_DIR.exists():
                gr.Examples(
                    [[str(EXAMPLE_DIR / f"{e}.wav"), "M009", e, "high"]
                     for e in ["happy", "angry", "sad", "surprised"] if (EXAMPLE_DIR / f"{e}.wav").exists()],
                    [audio, speaker, emotion, intensity],
                    label="Try an example")
        with gr.Tab("How it works"):
            gr.HTML(HOW)
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
    gr.HTML('<div id="footer">ProbTalk3D Studio · built on the authors’ pretrained model · '
            f'<a href="{REPO_URL}" target="_blank">GitHub</a> · <a href="{PAPER_URL}" target="_blank">Paper</a></div>')

    event = go.click(generate, [audio, speaker, emotion, intensity, temperature, sample, takes],
                     [video, npy, info], api_name="generate")
    try:                                   # Gradio 6 name for hiding a helper step from the API
        event.then(summary, info, stats, api_visibility="private")
    except TypeError:                      # older Gradio
        event.then(summary, info, stats, api_name=False)

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name=os.environ.get("HOST", "127.0.0.1"),
                                  server_port=int(os.environ.get("PORT", "7860")), **launch_kw)

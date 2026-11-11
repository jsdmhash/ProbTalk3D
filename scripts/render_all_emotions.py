import os, sys, subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

os.makedirs("demo_videos", exist_ok=True)
audio_dir = "results/generation/test_audio"
npy_dir = "results/generation/vqvae_pred/stage_2/0.2"

# 1. Map the 8 emotions
emotions = [
    ("angry", "angry_W016_disgusted_low.npy"),
    ("contempt", "contempt_W014_contempt_medium.npy"),
    ("disgusted", "disgusted_M003_surprised_high.npy"),
    ("fear", "fear_W018_neutral_high.npy"),
    ("happy", "happy_M009_happy_medium.npy"),
    ("neutral", "neutral_M028_angry_low.npy"),
    ("sad", "sad_M009_neutral_high.npy"),
    ("surprised", "surprised_W021_contempt_high.npy"),
]

print("Rendering 8 Emotion Videos...")
for idx, (emo, npy_name) in enumerate(emotions, 1):
    npy_path = os.path.join(npy_dir, npy_name)
    wav_path = os.path.join(audio_dir, f"{emo}.wav")
    out_mp4 = f"demo_videos/{idx:02d}_{emo}.mp4"
    if os.path.exists(out_mp4):
        print(f"[{idx}/8] {out_mp4} already exists.")
    elif not os.path.exists(npy_path):
        print(f"[{idx}/8] Skipping {emo}: {npy_path} not found. Run generation.py first.")
    else:
        print(f"[{idx}/8] Rendering {emo} -> {out_mp4}...")
        cmd = [sys.executable, "render_preview.py", ".", npy_path, out_mp4, wav_path]
        subprocess.run(cmd, check=True)

# 2. Render 3 Diversity Samples of happy.wav
multi_dir = "colab_results/ProbTalk3D_results/generation/vqvae_pred/stage_2/0.2_multi"
if os.path.exists(multi_dir):
    print("\nRendering 3 Non-Deterministic Diversity Samples...")
    for s_idx in [1, 2, 3]:
        npy_path = os.path.join(multi_dir, f"happy_M009_happy_medium({s_idx}).npy")
        wav_path = os.path.join(audio_dir, "happy.wav")
        out_mp4 = f"demo_videos/diversity_happy_sample{s_idx}.mp4"
        if os.path.exists(out_mp4):
            print(f"{out_mp4} already exists.")
        elif not os.path.exists(npy_path):
            print(f"Skipping Sample {s_idx}: {npy_path} not found.")
        else:
            print(f"Rendering Diversity Sample {s_idx} -> {out_mp4}...")
            cmd = [sys.executable, "render_preview.py", ".", npy_path, out_mp4, wav_path]
            subprocess.run(cmd, check=True)

print(f"\nDone. All demo videos are in: {os.path.abspath('demo_videos')}")


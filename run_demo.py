#!/usr/bin/env python3
"""
ProbTalk3D Live Demo Script
Runs 1 clip inference and renders the 3D FLAME preview video in ~30 seconds.
Saves output cleanly into 'demo_videos/demo_live_output.mp4'.
"""
import os, sys, subprocess, time

print("="*60)
print("  ProbTalk3D: Live Inference & 3D Video Demo")
print("="*60)

start_time = time.time()
python_bin = sys.executable

# 1. Setup single input clip
audio_dir = "results/generation/test_audio"
clip_name = "happy"
wav_file = os.path.join(audio_dir, f"{clip_name}.wav")
os.makedirs("demo_input", exist_ok=True)
subprocess.run(f"cp {wav_file} demo_input/", shell=True, check=True)

# 2. Run inference
print("\n[Step 1/2] Running ProbTalk3D inference on CPU...")
cmd_inf = (
    f"{python_bin} generation.py "
    "folder=model_weights/ProbTalk3D/stage_2 "
    "input_path=demo_input "
    'id=["M009"] emotion=["happy"] intensity=[1] '
    "number_of_samples=1"
)
subprocess.run(cmd_inf, shell=True, check=True)
print(f"✔ Inference completed in {time.time() - start_time:.1f}s")

# 3. Locate generated .npy output
output_npy = "results/generation/vqvae_pred/stage_2/0.2/happy_M009_happy_medium.npy"
if not os.path.exists(output_npy):
    print(f"Error: expected output {output_npy} not found!")
    sys.exit(1)

# 4. Render 3D preview video into demo_videos/
os.makedirs("demo_videos", exist_ok=True)
out_video = os.path.join("demo_videos", "demo_live_output.mp4")
print(f"\n[Step 2/2] Rendering 3D FLAME mesh video to {out_video}...")
render_start = time.time()
cmd_render = f"{python_bin} render_preview.py . {output_npy} {out_video} {wav_file}"
subprocess.run(cmd_render, shell=True, check=True)
print(f"✔ 3D video rendered in {time.time() - render_start:.1f}s")

total_time = time.time() - start_time
print("\n" + "="*60)
print(f"  SUCCESS! Live Demo finished in {total_time:.1f} seconds.")
print(f"  Output Video: {os.path.abspath(out_video)}")
print("="*60)

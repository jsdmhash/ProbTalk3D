#!/usr/bin/env python3
"""
ProbTalk3D - Interactive Dynamic Generator
Allows you to choose any audio file, select any emotion (mood),
adjust intensity, set speaker character, and render the 3D talking video on demand!
All generated videos are stored cleanly in the dedicated 'interactive_videos/' folder.
"""
import os, sys, subprocess, shutil

EMOTIONS = ["neutral", "happy", "sad", "surprised", "fear", "disgusted", "angry", "contempt"]
INTENSITIES = ["low", "medium", "high"]

def main():
    print("=" * 65)
    print("  ProbTalk3D: Interactive Dynamic Face Animation")
    print("=" * 65)

    # 1. Choose Audio File
    print("\n[1] Select Speech Audio:")
    test_audio_dir = "results/generation/test_audio"
    available_wavs = sorted([f for f in os.listdir(test_audio_dir) if f.endswith(".wav")])
    for i, w in enumerate(available_wavs, 1):
        print(f"  {i}. {w}")
    print("  Or type a custom path to any .wav file")
    
    choice = input("\nEnter choice [1-8 or file path, default: 5 (happy.wav)]: ").strip()
    if not choice:
        wav_file = os.path.join(test_audio_dir, "happy.wav")
    elif choice.isdigit() and 1 <= int(choice) <= len(available_wavs):
        wav_file = os.path.join(test_audio_dir, available_wavs[int(choice)-1])
    else:
        wav_file = choice

    if not os.path.exists(wav_file):
        print(f"Error: Audio file '{wav_file}' not found.")
        return

    clip_stem = os.path.splitext(os.path.basename(wav_file))[0]
    print(f"✔ Selected Audio: {wav_file}")

    # 2. Select Emotion / Mood
    print("\n[2] Select Emotion / Mood:")
    for i, emo in enumerate(EMOTIONS, 1):
        print(f"  {i}. {emo.capitalize()}")
    emo_choice = input("\nEnter emotion [1-8, default: 2 (Happy)]: ").strip()
    if emo_choice.isdigit() and 1 <= int(emo_choice) <= len(EMOTIONS):
        selected_emo = EMOTIONS[int(emo_choice)-1]
    else:
        selected_emo = "happy"
    print(f"✔ Selected Emotion: {selected_emo}")

    # 3. Select Intensity
    print("\n[3] Select Emotion Intensity:")
    print("  0. Low (Subtle)")
    print("  1. Medium (Natural conversation)")
    print("  2. High (Exaggerated / Dramatic)")
    int_choice = input("\nEnter intensity [0-2, default: 1 (Medium)]: ").strip()
    if int_choice in ["0", "1", "2"]:
        selected_int = int(int_choice)
    else:
        selected_int = 1
    print(f"✔ Selected Intensity: {INTENSITIES[selected_int]} ({selected_int})")

    # 4. Select Speaker Character ID
    print("\n[4] Select Character Identity (e.g. M009, M003, W014, W023):")
    spk_choice = input("Enter speaker ID [default: M009]: ").strip().upper()
    selected_spk = spk_choice if spk_choice else "M009"
    print(f"✔ Selected Character: {selected_spk}")

    # 5. Select Temperature / Diversity Knob
    print("\n[5] Select Diversity Temperature (0.05 = predictable, 1.0 = expressive):")
    temp_choice = input("Enter temperature [default: 0.2]: ").strip()
    try:
        selected_temp = float(temp_choice) if temp_choice else 0.2
    except ValueError:
        selected_temp = 0.2
    print(f"✔ Selected Temperature: {selected_temp}")

    # Prepare temporary dynamic directory
    dyn_dir = "dynamic_input"
    os.makedirs(dyn_dir, exist_ok=True)
    for f in os.listdir(dyn_dir):
        os.remove(os.path.join(dyn_dir, f))
    shutil.copy(wav_file, os.path.join(dyn_dir, f"{clip_stem}.wav"))

    # Prepare dedicated output directory
    out_dir = "interactive_videos"
    os.makedirs(out_dir, exist_ok=True)

    # Run generation
    print("\n" + "=" * 65)
    print("  [Step 1/2] Generating 3D Motion with ProbTalk3D...")
    print("=" * 65)
    cmd_gen = [
        sys.executable, "generation.py",
        "folder=model_weights/ProbTalk3D/stage_2",
        f"input_path={dyn_dir}",
        f"id=[\"{selected_spk}\"]",
        f"emotion=[\"{selected_emo}\"]",
        f"intensity=[{selected_int}]",
        f"temperature={selected_temp}",
        "number_of_samples=1"
    ]
    subprocess.run(cmd_gen, check=True)

    # Locate generated output
    npy_out = f"results/generation/vqvae_pred/stage_2/{selected_temp}/{clip_stem}_{selected_spk}_{selected_emo}_{INTENSITIES[selected_int]}.npy"
    if not os.path.exists(npy_out):
        import glob
        matches = glob.glob(f"results/generation/vqvae_pred/stage_2/{selected_temp}/*.npy")
        if matches:
            npy_out = matches[0]
        else:
            print(f"Error: Generated output not found at {npy_out}")
            return

    # Render video directly into interactive_videos/
    print("\n" + "=" * 65)
    print("  [Step 2/2] Rendering 3D FLAME Mesh Video...")
    print("=" * 65)
    video_filename = f"interactive_{clip_stem}_{selected_emo}_{selected_spk}_int{selected_int}_T{selected_temp}.mp4"
    out_video_path = os.path.join(out_dir, video_filename)
    cmd_render = [sys.executable, "render_preview.py", ".", npy_out, out_video_path, wav_file]
    subprocess.run(cmd_render, check=True)

    print("\n" + "=" * 65)
    print("  SUCCESSFULLY GENERATED DYNAMIC ANIMATION!")
    print(f"  Emotion:    {selected_emo.upper()} (Intensity: {INTENSITIES[selected_int]})")
    print(f"  Character:  {selected_spk}")
    print(f"  Saved To:   {os.path.abspath(out_video_path)}")
    print("=" * 65)

if __name__ == "__main__":
    main()

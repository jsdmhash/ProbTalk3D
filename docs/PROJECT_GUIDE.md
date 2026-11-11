1# ProbTalk3D: Comprehensive Project Guide & Plain-English Reference
**Course:** ITCS 6166 · Generative AI as a Service · Fall 2026  
**Presenters:** Jyothi · Sruthi  
**Paper:** *ProbTalk3D: Non-Deterministic 3D Facial Animation from Speech* (SIGGRAPH MIG 2024, arXiv:2409.07966)

---

## 1. Executive Summary: What is This Project?

Imagine an animated movie or video game. Normally, an animator spends days manually animating a 3D character's mouth and expressions to match an actor's voice line. 

**ProbTalk3D** is a Generative AI system that automates this:
1. You provide any voice recording (a `.wav` speech file).
2. You pick an emotion (like happy, sad, angry, surprised) and a character.
3. The AI generates a realistic, synchronized 3D talking face.

### The Breakthrough: Non-Deterministic (Creative) Animation
Earlier AI models were **deterministic**: if you ran the same speech line five times, the mouth and face moved identically every time—looking robotic and repetitive.

In real life, human actors never perform a line identically twice. One take might have a subtle eyebrow raise; another might have an expressive smile or different jaw emphasis. **ProbTalk3D introduces controlled probabilistic randomness:** every run yields a slightly different, natural facial performance while staying in perfect lip-sync!

---

## 2. Complete Glossary & Decoded Abbreviations

Every single abbreviation used in this project decoded into plain English:

| Abbreviation | Full Name | Meaning in Plain English |
|---|---|---|
| **AI** | **A**rtificial **I**ntelligence | Computer programs trained to perform tasks requiring human-like intelligence. |
| **GenAI** | **Gen**erative **A**rtificial **I**ntelligence | AI that creates *new* content (video, audio, 3D mesh motion) rather than just labeling data. |
| **ProbTalk3D** | **Prob**abilistic **Talk**ing **3D** Face | The name of our project. *"Probabilistic"* means it uses probability to generate diverse motions. |
| **FLAME** | **F**aces **L**earned with an **A**rticulated **M**odel and **E**xpressions | The industry-standard digital 3D human head model consisting of **5,023 3D dots** (vertices) connected into triangles. |
| **HuBERT** | **Hu**dden-Unit **B**idirectional **E**ncoder **R**epresentations from **T**ransformers | A speech-understanding model created by Meta (Facebook) that turns sound waves into **768 numbers** describing phonetic syllables. |
| **VQ-VAE** | **V**ector **Q**uantized **V**ariational **A**uto**E**ncoder | A deep neural network that compresses complex motion into a discrete **"dictionary"** of 256 realistic human facial poses. |
| **FPS** | **F**rames **P**er **S**econd | Video frame rate. This project runs at **25 FPS** (each video frame is 40 ms). |
| **kHz** | **K**ilo**h**ert**z** | Sound sampling frequency. Audio is standardized to **16 kHz** (16,000 measurements per second). |
| **GPU** | **G**raphics **P**rocessing **U**nit | High-speed parallel computation chip used for AI training and inference (e.g., Nvidia Tesla T4 in Google Colab). |
| **CPU** | **C**entral **P**rocessing **U**nit | Standard computer processor. |
| **VRAM** | **V**ideo **R**andom **A**ccess **M**emory | Memory built directly onto the GPU card (our T4 GPU has 15 GB VRAM). |
| **NPY** | **N**um**Py** Array File | Python file format storing raw grids/matrices of numbers. Model outputs motion as `.npy`. |
| **WAV** | **Wav**eform Audio File | Standard uncompressed audio recording format. |
| **MP4** | **M**oving **P**icture Experts Group Layer **4** | Standard compressed video format playable on any computer, phone, or browser. |
| **HF** | **H**ugging **F**ace | The premier cloud platform for open-source AI models and web demos. |
| **GUI** | **G**raphical **U**ser **I**nterface | Visual web page with clickable buttons, dropdowns, and sliders. |
| **API** | **A**pplication **P**rogramming **I**nterface | Digital doorway enabling programs to send audio and receive animations over the internet. |
| **SDK** | **S**oftware **D**evelopment **K**it | A library of helper code allowing developers to communicate with an API easily. |

---

## 3. How the Machine Works: The 5-Step Pipeline

```
[Audio .wav] ──► 1. HuBERT (Ears) ──► 2. Style Fusion ──► 3. Codebook (Dictionary) ──► 4. Decoder ──► 5. Renderer ──► [Video .mp4]
```

### Station 1: The Ears (HuBERT)
- Raw audio is loaded at 16 kHz and normalized.
- HuBERT turns speech waveforms into **768 acoustic features** at 50 fps, paired to 25 fps to match video.

### Station 2: The Style Fusion
- You select the speaker character (e.g. `M009`), an emotion (e.g. `happy`), and intensity (e.g. `medium`).
- The model projects this style into a vector and multiplies it with the audio features. Now the model knows **what is being said** and **how the speaker feels**.

### Station 3: The Motion Dictionary (VQ Codebook - The Core Innovation)
- Instead of generating facial points from scratch (which could result in twisted mouths or bulging eyes), the model chooses from a **learned dictionary of 256 realistic human facial poses** (trained on real 3D scans in Stage 1).
- **Where Randomness Lives:** Older models greedily pick the single nearest pose with `argmin`. ProbTalk3D computes distances to all 256 poses, converts them to probabilities using softmax, and **rolls a weighted die** (`torch.multinomial`).
- **Temperature Knob:** 
  - Low temperature ($0.05$): The die is heavily weighted toward the closest pose $\to$ almost predictable ($0.008$ diff).
  - High temperature ($1.0$): The probabilities flatten out, giving other valid poses a chance to be drawn $\to$ expressive variety ($0.038$ diff).

### Station 4: The Face Builder (Decoder)
- The chosen dictionary codes are reconstructed into **403 FLAME numbers per frame**:
  - **300 numbers:** Shape (identity of skull/nose — zeros for generic head).
  - **100 numbers:** Facial expression (first 50 actively control brows, cheeks, and lips).
  - **3 numbers:** Jaw angles (pitch, roll, yaw — driving mouth opening).
- Saved as a `.npy` file.

### Station 5: The Preview Renderer (`render_preview.py`)
- Takes the `.npy` array, applies parameters to the FLAME 3D head model, calculates surface lighting, renders lit 3D polygons, and muxes speech audio into an `.mp4` video.

---

## 4. Dedicated Media & Output Folder Structure

To keep the repository clean and ensure that media files never pollute git, outputs are organized into dedicated folders (all protected by `.gitignore`):

```
ProbTalk3D/
├── demo_videos/                 <-- VERIFIED DEMO SUITE (Fixed Replication Library)
│   ├── 01_angry.mp4             <-- 8 Emotion test videos rendered with audio
│   ├── 02_contempt.mp4
│   ├── 03_disgusted.mp4
│   ├── 04_fear.mp4
│   ├── 05_happy.mp4
│   ├── 06_neutral.mp4
│   ├── 07_sad.mp4
│   ├── 08_surprised.mp4
│   ├── diversity_happy_sample1.mp4  <-- 3 Non-deterministic diversity takes
│   ├── diversity_happy_sample2.mp4
│   ├── diversity_happy_sample3.mp4
│   ├── demo_live_output.mp4         <-- Generated by run_demo.py
│   └── happy_full.mp4               <-- Embedded video in presentation deck
│
├── interactive_videos/          <-- DYNAMIC GENERATOR OUTPUTS (User-Created)
│   └── interactive_*.mp4        <-- Generated on-demand via interactive_generate.py
│
├── docs/                        <-- PRESENTATION & SLIDES ASSETS
│   ├── ProbTalk3D Milestone 2 Replication.pptx  (15-Slide Master Deck)
│   ├── ProbTalk3D_M2_Speaker_Script.txt         (Sruthi & Jyothi 50/50 Script)
│   ├── different_outputs_showcase.png           (Slide 11 graphic)
│   └── emotion_gallery_grid.png
│
├── run_demo.py                  <-- Automated fast demo script (~33s)
└── interactive_generate.py      <-- Interactive terminal generator tool
```

---

## 5. Step-by-Step Instructions: Fast Demo Run (`run_demo.py`)

Use this script if the professor or classmate asks: *"Can you run the code and show me a generated 3D video right now?"*

### Step 1: Open Terminal in the ProbTalk3D Directory
```bash
cd /mnt/SHARED/ProbTalk3D
```

### Step 2: Execute the Demo Script
```bash
.venv/bin/python run_demo.py
```

### What Happens Automatically (Execution Timeline):
1. **Audio Prep (0.1s):** Copies `results/generation/test_audio/happy.wav` into `demo_input/`.
2. **AI Inference (13.9s):** Runs `generation.py` using CPU. HuBERT extracts features and ProbTalk3D predicts all 144 frames of FLAME facial parameters `(1, 144, 403)`.
3. **3D Mesh Rendering (19.8s):** `render_preview.py` takes the 403 parameters, reconstructs the 5,023-vertex 3D face mesh for each frame, computes lighting normals, renders lit shaded polygons, and combines with the original audio using FFmpeg.
4. **Total Elapsed Time:** **~33.7 seconds!**

### Step 3: View the Output Video
The generated video is saved automatically to:
```bash
/mnt/SHARED/ProbTalk3D/demo_videos/demo_live_output.mp4
```
You can play it in any media player (VLC, browser, or media viewer).

---

## 6. Step-by-Step Instructions: Interactive Dynamic Generator (`interactive_generate.py`)

Use this script when you want to show off custom controls—changing speech audio, selecting any of the 8 emotions, adjusting intensity, changing characters, or dialing the temperature.

### Step 1: Launch the Interactive Tool
```bash
cd /mnt/SHARED/ProbTalk3D
.venv/bin/python interactive_generate.py
```

### Step 2: Answer the Prompts in the Terminal

```
=================================================================
  ProbTalk3D: Interactive Dynamic Face Animation
=================================================================

[1] Select Speech Audio:
  1. angry.wav
  2. contempt.wav
  3. disgusted.wav
  4. fear.wav
  5. happy.wav
  6. neutral.wav
  7. sad.wav
  8. surprised.wav
  Or type a custom path to any .wav file

Enter choice [1-8 or file path, default: 5 (happy.wav)]: 8
✔ Selected Audio: results/generation/test_audio/surprised.wav

[2] Select Emotion / Mood:
  1. Neutral
  2. Happy
  3. Sad
  4. Surprised
  5. Fear
  6. Disgusted
  7. Angry
  8. Contempt

Enter emotion [1-8, default: 2 (Happy)]: 4
✔ Selected Emotion: surprised

[3] Select Emotion Intensity:
  0. Low (Subtle)
  1. Medium (Natural conversation)
  2. High (Exaggerated / Dramatic)

Enter intensity [0-2, default: 1 (Medium)]: 2
✔ Selected Intensity: high (2)

[4] Select Character Identity (e.g. M009, M003, W014, W023):
Enter speaker ID [default: M009]: W023
✔ Selected Character: W023

[5] Select Diversity Temperature (0.05 = predictable, 1.0 = expressive):
Enter temperature [default: 0.2]: 0.3
✔ Selected Temperature: 0.3
```

### Step 3: Automatic Inference and 3D Video Rendering
The script automatically:
1. Runs `generation.py` with your exact choices.
2. Generates the 3D FLAME parameter file `.npy`.
3. Calls `render_preview.py` to create the full shaded 3D head video with audio.

### Step 4: Where is Your Video Saved?
All interactive outputs are saved directly into the **`interactive_videos/`** folder with self-explanatory names:
```bash
/mnt/SHARED/ProbTalk3D/interactive_videos/interactive_surprised_surprised_W023_int2_T0.3.mp4
```

---

## 7. Course Milestones Breakdown

| Milestone | Goal | Deliverables | Status |
|---|---|---|---|
| **M0** | Paper Selection & Proposal | Slide deck introducing ProbTalk3D paper | ✅ Done |
| **M1** | Code Review & Understanding | Repository analysis, configuration tracing, identifying 6 compatibility risks | ✅ Done |
| **M2** | Replicate Inference in Colab | Real weights inference, 8 clips verified, non-determinism & temperature proved, 3D video demo, 15-slide deck | ✅ **Done (Current)** |
| **M3** | Host Model as GenAI Service | Cloud-hosted Gradio web app on Hugging Face Spaces + Python client API | 🔜 Next Milestone |
| **Bonus** | Stretch Goals | Mobile-friendly client app (+10) and HPC model training (+10) | ⏳ Planned |

---

## 8. Presentation Q&A Cheat-Sheet

| Question Asked | Your Clear, Confident Answer |
|---|---|
| *"What is the actual input and output?"* | *"The input is a 16 kHz `.wav` speech audio file with configurable emotion and speaker settings. The output is a `(1, T, 403)` NumPy array of FLAME 3D facial parameters at 25 fps, which our script renders into an MP4 video."* |
| *"Why does the face move differently on each run?"* | *"Instead of taking the single nearest pose with argmin, the Vector Quantizer converts distances into probabilities and draws poses using multinomial sampling."* |
| *"How did you verify this mathematically?"* | *"We ran the same audio and style 3 times: the outputs differed by an average distance of 0.017. When we set `sample=false`, the outputs were bitwise identical (0.000000 difference)."* |
| *"What did you fix to make the code run?"* | *"We bypassed the obsolete vendored HuBERT copy, dynamically remapped PyTorch weight-norm parameter keys, and passed `weights_only=False` for PyTorch 2.6+."* |
| *"Can it run live right now?"* | *"Yes! We have `run_demo.py`, which runs inference and renders a 3D talking video in 33 seconds on CPU."* |
| *"Can anyone choose their own voice and mood?"* | *"Yes! We have `interactive_generate.py` which lets anyone choose any .wav, any of 8 emotions, 3 intensities, 32 characters, and diversity temperatures."* |

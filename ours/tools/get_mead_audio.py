"""Download ONLY audio.tar for the 32 test speakers from the official MEAD Part0 Google Drive.
Skips video.tar (hundreds of GB). Safe to re-run: finished files are skipped."""
from pathlib import Path
import gdown

FOLDER = "https://drive.google.com/drive/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets/3DMEAD_preprocess/MEAD_audio_source"
speakers = [s.strip() for s in (ROOT / "datasets/mead-splits-pred/test").read_text().split() if s.strip()]
print("speakers needed:", len(speakers))

files = gdown.download_folder(FOLDER, skip_download=True, quiet=True)
audio = {Path(f.path).parent.name: f.id for f in files if Path(f.path).name == "audio.tar"}
print("audio.tar files found in Drive:", len(audio))

for i, spk in enumerate(speakers, 1):
    dest = OUT / spk / "audio.tar"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[{i}/{len(speakers)}] {spk}: already have it"); continue
    if spk not in audio:
        print(f"[{i}/{len(speakers)}] {spk}: NOT FOUND in Drive"); continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{i}/{len(speakers)}] {spk}: downloading", flush=True)
    gdown.download(id=audio[spk], output=str(dest), quiet=True, resume=True)
print("DONE")

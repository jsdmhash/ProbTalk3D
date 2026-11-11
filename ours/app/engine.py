"""
ProbTalk3D inference engine for the service.

Loads the authors' pretrained model ONCE and keeps it in memory, then answers
requests (audio + style + temperature) without restarting Python. It follows
generation.py step for step (same config merge, same checkpoint loading, same
denormalisation), but without Hydra's command-line runner, so it can live inside
a web server. The authors' code is imported, never modified.
"""
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

EMOTIONS = ["neutral", "happy", "sad", "surprised", "fear", "disgusted", "angry", "contempt"]
INTENSITIES = ["low", "medium", "high"]
SPEAKERS = [l.strip() for l in (ROOT / "datasets/mead-splits-pred/train").read_text().splitlines() if l.strip()]
MAX_SECONDS = float(os.environ.get("PROBTALK_MAX_SECONDS", "20"))


class ProbTalkEngine:
    def __init__(self, folder="model_weights/ProbTalk3D/stage_2", device=None):
        t0 = time.time()
        os.chdir(ROOT)                       # the authors' code uses paths relative to the repo root
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        import torch
        from omegaconf import OmegaConf
        import framework.launch.prepare      # noqa: F401  registers the authors' config resolvers
        from framework.model.utils.tools import load_checkpoint

        # generation.py resolves paths through Hydra's runtime; outside Hydra we point
        # the same resolvers at the repo root instead.
        def code_path(p=""):
            return str(ROOT / p)

        def absolute_path(p):
            return p if os.path.isabs(p) else str(ROOT / p)

        def last_ckpt(path, version, ckpt_name="last.ckpt"):
            return str(Path(absolute_path(str(path))) / f"lightning_logs/version_{version}/checkpoints" / ckpt_name)

        for name, fn in [("code_path", code_path), ("working_path", code_path),
                         ("absolute_path", absolute_path), ("get_last_checkpoint", last_ckpt)]:
            OmegaConf.register_new_resolver(name, fn, replace=True)

        # same merge as generation.py: config saved with the weights + generation.yaml on top
        prev = OmegaConf.load(ROOT / folder / ".hydra/config.yaml")
        gen = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/generation.yaml"))
        for k in ("defaults", "hydra", "data.batch_size"):
            gen.pop(k, None)
        gen.update(folder=folder, input_path="unused", path=OmegaConf.to_container(OmegaConf.load(ROOT / "configs/path.yaml")))
        cfg = OmegaConf.merge(prev, OmegaConf.create(gen))

        from hydra.utils import instantiate
        model = instantiate(cfg.model, nfeats=cfg.nfeats, split_path=cfg.data.split_path,
                            one_hot_dim=cfg.data.one_hot_dim, resumed_training=False,
                            logger_name="none", _recursive_=False)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        load_checkpoint(model, cfg.last_ckpt_path, eval_mode=True, device=model.device)
        model.k = 1

        with open(ROOT / "datasets/scaler_exp.pkl", "rb") as f:
            self.scaler_exp = pickle.load(f)
        with open(ROOT / "datasets/scaler_jaw.pkl", "rb") as f:
            self.scaler_jaw = pickle.load(f)

        self.torch = torch
        self.model = model
        self.device = str(model.device)
        self.load_seconds = time.time() - t0

    def load_audio(self, wav_path):
        import librosa
        from framework.data.tools.collate import audio_normalize
        speech, _ = librosa.load(wav_path, sr=16000)          # any input rate -> 16 kHz, as generation.py
        if len(speech) > MAX_SECONDS * 16000:
            speech = speech[: int(MAX_SECONDS * 16000)]
        return audio_normalize(speech)

    def generate(self, wav_path, speaker="M009", emotion="happy", intensity="medium",
                 temperature=0.2, sample=True, takes=1, seed=None):
        """Returns (list of (T, 403) arrays, style key, timing dict)."""
        if speaker not in SPEAKERS:
            raise ValueError(f"unknown speaker {speaker}")
        emo = EMOTIONS.index(emotion)
        ints = 0 if emotion == "neutral" else INTENSITIES.index(intensity)   # MEAD neutral has one level only
        keyid = f"{speaker}_x_{emo}_{ints}"

        t0 = time.time()
        audio = self.load_audio(wav_path)
        self.model.temperature = float(temperature)
        if seed is not None:
            random.seed(seed); np.random.seed(seed); self.torch.manual_seed(seed)
        out = []
        with self.torch.no_grad():
            for _ in range(max(1, int(takes))):
                pred = self.model({"audio": [audio], "keyid": [keyid]}, sample=bool(sample))   # [1, T, 53]
                pred = pred.squeeze(0).detach().cpu().numpy()
                exp = self.scaler_exp.inverse_transform(pred[:, :50])
                jaw = self.scaler_jaw.inverse_transform(pred[:, 50:])
                T = pred.shape[0]
                seq = np.concatenate([np.zeros((T, 300)), exp, np.zeros((T, 50)), jaw], axis=1)  # (T, 403)
                out.append(seq)
        timing = {"audio_seconds": round(len(audio) / 16000, 2), "generate_seconds": round(time.time() - t0, 2)}
        return out, keyid, timing

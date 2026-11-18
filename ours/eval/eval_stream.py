"""
Memory-safe version of the authors' evaluation.py (same model calls, same seed, same
metric formulas). The original keeps every predicted mesh of the whole test set in RAM
(tens of GB) and was killed on a 15 GB laptop at the final step. Here MVE and LVE are
accumulated as running sums, which gives identical values. Per-sequence scores are also
written to a CSV for the per-emotion / per-speaker breakdown.

  python ours/eval/eval_stream.py folder=model_weights/ProbTalk3D/stage_2 number_of_samples=10 temperature=0.2
Outputs: ours/results/eval_<tag>_summary.json and ours/results/eval_<tag>_per_sequence.csv
"""
import csv
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _n, _t in dict(bool=bool, int=int, float=float, complex=complex, object=object, unicode=str, str=str).items():
    if not hasattr(np, _n):
        setattr(np, _n, _t)

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import hydra  # noqa: E402
import torch  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import evaluation as ev  # noqa: E402  the authors' module: helpers, scalers, dicts
from framework.model.utils.tools import load_checkpoint, detach_to_numpy  # noqa: E402
from framework.data.tools.collate import collate_motion_and_audio  # noqa: E402
from deps.flame.flame_pytorch import FLAME  # noqa: E402


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="evaluation")
def main(newcfg: DictConfig):
    prevcfg = OmegaConf.load(Path(newcfg.folder) / ".hydra/config.yaml")
    cfg = OmegaConf.merge(prevcfg, newcfg)
    pl.seed_everything(cfg.seed)

    data_module = instantiate(cfg.data)
    model = instantiate(cfg.model, nfeats=data_module.nfeats, split_path=data_module.split_path,
                        one_hot_dim=data_module.one_hot_dim, resumed_training=False,
                        logger_name="none", _recursive_=False)
    if torch.cuda.is_available():
        model.to("cuda:0")
    load_checkpoint(model, cfg.last_ckpt_path, eval_mode=True, device=model.device)
    model.temperature = cfg.temperature
    model.k = cfg.k
    dataset = getattr(data_module, f"{cfg.split}_dataset")

    with open(Path(cfg.region_path) / "lve.txt") as f:
        mouth_map = [int(i) for i in f.read().split(",")]
    with open(Path(cfg.region_path) / "fdd.txt") as f:
        upper_map = [int(i) for i in f.read().split(",")]
    templates_ref = np.load(cfg.reference_path)

    tag = f"T{cfg.temperature}" if cfg.sample else "deterministic"
    out_dir = ROOT / "ours/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"eval_{tag}_per_sequence.csv"
    n = int(cfg.number_of_samples)

    mve_sum, mve_count = 0.0, 0
    lve_all, mee_all, ce_all, fdd_all = [], [], [], []
    diversity, seq_count, frame_count = 0.0, 0, 0
    rows = []
    with torch.no_grad():
        for keyid in dataset.keyids:
            motion_gt = np.load(Path(cfg.data.motion_path) / f"{keyid}.npy")
            spk, seq, emo, ints = keyid.split("_")
            batch = collate_motion_and_audio([dataset.load_keyid(keyid)])
            ce_lve_set, motion_set, verts_gt, flamelayer = [], [], None, None
            for i in range(n):
                motion_pred = model(batch.copy(), sample=cfg.sample, generation=False)
                if motion_pred.shape[1] < motion_gt.shape[1] and i == 0:
                    motion_gt = motion_gt[:, :motion_pred.shape[1], :]
                assert motion_gt.shape[1] == motion_pred.shape[1], "Length mismatch"
                shape = np.zeros((motion_gt.shape[1], 300))
                exp_jaw = detach_to_numpy(motion_pred.squeeze(0))
                inv_exp = np.concatenate((ev.scaler_exp.inverse_transform(exp_jaw[:, :50]),
                                          np.zeros((exp_jaw.shape[0], 50))), axis=1)
                inv_jaw = ev.scaler_jaw.inverse_transform(exp_jaw[:, 50:])
                if i == 0:
                    cfg.flame.batch_size = motion_gt.shape[1]
                    flamelayer = FLAME(cfg.flame).to(model.device)
                    exp_gt = np.concatenate((np.squeeze(motion_gt[:, :, 300:350]),
                                             np.zeros((motion_gt.shape[1], 50))), axis=1)
                    verts_gt = ev.transfer_to_vert(flamelayer, shape, exp_gt, np.squeeze(motion_gt[:, :, 400:]),
                                                   Path("unused"), model.device, 0, i)
                verts_pred = ev.transfer_to_vert(flamelayer, shape, inv_exp, inv_jaw,
                                                 Path("unused"), model.device, 0, i)
                ce_lve_set.append(ev.lve_compute(list(verts_gt), list(verts_pred), mouth_map))
                motion_set.append(verts_pred)

            T = motion_gt.shape[1]
            dist = np.linalg.norm(verts_gt - motion_set[0], axis=2)
            mve_sum += float(dist.sum()); mve_count += dist.size
            lve_seq = ce_lve_set[0]
            lve_all.extend(list(lve_seq))
            mee_seq = ev.lve_compute(list(verts_gt), list(np.mean(np.stack(motion_set), axis=0)), mouth_map)
            mee_all.extend(list(mee_seq))
            ce_seq = min(ce_lve_set, key=lambda x: np.sum(x))
            ce_all.extend(list(ce_seq))
            tmpl = templates_ref.reshape(1, 5023, 3)
            fdd_seq = ev.seq_std_compute(verts_gt - tmpl, upper_map) - ev.seq_std_compute(motion_set[0] - tmpl, upper_map)
            fdd_all.append(fdd_seq)
            np.random.shuffle(motion_set)
            half = n // 2
            div_seq = 0.0
            if half >= 1:
                for a, b in zip(motion_set[:half], motion_set[half:2 * half]):
                    div_seq += np.linalg.norm(a - b, axis=2).mean(axis=1).mean()
                div_seq /= half
            diversity += div_seq

            seq_count += 1
            frame_count += T
            rows.append(dict(keyid=keyid, speaker=spk, sentence=seq, emotion=ev.emo_dict[emo],
                             intensity=ev.int_dict[ints], frames=T, mve=float(dist.mean()),
                             lve=float(np.mean(lve_seq)), mee=float(np.mean(mee_seq)),
                             ce=float(np.mean(ce_seq)), fdd=float(fdd_seq), diversity=float(div_seq)))
            print(f"Done sampling: {keyid}", flush=True)
            if seq_count % 25 == 0:
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {"temperature": float(cfg.temperature) if cfg.sample else None, "sampling": bool(cfg.sample),
               "number_of_samples": n, "sequences": seq_count, "frames": frame_count,
               "speakers": sorted({r["speaker"] for r in rows}),
               "MVE": mve_sum / mve_count, "LVE": float(np.mean(lve_all)), "MEE": float(np.mean(mee_all)),
               "CE": float(np.mean(ce_all)), "FDD": float(np.mean(fdd_all)), "Diversity": float(diversity / seq_count)}
    (out_dir / f"eval_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    for k in ["sequences", "frames", "MVE", "LVE", "MEE", "CE", "FDD", "Diversity"]:
        v = summary[k]
        print(f"{k}: {v:.4e}" if isinstance(v, float) else f"{k}: {v}", flush=True)


if __name__ == "__main__":
    main()

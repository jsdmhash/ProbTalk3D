"""
Pure-NumPy FLAME forward pass (no torch, smplx or chumpy needed).

Used by our renderer and the Hugging Face Space so they do not depend on
chumpy, which is hard to install on Python 3.12+. It reads the same
deps/flame/model/generic_model.pkl the authors ship; the only chumpy object
inside that file (shapedirs) is unpickled through a tiny stand-in class.

Input per frame: the 403 values ProbTalk3D writes
    [0:300] shape (zeros), [300:400] expression, [400:403] jaw rotation (axis-angle)
Output: (T, 5023, 3) vertex positions in metres, plus the triangle list.
"""
import pickle
import sys
import types
from pathlib import Path

import numpy as np

# FLAME 2020 eyeball vertex ranges (checked: two 2.5 cm spheres at the eye joints)
LEFT_EYE = np.arange(3931, 4477)
RIGHT_EYE = np.arange(4477, 5023)


def _install_chumpy_stub():
    """Let pickle rebuild chumpy.Ch objects as plain containers."""
    if "chumpy" in sys.modules:
        return
    class Ch:  # noqa: D401
        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {"state": state})
    pkg = types.ModuleType("chumpy")
    pkg.Ch = Ch
    for sub in ("ch", "reordering", "utils", "logic"):
        m = types.ModuleType(f"chumpy.{sub}")
        m.Ch = Ch
        m.Select = type("Select", (Ch,), {})
        m.transpose = type("transpose", (Ch,), {})
        sys.modules[f"chumpy.{sub}"] = m
        setattr(pkg, sub, m)
    sys.modules["chumpy"] = pkg


def _as_array(x):
    if hasattr(x, "x"):              # chumpy stand-in
        x = x.x
    if hasattr(x, "toarray"):        # scipy sparse
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def _rodrigues(r):
    """r: (N, 3) axis-angle -> (N, 3, 3) rotation matrices."""
    angle = np.linalg.norm(r, axis=1, keepdims=True) + 1e-8
    axis = r / angle
    c, s = np.cos(angle)[:, :, None], np.sin(angle)[:, :, None]
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = np.zeros_like(x)
    K = np.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=1).reshape(-1, 3, 3)
    I = np.eye(3)[None]
    return I + s * K + (1 - c) * (K @ K)


class FlameNP:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "deps/flame/model/generic_model.pkl"
        _install_chumpy_stub()
        with open(model_path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        self.v_template = _as_array(d["v_template"])             # (V, 3)
        self.shapedirs = _as_array(d["shapedirs"])               # (V, 3, 400)
        self.posedirs = _as_array(d["posedirs"])                 # (V, 3, 36)
        self.J_regressor = _as_array(d["J_regressor"])           # (5, V)
        self.weights = _as_array(d["weights"])                   # (V, 5)
        self.faces = np.asarray(d["f"], dtype=np.int64)          # (F, 3)
        parents = np.asarray(d["kintree_table"][0]).astype(np.int64)
        parents[0] = -1
        self.parents = parents

    def forward(self, seq):
        """seq: (T, 403) or (1, T, 403) ProbTalk3D output -> (T, V, 3)."""
        seq = np.asarray(seq, dtype=np.float64).reshape(-1, 403)
        T = seq.shape[0]
        shape, exp, jaw = seq[:, :300], seq[:, 300:400], seq[:, 400:403]
        coeffs = np.concatenate([shape, exp], axis=1)                     # (T, 400)
        v_shaped = self.v_template[None] + np.einsum("vcp,tp->tvc", self.shapedirs, coeffs)
        J = np.einsum("jv,tvc->tjc", self.J_regressor, v_shaped)          # (T, 5, 3)

        # full pose: global, neck, jaw, left eye, right eye (only jaw is predicted)
        pose = np.zeros((T, 5, 3))
        pose[:, 2] = jaw
        R = _rodrigues(pose.reshape(-1, 3)).reshape(T, 5, 3, 3)
        pose_feature = (R[:, 1:] - np.eye(3)).reshape(T, 36)
        v_posed = v_shaped + np.einsum("vcp,tp->tvc", self.posedirs, pose_feature)

        # rigid transforms down the kinematic chain
        G = np.zeros((T, 5, 4, 4))
        for j in range(5):
            Tj = np.zeros((T, 4, 4))
            Tj[:, :3, :3] = R[:, j]
            rel = J[:, j] if self.parents[j] < 0 else J[:, j] - J[:, self.parents[j]]
            Tj[:, :3, 3] = rel
            Tj[:, 3, 3] = 1
            G[:, j] = Tj if self.parents[j] < 0 else G[:, self.parents[j]] @ Tj
        # remove rest-pose joint location
        A = G.copy()
        A[:, :, :3, 3] -= np.einsum("tjab,tjb->tja", G[:, :, :3, :3], J)

        Tv = np.einsum("vj,tjab->tvab", self.weights, A)                  # (T, V, 4, 4)
        verts = np.einsum("tvab,tvb->tva", Tv[:, :, :3, :3], v_posed) + Tv[:, :, :3, 3]
        return verts

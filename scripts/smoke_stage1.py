#!/usr/bin/env python
"""Self-test for the Stage-I LifeFM KD objective (no teacher checkpoint required).

Builds a compact student, synthesizes teacher embeddings, computes the top-k SVD basis, and runs
`train_lifefm_kd` for one epoch. Verifies that the loss is finite and that the student parameters change.

Usage: python scripts/smoke_stage1.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_k_probe_pulseppg import make_student_builder          # noqa: E402
from run_lifefm_kd import PP_STUDENT_CONFIGS, train_lifefm_kd  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_cfg = PP_STUDENT_CONFIGS["BF20_NB8"]
    d_full = 512
    k = 40                                    # B_KD for the BF20 student (B_S = 40)

    n = 256
    X = torch.randn(n, 1, 400)                 # 50 Hz, 8 s windows
    Z = rng.standard_normal((n, d_full)).astype(np.float32)
    mu = Z.mean(axis=0)
    _, _, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    Vt = Vt.astype(np.float32)
    V_k = torch.as_tensor(Vt[:k], device=dev)
    ds = TensorDataset(X, torch.as_tensor(Z @ Vt[:k].T, dtype=torch.float32))

    build = make_student_builder(dev, d_full)
    student = build(0, student_cfg)
    before = [p.detach().clone() for p in student.parameters() if p.requires_grad]

    student = train_lifefm_kd(student, ds, V_k, epochs=1, dev=dev, lam=0.1)

    after = [p.detach() for p in student.parameters() if p.requires_grad]
    changed = any(not torch.equal(a, b) for a, b in zip(before, after))
    finite = all(torch.isfinite(p).all() for p in after)
    print(f"[smoke stage1] student={student_cfg} k={k} n={n} params_changed={changed} finite={finite}")
    assert changed and finite, "Stage-I smoke failed"
    print("[smoke stage1] OK")


if __name__ == "__main__":
    main()

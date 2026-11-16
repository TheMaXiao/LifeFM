#!/usr/bin/env python
"""
NS-KD Experiment: Null-Space Suppression KD vs Subspace MSE
============================================================

Compares:
  - subspace_mse  (naive baseline)
  - ns_kd (λ)     (+ null-space suppression penalty)

For k ∈ {25, 50} (or via --k-values), 3 seeds, on PulsePPG and NormWear.

NS-KD loss:
    L = ||V_k (z_s - z_t)||^2  +  λ * ||(I - V_k^T V_k) z_s||^2
      = kd_match_in_subspace   +  λ * null_space_suppression

The second term forces the student's output to lie inside V_k, eliminating
random-initialization noise in the D−k orthogonal directions that receive no
training signal from the KD objective alone.

Usage:
    # PulsePPG, GPU 0
    CUDA_VISIBLE_DEVICES=0 python -u run_ns_kd.py --teacher pulseppg 2>&1 | tee logs/ns_kd_pulseppg.log

    # NormWear, GPU 1
    CUDA_VISIBLE_DEVICES=1 python -u run_ns_kd.py --teacher normwear 2>&1 | tee logs/ns_kd_normwear.log
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from downstream_eval import run_all_tasks, set_fp16, set_gpu_probe

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

KD_LR = 1e-4
KD_BS = 64
KD_EPOCHS = 30
EVAL_HZ = 50


# ─────────────────────────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

# PulsePPG student configs: name -> (base_filters, n_block, ~params, ~compression)
PP_STUDENT_CONFIGS = {
    "BF20_NB8":  dict(base_filters=20, n_block=8),   # 189K,  150x, enc_out=40
    "BF32_NB8":  dict(base_filters=32, n_block=8),   # 463K,   61x, enc_out=64
    "BF48_NB8":  dict(base_filters=48, n_block=8),   # 1.0M,   28x, enc_out=96  (baseline)
    "BF64_NB6":  dict(base_filters=64, n_block=6),   # 1.06M,  27x, enc_out=128 (wide ~iso-param)
    "BF80_NB5":  dict(base_filters=80, n_block=5),   # 1.07M,  27x, enc_out=160 (wider ~iso-param)
    # ── scaling matrix additions (10x / 5x compression) ──
    "BF96_NB8":  dict(base_filters=96, n_block=8),   # 3.96M,  7.2x, enc_out=192 (10x bucket)
    "BF112_NB8": dict(base_filters=112, n_block=8),  # 5.37M,  5.3x, enc_out=224 (~5M, PaPaGei-S head-to-head)
}

# NormWear student configs: name -> ViT hyperparams
NW_STUDENT_CONFIGS = {
    "base":  dict(embed_dim=256, depth=4, num_heads=4, dec_dim=192, dec_depth=2),  # 6.2M, 22x, ED=256
    "wide":  dict(embed_dim=384, depth=2, num_heads=6, dec_dim=128, dec_depth=1),  # 6.4M, 21x, ED=384
    # ── scaling matrix additions ──
    "large": dict(embed_dim=384, depth=4, num_heads=6, dec_dim=192, dec_depth=2),  # 12.2M, 11x  (10x bucket)
    "mid":   dict(embed_dim=192, depth=3, num_heads=4, dec_dim=128, dec_depth=1),  # 2.88M, 47x  (30x bucket)
    "small": dict(embed_dim=128, depth=3, num_heads=4, dec_dim=96,  dec_depth=1),  # 1.48M, 92x  (100x bucket)
    # ── 500K-2M sub-range fillers (paper §size-sweep low end) ──
    "tiny":  dict(embed_dim=128, depth=2, num_heads=4, dec_dim=96,  dec_depth=1),  # 1.09M, 28x
    "xs":    dict(embed_dim=96,  depth=2, num_heads=4, dec_dim=64,  dec_depth=1),  # 0.72M, 43x
}

# PaPaGei-S student configs: ResNet1D (no MoE heads) at PaPaGei native fs=125Hz, kernel=3, stride=2.
# Teacher = PaPaGei-S = 5.79M params; emb_dim = 512.
PG_STUDENT_CONFIGS = {
    "BF32_NB12": dict(base_filters=32, n_block=12),  # 557K,  10x
    "BF32_NB8":  dict(base_filters=32, n_block=8),   # 153K,  38x
    "BF16_NB8":  dict(base_filters=16, n_block=8),   # 47K,  122x
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", choices=["pulseppg", "normwear", "papagei"], default="normwear")
    p.add_argument("--student", choices=list(PP_STUDENT_CONFIGS.keys()),
                   default="BF20_NB8",
                   help="PulsePPG student architecture (ignored for normwear/papagei)")
    p.add_argument("--nw-student", choices=list(NW_STUDENT_CONFIGS.keys()),
                   default="base",
                   help="NormWear student architecture (ignored for pulseppg/papagei)")
    p.add_argument("--pg-student", choices=list(PG_STUDENT_CONFIGS.keys()),
                   default="BF32_NB12",
                   help="PaPaGei student architecture (ignored for pulseppg/normwear)")
    p.add_argument("--baseline-only", action="store_true",
                   help="Only compute teacher baseline (no student training). "
                        "Used for one-shot evaluation of the foundation model itself.")
    p.add_argument("--k-values", type=int, nargs="+", default=[25, 50])
    p.add_argument("--adaptive-k", action="store_true",
                   help="Override --k-values with adaptive k = min(k_EV(tau), d_student) "
                        "computed from teacher pool SVD. tau set by --adaptive-tau.")
    p.add_argument("--adaptive-tau", type=float, default=0.90,
                   help="Cumulative explained-variance threshold for adaptive k (default 0.90, "
                        "matches historical k=40 experiments on D=512 mixed pool).")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                   help="NS penalty weights to sweep")
    p.add_argument("--methods", nargs="+",
                   default=["subspace_mse", "ns_kd"],
                   choices=["subspace_mse", "ns_kd", "rkd", "pse_kd", "full_mse", "ns_rkd",
                            "lower_bound", "pkt", "sp", "crd", "seed", "disco", "norm",
                            "ns_kd_isig", "ns_kd_aux_crd", "aligned_subspace_crd"],
                   help="Which method groups to run")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0],
                   help="ns_kd_isig: per-direction weight exponent (w_i = (sigma_i/sigma_max)**-alpha, mean-normalised)")
    p.add_argument("--isig-lam", type=float, default=0.1,
                   help="ns_kd_isig: null-space penalty (fixed; not swept)")
    p.add_argument("--lambdas-rkd", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                   help="NS-RKD: RKD penalty weights to sweep (lambda_r); NS penalty fixed at 0.1)")
    p.add_argument("--lambdas-crd", type=float, nargs="+", default=[0.5, 1.0],
                   help="ns_kd_aux_crd: CRD auxiliary weights to sweep (lambda_c); NS penalty fixed at 0.1)")
    p.add_argument("--pse-stages", type=int, nargs="+", default=[5, 25, 50],
                   help="PSE-KD curriculum stages (increasing k values); "
                        "final stage k must be in --k-values")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=KD_EPOCHS)
    p.add_argument("--n-data", type=int, default=10000)
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--gpu-probe", action="store_true")
    p.add_argument("--pool", type=str, default="mixed",
                   choices=["mixed", "pulsedb", "dalia", "sdb"],
                   help="KD pool source. 'mixed' = dalia+sdb+pulsedb (default); "
                        "single-source variants enable pool-ablation experiments.")
    p.add_argument("--pool-seed", type=int, default=None,
                   help="Optional pool subsample seed (for pool-instability stress tests). "
                        "If None, uses the global SEED (default).")
    p.add_argument("--exp-suffix", type=str, default="",
                   help="Append suffix to experiment output dir (for parallel runs)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Trainers
# ─────────────────────────────────────────────────────────────────────────────

def _disable_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d)):
            m.p = 0.0


def train_subspace_mse(student, ds, V_k_t, epochs, dev, lr=KD_LR):
    """Baseline: subspace MSE (no null-space term)."""
    student = student.to(dev).train()
    _disable_dropout(student)
    ld = DataLoader(ds, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    sc = torch.amp.GradScaler(device="cuda")
    V = V_k_t.to(dev)

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, t_proj in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            t_proj = t_proj.to(dev, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b).float()
                loss = F.mse_loss(z_s @ V.T, t_proj)
            sc.scale(loss).backward()
            sc.step(opt)
            sc.update()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


def train_full_mse(student, ds_full, epochs, dev, lr=KD_LR):
    """Full-D MSE KD baseline: direct MSE on all d_full teacher embedding dimensions.
    No SVD, no subspace projection — the most naive feature KD baseline.
    ds_full must yield (x_batch, z_t_full).
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    ld  = DataLoader(ds_full, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    sc  = torch.amp.GradScaler(device="cuda")

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, z_t in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            z_t = z_t.to(dev, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s  = student(x_b).float()
                loss = F.mse_loss(z_s, z_t)
            sc.scale(loss).backward()
            sc.step(opt)
            sc.update()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss  = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


def train_ns_kd_isig(student, ds, V_k_t, w_k_t, epochs, dev, lr=KD_LR, lam=0.1):
    """NS-KD with inverse-sigma per-direction reweighting.

    Loss = mean_i [ w_i * (V_k^i (z_s - z_t))^2 ] + lam * ||(I - V_k^T V_k) z_s||^2

    w_k_t: (k,) per-direction weights; expected mean(w) = 1 (normalised by caller).
    Larger w on lower-variance directions inside V_k re-balances the gradient so that
    sub-leading directions (which carry rare-task-specific signal) are not crushed by
    the few high-variance ones. Uses only teacher SVD spectrum -- no task labels.
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    ld = DataLoader(ds, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    sc = torch.amp.GradScaler(device="cuda")
    V = V_k_t.to(dev)        # (k, D)
    w = w_k_t.to(dev).float() # (k,)

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, t_proj in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            t_proj = t_proj.to(dev, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b).float()
                z_s_proj = z_s @ V.T               # (B, k)
                err2 = (z_s_proj - t_proj) ** 2    # (B, k)
                kd_loss = (err2 * w[None, :]).mean()
                z_s_recon = z_s_proj @ V           # (B, D)
                ns_loss = F.mse_loss(z_s, z_s_recon)
                loss = kd_loss + lam * ns_loss
            sc.scale(loss).backward()
            sc.step(opt)
            sc.update()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


def train_ns_kd(student, ds, V_k_t, epochs, dev, lr=KD_LR, lam=0.5):
    """NS-KD: subspace MSE + null-space suppression.

    Loss = ||V_k (z_s - z_t)||^2  +  λ * ||(I - V_k^T V_k) z_s||^2

    The second term penalises components of z_s that fall outside V_k.
    Computed as: z_s_recon = (z_s @ V^T) @ V  →  ns_loss = mse(z_s, z_s_recon)
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    ld = DataLoader(ds, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    sc = torch.amp.GradScaler(device="cuda")
    V = V_k_t.to(dev)  # (k, D)

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, t_proj in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            t_proj = t_proj.to(dev, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b).float()
                z_s_proj = z_s @ V.T          # (B, k): projection onto V_k
                kd_loss = F.mse_loss(z_s_proj, t_proj)
                z_s_recon = z_s_proj @ V       # (B, D): back-projection = V_k^T V_k z_s
                ns_loss = F.mse_loss(z_s, z_s_recon)   # ||(I - V_k^T V_k) z_s||^2 / D
                loss = kd_loss + lam * ns_loss
            sc.scale(loss).backward()
            sc.step(opt)
            sc.update()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


def train_pse_kd(student, ds_full, Vt_full_t, stages, epochs, dev, lr=KD_LR):
    """PSE-KD: Progressive Subspace Expansion.

    Trains in curriculum: subspace k1 → k2 → ... → k_final,
    splitting epochs evenly across stages.
    V_5 ⊂ V_25 ⊂ V_50 (nested top-k PCA), so no forgetting across stages.

    Vt_full_t: (max_k, D) tensor containing top-max_k PCA rows.
    stages: e.g. [5, 25, 50]
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    epochs_per_stage = max(1, epochs // len(stages))

    for stage_k in stages:
        V = Vt_full_t[:stage_k]   # (stage_k, D)
        ld  = DataLoader(ds_full, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
        opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
        sc  = torch.amp.GradScaler(device="cuda")
        best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
        for _ in range(epochs_per_stage):
            ep_loss = []
            for x_b, z_t in ld:
                x_b = x_b.to(dev, non_blocking=True)
                if x_b.ndim == 2:
                    x_b = x_b.unsqueeze(1)
                z_t  = z_t.to(dev, non_blocking=True).float()
                t_proj = z_t @ V.T   # (B, stage_k): on-the-fly projection
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    z_s  = student(x_b).float()
                    loss = F.mse_loss(z_s @ V.T, t_proj)
                sc.scale(loss).backward()
                sc.step(opt)
                sc.update()
                ep_loss.append(loss.item())
            ml = float(np.mean(ep_loss))
            if ml < best_loss:
                best_loss  = ml
                best_state = copy.deepcopy(student.state_dict())
        student.load_state_dict(best_state)  # carry best of this stage forward

    student.eval()
    return student


# ─────────────────────────────────────────────────────────────────────────────
#  RKD helpers + trainer
# ─────────────────────────────────────────────────────────────────────────────

def _rkd_distance_loss(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """RKD distance-wise loss over all pairs in a batch."""
    # pairwise L2 distances
    def _pdist(z):
        dot = z @ z.T
        sq  = dot.diagonal()
        d2  = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2 * dot).clamp(min=1e-12)
        return d2.sqrt()

    td = _pdist(z_t)   # (B, B)
    sd = _pdist(z_s)
    # normalise by mean distance (eq. 2 in Park et al. 2019)
    td = td / (td.mean() + 1e-8)
    sd = sd / (sd.mean() + 1e-8)
    # upper-triangle only (avoid double-counting)
    B = z_t.shape[0]
    mask = torch.triu(torch.ones(B, B, dtype=torch.bool, device=z_t.device), diagonal=1)
    return F.huber_loss(sd[mask], td[mask], delta=1.0)


def _rkd_angle_loss(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """RKD angle-wise loss over all triplets (i, j, k) in a batch."""
    # diff vectors e[i,j] = z[j] - z[i], shape (B, B, D)
    td = F.normalize(z_t.unsqueeze(0) - z_t.unsqueeze(1), p=2, dim=-1)  # (B, B, D)
    sd = F.normalize(z_s.unsqueeze(0) - z_s.unsqueeze(1), p=2, dim=-1)
    # cos of angle at anchor i: einsum via bmm → (B, B, B)
    cos_t = torch.bmm(td, td.permute(0, 2, 1))
    cos_s = torch.bmm(sd, sd.permute(0, 2, 1))
    return F.huber_loss(cos_s, cos_t, delta=1.0)


def train_rkd(student, ds_full, epochs, dev, lr=KD_LR, mu_d=25.0, mu_a=50.0):
    """RKD: Relational Knowledge Distillation (Park et al., 2019).

    Loss = mu_d * L_distance + mu_a * L_angle
    ds_full must yield (x_batch, z_t_full) with full-dim teacher embeddings.

    NOTE: loss is computed in fp32 (outside autocast) to avoid fp16 gradient
    overflow that causes GradScaler to skip every optimizer step.
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    ld  = DataLoader(ds_full, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, z_t in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            z_t = z_t.to(dev, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b)           # fp16 forward for speed
            z_s = z_s.float()                # cast to fp32 before loss
            # ── fp32 loss (stable gradients, no GradScaler needed) ──
            loss = mu_d * _rkd_distance_loss(z_s, z_t) + mu_a * _rkd_angle_loss(z_s, z_t)
            loss.backward()
            opt.step()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss  = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


def train_ns_rkd(student, ds_full, V_k_t, epochs, dev, lr=KD_LR,
                 lam=0.1, lam_r=0.5, mu_d=25.0, mu_a=50.0):
    """NS-RKD: Subspace MSE + Null-Space Suppression + Subspace-Projected RKD.

    L = ||V_k(z_s - z_t)||^2                            (subspace MSE)
      + lam  * ||(I - V_k^T V_k) z_s||^2               (null-space suppression)
      + lam_r * [mu_d L_dist(Pz_s,Pz_t) + mu_a L_ang(Pz_s,Pz_t)]  (subspace RKD)

    where P = V_k projection (k-dim subspace).

    Unlike full-D RKD, the relational loss operates on k-dim projections:
      - Numerically stable: bmm tensors are B×k (not B×D), no fp16 overflow
      - Focused: relational structure learning on high-variance directions only
      - No GradScaler needed (fp32 loss after fp16 forward)

    ds_full: yields (x_batch, z_t_full) with full-dim teacher embeddings.
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    ld = DataLoader(ds_full, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    V = V_k_t.to(dev)   # (k, D)

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, z_t in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            z_t    = z_t.to(dev, non_blocking=True).float()
            t_proj = z_t @ V.T    # (B, k): teacher subspace projection

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b)    # fp16 forward
            z_s = z_s.float()         # fp32 before all loss terms

            # ── NS-KD terms ──
            z_s_proj  = z_s @ V.T    # (B, k)
            kd_loss   = F.mse_loss(z_s_proj, t_proj)
            z_s_recon = z_s_proj @ V  # (B, D)
            ns_loss   = F.mse_loss(z_s, z_s_recon)

            # ── Subspace RKD terms (k-dim, fp32, stable) ──
            rkd_loss  = (mu_d * _rkd_distance_loss(z_s_proj, t_proj) +
                         mu_a * _rkd_angle_loss(z_s_proj, t_proj))

            loss = kd_loss + lam * ns_loss + lam_r * rkd_loss
            loss.backward()
            opt.step()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss  = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-arch NS-KD variants  (V1: aux-RKD head | V2: cov | V3: auto-λ)
# ─────────────────────────────────────────────────────────────────────────────
#
# Design constraints (per user):
#   - Fixed teacher subspace V_k from the FULL KD pool (no per-batch SVD).
#   - Adaptive k = min(k_EV(0.99), d_student) is supplied by setup_*().
#   - V1 keeps RKD on the FULL D-dim teacher embedding (NOT inside V_k) — this
#     was the failure mode of train_ns_rkd. The student gets an auxiliary head
#     h: R^D→R^D applied only on the RKD branch; NS-KD continues to operate on
#     the raw student output z_s. The head is discarded after training so the
#     deployed student is unchanged.
#   - V2 adds second-moment matching on the in-subspace projection so that the
#     student preserves the teacher's *cloud shape* in V_k as well as its
#     pointwise position.
#   - V3 replaces the fixed λ with an online EMA balancer that tracks the
#     in-subspace MSE / null-space penalty ratio per student.

class _RKDAuxHead(nn.Module):
    """Small projection head used only by V1 during training, then discarded.
    LayerNorm → Linear → GELU → Linear, output dim = D (teacher dim).
    """
    def __init__(self, D: int, hidden: int = None):
        super().__init__()
        h = hidden or D
        self.ln = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, D)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.ln(x))))


def train_ns_kd_aux_rkd(student, ds_full, V_k_t, epochs, dev, lr=KD_LR,
                        lam=0.1, lam_r=1.0, mu_d=25.0, mu_a=50.0):
    """V1: NS-KD + auxiliary FULL-D RKD via a learnable alignment head.

    L = ||V_k(z_s - z_t)||^2                       (subspace MSE)
      + lam   * ||(I - V_k^T V_k) z_s||^2          (null-space suppression)
      + lam_r * RKD(h(z_s), z_t)                   (relational on FULL D)

    The head h decouples the relational target from the geometric target so the
    student's z_s isn't pulled in two competing directions at every coordinate.
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    D = V_k_t.shape[1]
    head = _RKDAuxHead(D).to(dev).train()
    ld = DataLoader(ds_full, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    params = list(student.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    V = V_k_t.to(dev)

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, z_t in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            z_t = z_t.to(dev, non_blocking=True).float()
            t_proj = z_t @ V.T

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b)
            z_s = z_s.float()

            # NS-KD core (subspace + null-space)
            z_s_proj = z_s @ V.T
            kd_loss = F.mse_loss(z_s_proj, t_proj)
            z_s_recon = z_s_proj @ V
            ns_loss = F.mse_loss(z_s, z_s_recon)

            # Auxiliary full-D RKD via head h(z_s) vs z_t (no subspace restriction)
            z_s_rkd = head(z_s)
            rkd_loss = (mu_d * _rkd_distance_loss(z_s_rkd, z_t) +
                        mu_a * _rkd_angle_loss(z_s_rkd, z_t))

            loss = kd_loss + lam * ns_loss + lam_r * rkd_loss
            loss.backward()
            opt.step()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


def train_aux_rkd(student, ds_full, V_k_t, epochs, dev, lr=KD_LR,
                  lam_r=1.0, mu_d=25.0, mu_a=50.0):
    """V1b: Subspace MSE + auxiliary FULL-D RKD via alignment head.
       Same as V1 but DROPS the null-space penalty.

    L = ||V_k(z_s - z_t)||^2  +  lam_r * RKD(h(z_s), z_t)

    Rationale (from V1 analysis):
      - V1's null-space term ||(I - V_k^T V_k) z_s||^2 amputates D-k dims of
        z_s. In rich-student regimes (e.g. xarch A) these dims carry the
        relational info RKD/CRD exploit, so amputating them defeats the
        auxiliary RKD head.
      - V1b keeps subspace MSE as a SOFT geometric anchor (no amputation)
        and lets the head h carry the relational target on the full-D z_s.
      - Predicted to: (a) match V1's win in capacity-floor regimes, AND
        (b) close the gap to RKD/CRD in rich-student regimes.
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    D = V_k_t.shape[1]
    head = _RKDAuxHead(D).to(dev).train()
    ld = DataLoader(ds_full, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    params = list(student.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    V = V_k_t.to(dev)

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, z_t in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            z_t = z_t.to(dev, non_blocking=True).float()
            t_proj = z_t @ V.T

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b)
            z_s = z_s.float()

            kd_loss = F.mse_loss(z_s @ V.T, t_proj)
            z_s_rkd = head(z_s)
            rkd_loss = (mu_d * _rkd_distance_loss(z_s_rkd, z_t) +
                        mu_a * _rkd_angle_loss(z_s_rkd, z_t))
            loss = kd_loss + lam_r * rkd_loss
            loss.backward()
            opt.step()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


class _CRDProjHead(nn.Module):
    """Projection head for contrastive losses. LN→Linear→GELU→Linear, output L2-normalised."""
    def __init__(self, in_dim: int, proj_dim: int = 128, hidden: int = None):
        super().__init__()
        h = hidden or max(in_dim, proj_dim)
        self.ln  = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, proj_dim)
    def forward(self, x):
        return F.normalize(self.fc2(self.act(self.fc1(self.ln(x)))), dim=1)


def _crd_infonce(z_s_n, z_t_n, tau=0.07):
    logits = z_s_n @ z_t_n.T / tau
    labels = torch.arange(z_s_n.shape[0], device=z_s_n.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def train_ns_kd_aux_crd(student, ds_full, V_k_t, epochs, dev, lr=KD_LR,
                        lam=0.1, lam_c=1.0, tau=0.07, proj_dim=128):
    """Method 1: NS-KD core (subspace MSE + null-space) + auxiliary FULL-space CRD.

    L = ||V_k(z_s - z_t)||^2
      + lam   * ||(I - V_k^T V_k) z_s||^2
      + lam_c * InfoNCE( h_s(z_s) , h_t(z_t) )    on FULL D, not the subspace

    Heads h_s, h_t are LN→MLP→L2-normalise. They keep the contrastive objective
    in a separate latent so the student's subspace coords aren't pulled in two
    directions, while letting CRD exploit full-D relational info — the part
    NS-KD can't transfer, especially in cross-arch / rich-student regimes.
    Heads are discarded after KD; deployed student is unchanged.
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    D = V_k_t.shape[1]
    head_s = _CRDProjHead(D, proj_dim).to(dev).train()
    head_t = _CRDProjHead(D, proj_dim).to(dev).train()
    ld = DataLoader(ds_full, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    params = list(student.parameters()) + list(head_s.parameters()) + list(head_t.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    V = V_k_t.to(dev)

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, z_t in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            z_t = z_t.to(dev, non_blocking=True).float()
            t_proj = z_t @ V.T

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b)
            z_s = z_s.float()

            # NS-KD core
            z_s_proj = z_s @ V.T
            kd_loss = F.mse_loss(z_s_proj, t_proj)
            ns_loss = F.mse_loss(z_s, z_s_proj @ V)

            # Auxiliary CRD on full-D via heads
            zs_n = head_s(z_s)
            zt_n = head_t(z_t)
            crd_loss = _crd_infonce(zs_n, zt_n, tau=tau)

            loss = kd_loss + lam * ns_loss + lam_c * crd_loss
            loss.backward()
            opt.step()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


def train_aligned_subspace_crd(student, ds_full, V_k_t, epochs, dev, lr=KD_LR,
                               lam=0.1, tau=0.07, proj_dim=128):
    """Method 2: Aligned-subspace CRD.

    Idea: avoid rigid coordinate regression. Both teacher (already projected to
    top-k subspace) and student go through small projection heads into a
    common latent, where contrastive matching is performed. NS-KD's null-space
    suppression on the raw student is kept as a regulariser to keep the student
    output clean (so downstream linear probes still work on z_s, not h_s(z_s)).

    L = lam * ||(I - V_k^T V_k) z_s||^2
      + InfoNCE( h_s(z_s) , h_t(V_k z_t) )

    The teacher input to its head is the k-dim projection (V_k z_t), so the
    contrastive geometry inherits NS-KD's "high-variance subspace = generalisable"
    prior, but the matching itself is a soft contrastive instead of a rigid MSE.
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    D = V_k_t.shape[1]
    k = V_k_t.shape[0]
    head_s = _CRDProjHead(D, proj_dim).to(dev).train()
    head_t = _CRDProjHead(k, proj_dim).to(dev).train()
    ld = DataLoader(ds_full, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    params = list(student.parameters()) + list(head_s.parameters()) + list(head_t.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    V = V_k_t.to(dev)

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, z_t in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            z_t = z_t.to(dev, non_blocking=True).float()
            t_proj = z_t @ V.T  # (B, k)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b)
            z_s = z_s.float()

            # Null-space regulariser on raw student
            z_s_proj = z_s @ V.T
            ns_loss = F.mse_loss(z_s, z_s_proj @ V)

            # Aligned contrastive: student raw → head_s ; teacher subspace → head_t
            zs_n = head_s(z_s)
            zt_n = head_t(t_proj)
            crd_loss = _crd_infonce(zs_n, zt_n, tau=tau)

            loss = crd_loss + lam * ns_loss
            loss.backward()
            opt.step()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


def train_ns_kd_cov(student, ds, V_k_t, epochs, dev, lr=KD_LR,
                    lam=0.1, beta=0.5):
    """V2: NS-KD + subspace covariance matching.

    L = ||V_k(z_s - z_t)||^2
      + lam  * ||(I - V_k^T V_k) z_s||^2
      + beta * ||Cov(P z_s) - Cov(P z_t)||_F^2 / k

    Captures the second-moment ("shape") of the student's projected cloud,
    which pure pointwise MSE doesn't constrain when the per-sample residual
    is anisotropic. Stays inside V_k so it inherits NS-KD's pool robustness.
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    ld = DataLoader(ds, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    sc = torch.amp.GradScaler(device="cuda")
    V = V_k_t.to(dev)
    k = V.shape[0]

    def _cov(z):  # (B, k) → (k, k)
        zc = z - z.mean(dim=0, keepdim=True)
        return (zc.T @ zc) / max(1, z.shape[0] - 1)

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, t_proj in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            t_proj = t_proj.to(dev, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b).float()
                z_s_proj = z_s @ V.T
                kd_loss = F.mse_loss(z_s_proj, t_proj)
                z_s_recon = z_s_proj @ V
                ns_loss = F.mse_loss(z_s, z_s_recon)
            # Covariance loss in fp32 outside autocast (more stable for k×k mat)
            C_s = _cov(z_s_proj)
            C_t = _cov(t_proj)
            cov_loss = (C_s - C_t).pow(2).mean()
            loss = kd_loss + lam * ns_loss + beta * cov_loss
            sc.scale(loss).backward()
            sc.step(opt); sc.update()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


def train_ns_kd_auto(student, ds, V_k_t, epochs, dev, lr=KD_LR,
                     target_ratio=0.1, ema=0.9, lam_clip=(1e-3, 10.0)):
    """V3: NS-KD with online auto-balanced λ (per-student adaptation).

    Adaptive λ: λ_t is updated each step so that
        lam_t * ema(ns_loss) ≈ target_ratio * ema(kd_loss)
    This makes the null-space penalty's contribution a fixed fraction of the
    KD term regardless of the student's natural output magnitude — addressing
    the per-student calibration problem (small students saturate ns_loss
    quickly; large ones never reach saturation under fixed λ).

    Combined with the adaptive k (already supplied by setup_*), this is the
    "fully adaptive" NS-KD variant.
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    ld = DataLoader(ds, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    sc = torch.amp.GradScaler(device="cuda")
    V = V_k_t.to(dev)

    ema_kd, ema_ns = None, None
    lam_cur = 0.1
    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, t_proj in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            t_proj = t_proj.to(dev, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b).float()
                z_s_proj = z_s @ V.T
                kd_loss = F.mse_loss(z_s_proj, t_proj)
                z_s_recon = z_s_proj @ V
                ns_loss = F.mse_loss(z_s, z_s_recon)
                loss = kd_loss + lam_cur * ns_loss
            sc.scale(loss).backward()
            sc.step(opt); sc.update()
            # Update EMAs (fp32, no grad)
            with torch.no_grad():
                kdv = float(kd_loss.detach())
                nsv = float(ns_loss.detach())
                ema_kd = kdv if ema_kd is None else ema * ema_kd + (1 - ema) * kdv
                ema_ns = nsv if ema_ns is None else ema * ema_ns + (1 - ema) * nsv
                if ema_ns > 1e-12:
                    lam_cur = float(np.clip(target_ratio * ema_kd / ema_ns,
                                            lam_clip[0], lam_clip[1]))
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss = ml
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.eval()
    return student


# ─────────────────────────────────────────────────────────────────────────────
#  External representation-KD baselines (PKT, SP, CRD, SEED, DisCo, NORM)
# ─────────────────────────────────────────────────────────────────────────────
#
# All baselines operate on full-D teacher embeddings (ds_full = (X, z_t_full))
# with a student that produces D-dim output. They mirror the published loss
# formulations adapted to our SSL no-classes setting (no logit KD applies).

def _kd_loop(student, ds_full, epochs, dev, lr, loss_fn, fp16=True):
    """Generic train loop with best-state checkpointing.

    loss_fn(z_s, z_t) -> scalar tensor (called in fp32 even when forward is fp16).
    """
    student = student.to(dev).train()
    _disable_dropout(student)
    ld  = DataLoader(ds_full, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)

    best_loss, best_state = float("inf"), copy.deepcopy(student.state_dict())
    for _ in range(epochs):
        ep_loss = []
        for x_b, z_t in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            z_t = z_t.to(dev, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            if fp16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    z_s = student(x_b)
                z_s = z_s.float()
            else:
                z_s = student(x_b).float()
            loss = loss_fn(z_s, z_t)
            loss.backward()
            opt.step()
            ep_loss.append(loss.item())
        ml = float(np.mean(ep_loss))
        if ml < best_loss:
            best_loss = ml
            best_state = copy.deepcopy(student.state_dict())
    student.load_state_dict(best_state)
    student.eval()
    return student


def train_pkt(student, ds_full, epochs, dev, lr=KD_LR, eps=1e-7):
    """PKT — Probabilistic Knowledge Transfer (Passalis & Tefas, ECCV 2018).

    Compute pairwise cosine similarities within the batch, convert each row
    to a probability distribution (positive cosine + L1 normalize), then
    minimise KL(p_t || p_s).
    """
    def _cos_to_prob(z):
        z = F.normalize(z, dim=1, eps=eps)
        s = (z @ z.T + 1.0) * 0.5            # in [0, 1]
        s = s - torch.diag(torch.diag(s))    # zero self-sim → exclude diagonal
        s = s / (s.sum(dim=1, keepdim=True) + eps)
        return s

    def _loss(z_s, z_t):
        p_s = _cos_to_prob(z_s)
        p_t = _cos_to_prob(z_t)
        # KL(p_t || p_s) summed over rows
        return (p_t * (torch.log(p_t + eps) - torch.log(p_s + eps))).sum(dim=1).mean()

    return _kd_loop(student, ds_full, epochs, dev, lr, _loss)


def train_sp(student, ds_full, epochs, dev, lr=KD_LR):
    """SP — Similarity-Preserving KD (Tung & Mori, ICCV 2019).

    Match row-normalized batch Gram matrices (B×B): G = norm(z z^T).
    """
    def _gram(z):
        g = z @ z.T
        return F.normalize(g, p=2, dim=1)

    def _loss(z_s, z_t):
        return F.mse_loss(_gram(z_s), _gram(z_t))

    return _kd_loop(student, ds_full, epochs, dev, lr, _loss)


def train_crd(student, ds_full, epochs, dev, lr=KD_LR, tau=0.07):
    """CRD — Contrastive Representation Distillation (Tian et al., ICLR 2020).

    Batch-only InfoNCE: positives = matched (z_s_i, z_t_i); negatives = the
    other B-1 teacher embeddings in the batch. Symmetric (s2t + t2s).
    """
    def _loss(z_s, z_t):
        z_s_n = F.normalize(z_s, dim=1)
        z_t_n = F.normalize(z_t, dim=1)
        logits_st = z_s_n @ z_t_n.T / tau          # (B, B)
        labels = torch.arange(z_s.shape[0], device=z_s.device)
        l1 = F.cross_entropy(logits_st, labels)
        l2 = F.cross_entropy(logits_st.T, labels)
        return 0.5 * (l1 + l2)

    return _kd_loop(student, ds_full, epochs, dev, lr, _loss)


def train_seed(student, ds_full, epochs, dev, lr=KD_LR, tau_s=0.1, tau_t=0.04):
    """SEED — Self-supervised Distillation with queue (Fang et al., ICLR 2021).

    We use a *batch-as-queue* simplification (no momentum encoder, no growing
    queue): for each sample i, build similarity distribution over the other
    B-1 teacher embeddings under both student and teacher, then minimize
    KL(p_t || p_s). Mirrors SEED's contrastive distribution distillation
    objective at small scale.
    """
    def _loss(z_s, z_t):
        z_s_n = F.normalize(z_s, dim=1)
        z_t_n = F.normalize(z_t, dim=1)
        sims_s = z_s_n @ z_t_n.T / tau_s   # student logits over teacher anchors
        sims_t = z_t_n @ z_t_n.T / tau_t   # teacher's own distribution
        B = z_s.shape[0]
        mask = torch.eye(B, device=z_s.device, dtype=torch.bool)
        sims_s = sims_s.masked_fill(mask, -1e9)
        sims_t = sims_t.masked_fill(mask, -1e9)
        log_p_s = F.log_softmax(sims_s, dim=1)
        p_t = F.softmax(sims_t, dim=1)
        return -(p_t * log_p_s).sum(dim=1).mean()

    return _kd_loop(student, ds_full, epochs, dev, lr, _loss)


def train_disco(student, ds_full, epochs, dev, lr=KD_LR,
                w_mse=1.0, w_cons=1.0, tau=0.2):
    """DisCo — Distilled Contrastive learning (Gao et al., ECCV 2022).

    Without paired augmentations available here we keep the spirit:
    - teacher–student feature consistency (MSE on L2-normalized features)
    - student–teacher contrastive alignment (InfoNCE within batch)
    L = w_mse * (1 - mean cos(z_s, z_t)) + w_cons * InfoNCE(z_s, z_t)
    """
    def _loss(z_s, z_t):
        z_s_n = F.normalize(z_s, dim=1)
        z_t_n = F.normalize(z_t, dim=1)
        cos_loss = 1.0 - (z_s_n * z_t_n).sum(dim=1).mean()
        logits = z_s_n @ z_t_n.T / tau
        labels = torch.arange(z_s.shape[0], device=z_s.device)
        nce = 0.5 * (F.cross_entropy(logits, labels) +
                     F.cross_entropy(logits.T, labels))
        return w_mse * cos_loss + w_cons * nce

    return _kd_loop(student, ds_full, epochs, dev, lr, _loss)


def train_norm(student, ds_full, epochs, dev, lr=KD_LR):
    """NORM — Normalized direction matching KD (Liu et al., ICLR 2023).

    Match L2-normalized features and L2-normalized batch-mean-centered
    features ("direction" + "shape"):
        L = MSE(L2(z_s), L2(z_t)) + MSE(L2(z_s - μ_s), L2(z_t - μ_t))
    Captures both per-sample direction and the within-batch geometry.
    """
    def _loss(z_s, z_t):
        zs1 = F.normalize(z_s, dim=1)
        zt1 = F.normalize(z_t, dim=1)
        zs2 = F.normalize(z_s - z_s.mean(dim=0, keepdim=True), dim=1)
        zt2 = F.normalize(z_t - z_t.mean(dim=0, keepdim=True), dim=1)
        return F.mse_loss(zs1, zt1) + F.mse_loss(zs2, zt2)

    return _kd_loop(student, ds_full, epochs, dev, lr, _loss)


def train_lower_bound(student, *_args, **_kwargs):
    """Lower bound: random-init student with NO knowledge distillation."""
    return student.eval()


# ─────────────────────────────────────────────────────────────────────────────
#  PulsePPG setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_pulseppg(args, dev, out_dir, student_name="BF20_NB8"):
    """Load teacher, build mixed pool, return (X, Z, build_student, eval_fn, teacher_baseline)."""
    import run_k_probe_pulseppg as ppg_mod

    print("\n-- Loading PulsePPG teacher --")
    teacher, teacher_name = ppg_mod.load_pulseppg_teacher(dev)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {teacher_name}, params={n_teacher:,}")

    cache_X = out_dir / "X_mix.pt"
    cache_Z = out_dir / "Z_mix.npy"
    if cache_X.exists() and cache_Z.exists():
        print("-- Loading cached embeddings --")
        X_all = torch.load(str(cache_X), weights_only=True)
        Z_all = np.load(str(cache_Z))
        print(f"  X={tuple(X_all.shape)}, Z={Z_all.shape}")
    else:
        pool = getattr(args, "pool", "mixed")
        sources_filter = None if pool == "mixed" else [pool]
        pool_seed = getattr(args, "pool_seed", None)
        print(f"-- Building pool: pool={pool}, n={args.n_data}, sources_filter={sources_filter}, pool_seed={pool_seed}")
        X_all, Z_all = ppg_mod.build_mixed_pool(teacher, dev, args.n_data,
                                                sources_filter=sources_filter,
                                                pool_seed=pool_seed)
        torch.save(X_all, str(cache_X))
        np.save(str(cache_Z), Z_all)

    d_full = int(Z_all.shape[1])
    build_student = ppg_mod.make_student_builder(dev, d_full)
    student_cfg = PP_STUDENT_CONFIGS[student_name]
    print(f"  Student: {student_name} (bf={student_cfg['base_filters']}, nb={student_cfg['n_block']})")

    data_dirs = {
        "wesad":         "pulseppg/data/datasets/wesad",
        "sdb":           "pulseppg/data/datasets/sdb",
        "wfppg":         "pulseppg/data/datasets/wfppg",
        "dalia":         "pulseppg/data/datasets/dalia",
        "ppgbp":         "pulseppg/data/datasets/ppgbp",
        "mimicperform":  "pulseppg/data/datasets/mimicperform",
        "ppgarrhythmia": "pulseppg/data/datasets/ppgarrhythmia",
        "earset":        "pulseppg/data/datasets/earset",
        "bidmc":         "pulseppg/data/datasets/bidmc",
        "vitaldb":       "pulseppg/data/datasets/vitaldb",
    }

    teacher_cache = out_dir / "teacher_baseline.json"
    if teacher_cache.exists():
        print("-- Loading cached teacher baseline --")
        teacher_baseline = json.loads(teacher_cache.read_text())
    else:
        teacher_baseline = {}

    # Determine which TASK_REGISTRY tasks are missing from the cache; compute
    # teacher metrics only on those (cheaper than full re-evaluation).
    from downstream_eval import TASK_REGISTRY
    desired = sorted(t for t in TASK_REGISTRY if t != "sportppg_activity")
    missing = [t for t in desired if t not in teacher_baseline]
    if missing:
        print(f"-- Computing teacher baseline on {len(missing)} missing tasks: {missing}")
        t_results = run_all_tasks(
            teacher, str(dev), data_dirs, tasks=missing,
            hz=EVAL_HZ, skip_missing=True, max_samples=args.max_samples,
        )
        for t, m in t_results.items():
            teacher_baseline[t] = {"value": float(m["primary"]),
                                   "metric": m["primary_metric"]}
        teacher_cache.write_text(json.dumps(teacher_baseline, indent=2))

    eval_tasks = list(teacher_baseline.keys())

    def eval_fn(student):
        student.eval()
        res = run_all_tasks(
            student, str(dev), data_dirs, tasks=eval_tasks,
            hz=EVAL_HZ, skip_missing=True, max_samples=args.max_samples,
        )
        rets = {}
        for tname, m in res.items():
            ref = teacher_baseline[tname]
            val = float(m["primary"])
            if ppg_mod.metric_is_higher_better(ref["metric"]):
                rets[tname] = val / ref["value"] if ref["value"] != 0 else float("nan")
            else:
                rets[tname] = ref["value"] / val if val != 0 else float("nan")
        return float(np.nanmean(list(rets.values()))), float(np.nanmin(list(rets.values()))), rets

    def build_fn(seed):
        return build_student(seed, student_cfg)

    return X_all, Z_all, d_full, build_fn, eval_fn, n_teacher


# ─────────────────────────────────────────────────────────────────────────────
#  NormWear setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_normwear(args, dev, out_dir):
    """Load teacher, return (X, Z, build_student, eval_fn, teacher_baseline)."""
    import normwear_kd_experiment as nkd

    D_NW = 768
    cfg = nkd.ExperimentConfig()

    nw_student_name = getattr(args, "nw_student", "base")
    nw_cfg = NW_STUDENT_CONFIGS[nw_student_name]

    print("\n-- Loading NormWear teacher --")
    teacher = nkd.NormWearTeacherEncoder(
        cfg.teacher_checkpoint,
        device=dev,
        sampling_rate=cfg.teacher_sampling_rate,
        patch_pool=cfg.teacher_patch_pool,
        channel_pool=cfg.teacher_channel_pool,
    )
    n_teacher = nkd.count_parameters(teacher)
    print(f"  Teacher params: {n_teacher:,}")

    # Use cached 10K mixed embeddings from k-sweep
    KS_DIR = Path("kd_sweep_results/normwear_preliminary/exp_k_sweep")
    X_all = torch.load(str(KS_DIR / "X_mix_10k.pt"), weights_only=True)
    Z_all = np.load(str(KS_DIR / "Z_mix_10k.npy"))
    print(f"  Loaded cached embeddings: X={tuple(X_all.shape)}, Z={Z_all.shape}")
    d_full = int(Z_all.shape[1])

    print(f"  NormWear student: {nw_student_name} "
          f"(embed_dim={nw_cfg['embed_dim']}, depth={nw_cfg['depth']}, "
          f"num_heads={nw_cfg['num_heads']})")

    def build_fn(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        encoder = nkd.NormWearStudentEncoder(
            device=dev,
            sampling_rate=cfg.teacher_sampling_rate,
            patch_pool=cfg.teacher_patch_pool,
            channel_pool=cfg.teacher_channel_pool,
            embed_dim=nw_cfg["embed_dim"],
            depth=nw_cfg["depth"],
            num_heads=nw_cfg["num_heads"],
            decoder_embed_dim=nw_cfg["dec_dim"],
            decoder_depth=nw_cfg["dec_depth"],
            mlp_ratio=4.0,
            fuse_freq=2,
            nvar=1,
        ).to(dev)
        student = nkd.StudentOutputAdapter(encoder).to(dev)
        with torch.no_grad():
            _x = torch.zeros(2, 1, 500, device=dev)
            student.attach_output_head(_x, dev, output_dim=D_NW)
        return student.to(dev)

    # Teacher baseline
    baseline_cache = Path("kd_sweep_results/normwear_preliminary/exp_vw_kd_stage1/cache/teacher_baseline.json")
    teacher_baseline = json.loads(baseline_cache.read_text())
    eval_tasks = list(teacher_baseline.keys())
    NW_TASK_HIGHER = {
        "f1_macro": True, "accuracy": True, "r2": True,
        "mae": False, "rmse": False,
    }

    DATA_DIRS = {
        "wesad":         "pulseppg/data/datasets/wesad",
        "sdb":           "pulseppg/data/datasets/sdb",
        "wfppg":         "pulseppg/data/datasets/wfppg",
        "dalia":         "pulseppg/data/datasets/dalia",
        "ppgbp":         "pulseppg/data/datasets/ppgbp",
        "mimicperform":  "pulseppg/data/datasets/mimicperform",
        "ppgarrhythmia": "pulseppg/data/datasets/ppgarrhythmia",
        "earset":        "pulseppg/data/datasets/earset",
        "bidmc":         "pulseppg/data/datasets/bidmc",
        "vitaldb":       "pulseppg/data/datasets/vitaldb",
        "sportppg": {
            "data_root": str(cfg.sport_data_root),
            "gt_root":   str(cfg.sport_gt_root),
        },
    }

    def eval_fn(student):
        student.eval()
        res = run_all_tasks(
            student, str(dev), DATA_DIRS, tasks=eval_tasks,
            hz=EVAL_HZ, skip_missing=True, max_samples=args.max_samples,
        )
        rets = {}
        for tname, m in res.items():
            ref = teacher_baseline[tname]
            val = float(m["primary"])
            higher = NW_TASK_HIGHER.get(ref["metric"], True)
            if higher:
                rets[tname] = val / ref["value"] if ref["value"] != 0 else float("nan")
            else:
                rets[tname] = ref["value"] / val if val != 0 else float("nan")
        return float(np.nanmean(list(rets.values()))), float(np.nanmin(list(rets.values()))), rets

    return X_all, Z_all, d_full, build_fn, eval_fn, n_teacher


# ─────────────────────────────────────────────────────────────────────────────
#  PaPaGei-S setup  (3rd teacher; 5.79M params; native 125Hz, 1250-sample windows)
# ─────────────────────────────────────────────────────────────────────────────

PAPAGEI_REPO  = "/home/xiaoma/foundation_model/papagei-foundation-model"
PAPAGEI_CKPT  = f"{PAPAGEI_REPO}/weights/papagei_s.pt"
PAPAGEI_FS    = 125         # PaPaGei native sampling rate
PAPAGEI_LEN   = 1250        # 10-s window @ 125 Hz
PAPAGEI_DIM   = 512


class _PapaGeiEmbAdapter(nn.Module):
    """Wrap PaPaGei ResNet1D / ResNet1DMoE to expose a (B, D) embedding.

    PaPaGei forwards return tuples:
      ResNet1DMoE: (out_dense_512, out_moe1, out_moe2, out_emb_channels)
      ResNet1D:    (out_dense_n_classes, out_emb_channels)

    mode='emb' returns the pre-head pooled feature (out_emb), used for the
        TEACHER (matches PaPaGei's downstream-feature convention; for the
        teacher this is 512-d because base_filters=32, n_block=18 → channels=512).
    mode='dense' returns the dense head output, used for STUDENTS so the dense
        layer (Linear(channels, d_full)) acts as the learnable projection to
        teacher dim, mirroring setup_pulseppg's nn.Sequential(enc, proj).
    """
    def __init__(self, backbone, mode: str = "emb"):
        super().__init__()
        assert mode in ("emb", "dense")
        self.backbone = backbone
        self.mode = mode

    def forward(self, x):
        out = self.backbone(x)
        return out[-1] if self.mode == "emb" else out[0]


def _load_papagei_teacher(dev):
    sys.path.insert(0, PAPAGEI_REPO)
    from models.resnet import ResNet1DMoE
    backbone = ResNet1DMoE(
        in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1,
        n_block=18, n_classes=PAPAGEI_DIM, n_experts=3,
    )
    state = torch.load(PAPAGEI_CKPT, map_location="cpu")
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    backbone.load_state_dict(cleaned, strict=True)
    backbone.eval()
    teacher = _PapaGeiEmbAdapter(backbone, mode="emb").to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _build_papagei_pool(teacher, dev, n_total, sources_filter=None, pool_seed=None):
    """Build a 125Hz/1250-sample mixed pool and extract teacher embeddings."""
    import run_k_probe_pulseppg as ppg_mod

    ds_dalia = ppg_mod.DaliaPublicNPYDataset("pulseppg/data/datasets/dalia",
                                             target_len=PAPAGEI_LEN)
    ds_sdb   = ppg_mod.SDBPublicNPYDataset("pulseppg/data/datasets/sdb")
    sources = [("dalia", ds_dalia), ("sdb", ds_sdb)]
    pulsedb_ppg = Path("/home/xiaoma/datasets/ppg/PulseDB/subset_3M_h5_ppg.npy")
    if pulsedb_ppg.exists():
        sources.append(("pulsedb",
                        ppg_mod.PPGNPYDataset(str(pulsedb_ppg), target_len=PAPAGEI_LEN)))
    if sources_filter is not None:
        sf = set(sources_filter)
        sources = [(n, d) for (n, d) in sources if n in sf]
    n_per = max(1, n_total // len(sources))
    rng = np.random.default_rng(ppg_mod.SEED if pool_seed is None else int(pool_seed))

    from torch.utils.data import DataLoader, Subset
    all_x, all_z = [], []
    print("\n-- Extracting PaPaGei mixed embeddings (125Hz, 1250-len) --")
    for name, ds in sources:
        n_use = min(n_per, len(ds))
        idx = np.sort(rng.choice(len(ds), n_use, replace=False))
        loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256,
                            shuffle=False, num_workers=0)
        xp, zp = [], []
        with torch.inference_mode(), torch.cuda.amp.autocast():
            for batch in loader:
                x = batch.to(dev).float()
                if x.ndim == 2:
                    x = x.unsqueeze(1)
                # SDBPublicNPYDataset may yield 50Hz; resample to 125Hz if needed.
                if x.shape[-1] != PAPAGEI_LEN:
                    x = F.interpolate(x, size=PAPAGEI_LEN, mode="linear",
                                      align_corners=False)
                z = teacher(x).float()
                if z.ndim > 2:
                    z = z.flatten(1)
                xp.append(x.cpu())
                zp.append(z.cpu().numpy().astype(np.float32))
        x_cat = torch.cat(xp)
        z_cat = np.concatenate(zp)
        all_x.append(x_cat); all_z.append(z_cat)
        print(f"  {name}: {len(z_cat)}")
    X_all = torch.cat(all_x)
    Z_all = np.concatenate(all_z)
    print(f"  Total: X={tuple(X_all.shape)}, Z={Z_all.shape}")
    return X_all, Z_all


def _make_papagei_student_builder(dev, d_full):
    """Build_student(seed, cfg) → ResNet1D student wrapped to expose (B, d_full).

    Uses PaPaGei's own ResNet1D family (not PulsePPG's) for fair architectural lineage.
    """
    sys.path.insert(0, PAPAGEI_REPO)
    from models.resnet import ResNet1D

    def build_student(seed, student_cfg):
        torch.manual_seed(seed); np.random.seed(seed)
        backbone = ResNet1D(
            in_channels=1, base_filters=student_cfg["base_filters"],
            kernel_size=3, stride=2, groups=1, n_block=student_cfg["n_block"],
            n_classes=d_full, downsample_gap=2, increasefilter_gap=4,
            use_bn=True, use_do=True, verbose=False,
        )
        return _PapaGeiEmbAdapter(backbone, mode="dense").to(dev)

    return build_student


def setup_papagei(args, dev, out_dir, student_name="BF32_NB12"):
    """Mirror of setup_pulseppg / setup_normwear for the PaPaGei-S teacher."""
    print("\n-- Loading PaPaGei-S teacher --")
    teacher = _load_papagei_teacher(dev)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher params: {n_teacher:,}")

    cache_X = out_dir / "X_mix.pt"
    cache_Z = out_dir / "Z_mix.npy"
    if cache_X.exists() and cache_Z.exists():
        print("-- Loading cached PaPaGei embeddings --")
        X_all = torch.load(str(cache_X), weights_only=True)
        Z_all = np.load(str(cache_Z))
    else:
        pool = getattr(args, "pool", "mixed")
        sources_filter = None if pool == "mixed" else [pool]
        X_all, Z_all = _build_papagei_pool(
            teacher, dev, args.n_data,
            sources_filter=sources_filter,
            pool_seed=getattr(args, "pool_seed", None),
        )
        torch.save(X_all, str(cache_X))
        np.save(str(cache_Z), Z_all)
    print(f"  X={tuple(X_all.shape)}, Z={Z_all.shape}")

    d_full = int(Z_all.shape[1])
    build_student = _make_papagei_student_builder(dev, d_full)
    student_cfg = PG_STUDENT_CONFIGS[student_name]
    print(f"  Student: {student_name} "
          f"(bf={student_cfg['base_filters']}, nb={student_cfg['n_block']})")

    data_dirs = {
        "wesad":         "pulseppg/data/datasets/wesad",
        "sdb":           "pulseppg/data/datasets/sdb",
        "wfppg":         "pulseppg/data/datasets/wfppg",
        "dalia":         "pulseppg/data/datasets/dalia",
        "ppgbp":         "pulseppg/data/datasets/ppgbp",
        "mimicperform":  "pulseppg/data/datasets/mimicperform",
        "ppgarrhythmia": "pulseppg/data/datasets/ppgarrhythmia",
        "earset":        "pulseppg/data/datasets/earset",
        "bidmc":         "pulseppg/data/datasets/bidmc",
        "vitaldb":       "pulseppg/data/datasets/vitaldb",
    }

    teacher_cache = out_dir / "teacher_baseline.json"
    teacher_baseline = json.loads(teacher_cache.read_text()) if teacher_cache.exists() else {}

    from downstream_eval import TASK_REGISTRY
    desired = sorted(t for t in TASK_REGISTRY if t != "sportppg_activity")
    missing = [t for t in desired if t not in teacher_baseline]
    if missing:
        print(f"-- Computing PaPaGei teacher baseline on {len(missing)} tasks "
              f"(hz={PAPAGEI_FS}): {missing}")
        t_results = run_all_tasks(
            teacher, str(dev), data_dirs, tasks=missing,
            hz=PAPAGEI_FS, skip_missing=True, max_samples=args.max_samples,
        )
        for t, m in t_results.items():
            teacher_baseline[t] = {"value": float(m["primary"]),
                                   "metric": m["primary_metric"]}
        teacher_cache.write_text(json.dumps(teacher_baseline, indent=2))

    eval_tasks = list(teacher_baseline.keys())

    # PaPaGei downstream eval also runs at 125Hz. Students share the same backbone
    # family + sample-length convention, so they consume the same 1250-sample input.
    import run_k_probe_pulseppg as ppg_mod

    def eval_fn(student):
        student.eval()
        res = run_all_tasks(
            student, str(dev), data_dirs, tasks=eval_tasks,
            hz=PAPAGEI_FS, skip_missing=True, max_samples=args.max_samples,
        )
        rets = {}
        for tname, m in res.items():
            ref = teacher_baseline[tname]
            val = float(m["primary"])
            if ppg_mod.metric_is_higher_better(ref["metric"]):
                rets[tname] = val / ref["value"] if ref["value"] != 0 else float("nan")
            else:
                rets[tname] = ref["value"] / val if val != 0 else float("nan")
        return float(np.nanmean(list(rets.values()))), float(np.nanmin(list(rets.values()))), rets

    def build_fn(seed):
        return build_student(seed, student_cfg)

    return X_all, Z_all, d_full, build_fn, eval_fn, n_teacher


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        raise RuntimeError("CUDA required.")

    set_fp16(True)
    if args.gpu_probe:
        set_gpu_probe(str(dev))

    # output dir encodes student name
    if args.teacher == "pulseppg" and args.student != "BF20_NB8":
        exp_name = f"exp_ns_kd_{args.student}"
    elif args.teacher == "normwear" and args.nw_student != "base":
        exp_name = f"exp_ns_kd_nw_{args.nw_student}"
    elif args.teacher == "papagei":
        if args.baseline_only:
            exp_name = "exp_papagei_baseline"
        else:
            exp_name = f"exp_ns_kd_pg_{args.pg_student}"
    else:
        exp_name = "exp_ns_kd"
    out_dir = Path(f"kd_sweep_results/{args.teacher}_preliminary/{exp_name}{args.exp_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ns_kd_results.csv"

    nw_student_name = getattr(args, "nw_student", "base")
    print("=" * 70)
    if args.teacher == "pulseppg":
        print(f"  NS-KD Experiment  —  teacher={args.teacher}  student={args.student}")
    elif args.teacher == "papagei":
        print(f"  NS-KD Experiment  —  teacher=papagei  pg_student={args.pg_student}"
              f"{'  [BASELINE-ONLY]' if args.baseline_only else ''}")
    else:
        print(f"  NS-KD Experiment  —  teacher={args.teacher}  nw_student={nw_student_name}")
    print("=" * 70)
    print(f"  k_values={args.k_values}  lambdas={args.lambdas}  n_seeds={args.n_seeds}")
    print(f"  epochs={args.epochs}  n_data={args.n_data}")
    print(f"  out: {ckpt_path}")

    # ── Setup teacher-specific infrastructure ──
    if args.teacher == "pulseppg":
        X_all, Z_all, d_full, build_fn, eval_fn, n_teacher = setup_pulseppg(
            args, dev, out_dir, student_name=args.student)
    elif args.teacher == "papagei":
        X_all, Z_all, d_full, build_fn, eval_fn, n_teacher = setup_papagei(
            args, dev, out_dir, student_name=args.pg_student)
        if args.baseline_only:
            print("\n-- baseline-only mode: teacher_baseline.json written, exiting --")
            return
    else:
        X_all, Z_all, d_full, build_fn, eval_fn, n_teacher = setup_normwear(args, dev, out_dir)

    print(f"\n  D_FULL={d_full},  N={len(Z_all)}")
    s0 = build_fn(0)
    n_student = sum(p.numel() for p in s0.parameters())
    print(f"  Student params={n_student:,},  compression={n_teacher / n_student:.1f}x")
    del s0; gc.collect(); torch.cuda.empty_cache()

    # ── Resume support ──
    all_rows = []
    done = set()
    if ckpt_path.exists():
        prev = pd.read_csv(ckpt_path)
        all_rows = prev.to_dict("records")
        done = {(r["method"], r["k"], r["seed"]) for r in all_rows}
        print(f"\n-- Resume: {len(done)} runs already done --")

    # ── Adaptive k override (must run BEFORE method grid so _kfirst is correct) ──
    if getattr(args, "adaptive_k", False):
        rng_a = np.random.default_rng(42)
        idx_a = (np.sort(rng_a.choice(len(Z_all), args.n_data, replace=False))
                 if args.n_data < len(Z_all) else np.arange(len(Z_all)))
        Z_a = Z_all[idx_a]
        Zc_a = (Z_a - Z_a.mean(axis=0)).astype(np.float64)
        _, S_a, _ = np.linalg.svd(Zc_a, full_matrices=False)
        ev_a = (S_a ** 2)
        cum_a = np.cumsum(ev_a) / ev_a.sum()
        k_ev = int(np.searchsorted(cum_a, float(args.adaptive_tau)) + 1)
        try:
            _stu_probe = build_fn(0)
            with torch.inference_mode():
                xp = torch.zeros(2, 1, 500, device=dev)
                d_student = int(_stu_probe(xp).shape[-1])
            del _stu_probe; gc.collect(); torch.cuda.empty_cache()
        except Exception:
            d_student = d_full
        k_adaptive = min(k_ev, d_student, d_full)
        print(f"\n[adaptive-k] tau={args.adaptive_tau}  k_EV={k_ev}  d_student={d_student}  "
              f"d_full={d_full}  ->  k={k_adaptive}")
        args.k_values = [k_adaptive]

    # ── Build method grid ──
    # Each entry: (label, lam_or_None, method_type, target_k_or_None)
    #   method_type: 'subspace_mse' | 'ns_kd' | 'rkd' | 'pse_kd'
    #   target_k: for pse_kd = stages[-1]; for others = None (run at every k)
    methods = []
    if "subspace_mse" in args.methods:
        methods.append(("subspace_mse", None, "subspace_mse", None))
    if "ns_kd" in args.methods:
        for lam in args.lambdas:
            methods.append((f"ns_kd_lam{lam}", lam, "ns_kd", None))
    if "ns_kd_isig" in args.methods:
        for alpha in args.alphas:
            # lam field is overloaded to carry alpha; dispatcher reads args.isig_lam for ns penalty
            methods.append((f"ns_kd_isig_a{alpha}", alpha, "ns_kd_isig", None))
    if "full_mse" in args.methods:
        methods.append(("full_mse", None, "full_mse", None))
    if "rkd" in args.methods:
        methods.append(("rkd", None, "rkd", None))
    if "ns_rkd" in args.methods:
        for lam_r in args.lambdas_rkd:
            # lam_ns is fixed at 0.1 (best from NS-KD ablation); lam_r is swept
            methods.append((f"ns_rkd_lr{lam_r}", lam_r, "ns_rkd", None))
    if "ns_kd_aux_crd" in args.methods:
        for lam_c in getattr(args, "lambdas_crd", [1.0]):
            methods.append((f"ns_kd_aux_crd_lc{lam_c}", lam_c, "ns_kd_aux_crd", None))
    if "aligned_subspace_crd" in args.methods:
        methods.append(("aligned_subspace_crd", None, "aligned_subspace_crd", None))
    if "pse_kd" in args.methods:
        stages = sorted(args.pse_stages)
        label  = "pse_kd_" + "_".join(str(s) for s in stages)
        methods.append((label, None, "pse_kd", stages[-1]))
    # External representation-KD baselines + lower bound
    # (target_k = first k_value so they only run once per seed, not per k)
    _kfirst = sorted(args.k_values)[0] if args.k_values else None
    for _m in ("lower_bound", "pkt", "sp", "crd", "seed", "disco", "norm"):
        if _m in args.methods:
            methods.append((_m, None, _m, _kfirst))

    n_total = sum(
        (1 if tgt is None else int(tgt in args.k_values))
        for _, _, _, tgt in methods
    ) * len(args.k_values if True else []) * args.n_seeds  # approximate
    n_total = args.n_seeds * sum(
        len(args.k_values) if tgt is None else (1 if tgt in args.k_values else 0)
        for _, _, _, tgt in methods
    )
    run_count = 0
    print(f"\n-- Total runs planned: {n_total} --\n")

    # ── Main experiment loop ──
    for k in sorted(args.k_values):
        if k > d_full:
            print(f"  skip k={k}: k > D_FULL ({d_full})")
            continue

        print(f"\n{'═' * 60}")
        print(f"  k={k}")
        print(f"{'═' * 60}")

        # Compute SVD once per k (same data, same V_k for all methods)
        rng = np.random.default_rng(42)
        if args.n_data < len(Z_all):
            idx = np.sort(rng.choice(len(Z_all), args.n_data, replace=False))
        else:
            idx = np.arange(len(Z_all))

        X_sub = X_all[idx]
        Z_sub = Z_all[idx]

        mu = Z_sub.mean(axis=0)
        Zc = (Z_sub - mu).astype(np.float64)
        _, S_full, Vt = np.linalg.svd(Zc, full_matrices=False)
        Vt = Vt.astype(np.float32)
        S_full = S_full.astype(np.float32)
        del Zc

        V_k = Vt[:k]
        S_k = S_full[:k]
        Z_proj = Z_sub @ V_k.T     # (N, k): pre-computed subspace projections
        V_k_t = torch.as_tensor(V_k, dtype=torch.float32, device=dev)
        # Full Vt tensor (for PSE-KD curriculum; also used by RKD)
        Vt_full_t = torch.as_tensor(Vt, dtype=torch.float32, device=dev)
        ds_train = TensorDataset(X_sub, torch.as_tensor(Z_proj, dtype=torch.float32))
        # Full-dim dataset for RKD/PSE-KD (uses original Z, not subspace projection)
        ds_full  = TensorDataset(X_sub, torch.as_tensor(Z_sub.astype(np.float32)))

        for label, lam, mtype, target_k in methods:
            # PSE-KD only runs at its designated final k
            if target_k is not None and target_k != k:
                continue
            for seed in range(args.n_seeds):
                key = (label, k, seed)
                if key in done:
                    prev_row = next((r for r in all_rows if r["method"] == label
                                     and int(r["k"]) == k and int(r["seed"]) == seed), None)
                    if prev_row:
                        print(f"  {label:24s} k={k} seed={seed}: avg={prev_row['avg_ret']:.4f} [cached]")
                    continue

                t0 = time.time()
                stu = build_fn(77 + seed * 1000)

                if mtype == "subspace_mse":
                    stu = train_subspace_mse(stu, ds_train, V_k_t, args.epochs, dev)
                elif mtype == "ns_kd":
                    stu = train_ns_kd(stu, ds_train, V_k_t, args.epochs, dev, lam=lam)
                elif mtype == "ns_kd_isig":
                    # lam field carries alpha here
                    alpha = float(lam)
                    sk = torch.as_tensor(S_k, dtype=torch.float32, device=dev)
                    rel = sk / sk.max().clamp(min=1e-12)
                    w = rel.clamp(min=1e-6) ** (-alpha)
                    w = w / w.mean()  # normalise so mean(w)=1
                    stu = train_ns_kd_isig(stu, ds_train, V_k_t, w,
                                            args.epochs, dev, lam=args.isig_lam)
                elif mtype == "full_mse":
                    stu = train_full_mse(stu, ds_full, args.epochs, dev)
                elif mtype == "rkd":
                    stu = train_rkd(stu, ds_full, args.epochs, dev)
                elif mtype == "ns_rkd":
                    # lam field stores lam_r; lam_ns fixed at 0.1
                    stu = train_ns_rkd(stu, ds_full, V_k_t, args.epochs, dev,
                                       lam=0.1, lam_r=lam)
                elif mtype == "ns_kd_aux_crd":
                    # lam field stores lam_c (CRD weight); lam (NS) fixed at 0.1
                    stu = train_ns_kd_aux_crd(stu, ds_full, V_k_t, args.epochs, dev,
                                              lam=0.1, lam_c=float(lam))
                elif mtype == "aligned_subspace_crd":
                    stu = train_aligned_subspace_crd(stu, ds_full, V_k_t, args.epochs, dev,
                                                     lam=0.1)
                elif mtype == "pse_kd":
                    stu = train_pse_kd(stu, ds_full, Vt_full_t, stages, args.epochs, dev)
                elif mtype == "pkt":
                    stu = train_pkt(stu, ds_full, args.epochs, dev)
                elif mtype == "sp":
                    stu = train_sp(stu, ds_full, args.epochs, dev)
                elif mtype == "crd":
                    stu = train_crd(stu, ds_full, args.epochs, dev)
                elif mtype == "seed":
                    stu = train_seed(stu, ds_full, args.epochs, dev)
                elif mtype == "disco":
                    stu = train_disco(stu, ds_full, args.epochs, dev)
                elif mtype == "norm":
                    stu = train_norm(stu, ds_full, args.epochs, dev)
                elif mtype == "lower_bound":
                    stu = train_lower_bound(stu)

                avg_ret, floor_ret, per_task = eval_fn(stu)
                elapsed = time.time() - t0

                row = {
                    "method": label,
                    "k": k,
                    "seed": seed,
                    "lam": lam if lam is not None else 0.0,
                    "avg_ret": avg_ret,
                    "floor_ret": floor_ret,
                    "n_tasks": len(per_task),
                    "elapsed_s": elapsed,
                }
                for tname, ret in per_task.items():
                    row[f"ret_{tname}"] = ret

                all_rows.append(row)
                done.add(key)
                run_count += 1

                print(f"  {label:24s} k={k} seed={seed}: avg={avg_ret:.4f} floor={floor_ret:.4f} [{elapsed:.0f}s]")
                pd.DataFrame(all_rows).to_csv(ckpt_path, index=False)

                del stu; gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Summary  ({run_count} new runs)")
    print(f"{'=' * 70}")
    df = pd.DataFrame(all_rows)
    summary = df.groupby(["method", "k"])[["avg_ret", "floor_ret"]].agg(["mean", "std"]).round(4)
    print(summary.to_string())

    print(f"\n  Saved: {ckpt_path}")
    print("  Done!")


if __name__ == "__main__":
    main()

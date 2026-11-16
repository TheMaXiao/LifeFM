#!/usr/bin/env python
"""
NS-KD Experiment: Null-Space Suppression KD vs Subspace MSE
============================================================

Compares:
  - subspace_mse  (naive baseline)
  - lifefm_kd (λ)     (+ null-space suppression penalty)

For k ∈ {25, 50} (or via --k-values), 3 seeds, on PulsePPG and NormWear.

NS-KD loss:
    L = ||V_k (z_s - z_t)||^2  +  λ * ||(I - V_k^T V_k) z_s||^2
      = kd_match_in_subspace   +  λ * null_space_suppression

The second term forces the student's output to lie inside V_k, eliminating
random-initialization noise in the D−k orthogonal directions that receive no
training signal from the KD objective alone.

Usage:
    # PulsePPG, GPU 0
    CUDA_VISIBLE_DEVICES=0 python -u run_lifefm_kd.py --teacher pulseppg 2>&1 | tee logs/ns_kd_pulseppg.log

    # NormWear, GPU 1
    CUDA_VISIBLE_DEVICES=1 python -u run_lifefm_kd.py --teacher normwear 2>&1 | tee logs/ns_kd_normwear.log
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
    p.add_argument("--lambdas", "--lambda-sup", dest="lambdas", type=float, nargs="+", default=[0.1],
                   help="Non-core suppression weight lambda_sup (paper: 0.1). Pass multiple values to sweep.")
    p.add_argument("--methods", nargs="+", default=["lifefm_kd"],
                   choices=["lifefm_kd"],
                   help="Stage-I objective. Only LifeFM's core-alignment + non-core-suppression KD is supported.")
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








def train_lifefm_kd(student, ds, V_k_t, epochs, dev, lr=KD_LR, lam=0.5):
    """LifeFM Stage-I KD: core alignment + non-core suppression.

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
    #   method_type: 'subspace_mse' | 'lifefm_kd' | 'rkd' | 'pse_kd'
    #   target_k: for pse_kd = stages[-1]; for others = None (run at every k)
    methods = []
    if "subspace_mse" in args.methods:
        methods.append(("subspace_mse", None, "subspace_mse", None))
    if "lifefm_kd" in args.methods:
        for lam in args.lambdas:
            methods.append((f"ns_kd_lam{lam}", lam, "lifefm_kd", None))
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

                if mtype != "lifefm_kd":
                    raise ValueError(f"unsupported Stage-I method: {mtype}")
                stu = train_lifefm_kd(stu, ds_train, V_k_t, args.epochs, dev, lam=lam)

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

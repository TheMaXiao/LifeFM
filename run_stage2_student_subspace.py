#!/usr/bin/env python3
"""Stage-2 custom multi-task adaptation methods for compressed PulsePPG students.

This script extends the first Stage-2 prototype with:

1. sp_mtft
   Shared-Private Subspace Multi-Task Fine-Tuning. A frozen Stage-1 student is
   used to discover a student-native shared/private basis on the Stage-2 train
   data. Shared coordinates are kept fixed through the frozen anchor encoder,
   while task-specific masks control which private coordinates each task can
   use and update.

2. sp_gradnorm
   The same shared/private student-space structure, now combined with
   GradNorm adaptive task balancing.

3. soft_sp_mtft / soft_sp_gradnorm
   Soft shared-protection variants. Shared coordinates come from the current
   encoder instead of being frozen from the Stage-1 anchor, and a light shared
   anchor penalty keeps those shared coordinates close to the frozen Stage-1
   representation. `soft_sp_gradnorm` applies GradNorm only to the task losses;
   subspace regularizers are added separately to the total loss.

4. pcgrad_joint
   Joint fine-tuning with PCGrad-style gradient surgery.

5. gradnorm_joint
   Joint fine-tuning with GradNorm adaptive loss balancing.

6. pace_mtl
   Progress-balanced Adaptive Conflict-aware Edge MTL. Starts from joint
   fine-tuning, adds floor-aware progress balancing, applies PCGrad-style
   projection only when shared encoder gradients show genuine negative
   conflict, and adds tiny task-private late residuals so tasks can disagree
   without cloning the full encoder.

7. physioscale_mtl
   Physiological Scale-Factorized Multi-Task Adaptation. A tiny factor bank
   splits the late representation into rhythm, morphology, and context factors.
   Each task learns a soft routing over those shared physiological factors
   instead of receiving a private encoder.

8. uniphys_mtl
   A universal physiological multi-task variant that keeps the shared encoder
   embedding as the base feature, adds tiny residual multi-scale corrections,
   and directly optimizes both average and worst-task normalized loss.

Outputs use the same CSV schema as run_stage2_multitask_ppg.py so the results
can be merged with the earlier Stage-2 baselines by merge_stage2_results.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from run_stage2_multitask_ppg import (
    ROOT,
    DATASET_SPECS,
    DEFAULT_CKPT_DIRS,
    TaskData,
    TaskRuntime,
    TaskHead,
    MultiTaskModel,
    add_relative_scores,
    aggregate_over_seeds,
    count_params,
    disable_dropout,
    ensure_temporal_cols,
    evaluate,
    init_regression_biases,
    load_multitask_dataset,
    load_stage1_full,
    method_completed,
    apply_task_specific_input_windows,
    resolve_input_window_sec_map,
    resolve_scale_grid_sec,
    resolve_task_supports_sec,
    records_from_metrics,
    resolve_device,
    set_seed,
    task_losses,
    validation_loss,
    make_task_loaders,
)


DEFAULT_OUT_ROOT = ROOT / "kd_sweep_results/stage2_student_subspace"
HARD_SP_METHODS = {"sp_mtft", "sp_gradnorm"}
SOFT_SP_METHODS = {"soft_sp_mtft", "soft_sp_gradnorm"}
SP_METHODS = HARD_SP_METHODS | SOFT_SP_METHODS
GRADNORM_METHODS = {"gradnorm_joint", "sp_gradnorm", "soft_sp_gradnorm"}
PACE_METHODS = {"pace_mtl"}
PHYSIOSCALE_METHODS = {"physioscale_mtl"}
UNIPHYS_METHODS = {"uniphys_mtl"}
PHYSIO_FACTOR_NAMES = ("rhythm", "morphology", "context")


REFERENCE_MULTITASK_RESULTS: Dict[str, List[Path]] = {
    "bidmc": [
        ROOT / "kd_sweep_results/stage2_multitask_ppg/bidmc/stage2_results.csv",
    ],
    "ppgbp": [
        ROOT / "kd_sweep_results/stage2_multitask_ppg/ppgbp/stage2_results.csv",
        ROOT / "kd_sweep_results/stage2_multitask_ppg_final/ppgbp/stage2_results.csv",
    ],
    "wesad": [
        ROOT / "kd_sweep_results/stage2_multitask_ppg/wesad/stage2_results.csv",
        ROOT / "kd_sweep_results/stage2_multitask_ppg_final/wesad/stage2_results.csv",
    ],
    "dalia": [
        ROOT / "kd_sweep_results/stage2_multitask_ppg/dalia/stage2_results.csv",
        ROOT / "kd_sweep_results/stage2_multitask_ppg_corrected/dalia/stage2_results.csv",
        ROOT / "kd_sweep_results/stage2_multitask_ppg_final/dalia_with_stage2v4/stage2_results.csv",
    ],
    "sportppg": [
        ROOT / "kd_sweep_results/stage2_multitask_ppg/sportppg/stage2_results.csv",
        ROOT / "kd_sweep_results/stage2_multitask_ppg_corrected/sportppg/stage2_results.csv",
        ROOT / "kd_sweep_results/stage2_multitask_ppg_sportppg/sportppg/stage2_results.csv",
    ],
}


@dataclass
class StudentSubspaceSpec:
    mean: torch.Tensor
    basis_shared: torch.Tensor
    basis_private: torch.Tensor
    task_private_init: Dict[str, torch.Tensor]
    candidate_dim: int
    shared_dim: int
    private_dim: int
    shared_var_covered: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="dalia")
    ap.add_argument("--students", nargs="+", default=["BF20_NB8"])
    ap.add_argument("--kd-method", default="ns_kd", choices=["ns_kd", "subspace_mse"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["sp_mtft", "sp_gradnorm", "pcgrad_joint", "gradnorm_joint"],
        choices=["sp_mtft", "sp_gradnorm", "soft_sp_mtft", "soft_sp_gradnorm", "pcgrad_joint", "gradnorm_joint", "pace_mtl", "physioscale_mtl", "uniphys_mtl"],
    )
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--shared-var-thresh", type=float, default=0.95)
    ap.add_argument("--shared-frac", type=float, default=0.5)
    ap.add_argument("--shared-anchor", type=float, default=0.05)
    ap.add_argument("--mask-sparse", type=float, default=0.02)
    ap.add_argument("--mask-overlap", type=float, default=0.05)
    ap.add_argument("--gradnorm-alpha", type=float, default=1.5)
    ap.add_argument("--gradnorm-lr", type=float, default=2.5e-2)
    ap.add_argument("--pace-progress-alpha", type=float, default=1.0)
    ap.add_argument("--pace-floor-beta", type=float, default=0.75)
    ap.add_argument("--pace-conflict-tau", type=float, default=0.05)
    ap.add_argument("--pace-ema", type=float, default=0.9)
    ap.add_argument("--pace-private-rank", type=int, default=4)
    ap.add_argument("--pace-private-penalty", type=float, default=5e-4)
    ap.add_argument("--physio-bottleneck", type=int, default=16)
    ap.add_argument("--physio-diversity", type=float, default=0.05)
    ap.add_argument("--physio-router-prior", type=float, default=0.02)
    ap.add_argument("--uniphys-bottleneck", type=int, default=16)
    ap.add_argument("--uniphys-diversity", type=float, default=0.02)
    ap.add_argument("--uniphys-floor-weight", type=float, default=0.5)
    ap.add_argument("--uniphys-router-prior", type=float, default=0.0)
    ap.add_argument("--uniphys-worst-k", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument(
        "--input-window-sec-map",
        type=str,
        default="",
        help='JSON dict of task-specific raw input windows in seconds, e.g. {"activity":2.5,"hr":10}',
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    return ap.parse_args()


def default_temporal_config_json(dataset: str, task_data: List[TaskData], hz: int) -> str:
    native_sec = float(max(int(td.split_X["train"].shape[-1]) for td in task_data)) / float(hz)
    payload = {
        "policy": "shared_window",
        "task_supports_sec": resolve_task_supports_sec(
            dataset,
            [td.runtime.spec.name for td in task_data],
            native_sec=native_sec,
        ),
        "scales_sec": resolve_scale_grid_sec(dataset, native_sec=native_sec),
    }
    return json.dumps(payload, sort_keys=True)


def load_reference_frozen_rows(dataset: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in REFERENCE_MULTITASK_RESULTS.get(dataset, []):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        frames.append(df[df["method"] == "frozen_heads"].copy())
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = ensure_temporal_cols(df)
    return df.drop_duplicates(
        subset=[
            "dataset",
            "student",
            "kd_method",
            "seed",
            "method",
            "task",
            "encoder_update_scope",
            "temporal_policy",
            "temporal_config",
            "input_window_policy",
            "input_window_sec_map",
        ],
        keep="last",
    )


def score_rows_with_frozen_reference(
    rows_df: pd.DataFrame,
    frozen_df: pd.DataFrame,
    methods: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if rows_df.empty:
        return rows_df, pd.DataFrame()
    keep_methods = set(rows_df["method"].dropna().astype(str).tolist())
    merged = rows_df if frozen_df.empty else pd.concat([frozen_df, rows_df], ignore_index=True)
    scored, summary = add_relative_scores(merged)
    scored = scored[scored["method"].isin(keep_methods)].copy()
    summary = summary[summary["method"].isin(keep_methods)].copy()
    return scored, summary


def encode_array(
    encoder: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if X.ndim == 2:
        X = X[:, None, :]
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    outs: List[np.ndarray] = []
    encoder = encoder.to(device).eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            z = encoder(xb)
            if z.ndim == 3:
                z = z.mean(-1)
            elif z.ndim > 2:
                z = z.flatten(1)
            outs.append(z.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def unique_train_arrays(task_data: List[TaskData]) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    seen: set[Tuple[int, Tuple[int, ...]]] = set()
    for td in task_data:
        X = td.split_X["train"]
        key = (int(X.__array_interface__["data"][0]), tuple(X.shape))
        if key in seen:
            continue
        seen.add(key)
        arrays.append(X)
    return arrays


def regression_support(coords: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-6)
    c = coords.astype(np.float32)
    c = (c - c.mean(0, keepdims=True)) / (c.std(0, keepdims=True) + 1e-6)
    sup = np.abs((c * y[:, None]).mean(0))
    if np.max(sup) > 0:
        sup = sup / (np.max(sup) + 1e-6)
    return sup.astype(np.float32)


def classification_support(coords: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = y.astype(np.int64)
    c = coords.astype(np.float32)
    mu = c.mean(0)
    total = c.var(0) + 1e-6
    between = np.zeros(c.shape[1], dtype=np.float32)
    for cls in np.unique(y):
        mask = y == cls
        if mask.sum() == 0:
            continue
        diff = c[mask].mean(0) - mu
        between += float(mask.mean()) * (diff * diff).astype(np.float32)
    sup = between / total
    if np.max(sup) > 0:
        sup = sup / (np.max(sup) + 1e-6)
    return sup.astype(np.float32)


def build_student_subspace(
    anchor_encoder: nn.Module,
    task_data: List[TaskData],
    device: torch.device,
    batch_size: int,
    shared_var_thresh: float,
    shared_frac: float,
) -> StudentSubspaceSpec:
    feat_blocks = [encode_array(anchor_encoder, X, device, batch_size) for X in unique_train_arrays(task_data)]
    Z_all = np.concatenate(feat_blocks, axis=0)
    mu = Z_all.mean(0).astype(np.float32)
    Zc = Z_all - mu[None, :]

    _, svals, vt = np.linalg.svd(Zc, full_matrices=False)
    eig = (svals.astype(np.float32) ** 2) / max(1, Zc.shape[0] - 1)
    var_ratio = eig / max(1e-6, float(eig.sum()))
    cum = np.cumsum(var_ratio)
    candidate_dim = int(np.searchsorted(cum, shared_var_thresh, side="left") + 1)
    candidate_dim = max(1, min(candidate_dim, vt.shape[0]))

    basis = vt.astype(np.float32)
    cand_basis = basis[:candidate_dim]
    task_supports: List[np.ndarray] = []
    for td in task_data:
        z_task = encode_array(anchor_encoder, td.split_X["train"], device, batch_size)
        coords = (z_task - mu[None, :]) @ cand_basis.T
        if td.runtime.spec.kind == "reg":
            sup = regression_support(coords, td.split_y["train"])
        else:
            sup = classification_support(coords, td.split_y["train"])
        task_supports.append(sup)

    sup_mat = np.stack(task_supports, axis=0) if task_supports else np.zeros((0, candidate_dim), dtype=np.float32)
    mean_sup = sup_mat.mean(0) if len(sup_mat) else np.zeros(candidate_dim, dtype=np.float32)
    std_sup = sup_mat.std(0) if len(sup_mat) else np.zeros(candidate_dim, dtype=np.float32)
    consistency = 1.0 - np.minimum(1.0, std_sup / (mean_sup + 1e-6))
    shared_score = var_ratio[:candidate_dim] * mean_sup * consistency

    shared_dim = max(1, int(math.ceil(candidate_dim * shared_frac)))
    shared_dim = min(shared_dim, candidate_dim)
    shared_idx = np.argsort(-shared_score)[:shared_dim]
    shared_idx = np.sort(shared_idx.astype(np.int64))
    private_idx = np.array([i for i in range(basis.shape[0]) if i not in set(shared_idx.tolist())], dtype=np.int64)

    full_support = np.zeros((len(task_data), basis.shape[0]), dtype=np.float32)
    if len(task_supports):
        full_support[:, :candidate_dim] = sup_mat

    task_private_init: Dict[str, torch.Tensor] = {}
    for td, sup in zip(task_data, full_support):
        if len(private_idx) == 0:
            init = np.zeros((0,), dtype=np.float32)
        else:
            priv = sup[private_idx]
            if np.max(priv) > 0:
                priv = priv / (np.max(priv) + 1e-6)
                init = 0.25 + 0.5 * priv
            else:
                init = np.full((len(private_idx),), 0.5, dtype=np.float32)
        task_private_init[td.runtime.spec.name] = torch.as_tensor(init, dtype=torch.float32)

    return StudentSubspaceSpec(
        mean=torch.from_numpy(mu),
        basis_shared=torch.from_numpy(basis[shared_idx]),
        basis_private=torch.from_numpy(basis[private_idx]),
        task_private_init=task_private_init,
        candidate_dim=int(candidate_dim),
        shared_dim=int(len(shared_idx)),
        private_dim=int(len(private_idx)),
        shared_var_covered=float(cum[candidate_dim - 1]),
    )


class SharedPrivateMTModel(MultiTaskModel):
    def __init__(
        self,
        encoder: nn.Module,
        anchor_encoder: nn.Module,
        enc_dim: int,
        tasks: List[TaskRuntime],
        dropout: float,
        spec: StudentSubspaceSpec,
    ) -> None:
        super().__init__(encoder, enc_dim, tasks, dropout)
        self.anchor_encoder = anchor_encoder
        disable_dropout(self.anchor_encoder)
        self.anchor_encoder.requires_grad_(False)
        self.register_buffer("basis_mean", spec.mean.float())
        self.register_buffer("basis_shared", spec.basis_shared.float())
        self.register_buffer("basis_private", spec.basis_private.float())
        self.private_logits = nn.ParameterDict()
        for rt in tasks:
            init = spec.task_private_init[rt.spec.name]
            if init.numel():
                init = init.clamp(0.05, 0.95)
                logit = torch.log(init / (1.0 - init))
            else:
                logit = init
            self.private_logits[rt.spec.name] = nn.Parameter(logit)

    def encode_anchor(self, x: torch.Tensor, task_name: Optional[str] = None) -> torch.Tensor:
        return self.encode_with_encoder(self.anchor_encoder, x, task_name=task_name)

    def mask_regularization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = [torch.sigmoid(v) for v in self.private_logits.values()]
        if not masks:
            zero = self.basis_mean.new_tensor(0.0)
            return zero, zero
        sparse = torch.stack([m.mean() for m in masks]).mean()
        overlaps: List[torch.Tensor] = []
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                overlaps.append((masks[i] * masks[j]).mean())
        if overlaps:
            overlap = torch.stack(overlaps).mean()
        else:
            overlap = sparse.new_tensor(0.0)
        return sparse, overlap

    def task_features(self, x: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        zc = z - self.basis_mean.unsqueeze(0)
        with torch.no_grad():
            z0 = self.encode_anchor(x)
            z0c = z0 - self.basis_mean.unsqueeze(0)
        if self.basis_shared.numel():
            shared_coords = z0c @ self.basis_shared.T
            shared = shared_coords @ self.basis_shared
        else:
            shared = torch.zeros_like(zc)
        if self.basis_private.numel():
            private_coords = zc @ self.basis_private.T
        else:
            private_coords = zc.new_zeros((zc.shape[0], 0))

        feats: Dict[str, torch.Tensor] = {}
        for name in self.heads:
            if self.basis_private.numel():
                mask = torch.sigmoid(self.private_logits[name]).unsqueeze(0)
                private = (private_coords * mask) @ self.basis_private
            else:
                private = torch.zeros_like(shared)
            feats[name] = self.basis_mean.unsqueeze(0) + shared + private
        return feats

    def forward(self, x: torch.Tensor, task_name: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.encode(x, task_name=task_name)
        feats = self.task_features(x, z)
        return z, {name: self.heads[name](feat) for name, feat in feats.items()}


class SoftSharedPrivateMTModel(SharedPrivateMTModel):
    def shared_anchor_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.basis_shared.numel():
            return z.new_tensor(0.0)
        zc = z - self.basis_mean.unsqueeze(0)
        current_shared = zc @ self.basis_shared.T
        with torch.no_grad():
            z0 = self.encode_anchor(x)
            z0c = z0 - self.basis_mean.unsqueeze(0)
            anchor_shared = z0c @ self.basis_shared.T
        return F.mse_loss(current_shared, anchor_shared)

    def task_features(self, x: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        del x
        zc = z - self.basis_mean.unsqueeze(0)
        if self.basis_shared.numel():
            shared_coords = zc @ self.basis_shared.T
            shared = shared_coords @ self.basis_shared
        else:
            shared = torch.zeros_like(zc)
        if self.basis_private.numel():
            private_coords = zc @ self.basis_private.T
        else:
            private_coords = zc.new_zeros((zc.shape[0], 0))

        feats: Dict[str, torch.Tensor] = {}
        for name in self.heads:
            if self.basis_private.numel():
                mask = torch.sigmoid(self.private_logits[name]).unsqueeze(0)
                private = (private_coords * mask) @ self.basis_private
            else:
                private = torch.zeros_like(shared)
            feats[name] = self.basis_mean.unsqueeze(0) + shared + private
        return feats


class LowRankTaskResidual(nn.Module):
    def __init__(self, dim: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(0.10))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(F.gelu(self.down(self.norm(z)))))
        return z + torch.tanh(self.scale) * delta

    def penalty(self) -> torch.Tensor:
        return self.scale.pow(2) + 0.1 * self.up.weight.pow(2).mean()


class PaceMTModel(MultiTaskModel):
    def __init__(
        self,
        encoder: nn.Module,
        enc_dim: int,
        tasks: List[TaskRuntime],
        dropout: float,
        private_rank: int,
    ) -> None:
        super().__init__(encoder, enc_dim, tasks, dropout)
        private_dropout = min(0.2, max(0.0, dropout * 0.5))
        self.task_residuals = nn.ModuleDict({
            rt.spec.name: LowRankTaskResidual(enc_dim, private_rank, private_dropout)
            for rt in tasks
        })

    def task_features(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: self.task_residuals[name](z) for name in self.heads}

    def private_penalty(self) -> torch.Tensor:
        if not self.task_residuals:
            return next(self.parameters()).new_tensor(0.0)
        return torch.stack([mod.penalty() for mod in self.task_residuals.values()]).mean()

    def forward(self, x: torch.Tensor, task_name: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.encode(x, task_name=task_name)
        feats = self.task_features(z)
        return z, {name: self.heads[name](feat) for name, feat in feats.items()}


def physio_task_prior(task_name: str, task_kind: str) -> torch.Tensor:
    name = task_name.lower()
    if "activity" in name:
        prior = [0.15, 0.15, 0.70]
    elif "hypertension" in name:
        prior = [0.15, 0.50, 0.35]
    elif "sysbp" in name or "diasbp" in name or name.endswith("bp"):
        prior = [0.20, 0.65, 0.15]
    elif "hr" in name:
        prior = [0.60, 0.25, 0.15]
    elif "stress" in name or name in {"binary", "multiclass"}:
        prior = [0.25, 0.20, 0.55]
    elif task_kind == "reg":
        prior = [0.45, 0.40, 0.15]
    else:
        prior = [0.25, 0.25, 0.50]
    return torch.tensor(prior, dtype=torch.float32)


class TemporalPhysioFactor(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck: int,
        kernel_size: int,
        smooth_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(1, affine=False)
        if smooth_kernel > 1:
            self.pool = nn.AvgPool1d(smooth_kernel, stride=1, padding=smooth_kernel // 2)
        else:
            self.pool = nn.Identity()
        self.down = nn.Conv1d(1, bottleneck, kernel_size=1, bias=False)
        self.temporal = nn.Conv1d(
            bottleneck,
            bottleneck,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=bottleneck,
            bias=False,
        )
        self.up = nn.Conv1d(bottleneck, dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.10))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.temporal.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def signal_view(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected waveform tensor with 2 or 3 dims, got {tuple(x.shape)}")
        if x.shape[1] != 1:
            x = x.mean(1, keepdim=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.signal_view(x)
        x = self.pool(s)
        x = self.down(x)
        x = self.temporal(x)
        x = self.dropout(F.gelu(x))
        x = self.up(x)
        return torch.tanh(self.scale) * x.mean(-1)


class ContextPhysioFactor(nn.Module):
    def __init__(self, dim: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(1, affine=False)
        self.key = nn.Conv1d(1, bottleneck, kernel_size=1, bias=False)
        self.score = nn.Conv1d(bottleneck, 1, kernel_size=1, bias=True)
        self.value = nn.Conv1d(1, dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.10))
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.value.weight, std=0.02)

    def signal_view(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected waveform tensor with 2 or 3 dims, got {tuple(x.shape)}")
        if x.shape[1] != 1:
            x = x.mean(1, keepdim=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.signal_view(x)
        x = self.norm(s)
        attn_logits = self.score(F.gelu(self.key(x))).squeeze(1)
        attn = torch.softmax(attn_logits, dim=-1).unsqueeze(1)
        value = self.value(self.dropout(x))
        ctx = (value * attn).sum(-1)
        return torch.tanh(self.scale) * ctx


class PhysioScaleMTModel(MultiTaskModel):
    def __init__(
        self,
        encoder: nn.Module,
        enc_dim: int,
        tasks: List[TaskRuntime],
        dropout: float,
        bottleneck: int,
    ) -> None:
        super().__init__(encoder, enc_dim, tasks, dropout)
        branch_dropout = min(0.2, max(0.0, dropout * 0.5))
        self.factor_modules = nn.ModuleDict({
            "rhythm": TemporalPhysioFactor(enc_dim, bottleneck, kernel_size=9, smooth_kernel=5, dropout=branch_dropout),
            "morphology": TemporalPhysioFactor(enc_dim, bottleneck, kernel_size=3, smooth_kernel=1, dropout=branch_dropout),
            "context": ContextPhysioFactor(enc_dim, bottleneck, dropout=branch_dropout),
        })
        self.factor_norms = nn.ModuleDict({name: nn.LayerNorm(enc_dim) for name in PHYSIO_FACTOR_NAMES})
        self.router_logits = nn.ParameterDict()
        self._router_prior_names: Dict[str, str] = {}
        for rt in tasks:
            prior = physio_task_prior(rt.spec.name, rt.spec.kind).clamp_min(1e-4)
            key = f"router_prior__{rt.spec.name}"
            self._router_prior_names[rt.spec.name] = key
            self.register_buffer(key, prior)
            self.router_logits[rt.spec.name] = nn.Parameter(torch.log(prior))

    def get_router_prior(self, task_name: str) -> torch.Tensor:
        return getattr(self, self._router_prior_names[task_name])

    def router_weight_dict(self) -> Dict[str, torch.Tensor]:
        return {name: torch.softmax(logits, dim=0) for name, logits in self.router_logits.items()}

    def router_prior_loss(self) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for name, logits in self.router_logits.items():
            probs = torch.softmax(logits, dim=0)
            prior = self.get_router_prior(name).to(probs)
            losses.append(torch.sum(probs * (torch.log(probs.clamp_min(1e-6)) - torch.log(prior.clamp_min(1e-6)))))
        if not losses:
            return next(self.parameters()).new_tensor(0.0)
        return torch.stack(losses).mean()

    def factor_diversity_loss(self, deltas: Dict[str, torch.Tensor]) -> torch.Tensor:
        vals: List[torch.Tensor] = []
        names = list(deltas)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                fi = F.normalize(deltas[names[i]], dim=1)
                fj = F.normalize(deltas[names[j]], dim=1)
                vals.append(F.cosine_similarity(fi, fj, dim=1).pow(2).mean())
        if not vals:
            return next(self.parameters()).new_tensor(0.0)
        return torch.stack(vals).mean()

    def compute_factors(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        deltas = {name: module(x) for name, module in self.factor_modules.items()}
        factors = {name: self.factor_norms[name](z + deltas[name]) for name in deltas}
        return factors, deltas

    def forward_with_aux(self, x: torch.Tensor, task_name: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        z = self.encode(x, task_name=task_name)
        factors, deltas = self.compute_factors(x, z)
        router_weights = self.router_weight_dict()
        outputs: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            weights = router_weights[name]
            feat = sum(weights[idx] * factors[factor_name] for idx, factor_name in enumerate(PHYSIO_FACTOR_NAMES))
            outputs[name] = head(feat)
        aux = {
            "factor_diversity": self.factor_diversity_loss(deltas),
            "router_prior": self.router_prior_loss(),
            "router_weights": router_weights,
        }
        return z, outputs, aux

    def forward(self, x: torch.Tensor, task_name: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z, outputs, _ = self.forward_with_aux(x, task_name=task_name)
        return z, outputs


class UniPhysMTModel(PhysioScaleMTModel):
    def forward_with_aux(self, x: torch.Tensor, task_name: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        z = self.encode(x, task_name=task_name)
        _, deltas = self.compute_factors(x, z)
        router_weights = self.router_weight_dict()
        outputs: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            weights = router_weights[name]
            delta = sum(weights[idx] * deltas[factor_name] for idx, factor_name in enumerate(PHYSIO_FACTOR_NAMES))
            outputs[name] = head(z + delta)
        aux = {
            "factor_diversity": self.factor_diversity_loss(deltas),
            "router_prior": self.router_prior_loss(),
            "router_weights": router_weights,
        }
        return z, outputs, aux


def set_trainable(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = True
    if hasattr(model, "heads"):
        for p in model.heads.parameters():
            p.requires_grad = True
    if hasattr(model, "private_logits"):
        for p in model.private_logits.parameters():
            p.requires_grad = True
    if hasattr(model, "router_logits"):
        for p in model.router_logits.parameters():
            p.requires_grad = True
    if hasattr(model, "task_residuals"):
        for p in model.task_residuals.parameters():
            p.requires_grad = True
    if hasattr(model, "factor_modules"):
        for p in model.factor_modules.parameters():
            p.requires_grad = True
    if hasattr(model, "factor_norms"):
        for p in model.factor_norms.parameters():
            p.requires_grad = True


def make_optimizer(
    model: nn.Module,
    head_lr: float,
    encoder_lr_mult: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    enc_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            enc_params.append(p)
        else:
            other_params.append(p)
    groups = []
    if enc_params:
        groups.append(
            {
                "params": enc_params,
                "lr": head_lr * encoder_lr_mult,
                "weight_decay": weight_decay,
            }
        )
    if other_params:
        groups.append(
            {
                "params": other_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
            }
        )
    return torch.optim.AdamW(groups)


def deploy_param_count(model: nn.Module) -> int:
    total = 0
    if hasattr(model, "encoder"):
        total += count_params(model.encoder)
    if hasattr(model, "heads"):
        total += count_params(model.heads)
    if hasattr(model, "private_logits"):
        total += count_params(model.private_logits)
    if hasattr(model, "router_logits"):
        total += count_params(model.router_logits)
    if hasattr(model, "task_residuals"):
        total += count_params(model.task_residuals)
    if hasattr(model, "factor_modules"):
        total += count_params(model.factor_modules)
    if hasattr(model, "factor_norms"):
        total += count_params(model.factor_norms)
    return int(total)


def flatten_grads(grads: Sequence[Optional[torch.Tensor]], params: Sequence[nn.Parameter]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            parts.append(torch.zeros_like(p, memory_format=torch.contiguous_format).reshape(-1))
        else:
            parts.append(g.reshape(-1))
    return torch.cat(parts) if parts else torch.zeros(0)


def assign_flat_grad(params: Sequence[nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        g = flat[offset:offset + n].view_as(p).clone()
        p.grad = g
        offset += n


def pcgrad_merge(task_grads: torch.Tensor) -> torch.Tensor:
    if task_grads.ndim != 2 or task_grads.shape[0] == 0:
        raise ValueError("task_grads must have shape [n_tasks, n_params]")
    proj = task_grads.clone()
    for i in range(proj.shape[0]):
        for j in range(proj.shape[0]):
            if i == j:
                continue
            gij = torch.dot(proj[i], task_grads[j])
            if gij < 0:
                denom = task_grads[j].pow(2).sum().clamp_min(1e-12)
                proj[i] = proj[i] - gij / denom * task_grads[j]
    return proj.mean(0)


def conditional_pcgrad_merge(task_grads: torch.Tensor, tau: float) -> Tuple[torch.Tensor, int, int]:
    if task_grads.ndim != 2 or task_grads.shape[0] == 0:
        raise ValueError("task_grads must have shape [n_tasks, n_params]")
    proj = task_grads.clone()
    n_conflicts = 0
    n_pairs = 0
    for i in range(proj.shape[0]):
        for j in range(proj.shape[0]):
            if i == j:
                continue
            gi = proj[i]
            gj = task_grads[j]
            ni = gi.norm().clamp_min(1e-12)
            nj = gj.norm().clamp_min(1e-12)
            if float(ni.item()) <= 1e-12 or float(nj.item()) <= 1e-12:
                continue
            cos = torch.dot(gi, gj) / (ni * nj)
            n_pairs += 1
            if cos < -tau:
                denom = gj.pow(2).sum().clamp_min(1e-12)
                proj[i] = gi - torch.dot(gi, gj) / denom * gj
                n_conflicts += 1
    return proj.mean(0), n_conflicts, n_pairs


def run_epoch_batches(
    train_loaders: Dict[str, DataLoader],
    tasks: List[TaskRuntime],
) -> Iterable[Tuple[TaskRuntime, torch.Tensor, torch.Tensor]]:
    task_iters = {rt.spec.name: iter(train_loaders[rt.spec.name]) for rt in tasks}
    n_steps = max(len(train_loaders[rt.spec.name]) for rt in tasks)
    for _ in range(n_steps):
        for rt in tasks:
            name = rt.spec.name
            try:
                xb, yb = next(task_iters[name])
            except StopIteration:
                task_iters[name] = iter(train_loaders[name])
                xb, yb = next(task_iters[name])
            yield rt, xb, yb


@torch.no_grad()
def validation_losses_by_task(
    model: nn.Module,
    val_loaders: Dict[str, DataLoader],
    tasks: List[TaskRuntime],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for rt in tasks:
        name = rt.spec.name
        task_vals: List[float] = []
        for xb, yb in val_loaders[name]:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            _, preds = model(xb)
            losses = task_losses(preds, {name: yb}, [rt])
            if losses:
                task_vals.append(float(losses[name].item()))
        out[name] = float(np.mean(task_vals)) if task_vals else float("inf")
    return out


def pace_task_weights(
    train_ema: torch.Tensor,
    init_train: torch.Tensor,
    val_ema: Optional[torch.Tensor],
    init_val: Optional[torch.Tensor],
    progress_alpha: float,
    floor_beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_ratio = train_ema.clamp_min(1e-6) / init_train.clamp_min(1e-6)
    progress = train_ratio / train_ratio.mean().clamp_min(1e-6)
    if val_ema is None or init_val is None:
        floor = torch.ones_like(progress)
    else:
        val_ratio = val_ema.clamp_min(1e-6) / init_val.clamp_min(1e-6)
        floor = val_ratio / val_ratio.max().clamp_min(1e-6)
    raw = progress.pow(progress_alpha)
    raw = raw * (1.0 + floor_beta * floor)
    weights = len(train_ratio) * raw / raw.sum().clamp_min(1e-6)
    return weights.detach(), progress.detach(), floor.detach()


def train_pace_variant(
    model: PaceMTModel,
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, DataLoader],
    val_loaders: Dict[str, DataLoader],
    test_loaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    patience: int,
    head_lr: float,
    encoder_lr_mult: float,
    weight_decay: float,
    pace_progress_alpha: float,
    pace_floor_beta: float,
    pace_conflict_tau: float,
    pace_ema: float,
    pace_private_penalty: float,
    log_prefix: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    model = model.to(device)
    set_trainable(model)
    init_regression_biases(model, tasks)

    opt = make_optimizer(model, head_lr, encoder_lr_mult, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    shared_params = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("encoder.")]

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0

    init_train_losses: Optional[torch.Tensor] = None
    train_ema: Optional[torch.Tensor] = None
    init_val_losses: Optional[torch.Tensor] = None
    val_ema: Optional[torch.Tensor] = None

    last_private = 0.0
    last_conflict_rate = 0.0
    last_weights: List[float] = [1.0 for _ in tasks]
    last_progress: List[float] = [1.0 for _ in tasks]
    last_floor: List[float] = [1.0 for _ in tasks]

    for ep in range(epochs):
        model.train()
        train_losses: List[float] = []
        task_iters = {rt.spec.name: iter(train_loaders[rt.spec.name]) for rt in tasks}
        n_steps = max(len(train_loaders[rt.spec.name]) for rt in tasks)
        for _ in range(n_steps):
            per_task_losses: List[torch.Tensor] = []
            batches: List[Tuple[TaskRuntime, torch.Tensor, torch.Tensor]] = []
            for rt in tasks:
                name = rt.spec.name
                try:
                    xb, yb = next(task_iters[name])
                except StopIteration:
                    task_iters[name] = iter(train_loaders[name])
                    xb, yb = next(task_iters[name])
                batches.append((rt, xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)))

            for rt, xb, yb in batches:
                _, out = model(xb)
                per_task_losses.append(task_losses(out, {rt.spec.name: yb}, [rt])[rt.spec.name])
            if not per_task_losses:
                continue

            current_losses = torch.as_tensor(
                [float(loss.detach().item()) for loss in per_task_losses],
                device=device,
                dtype=torch.float32,
            ).clamp_min_(1e-6)
            if init_train_losses is None:
                init_train_losses = current_losses.clone()
                train_ema = current_losses.clone()
            else:
                assert train_ema is not None
                train_ema = pace_ema * train_ema + (1.0 - pace_ema) * current_losses

            assert init_train_losses is not None and train_ema is not None
            weights, progress_ratio, floor_ratio = pace_task_weights(
                train_ema=train_ema,
                init_train=init_train_losses,
                val_ema=val_ema,
                init_val=init_val_losses,
                progress_alpha=pace_progress_alpha,
                floor_beta=pace_floor_beta,
            )

            task_stack = torch.stack(per_task_losses)
            task_grads: List[torch.Tensor] = []
            if shared_params:
                for i, loss_i in enumerate(per_task_losses):
                    grads = torch.autograd.grad(
                        weights[i] * loss_i,
                        shared_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    task_grads.append(flatten_grads(grads, shared_params))

            total_task_loss = torch.sum(weights * task_stack)
            private_penalty = model.private_penalty()
            total_loss = total_task_loss + pace_private_penalty * private_penalty

            opt.zero_grad(set_to_none=True)
            total_loss.backward()

            if task_grads and shared_params:
                merged, n_conflicts, n_pairs = conditional_pcgrad_merge(torch.stack(task_grads), pace_conflict_tau)
                assign_flat_grad(shared_params, merged)
                last_conflict_rate = float(n_conflicts / max(1, n_pairs))
            else:
                last_conflict_rate = 0.0

            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()

            last_private = float(private_penalty.detach().item())
            last_weights = [float(v) for v in weights.detach().cpu().tolist()]
            last_progress = [float(v) for v in progress_ratio.detach().cpu().tolist()]
            last_floor = [float(v) for v in floor_ratio.detach().cpu().tolist()]
            train_losses.append(float(total_loss.detach().item()))

        sched.step()
        per_task_val = validation_losses_by_task(model, val_loaders, tasks, device)
        val = float(np.mean(list(per_task_val.values()))) if per_task_val else float("inf")
        current_val = torch.as_tensor(
            [per_task_val[rt.spec.name] for rt in tasks],
            device=device,
            dtype=torch.float32,
        ).clamp_min_(1e-6)
        if init_val_losses is None:
            init_val_losses = current_val.clone()
            val_ema = current_val.clone()
        else:
            assert val_ema is not None
            val_ema = pace_ema * val_ema + (1.0 - pace_ema) * current_val

        improved = val < best_val
        if improved:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if improved or (ep + 1) % 10 == 0:
            weight_str = ",".join(f"{rt.spec.name}:{w:.2f}" for rt, w in zip(tasks, last_weights))
            prog_str = ",".join(f"{rt.spec.name}:{w:.2f}" for rt, w in zip(tasks, last_progress))
            floor_str = ",".join(f"{rt.spec.name}:{w:.2f}" for rt, w in zip(tasks, last_floor))
            print(
                f"    {log_prefix}ep{ep:03d} train={np.mean(train_losses):.4f} "
                f"val_loss={val:.4f} best={best_val:.4f} bad={bad} "
                f"conflict={last_conflict_rate:.3f} private={last_private:.4g} "
                f"w[{weight_str}] prog[{prog_str}] floor[{floor_str}]",
                flush=True,
            )
        if bad >= patience:
            print(f"    {log_prefix}early-stop @ ep{ep}", flush=True)
            break

    model.load_state_dict(best_state)
    metrics = evaluate(model, test_loaders, tasks, device)
    return metrics, {"val_loss_best": float(best_val), "epochs_trained": int(ep + 1)}


def train_physioscale_variant(
    model: PhysioScaleMTModel,
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, DataLoader],
    val_loaders: Dict[str, DataLoader],
    test_loaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    patience: int,
    head_lr: float,
    encoder_lr_mult: float,
    weight_decay: float,
    physio_diversity: float,
    physio_router_prior: float,
    log_prefix: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    model = model.to(device)
    set_trainable(model)
    init_regression_biases(model, tasks)

    opt = make_optimizer(model, head_lr, encoder_lr_mult, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    init_losses: Optional[torch.Tensor] = None
    last_div = 0.0
    last_prior = 0.0
    last_router: Dict[str, List[float]] = {rt.spec.name: [1.0 / len(PHYSIO_FACTOR_NAMES)] * len(PHYSIO_FACTOR_NAMES) for rt in tasks}

    for ep in range(epochs):
        model.train()
        train_losses: List[float] = []
        task_iters = {rt.spec.name: iter(train_loaders[rt.spec.name]) for rt in tasks}
        n_steps = max(len(train_loaders[rt.spec.name]) for rt in tasks)
        for _ in range(n_steps):
            per_task_losses: List[torch.Tensor] = []
            per_div: List[torch.Tensor] = []
            batches: List[Tuple[TaskRuntime, torch.Tensor, torch.Tensor]] = []
            for rt in tasks:
                name = rt.spec.name
                try:
                    xb, yb = next(task_iters[name])
                except StopIteration:
                    task_iters[name] = iter(train_loaders[name])
                    xb, yb = next(task_iters[name])
                batches.append((rt, xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)))

            for rt, xb, yb in batches:
                _, out, aux = model.forward_with_aux(xb)
                loss_i = task_losses(out, {rt.spec.name: yb}, [rt])[rt.spec.name]
                per_task_losses.append(loss_i)
                per_div.append(aux["factor_diversity"])
                last_router = {
                    name: [float(v) for v in weights.detach().cpu().tolist()]
                    for name, weights in aux["router_weights"].items()
                }

            if not per_task_losses:
                continue

            loss_vec = torch.stack(per_task_losses)
            if init_losses is None:
                init_losses = loss_vec.detach().clamp_min(1e-6)
            scaled_loss = torch.mean(loss_vec / init_losses.clamp_min(1e-6))
            diversity_loss = torch.stack(per_div).mean() if per_div else loss_vec.new_tensor(0.0)
            prior_loss = model.router_prior_loss()
            total_loss = scaled_loss + physio_diversity * diversity_loss + physio_router_prior * prior_loss

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()

            last_div = float(diversity_loss.detach().item())
            last_prior = float(prior_loss.detach().item())
            train_losses.append(float(total_loss.detach().item()))

        sched.step()
        val = validation_loss(model, val_loaders, tasks, device)
        improved = val < best_val
        if improved:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if improved or (ep + 1) % 10 == 0:
            router_str = " | ".join(
                f"{name}:r{weights[0]:.2f}/m{weights[1]:.2f}/c{weights[2]:.2f}"
                for name, weights in last_router.items()
            )
            print(
                f"    {log_prefix}ep{ep:03d} train={np.mean(train_losses):.4f} "
                f"val_loss={val:.4f} best={best_val:.4f} bad={bad} "
                f"div={last_div:.4g} prior={last_prior:.4g} routers[{router_str}]",
                flush=True,
            )
        if bad >= patience:
            print(f"    {log_prefix}early-stop @ ep{ep}", flush=True)
            break

    model.load_state_dict(best_state)
    metrics = evaluate(model, test_loaders, tasks, device)
    router_meta: Dict[str, Any] = {}
    for name, weights in model.router_weight_dict().items():
        vals = [float(v) for v in weights.detach().cpu().tolist()]
        router_meta[f"router_{name}_rhythm"] = vals[0]
        router_meta[f"router_{name}_morphology"] = vals[1]
        router_meta[f"router_{name}_context"] = vals[2]
    router_meta.update({"val_loss_best": float(best_val), "epochs_trained": int(ep + 1)})
    return metrics, router_meta


def train_uniphys_variant(
    model: UniPhysMTModel,
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, DataLoader],
    val_loaders: Dict[str, DataLoader],
    test_loaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    patience: int,
    head_lr: float,
    encoder_lr_mult: float,
    weight_decay: float,
    uniphys_diversity: float,
    uniphys_floor_weight: float,
    uniphys_router_prior: float,
    uniphys_worst_k: int,
    log_prefix: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    model = model.to(device)
    set_trainable(model)
    init_regression_biases(model, tasks)

    opt = make_optimizer(model, head_lr, encoder_lr_mult, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    init_losses: Optional[torch.Tensor] = None
    last_div = 0.0
    last_prior = 0.0
    last_worst = 0.0
    last_router: Dict[str, List[float]] = {
        rt.spec.name: [1.0 / len(PHYSIO_FACTOR_NAMES)] * len(PHYSIO_FACTOR_NAMES)
        for rt in tasks
    }
    worst_k = uniphys_worst_k if uniphys_worst_k > 0 else (1 if len(tasks) <= 2 else 2)
    worst_k = max(1, min(worst_k, len(tasks)))

    for ep in range(epochs):
        model.train()
        train_losses: List[float] = []
        task_iters = {rt.spec.name: iter(train_loaders[rt.spec.name]) for rt in tasks}
        n_steps = max(len(train_loaders[rt.spec.name]) for rt in tasks)
        for _ in range(n_steps):
            per_task_losses: List[torch.Tensor] = []
            per_div: List[torch.Tensor] = []
            batches: List[Tuple[TaskRuntime, torch.Tensor, torch.Tensor]] = []
            for rt in tasks:
                name = rt.spec.name
                try:
                    xb, yb = next(task_iters[name])
                except StopIteration:
                    task_iters[name] = iter(train_loaders[name])
                    xb, yb = next(task_iters[name])
                batches.append((rt, xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)))

            for rt, xb, yb in batches:
                _, out, aux = model.forward_with_aux(xb)
                loss_i = task_losses(out, {rt.spec.name: yb}, [rt])[rt.spec.name]
                per_task_losses.append(loss_i)
                per_div.append(aux["factor_diversity"])
                last_router = {
                    name: [float(v) for v in weights.detach().cpu().tolist()]
                    for name, weights in aux["router_weights"].items()
                }

            if not per_task_losses:
                continue

            loss_vec = torch.stack(per_task_losses)
            if init_losses is None:
                init_losses = loss_vec.detach().clamp_min(1e-6)
            scaled = loss_vec / init_losses.clamp_min(1e-6)
            main_loss = scaled.mean()
            worst_loss = torch.topk(scaled, k=worst_k).values.mean()
            diversity_loss = torch.stack(per_div).mean() if per_div else loss_vec.new_tensor(0.0)
            prior_loss = model.router_prior_loss()
            total_loss = main_loss + uniphys_floor_weight * worst_loss + uniphys_diversity * diversity_loss
            if uniphys_router_prior > 0:
                total_loss = total_loss + uniphys_router_prior * prior_loss

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()

            last_div = float(diversity_loss.detach().item())
            last_prior = float(prior_loss.detach().item())
            last_worst = float(worst_loss.detach().item())
            train_losses.append(float(total_loss.detach().item()))

        sched.step()
        val = validation_loss(model, val_loaders, tasks, device)
        improved = val < best_val
        if improved:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if improved or (ep + 1) % 10 == 0:
            router_str = " | ".join(
                f"{name}:r{weights[0]:.2f}/m{weights[1]:.2f}/c{weights[2]:.2f}"
                for name, weights in last_router.items()
            )
            print(
                f"    {log_prefix}ep{ep:03d} train={np.mean(train_losses):.4f} "
                f"val_loss={val:.4f} best={best_val:.4f} bad={bad} "
                f"worst={last_worst:.4g} div={last_div:.4g} prior={last_prior:.4g} "
                f"routers[{router_str}]",
                flush=True,
            )
        if bad >= patience:
            print(f"    {log_prefix}early-stop @ ep{ep}", flush=True)
            break

    model.load_state_dict(best_state)
    metrics = evaluate(model, test_loaders, tasks, device)
    router_meta: Dict[str, Any] = {}
    for name, weights in model.router_weight_dict().items():
        vals = [float(v) for v in weights.detach().cpu().tolist()]
        router_meta[f"router_{name}_rhythm"] = vals[0]
        router_meta[f"router_{name}_morphology"] = vals[1]
        router_meta[f"router_{name}_context"] = vals[2]
    router_meta.update({"val_loss_best": float(best_val), "epochs_trained": int(ep + 1)})
    return metrics, router_meta


def train_variant(
    model: nn.Module,
    method: str,
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, DataLoader],
    val_loaders: Dict[str, DataLoader],
    test_loaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    patience: int,
    head_lr: float,
    encoder_lr_mult: float,
    weight_decay: float,
    mask_sparse: float,
    mask_overlap: float,
    shared_anchor: float,
    gradnorm_alpha: float,
    gradnorm_lr: float,
    log_prefix: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    model = model.to(device)
    set_trainable(model)
    init_regression_biases(model, tasks)
    if hasattr(model, "anchor_encoder"):
        model.anchor_encoder = model.anchor_encoder.to(device).eval()
        model.anchor_encoder.requires_grad_(False)

    opt = make_optimizer(model, head_lr, encoder_lr_mult, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    task_logits = None
    task_opt = None
    init_losses = None
    if method in GRADNORM_METHODS:
        task_logits = nn.Parameter(torch.zeros(len(tasks), device=device))
        task_opt = torch.optim.Adam([task_logits], lr=gradnorm_lr)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    last_sparse = 0.0
    last_overlap = 0.0
    last_shared_anchor = 0.0
    last_aux = 0.0

    for ep in range(epochs):
        model.train()
        if hasattr(model, "anchor_encoder"):
            model.anchor_encoder.eval()

        train_losses: List[float] = []
        task_iters = {rt.spec.name: iter(train_loaders[rt.spec.name]) for rt in tasks}
        n_steps = max(len(train_loaders[rt.spec.name]) for rt in tasks)
        for _ in range(n_steps):
            per_task_losses: List[torch.Tensor] = []
            per_task_z: List[torch.Tensor] = []
            batches: List[Tuple[TaskRuntime, torch.Tensor, torch.Tensor]] = []
            for rt in tasks:
                name = rt.spec.name
                try:
                    xb, yb = next(task_iters[name])
                except StopIteration:
                    task_iters[name] = iter(train_loaders[name])
                    xb, yb = next(task_iters[name])
                batches.append((rt, xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)))

            if method == "pcgrad_joint":
                opt.zero_grad(set_to_none=True)
                params = [p for p in model.parameters() if p.requires_grad]
                task_grads: List[torch.Tensor] = []
                task_scalars: List[float] = []
                for i, (rt, xb, yb) in enumerate(batches):
                    z, out = model(xb)
                    loss_i = task_losses(out, {rt.spec.name: yb}, [rt])[rt.spec.name]
                    grads = torch.autograd.grad(
                        loss_i,
                        params,
                        retain_graph=i < len(batches) - 1,
                        allow_unused=True,
                    )
                    task_grads.append(flatten_grads(grads, params))
                    task_scalars.append(float(loss_i.detach().item()))
                merged = pcgrad_merge(torch.stack(task_grads))
                assign_flat_grad(params, merged)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                loss_scalar = float(np.mean(task_scalars))
                train_losses.append(loss_scalar)
                continue

            for rt, xb, yb in batches:
                z, out = model(xb)
                loss_i = task_losses(out, {rt.spec.name: yb}, [rt])[rt.spec.name]
                per_task_losses.append(loss_i)
                per_task_z.append(z)
            if not per_task_losses:
                continue

            sparse = per_task_losses[0].new_tensor(0.0)
            overlap = sparse
            soft_anchor_loss = sparse
            if method in SP_METHODS and hasattr(model, "mask_regularization"):
                sparse, overlap = model.mask_regularization()
                last_sparse = float(sparse.detach().item())
                last_overlap = float(overlap.detach().item())
            if method in SOFT_SP_METHODS and hasattr(model, "shared_anchor_loss"):
                regs = [
                    model.shared_anchor_loss(xb, z)
                    for (_, xb, _), z in zip(batches, per_task_z)
                ]
                if regs:
                    soft_anchor_loss = torch.stack(regs).mean()
                    last_shared_anchor = float(soft_anchor_loss.detach().item())

            if method in {"sp_mtft", "soft_sp_mtft"}:
                loss = torch.stack(per_task_losses).mean() + mask_sparse * sparse + mask_overlap * overlap
                if method == "soft_sp_mtft":
                    loss = loss + shared_anchor * soft_anchor_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                opt.step()
                train_losses.append(float(loss.detach().item()))
                continue

            if method in GRADNORM_METHODS:
                assert task_logits is not None and task_opt is not None
                losses_tensor = torch.stack(per_task_losses)
                if init_losses is None:
                    init_losses = losses_tensor.detach().clamp_min(1e-6)
                weights = len(tasks) * torch.softmax(task_logits, dim=0)
                enc_params = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("encoder.")]
                G_list: List[torch.Tensor] = []
                for i, loss_i in enumerate(per_task_losses):
                    grads = torch.autograd.grad(
                        weights[i] * loss_i,
                        enc_params,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )
                    flat = flatten_grads(grads, enc_params)
                    G_list.append(flat.norm(2))
                G = torch.stack(G_list)
                loss_ratio = losses_tensor.detach().clamp_min(1e-6) / init_losses
                inv_rate = loss_ratio / loss_ratio.mean()
                target = G.detach().mean() * inv_rate.pow(gradnorm_alpha)
                gradnorm_loss = torch.abs(G - target).sum()
                total_loss = torch.sum(weights.detach() * losses_tensor)
                if method in SP_METHODS:
                    total_loss = total_loss + mask_sparse * sparse + mask_overlap * overlap
                if method == "soft_sp_gradnorm":
                    total_loss = total_loss + shared_anchor * soft_anchor_loss

                task_opt.zero_grad(set_to_none=True)
                opt.zero_grad(set_to_none=True)
                gradnorm_loss.backward(retain_graph=True)
                task_opt.step()

                task_opt.zero_grad(set_to_none=True)
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                opt.step()

                last_aux = float(gradnorm_loss.detach().item())
                train_losses.append(float(total_loss.detach().item()))
                continue

            raise ValueError(f"Unsupported method: {method}")

        sched.step()
        val = validation_loss(model, val_loaders, tasks, device)
        improved = val < best_val
        if improved:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if improved or (ep + 1) % 10 == 0:
            extra = ""
            if method in {"sp_mtft", "sp_gradnorm", "soft_sp_mtft", "soft_sp_gradnorm"}:
                extra = f" sparse={last_sparse:.4g} overlap={last_overlap:.4g}"
                if method in SOFT_SP_METHODS:
                    extra += f" shared_anchor={last_shared_anchor:.4g}"
            if method in GRADNORM_METHODS:
                extra = f" gradnorm={last_aux:.4g}"
                if method in SP_METHODS:
                    extra += f" sparse={last_sparse:.4g} overlap={last_overlap:.4g}"
                if method == "soft_sp_gradnorm":
                    extra += f" shared_anchor={last_shared_anchor:.4g}"
            print(
                f"    {log_prefix}ep{ep:03d} train={np.mean(train_losses):.4f} "
                f"val_loss={val:.4f} best={best_val:.4f} bad={bad}{extra}",
                flush=True,
            )
        if bad >= patience:
            print(f"    {log_prefix}early-stop @ ep{ep}", flush=True)
            break

    model.load_state_dict(best_state)
    metrics = evaluate(model, test_loaders, tasks, device)
    return metrics, {"val_loss_best": float(best_val), "epochs_trained": int(ep + 1)}


def run_method(
    method: str,
    student: str,
    seed: int,
    kd_method: str,
    ckpt_dir: Path,
    task_data: List[TaskData],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, Any]]:
    tasks = [td.runtime for td in task_data]
    temporal_config_json = default_temporal_config_json(args.dataset, task_data, args.hz)
    enc, _, ck_meta = load_stage1_full(student, ckpt_dir, kd_method, seed, device)
    train_loaders, val_loaders, test_loaders = make_task_loaders(task_data, args.batch_size)
    spec = None

    if method in SP_METHODS:
        anchor_enc, _, _ = load_stage1_full(student, ckpt_dir, kd_method, seed, device)
        spec = build_student_subspace(
            anchor_encoder=anchor_enc,
            task_data=task_data,
            device=device,
            batch_size=args.batch_size,
            shared_var_thresh=args.shared_var_thresh,
            shared_frac=args.shared_frac,
        )
        if method in SOFT_SP_METHODS:
            model = SoftSharedPrivateMTModel(enc, anchor_enc, int(ck_meta["enc_dim"]), tasks, args.dropout, spec)
        else:
            model = SharedPrivateMTModel(enc, anchor_enc, int(ck_meta["enc_dim"]), tasks, args.dropout, spec)
    elif method in PACE_METHODS:
        model = PaceMTModel(
            enc,
            int(ck_meta["enc_dim"]),
            tasks,
            args.dropout,
            args.pace_private_rank,
        )
    elif method in PHYSIOSCALE_METHODS:
        model = PhysioScaleMTModel(
            enc,
            int(ck_meta["enc_dim"]),
            tasks,
            args.dropout,
            args.physio_bottleneck,
        )
    elif method in UNIPHYS_METHODS:
        model = UniPhysMTModel(
            enc,
            int(ck_meta["enc_dim"]),
            tasks,
            args.dropout,
            args.uniphys_bottleneck,
        )
    else:
        model = MultiTaskModel(enc, int(ck_meta["enc_dim"]), tasks, args.dropout)

    log_prefix = f"[{student}/s{seed}/{method}] "
    t0 = time.time()
    if method in PACE_METHODS:
        metrics, meta = train_pace_variant(
            model=model,
            tasks=tasks,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            head_lr=args.head_lr,
            encoder_lr_mult=args.encoder_lr_mult,
            weight_decay=args.weight_decay,
            pace_progress_alpha=args.pace_progress_alpha,
            pace_floor_beta=args.pace_floor_beta,
            pace_conflict_tau=args.pace_conflict_tau,
            pace_ema=args.pace_ema,
            pace_private_penalty=args.pace_private_penalty,
            log_prefix=log_prefix,
        )
    elif method in PHYSIOSCALE_METHODS:
        metrics, meta = train_physioscale_variant(
            model=model,
            tasks=tasks,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            head_lr=args.head_lr,
            encoder_lr_mult=args.encoder_lr_mult,
            weight_decay=args.weight_decay,
            physio_diversity=args.physio_diversity,
            physio_router_prior=args.physio_router_prior,
            log_prefix=log_prefix,
        )
    elif method in UNIPHYS_METHODS:
        metrics, meta = train_uniphys_variant(
            model=model,
            tasks=tasks,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            head_lr=args.head_lr,
            encoder_lr_mult=args.encoder_lr_mult,
            weight_decay=args.weight_decay,
            uniphys_diversity=args.uniphys_diversity,
            uniphys_floor_weight=args.uniphys_floor_weight,
            uniphys_router_prior=args.uniphys_router_prior,
            uniphys_worst_k=args.uniphys_worst_k,
            log_prefix=log_prefix,
        )
    else:
        metrics, meta = train_variant(
            model=model,
            method=method,
            tasks=tasks,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            head_lr=args.head_lr,
            encoder_lr_mult=args.encoder_lr_mult,
            weight_decay=args.weight_decay,
            mask_sparse=args.mask_sparse,
            mask_overlap=args.mask_overlap,
            shared_anchor=args.shared_anchor,
            gradnorm_alpha=args.gradnorm_alpha,
            gradnorm_lr=args.gradnorm_lr,
            log_prefix=log_prefix,
        )

    base = {
        "dataset": args.dataset,
        "student": student,
        "kd_method": kd_method,
        "seed": seed,
        "method": method,
        "encoder_update_scope": "full",
        "temporal_policy": "shared_window",
        "temporal_config": temporal_config_json,
        "input_window_policy": "task_specific_input" if args.input_window_sec_map else "shared_native",
        "input_window_sec_map": json.dumps(
            resolve_input_window_sec_map(
                [rt.spec.name for rt in tasks],
                override_json=args.input_window_sec_map,
            ),
            sort_keys=True,
        ),
        "head_lr": args.head_lr,
        "encoder_lr_mult": args.encoder_lr_mult,
        "deploy_encoders": 1,
        "encoder_params": count_params(model.encoder),
        "total_params": deploy_param_count(model),
        "trainable_params": count_params(model, trainable_only=True),
        "stage1_ckpt": ck_meta["ckpt_path"],
        "time_s": float(time.time() - t0),
        "shared_dim": int(spec.shared_dim if spec is not None else 0),
        "private_dim": int(spec.private_dim if spec is not None else 0),
        "candidate_dim": int(spec.candidate_dim if spec is not None else 0),
        "shared_var_covered": float(spec.shared_var_covered if spec is not None else 0.0),
        "mask_sparse": float(args.mask_sparse if method in SP_METHODS else 0.0),
        "mask_overlap": float(args.mask_overlap if method in SP_METHODS else 0.0),
        "shared_anchor": float(args.shared_anchor if method in SOFT_SP_METHODS else 0.0),
        "gradnorm_alpha": float(args.gradnorm_alpha if method in GRADNORM_METHODS else 0.0),
        "pace_progress_alpha": float(args.pace_progress_alpha if method in PACE_METHODS else 0.0),
        "pace_floor_beta": float(args.pace_floor_beta if method in PACE_METHODS else 0.0),
        "pace_conflict_tau": float(args.pace_conflict_tau if method in PACE_METHODS else 0.0),
        "pace_private_rank": int(args.pace_private_rank if method in PACE_METHODS else 0),
        "pace_private_penalty": float(args.pace_private_penalty if method in PACE_METHODS else 0.0),
        "physio_bottleneck": int(args.physio_bottleneck if method in PHYSIOSCALE_METHODS else 0),
        "physio_diversity": float(args.physio_diversity if method in PHYSIOSCALE_METHODS else 0.0),
        "physio_router_prior": float(args.physio_router_prior if method in PHYSIOSCALE_METHODS else 0.0),
        "uniphys_bottleneck": int(args.uniphys_bottleneck if method in UNIPHYS_METHODS else 0),
        "uniphys_diversity": float(args.uniphys_diversity if method in UNIPHYS_METHODS else 0.0),
        "uniphys_floor_weight": float(args.uniphys_floor_weight if method in UNIPHYS_METHODS else 0.0),
        "uniphys_router_prior": float(args.uniphys_router_prior if method in UNIPHYS_METHODS else 0.0),
        "uniphys_worst_k": int(args.uniphys_worst_k if method in UNIPHYS_METHODS else 0),
    }
    rows = records_from_metrics(metrics, base, tasks, meta)
    if method in PHYSIOSCALE_METHODS or method in UNIPHYS_METHODS:
        for rec in rows:
            task_name = rec["task"]
            rec["router_rhythm"] = float(meta.get(f"router_{task_name}_rhythm", np.nan))
            rec["router_morphology"] = float(meta.get(f"router_{task_name}_morphology", np.nan))
            rec["router_context"] = float(meta.get(f"router_{task_name}_context", np.nan))
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[device] {device}", flush=True)
    print(f"[dataset] {args.dataset}", flush=True)
    frozen_reference_df = load_reference_frozen_rows(args.dataset)

    out_dir = args.out_root / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "stage2_results.csv"
    summary_csv = out_dir / "stage2_summary.csv"
    summary_by_student_csv = out_dir / "stage2_summary_by_student.csv"
    config_json = out_dir / "stage2_config.json"
    config_json.write_text(json.dumps(vars(args), indent=2, default=str))

    if args.resume and out_csv.exists():
        df = ensure_temporal_cols(pd.read_csv(out_csv))
        df, summary = score_rows_with_frozen_reference(df, frozen_reference_df, args.methods)
        summary.to_csv(summary_csv, index=False)
        aggregate_over_seeds(summary).to_csv(summary_by_student_csv, index=False)
        print(summary.to_string(index=False), flush=True)
        return

    existing_df = pd.DataFrame()
    if out_csv.exists() and not args.fresh:
        existing_df = ensure_temporal_cols(pd.read_csv(out_csv))
        print(f"[append] loaded {len(existing_df)} existing rows from {out_csv}", flush=True)

    all_rows: List[Dict[str, Any]] = existing_df.to_dict("records") if not existing_df.empty else []
    printed_data = False
    for student in args.students:
        ckpt_dir = DEFAULT_CKPT_DIRS.get(student)
        if ckpt_dir is None:
            ckpt_dir = ROOT / f"kd_sweep_results/pulseppg_preliminary/{student.lower()}_ckpts"
        for seed in args.seeds:
            set_seed(seed)
            task_data = load_multitask_dataset(
                args.dataset,
                hz=args.hz,
                limit_train=args.limit_train,
                limit_val=args.limit_val,
                limit_test=args.limit_test,
                seed=seed,
            )
            input_window_sec_map = resolve_input_window_sec_map(
                [td.runtime.spec.name for td in task_data],
                override_json=args.input_window_sec_map,
            )
            if input_window_sec_map:
                task_data = apply_task_specific_input_windows(task_data, args.hz, input_window_sec_map)
            if not printed_data:
                parts = []
                for td in task_data:
                    parts.append(
                        f"{td.runtime.spec.name}:"
                        f"tr{len(td.split_X['train'])}/va{len(td.split_X['val'])}/te{len(td.split_X['test'])}"
                    )
                print("[data] " + " | ".join(parts), flush=True)
                print("[tasks] " + ", ".join(f"{td.runtime.spec.name}:{td.runtime.spec.kind}" for td in task_data), flush=True)
                if input_window_sec_map:
                    print(f"[input-window-sec-map] {json.dumps(input_window_sec_map, sort_keys=True)}", flush=True)
                printed_data = True
            try:
                _, _, ck_meta = load_stage1_full(student, ckpt_dir, args.kd_method, seed, device)
            except Exception as e:
                print(f"[warn] {e}", flush=True)
                continue
            print(
                f"\n{'=' * 80}\n"
                f"student={student} seed={seed} kd={args.kd_method}\n"
                f"ckpt={ck_meta['ckpt_path']}\n"
                f"{'=' * 80}",
                flush=True,
            )
            for method in args.methods:
                temporal_config_json = default_temporal_config_json(args.dataset, task_data, args.hz)
                if method_completed(
                    existing_df,
                    args.dataset,
                    student,
                    args.kd_method,
                    seed,
                    method,
                    task_data,
                    "full",
                    "shared_window",
                    temporal_config_json,
                    "task_specific_input" if input_window_sec_map else "shared_native",
                    json.dumps(input_window_sec_map, sort_keys=True),
                ):
                    print(f"\n-- method: {method} [cached] --", flush=True)
                    continue
                print(f"\n-- method: {method} --", flush=True)
                try:
                    rows = run_method(
                        method=method,
                        student=student,
                        seed=seed,
                        kd_method=args.kd_method,
                        ckpt_dir=ckpt_dir,
                        task_data=task_data,
                        args=args,
                        device=device,
                    )
                except Exception:
                    import traceback

                    traceback.print_exc()
                    continue
                all_rows.extend(rows)
                df_tmp, summary_tmp = score_rows_with_frozen_reference(
                    ensure_temporal_cols(pd.DataFrame(all_rows)),
                    frozen_reference_df,
                    args.methods,
                )
                df_tmp.to_csv(out_csv, index=False)
                summary_tmp.to_csv(summary_csv, index=False)
                aggregate_over_seeds(summary_tmp).to_csv(summary_by_student_csv, index=False)

    if not all_rows:
        raise RuntimeError("No rows were produced.")

    df, summary = score_rows_with_frozen_reference(
        ensure_temporal_cols(pd.DataFrame(all_rows)),
        frozen_reference_df,
        args.methods,
    )
    df.to_csv(out_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    aggregate_over_seeds(summary).to_csv(summary_by_student_csv, index=False)
    print(f"\n[done] wrote {out_csv}", flush=True)
    print(f"[done] wrote {summary_csv}", flush=True)
    print(f"[done] wrote {summary_by_student_csv}", flush=True)
    print(summary[["dataset", "student", "seed", "method", "avg_rel_vs_frozen", "floor_rel_vs_frozen"]].to_string(index=False))


if __name__ == "__main__":
    main()

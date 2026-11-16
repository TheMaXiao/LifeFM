"""
downstream_eval.py — Unified linear-probe benchmark for PPG foundation models.

Covers tasks from:
  • PulsePPG paper (arXiv:2502.01108): 5 datasets, 11 tasks
  • PapaGei paper  (arXiv:2410.20542): 10 datasets, 20 tasks
  • PulseLM paper  (arXiv:2603.03331): 3 additional source datasets
  (tasks that share available local datasets are included)

Quick start
-----------
    from downstream_eval import run_task, run_all_tasks

    DATA_DIRS = {
        "wesad":          "pulseppg/data/datasets/wesad",
        "sdb":            "pulseppg/data/datasets/sdb",
        "wfppg":          "pulseppg/data/datasets/wfppg",
        "dalia":          "pulseppg/data/datasets/dalia",
        "ppgbp":          "pulseppg/data/datasets/ppgbp",
        # PulseLM Table-1 datasets (preprocess first – see pulseppg/data/process/):
        "mimicperform":   "pulseppg/data/datasets/mimicperform",
        "ppgarrhythmia":  "pulseppg/data/datasets/ppgarrhythmia",
        "earset":         "pulseppg/data/datasets/earset",
        # BIDMC + VitalDB (preprocess via pulseppg/data/process/{bidmc,vitaldb}.py):
        "bidmc":          "pulseppg/data/datasets/bidmc",
        "vitaldb":        "pulseppg/data/datasets/vitaldb",
        # optional – SportPPG needs extra keys (see TASK_REGISTRY notes):
        # "sportppg": {"data_root": "/path/...", "gt_root": "/path/..."},
    }

    results = run_all_tasks(encoder, device, DATA_DIRS, hz=50)
    for task, m in results.items():
        print(f"{task:30s}  {m['primary_metric']}={m['primary']:.4f}")

    # or a single task:
    m = run_task("sdb_binary", encoder, device, DATA_DIRS, hz=50)

Returned metric dict keys (all tasks)
--------------------------------------
  primary        – value of the primary metric
  primary_metric – name string ("f1_macro" | "mae")
  mae / rmse / r2                 (regression)
  f1_macro / acc / auroc / auprc  (classification)
  best_params                     (GridSearchCV best params)
  fold_results   (list – only for k-fold tasks)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize

# ──────────────────────────────────────────────────────────────────────────────
# Task registry
# ──────────────────────────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Pre-split (train/val/test .npy available) ─────────────────────────────
    # SDB: Sleep Disordered Breathing (PulsePPG T7 / PapaGei T15)
    "sdb_binary": {
        "type": "cls", "dataset": "sdb", "split_mode": "presplit",
        "x_file": "{split}_X_ppg_{hz}Hz.npy",
        "y_file": "{split}_y_sdb.npy",
        "label_transform": "binarize",   # (y > 0) → {0,1}
        "primary_metric": "f1_macro",
    },
    "sdb_multiclass": {
        "type": "cls", "dataset": "sdb", "split_mode": "presplit",
        "x_file": "{split}_X_ppg_{hz}Hz.npy",
        "y_file": "{split}_y_sdb.npy",
        "label_transform": None,
        "primary_metric": "f1_macro",
    },

    # PPG-DaLiA: HR regression (PulsePPG T3 / PapaGei T19)
    "ppgdalia_hr": {
        "type": "reg", "dataset": "dalia", "split_mode": "presplit",
        "x_file": "{split}_X_ppg_{hz}Hz.npy",
        "y_file": "{split}_y_hr.npy",          # no hz suffix
        "primary_metric": "mae",
    },
    # PPG-DaLiA: Activity 9-class (PulsePPG T4 / PapaGei T20)
    "ppgdalia_activity": {
        "type": "cls", "dataset": "dalia", "split_mode": "presplit",
        "x_file": "{split}_X_ppg_{hz}Hz.npy",
        "y_file": "{split}_y_act.npy",          # no hz suffix; float → int
        "label_transform": "to_int",
        "primary_metric": "f1_macro",
    },

    # PPG-BP: 4 tasks (PulsePPG T8-T11 / PapaGei T11-T14)
    "ppgbp_sysbp": {
        "type": "reg", "dataset": "ppgbp", "split_mode": "presplit",
        "x_file": "{split}_X_ppg_{hz}Hz.npy",
        "y_file": "{split}_y_sysbp_{hz}Hz.npy",
        "primary_metric": "mae",
    },
    "ppgbp_diasbp": {
        "type": "reg", "dataset": "ppgbp", "split_mode": "presplit",
        "x_file": "{split}_X_ppg_{hz}Hz.npy",
        "y_file": "{split}_y_diasbp_{hz}Hz.npy",
        "primary_metric": "mae",
    },
    "ppgbp_hr": {
        "type": "reg", "dataset": "ppgbp", "split_mode": "presplit",
        "x_file": "{split}_X_ppg_{hz}Hz.npy",
        "y_file": "{split}_y_hr_{hz}Hz.npy",
        "primary_metric": "mae",
    },
    "ppgbp_hypertension": {
        "type": "cls", "dataset": "ppgbp", "split_mode": "presplit",
        "x_file": "{split}_X_ppg_{hz}Hz.npy",
        "y_file": "{split}_y_ht_{hz}Hz.npy",
        "label_transform": "binarize",   # 4-class AHI → normal vs. any HT
        "primary_metric": "f1_macro",
    },

    # ── Subject-wise k-fold (single full .npy + subject IDs) ─────────────────
    # WESAD: Stress binary (PulsePPG T5 / PapaGei T17-T18 uses valence/arousal,
    #        but WESAD binary stress is the standard linear-probe task)
    "wesad_binary": {
        "type": "cls", "dataset": "wesad", "split_mode": "kfold",
        "subdir": "binary",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_stress_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "f1_macro",
    },
    # WESAD: Stress 4-class (PulsePPG T6)
    "wesad_multiclass": {
        "type": "cls", "dataset": "wesad", "split_mode": "kfold",
        "subdir": "multiclass",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_stress_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "f1_macro",
    },
    # WESAD: 60 s / stride-20 s majority-vote caches
    "wesad_binary_60s": {
        "type": "cls", "dataset": "wesad", "split_mode": "kfold",
        "subdir": "binary_len60_stride20",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_stress_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "f1_macro",
    },
    "wesad_multiclass_60s": {
        "type": "cls", "dataset": "wesad", "split_mode": "kfold",
        "subdir": "multiclass_len60_stride20",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_stress_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "f1_macro",
    },
    # WESAD: 20 s / stride-10 s majority-vote cache (binary only; multiclass keeps 60s)
    "wesad_binary_20s": {
        "type": "cls", "dataset": "wesad", "split_mode": "kfold",
        "subdir": "binary_len20_stride10",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_stress_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "f1_macro",
    },

    # WF-PPG HR regression (wrist / finger) – existing tasks
    "wfppg_wrist": {
        "type": "reg", "dataset": "wfppg", "split_mode": "kfold",
        "subdir": "wrist",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_hr_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "mae",
    },
    "wfppg_finger": {
        "type": "reg", "dataset": "wfppg", "split_mode": "kfold",
        "subdir": "finger",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_hr_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "mae",
    },

    # ── SportPPG: Activity (requires SportPPGConfig path dict) ────────────────
    "sportppg_activity": {
        "type": "cls", "dataset": "sportppg", "split_mode": "sportppg",
        "primary_metric": "f1_macro",
    },

    # ── PulseLM Table-1 datasets ──────────────────────────────────────────────
    # MIMIC PERform AF: binary AF detection (PulseLM dataset #9)
    # Preprocess with: pulseppg/data/process/mimicperform_af.py
    # Source: https://physionet.org/content/mimic-perform-af-dataset/1.0.0/
    "mimicperform_af": {
        "type": "cls", "dataset": "mimicperform", "split_mode": "kfold",
        "subdir": "",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_af_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "f1_macro",
    },

    # PPGArrhythmia: 6-class rhythm detection (PulseLM dataset #8)
    # Preprocess with: pulseppg/data/process/ppgarrhythmia.py
    # Source: https://doi.org/10.6084/m9.figshare.16946164
    # Classes: 0=sinus_rhythm  1=pvc  2=pac  3=vt  4=svt  5=af
    "ppgarrhythmia_6class": {
        "type": "cls", "dataset": "ppgarrhythmia", "split_mode": "kfold",
        "subdir": "",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_arrhythmia_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "f1_macro",
    },

    # EarSet: HR regression from in-ear PPG (PulseLM dataset #10)
    # Preprocess with: pulseppg/data/process/earset.py
    # Source: https://zenodo.org/records/7481901
    "earset_hr": {
        "type": "reg", "dataset": "earset", "split_mode": "kfold",
        "subdir": "",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_hr_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "mae",
    },

    # ── BIDMC (PPG + Respiration; 53 ICU recs) ──────────────────────────────
    # Preprocess with: pulseppg/data/process/bidmc.py
    # Source: https://physionet.org/content/bidmc/1.0.0/
    "bidmc_hr": {
        "type": "reg", "dataset": "bidmc", "split_mode": "kfold",
        "subdir": "",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_hr_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "mae",
    },
    "bidmc_rr": {
        "type": "reg", "dataset": "bidmc", "split_mode": "kfold",
        "subdir": "",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_rr_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "mae",
    },
    "bidmc_spo2": {
        "type": "reg", "dataset": "bidmc", "split_mode": "kfold",
        "subdir": "",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_spo2_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "mae",
    },

    # ── VitalDB SpO₂ (subset of cases with SNUADC/PLETH + PLETH_SPO2) ───────
    # Preprocess with: pulseppg/data/process/vitaldb.py
    # Source: https://vitaldb.net/
    "vitaldb_spo2": {
        "type": "reg", "dataset": "vitaldb", "split_mode": "kfold",
        "subdir": "",
        "x_file": "X_ppg_{hz}Hz.npy",
        "y_file": "y_spo2_{hz}Hz.npy",
        "subj_file": "subject_{hz}Hz.npy",
        "primary_metric": "mae",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

class _NpyDS(Dataset):
    """Minimal (x, y) dataset backed by numpy arrays."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.X[idx])).float()
        yi = self.y[idx]
        if np.issubdtype(type(yi), np.integer) or isinstance(yi, (int, np.integer)):
            return x, int(yi)
        return x, float(yi)


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(_NpyDS(X, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _forward_to_embedding(out: torch.Tensor | Any) -> torch.Tensor:
    """Accept (B,D) or (B,C,L) tensors; also handles dict outputs."""
    if isinstance(out, dict):
        for key in ("feat", "features", "embedding", "emb", "rep", "z", "proj", "logits"):
            if key in out and torch.is_tensor(out[key]):
                out = out[key]
                break
        else:
            out = next(v for v in out.values() if torch.is_tensor(v))
    if not torch.is_tensor(out):
        raise TypeError(f"Unexpected encoder output type: {type(out)}")
    if out.ndim == 2:
        return out
    if out.ndim == 3:
        return out.mean(dim=-1)
    return out.flatten(1)


_USE_FP16: bool = False   # set via set_fp16(True) for ~2.4× extraction speedup

def set_fp16(enabled: bool = True):
    """Enable/disable fp16 autocast during embedding extraction."""
    global _USE_FP16
    _USE_FP16 = enabled
    print(f"[downstream_eval] fp16 extraction {'enabled' if enabled else 'disabled'}", flush=True)


@torch.no_grad()
def _extract_all(encoder: nn.Module, X_all: np.ndarray, device: str,
                 batch_size: int = 512) -> np.ndarray:
    """Extract embeddings for the entire (N,1,L) array in batches."""
    encoder.eval()
    embs = []
    n = len(X_all)
    use_amp = _USE_FP16 and device != "cpu"
    for start in range(0, n, batch_size):
        xb = torch.from_numpy(np.asarray(X_all[start:start + batch_size])).float().to(device)
        if use_amp:
            with torch.cuda.amp.autocast():
                z = _forward_to_embedding(encoder(xb))
        else:
            z = _forward_to_embedding(encoder(xb))
        embs.append(z.cpu().float().numpy())
    return np.concatenate(embs, axis=0)  # (N, D)


@torch.no_grad()
def _extract_from_loader(encoder: nn.Module, loader: DataLoader,
                         device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (X_emb, y) from a DataLoader yielding (x, y) pairs."""
    encoder.eval()
    feats, labs = [], []
    for batch in loader:
        x, y = batch[0], batch[1]   # ignore extra fields
        x = x.to(device, non_blocking=True)
        z = _forward_to_embedding(encoder(x))
        feats.append(z.cpu().float().numpy())
        if torch.is_tensor(y):
            labs.append(y.cpu().numpy())
        else:
            labs.append(np.array(y))
    X_emb = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), dtype=np.float32)
    y_arr = np.concatenate(labs, axis=0) if labs else np.zeros((0,), dtype=np.float32)
    return X_emb, y_arr


# ──────────────────────────────────────────────────────────────────────────────
# GPU probe toggle  — call  set_gpu_probe("cuda:0")  before run_all_tasks
# to use PyTorch-based probes on GPU instead of sklearn (much faster).
# ──────────────────────────────────────────────────────────────────────────────

_GPU_PROBE_DEVICE: Optional[str] = None   # None → use sklearn (default)

def set_gpu_probe(device: Optional[str] = "cuda"):
    """Enable (device str) or disable (None) GPU-accelerated linear probes."""
    global _GPU_PROBE_DEVICE
    _GPU_PROBE_DEVICE = device
    if device:
        print(f"[downstream_eval] GPU probing enabled on {device}", flush=True)
    else:
        print("[downstream_eval] GPU probing disabled, using sklearn", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# GPU-accelerated probes (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

def _standardize_torch(X_tr: torch.Tensor, *others):
    """Z-score normalize using train-set statistics. Returns list of tensors."""
    mu = X_tr.mean(dim=0)
    std = X_tr.std(dim=0).clamp(min=1e-8)
    out = [(X_tr - mu) / std]
    for X in others:
        out.append((X - mu) / std)
    return out, mu, std


def _ridge_closed_form(X: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """Closed-form Ridge with automatic primal/dual selection + float64 for stability.

    Primal (D < N): w = (X^T X + αI)^{-1} X^T y       — (D×D) solve
    Dual   (D ≥ N): w = X^T (X X^T + αI)^{-1} y       — (N×N) solve
    """
    N, D = X.shape
    orig_dtype = X.dtype
    X64 = X.double()
    y64 = y.double()

    if D < N:  # primal
        XtX = X64.T @ X64
        XtX.diagonal().add_(alpha)
        Xty = X64.T @ y64
        w = torch.linalg.solve(XtX, Xty)
    else:  # dual — avoids rank-deficient D×D system
        XXt = X64 @ X64.T
        XXt.diagonal().add_(alpha)
        beta = torch.linalg.solve(XXt, y64)
        w = X64.T @ beta

    return w.to(orig_dtype)


def _kfold_indices(n: int, k: int, seed: int = 42, device: str = "cpu"):
    """Reproduce sklearn KFold(shuffle=True, random_state=seed) split.

    sklearn uses np.random.RandomState(seed).permutation(n) and then
    contiguous chunks (with first n%k chunks getting one extra).
    """
    rs = np.random.RandomState(seed)
    idx = rs.permutation(n)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out, s = [], 0
    for sz in sizes:
        out.append(torch.from_numpy(idx[s:s+sz]).long().to(device))
        s += sz
    return out


def _reg_probe_gpu(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_va: np.ndarray, y_va: np.ndarray,
                   X_te: np.ndarray, y_te: np.ndarray) -> Dict[str, Any]:
    """Ridge regression probe on GPU — sklearn-equivalent.

    Mirrors the CPU path: same StandardScaler, same alpha grid [0.1,1,10,100],
    same 3-fold KFold(shuffle=True, random_state=42), same neg-MSE scoring,
    same fit_intercept=True (handled by centering y).
    """
    dev = _GPU_PROBE_DEVICE
    X_tv_np = np.concatenate([X_tr, X_va])
    y_tv_np = np.concatenate([y_tr.astype(np.float32), y_va.astype(np.float32)])

    # ---- Standardize features (matches sklearn StandardScaler default ddof=0) ----
    X_tv64 = torch.from_numpy(X_tv_np).double().to(dev)
    X_te64 = torch.from_numpy(X_te.astype(np.float32).copy()).double().to(dev)
    mu = X_tv64.mean(0)
    # sklearn StandardScaler uses population std (ddof=0); torch.std defaults to ddof=1
    var = ((X_tv64 - mu) ** 2).mean(0)
    std = var.sqrt().clamp(min=1e-12)
    # sklearn replaces zero-std features by 1 (no scaling) — emulate:
    std = torch.where(std < 1e-10, torch.ones_like(std), std)
    X_tv = (X_tv64 - mu) / std
    X_te_s = (X_te64 - mu) / std

    y_tv = torch.from_numpy(y_tv_np).double().to(dev)

    alphas = [0.1, 1.0, 10.0, 100.0]
    n = X_tv.shape[0]
    folds = _kfold_indices(n, k=3, seed=42, device=dev)

    best_alpha, best_neg_mse = alphas[0], -1e30
    for alpha in alphas:
        fold_mses = []
        for fi in range(3):
            va_idx = folds[fi]
            tr_idx = torch.cat([folds[j] for j in range(3) if j != fi])
            Xtr, ytr = X_tv[tr_idx], y_tv[tr_idx]
            Xva, yva = X_tv[va_idx], y_tv[va_idx]
            # fit_intercept=True: center X, y on the training fold; sklearn
            # Ridge does this internally (X already centered if standardized,
            # but mean may not be exactly 0 on subset → re-center).
            xm = Xtr.mean(0); ym = ytr.mean()
            w = _ridge_closed_form(Xtr - xm, ytr - ym, alpha)
            preds = (Xva - xm) @ w + ym
            mse = -((yva - preds) ** 2).mean().item()
            fold_mses.append(mse)
        avg = float(np.mean(fold_mses))
        if avg > best_neg_mse:
            best_neg_mse = avg
            best_alpha = alpha

    # Refit on full train+val
    xm = X_tv.mean(0); ym = y_tv.mean()
    w = _ridge_closed_form(X_tv - xm, y_tv - ym, best_alpha)
    y_hat = ((X_te_s - xm) @ w + ym).cpu().numpy().astype(np.float32)
    y_te_f = y_te.astype(np.float32)

    mae = float(mean_absolute_error(y_te_f, y_hat))
    rmse = float(np.sqrt(np.mean((y_te_f - y_hat) ** 2)))
    r2 = float(r2_score(y_te_f, y_hat))

    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "best_params": {"alpha": best_alpha},
        "best_cv_mse": float(-best_neg_mse),
    }


def _logreg_lbfgs(X: torch.Tensor, y: torch.Tensor, C: float,
                  n_cls: int, max_iter: int = 500,
                  tol: float = 1e-4) -> tuple:
    """Multinomial logistic regression with L-BFGS, sklearn-equivalent.

    Loss = mean_i CE(softmax(X w_i + b), y_i) * N + 0.5 * (1/C) * ||W||^2
    (sklearn's LogisticRegression uses sum-CE not mean-CE, with same penalty).
    fit_intercept=True; intercept is NOT regularized.
    """
    dev = X.device
    N, D = X.shape
    W = torch.zeros(D, n_cls, device=dev, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_cls, device=dev, dtype=torch.float64, requires_grad=True)
    lam = 1.0 / C

    opt = torch.optim.LBFGS(
        [W, b], lr=1.0, max_iter=max_iter, tolerance_grad=tol,
        tolerance_change=tol * 1e-2, history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = X @ W + b
        # sum-CE to match sklearn's loss formulation
        ce = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
        loss = ce + 0.5 * lam * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def _cls_probe_gpu(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_va: np.ndarray, y_va: np.ndarray,
                   X_te: np.ndarray, y_te: np.ndarray) -> Dict[str, Any]:
    """Logistic-regression probe on GPU — sklearn-equivalent (LBFGS).

    Mirrors the CPU path: same StandardScaler, same C grid [0.1,1,10],
    same 3-fold KFold(shuffle=True, random_state=42), same f1_macro scoring,
    same LBFGS solver with multinomial CE + L2(W) + free intercept.
    """
    dev = _GPU_PROBE_DEVICE
    X_tv_np = np.concatenate([X_tr, X_va])
    y_tv_np = np.concatenate([y_tr, y_va]).astype(np.int64)

    X_tv64 = torch.from_numpy(X_tv_np).double().to(dev)
    X_te64 = torch.from_numpy(X_te.copy()).double().to(dev)
    mu = X_tv64.mean(0)
    std = ((X_tv64 - mu) ** 2).mean(0).sqrt().clamp(min=1e-12)
    std = torch.where(std < 1e-10, torch.ones_like(std), std)
    X_tv = (X_tv64 - mu) / std
    X_te_s = (X_te64 - mu) / std

    y_tv = torch.from_numpy(y_tv_np).long().to(dev)
    n_cls = int(y_tv.max().item()) + 1

    Cs = [0.1, 1.0, 10.0]
    n = X_tv.shape[0]
    folds = _kfold_indices(n, k=3, seed=42, device=dev)

    def _f1_macro(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        f1s = []
        for c in range(n_cls):
            tp = ((y_pred == c) & (y_true == c)).sum().double()
            fp = ((y_pred == c) & (y_true != c)).sum().double()
            fn = ((y_pred != c) & (y_true == c)).sum().double()
            if tp + fp + fn == 0:
                continue
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1s.append(2 * prec * rec / (prec + rec + 1e-12))
        return float(torch.stack(f1s).mean().item()) if f1s else 0.0

    best_C, best_score = Cs[0], -1.0
    for C in Cs:
        scores = []
        for fi in range(3):
            va_idx = folds[fi]
            tr_idx = torch.cat([folds[j] for j in range(3) if j != fi])
            W, b = _logreg_lbfgs(X_tv[tr_idx], y_tv[tr_idx], C, n_cls)
            with torch.no_grad():
                pred = (X_tv[va_idx] @ W + b).argmax(dim=1)
            scores.append(_f1_macro(y_tv[va_idx], pred))
        avg = float(np.mean(scores))
        if avg > best_score:
            best_score = avg
            best_C = C

    # Final fit on full train+val
    W, b = _logreg_lbfgs(X_tv, y_tv, best_C, n_cls)

    with torch.no_grad():
        logits_te = X_te_s @ W + b
        probs_t = torch.softmax(logits_te, dim=1)
        y_pred_t = logits_te.argmax(dim=1)

    y_pred = y_pred_t.cpu().numpy()
    probs = probs_t.cpu().numpy()

    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_te, y_pred, average="macro", zero_division=0)

    if n_cls == 2:
        auprc = average_precision_score(y_te, probs[:, 1])
        auroc = roc_auc_score(y_te, probs[:, 1])
    else:
        y_ohe = label_binarize(y_te, classes=np.arange(n_cls))
        valid = y_ohe.sum(axis=0) > 0
        auprc = (average_precision_score(y_ohe[:, valid], probs[:, valid], average="macro")
                 if valid.any() else float("nan"))
        try:
            auroc = roc_auc_score(y_te, probs, average="macro", multi_class="ovo")
        except ValueError:
            auroc = float("nan")

    return {
        "f1_macro": f1, "acc": acc, "precision": prec, "recall": rec,
        "auprc": auprc, "auroc": auroc,
        "best_params": {"C": best_C}, "best_cv_f1": float(best_score),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Probe dispatchers (sklearn or GPU based on _GPU_PROBE_DEVICE)
# ──────────────────────────────────────────────────────────────────────────────

def _cls_probe(X_tr: np.ndarray, y_tr: np.ndarray,
               X_va: np.ndarray, y_va: np.ndarray,
               X_te: np.ndarray, y_te: np.ndarray) -> Dict[str, Any]:
    """Logistic-regression probe — dispatches to GPU or sklearn."""
    if _GPU_PROBE_DEVICE is not None:
        return _cls_probe_gpu(X_tr, y_tr, X_va, y_va, X_te, y_te)
    X_tv = np.concatenate([X_tr, X_va])
    y_tv = np.concatenate([y_tr, y_va])
    scaler = StandardScaler().fit(X_tv)
    X_tv_s = scaler.transform(X_tv)

    grid = GridSearchCV(
        LogisticRegression(random_state=42, solver="lbfgs", max_iter=500),
        {"penalty": ["l2"], "C": [0.1, 1, 10]},
        cv=3, scoring="f1_macro", n_jobs=1, verbose=0,
    )
    grid.fit(X_tv_s, y_tv)

    X_te_s = scaler.transform(X_te)
    y_pred = grid.predict(X_te_s)
    probs = grid.predict_proba(X_te_s)

    n_cls = probs.shape[1]
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_te, y_pred, average="macro", zero_division=0)

    if n_cls == 2:
        auprc = average_precision_score(y_te, probs[:, 1])
        auroc = roc_auc_score(y_te, probs[:, 1])
    else:
        y_ohe = label_binarize(y_te, classes=np.arange(n_cls))
        valid = y_ohe.sum(axis=0) > 0
        auprc = (average_precision_score(y_ohe[:, valid], probs[:, valid], average="macro")
                 if valid.any() else float("nan"))
        try:
            auroc = roc_auc_score(y_te, probs, average="macro", multi_class="ovo")
        except ValueError:
            auroc = float("nan")

    return {
        "f1_macro": f1, "acc": acc, "precision": prec, "recall": rec,
        "auprc": auprc, "auroc": auroc,
        "best_params": grid.best_params_, "best_cv_f1": grid.best_score_,
    }


def _reg_probe(X_tr: np.ndarray, y_tr: np.ndarray,
               X_va: np.ndarray, y_va: np.ndarray,
               X_te: np.ndarray, y_te: np.ndarray) -> Dict[str, Any]:
    """Ridge regression probe — dispatches to GPU or sklearn."""
    if _GPU_PROBE_DEVICE is not None:
        return _reg_probe_gpu(X_tr, y_tr, X_va, y_va, X_te, y_te)
    X_tv = np.concatenate([X_tr, X_va])
    y_tv = np.concatenate([y_tr.astype(np.float32), y_va.astype(np.float32)])
    scaler = StandardScaler().fit(X_tv)
    X_tv_s = scaler.transform(X_tv)

    alphas = [0.1, 1.0, 10.0, 100.0]
    grid = GridSearchCV(
        Ridge(), {"alpha": alphas},
        cv=3, scoring="neg_mean_squared_error", n_jobs=1, verbose=0,
    )
    grid.fit(X_tv_s, y_tv)

    X_te_s = scaler.transform(X_te)
    y_hat = grid.predict(X_te_s).astype(np.float32)
    y_te_f = y_te.astype(np.float32)

    mae = float(mean_absolute_error(y_te_f, y_hat))
    rmse = float(np.sqrt(np.mean((y_te_f - y_hat) ** 2)))
    r2 = float(r2_score(y_te_f, y_hat))

    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "best_params": grid.best_params_,
        "best_cv_mse": float(-grid.best_score_),
    }


def _add_primary(result: Dict, primary_metric: str) -> Dict:
    result["primary"] = result[primary_metric]
    result["primary_metric"] = primary_metric
    return result


# ──────────────────────────────────────────────────────────────────────────────
# File-loading utilities
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_file(root: str, template: str, split: str, hz: int) -> str:
    name = template.format(split=split, hz=hz)
    return os.path.join(root, name)


def _load_y(path: str, transform: Optional[str]) -> np.ndarray:
    y = np.asarray(np.load(path))
    if transform == "binarize":
        y = (y > 0).astype(np.int64)
    elif transform == "to_int":
        y = y.astype(np.int64)
    return y


def _presplit_arrays(root: str, cfg: Dict, hz: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val/test X and y arrays for a pre-split task."""
    results = {}
    for split in ("train", "val", "test"):
        xp = _resolve_file(root, cfg["x_file"], split, hz)
        yp = _resolve_file(root, cfg["y_file"], split, hz)
        if not os.path.exists(xp):
            raise FileNotFoundError(f"Missing X file: {xp}")
        if not os.path.exists(yp):
            raise FileNotFoundError(f"Missing y file: {yp}")
        X = np.load(xp, mmap_mode="r")
        y = _load_y(yp, cfg.get("label_transform"))
        results[split] = (X, y)
    return (*results["train"], *results["val"], *results["test"])


def _kfold_arrays(root: str, cfg: Dict, hz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load full X, y and subject arrays for a k-fold task."""
    subdir = cfg.get("subdir", "")
    data_root = os.path.join(root, subdir) if subdir else root

    xp = os.path.join(data_root, cfg["x_file"].format(hz=hz))
    yp = os.path.join(data_root, cfg["y_file"].format(hz=hz))
    sp = os.path.join(data_root, cfg["subj_file"].format(hz=hz))

    for p in (xp, yp, sp):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file for k-fold task: {p}")

    X = np.load(xp, mmap_mode="r")
    y = np.load(yp, mmap_mode="r")
    subjects = np.load(sp, allow_pickle=True)
    subjects = np.asarray(subjects, dtype=str)
    return X, y, subjects


# ──────────────────────────────────────────────────────────────────────────────
# K-fold evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _subject_kfold_splits(subjects: np.ndarray, n_folds: int, seed: int
                          ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Yield index masks (tr_idx, va_idx, te_idx) for n_folds rounds.
    Each round: one subject group → test, next group → val, rest → train.
    """
    unique = np.array(sorted(set(subjects.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    # split subjects into n_folds groups (last group may be slightly smaller)
    fold_subjs = np.array_split(unique, n_folds)

    splits = []
    for i in range(n_folds):
        te_subjs = set(fold_subjs[i].tolist())
        va_subjs = set(fold_subjs[(i + 1) % n_folds].tolist())
        tr_subjs = set(unique.tolist()) - te_subjs - va_subjs

        tr_idx = np.where([s in tr_subjs for s in subjects])[0]
        va_idx = np.where([s in va_subjs for s in subjects])[0]
        te_idx = np.where([s in te_subjs for s in subjects])[0]
        splits.append((tr_idx, va_idx, te_idx))
    return splits


def _average_fold_results(fold_results: List[Dict]) -> Dict:
    """Average numeric metrics across folds; keep best_params from last fold."""
    avg: Dict[str, Any] = {}
    numeric_keys = [k for k in fold_results[0] if isinstance(fold_results[0][k], float)]
    for k in numeric_keys:
        vals = [r[k] for r in fold_results if not np.isnan(r[k])]
        avg[k] = float(np.mean(vals)) if vals else float("nan")
    avg["best_params"] = fold_results[-1].get("best_params", {})
    avg["fold_results"] = fold_results
    return avg


def _run_kfold(task_type: str, X_emb: np.ndarray, y: np.ndarray,
               subjects: np.ndarray, n_folds: int, seed: int,
               primary_metric: str) -> Dict:
    splits = _subject_kfold_splits(subjects, n_folds, seed)
    probe_fn = _cls_probe if task_type == "cls" else _reg_probe

    fold_results = []
    for tr_idx, va_idx, te_idx in splits:
        if len(te_idx) == 0 or len(tr_idx) == 0:
            continue
        res = probe_fn(X_emb[tr_idx], y[tr_idx],
                       X_emb[va_idx], y[va_idx],
                       X_emb[te_idx], y[te_idx])
        fold_results.append(res)

    if not fold_results:
        raise RuntimeError("All folds were empty — check subject arrays.")

    result = _average_fold_results(fold_results)
    return _add_primary(result, primary_metric)


# ──────────────────────────────────────────────────────────────────────────────
# Per-task runners
# ──────────────────────────────────────────────────────────────────────────────

def _subsample_split(X, y, max_n, rng):
    """Stratified-ish subsample to max_n if needed."""
    if max_n is None or len(X) <= max_n:
        return X, y
    idx = np.sort(rng.choice(len(X), max_n, replace=False))
    return X[idx], y[idx]


def _run_presplit(task: str, cfg: Dict, encoder: nn.Module, device: str,
                  root: str, hz: int, batch_size: int,
                  max_samples: Optional[int] = None,
                  _emb_cache: Optional[Dict] = None) -> Dict:
    X_tr, y_tr, X_va, y_va, X_te, y_te = _presplit_arrays(root, cfg, hz)

    # Cap per-split sizes proportionally (e.g. max_samples=50000 for sdb)
    if max_samples is not None:
        total = len(X_tr) + len(X_va) + len(X_te)
        if total > max_samples:
            ratio = max_samples / total
            rng = np.random.default_rng(42)
            X_tr, y_tr = _subsample_split(X_tr, y_tr, int(len(X_tr) * ratio), rng)
            X_va, y_va = _subsample_split(X_va, y_va, int(len(X_va) * ratio), rng)
            X_te, y_te = _subsample_split(X_te, y_te, int(len(X_te) * ratio), rng)
            print(f"  [{task}] Capped to {len(X_tr)+len(X_va)+len(X_te)} "
                  f"(from {total}, max_samples={max_samples})", flush=True)

    # Check embedding cache (avoid re-extracting for tasks sharing same X files)
    cache_key = (root, cfg["x_file"], hz)
    if _emb_cache is not None and cache_key in _emb_cache:
        cached = _emb_cache[cache_key]
        # Cached embeddings were from original arrays; if we subsampled we need
        # to re-extract (sizes won't match).  Only reuse if sizes match.
        if (len(cached["train"]) == len(X_tr) and
            len(cached["val"]) == len(X_va) and
            len(cached["test"]) == len(X_te)):
            print(f"  [{task}] Reusing cached embeddings", flush=True)
            X_tr_e, X_va_e, X_te_e = cached["train"], cached["val"], cached["test"]
        else:
            X_tr_e = _extract_all(encoder, X_tr, device, batch_size=batch_size)
            X_va_e = _extract_all(encoder, X_va, device, batch_size=batch_size)
            X_te_e = _extract_all(encoder, X_te, device, batch_size=batch_size)
    else:
        X_tr_e = _extract_all(encoder, X_tr, device, batch_size=batch_size)
        X_va_e = _extract_all(encoder, X_va, device, batch_size=batch_size)
        X_te_e = _extract_all(encoder, X_te, device, batch_size=batch_size)
        if _emb_cache is not None:
            _emb_cache[cache_key] = {
                "train": X_tr_e, "val": X_va_e, "test": X_te_e}

    probe_fn = _cls_probe if cfg["type"] == "cls" else _reg_probe
    result = probe_fn(X_tr_e, y_tr, X_va_e, y_va, X_te_e, y_te)
    return _add_primary(result, cfg["primary_metric"])


def _run_kfold_task(task: str, cfg: Dict, encoder: nn.Module, device: str,
                    root: str, hz: int, batch_size: int,
                    n_folds: int, seed: int,
                    max_samples: Optional[int] = None,
                    _emb_cache: Optional[Dict] = None) -> Dict:
    X_all, y_all, subjects = _kfold_arrays(root, cfg, hz)
    if cfg.get("label_transform") == "binarize":
        y_all = (y_all > 0).astype(np.int64)
    elif cfg.get("label_transform") == "to_int":
        y_all = y_all.astype(np.int64)

    # Cap total samples if needed
    if max_samples is not None and len(X_all) > max_samples:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(X_all), max_samples, replace=False))
        print(f"  [{task}] Capped to {max_samples} (from {len(X_all)})", flush=True)
        X_all, y_all, subjects = X_all[idx], y_all[idx], subjects[idx]

    # Check embedding cache
    subdir = cfg.get("subdir", "")
    cache_key = (root, subdir, cfg["x_file"], hz, len(X_all))
    if _emb_cache is not None and cache_key in _emb_cache:
        print(f"  [{task}] Reusing cached embeddings ({len(X_all)} samples)", flush=True)
        X_emb = _emb_cache[cache_key]
    else:
        print(f"  [{task}] Extracting embeddings for {len(X_all)} samples …", flush=True)
        X_emb = _extract_all(encoder, X_all, device, batch_size=batch_size)
        if _emb_cache is not None:
            _emb_cache[cache_key] = X_emb

    return _run_kfold(cfg["type"], X_emb, y_all, subjects, n_folds, seed,
                      cfg["primary_metric"])


def _run_sportppg(cfg: Dict, encoder: nn.Module, device: str,
                  sportppg_dir: Dict, hz: int, batch_size: int,
                  n_folds: int, seed: int) -> Dict:
    """SportPPG activity classification via subject-wise k-fold."""
    from sportppg import SportPPGConfig, SportPPGWindowDataset

    data_root = Path(sportppg_dir["data_root"])
    gt_root   = Path(sportppg_dir["gt_root"])
    subjects  = sportppg_dir.get("subjects",
                    [f"S{sid:02d}" for sid in range(1, 31)])
    acts      = sportppg_dir.get("acts",
                    ["stationary", "walking", "running",
                     "badminton", "tabletennis", "basketball"])
    hand      = sportppg_dir.get("hand", "L")

    sp_cfg = SportPPGConfig(
        data_root=data_root, gt_root=gt_root,
        subjects=subjects, acts=acts, hand=hand,
        use_acc=False, orig_fs=64, target_fs=hz,
        win_sec=10.0, hop_sec=2.0,
    )
    ds = SportPPGWindowDataset(sp_cfg, task="activity")

    # Pull all (x, label, subject) into arrays
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    feats, labs, subjs = [], [], []
    encoder.eval()
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device, non_blocking=True)
            z = _forward_to_embedding(encoder(x))
            feats.append(z.cpu().float().numpy())
            labs.append(y.cpu().numpy() if torch.is_tensor(y) else np.array(y))
            # meta["subject"] may be a list of strings
            if isinstance(meta, dict) and "subject" in meta:
                subjs.extend(meta["subject"])
            else:
                subjs.extend(["unk"] * len(y))

    X_emb = np.concatenate(feats, axis=0)
    y_all = np.concatenate(labs, axis=0).astype(np.int64)
    subjects_arr = np.array(subjs, dtype=str)

    return _run_kfold("cls", X_emb, y_all, subjects_arr, n_folds, seed,
                      cfg["primary_metric"])


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_task(
    task: str,
    encoder: nn.Module,
    device: str,
    data_dirs: Dict[str, Any],
    hz: int = 50,
    batch_size: int = 256,
    n_folds: int = 5,
    seed: int = 42,
    max_samples: Optional[int] = None,
    _emb_cache: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run a single downstream linear-probe task.

    Parameters
    ----------
    task        : task name from TASK_REGISTRY (e.g. "sdb_binary")
    encoder     : frozen backbone; called as encoder(x) → embedding
    device      : "cuda" | "cpu" | "cuda:N"
    data_dirs   : dict mapping dataset key → local path string
                  Required keys depend on the task.  Example::

                      {
                          "wesad":  "pulseppg/data/datasets/wesad",
                          "sdb":    "pulseppg/data/datasets/sdb",
                          "wfppg":  "pulseppg/data/datasets/wfppg",
                          "dalia":  "pulseppg/data/datasets/dalia",
                          "ppgbp":  "pulseppg/data/datasets/ppgbp",
                          "sportppg": {"data_root": "...", "gt_root": "..."},
                      }

    hz          : target sampling rate (must match preprocessed .npy files)
    batch_size  : embedding extraction batch size
    n_folds     : number of folds for subject-wise k-fold tasks
    seed        : random seed for subject shuffling

    Returns
    -------
    dict with keys: primary, primary_metric, and task-specific metrics
    """
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task}'. Available: {list(TASK_REGISTRY)}")

    cfg = TASK_REGISTRY[task]
    dataset_key = cfg["dataset"]
    split_mode = cfg["split_mode"]

    encoder = encoder.to(device).eval()

    if split_mode == "presplit":
        if dataset_key not in data_dirs:
            raise KeyError(f"Task '{task}' needs data_dirs['{dataset_key}']")
        return _run_presplit(task, cfg, encoder, device,
                             str(data_dirs[dataset_key]), hz, batch_size,
                             max_samples=max_samples, _emb_cache=_emb_cache)

    elif split_mode == "kfold":
        if dataset_key not in data_dirs:
            raise KeyError(f"Task '{task}' needs data_dirs['{dataset_key}']")
        return _run_kfold_task(task, cfg, encoder, device,
                               str(data_dirs[dataset_key]), hz, batch_size,
                               n_folds, seed,
                               max_samples=max_samples, _emb_cache=_emb_cache)

    elif split_mode == "sportppg":
        if "sportppg" not in data_dirs:
            raise KeyError("Task 'sportppg_activity' needs data_dirs['sportppg'] = "
                           "{'data_root': ..., 'gt_root': ...}")
        sportppg_dir = data_dirs["sportppg"]
        if isinstance(sportppg_dir, str):
            raise ValueError("data_dirs['sportppg'] must be a dict with 'data_root' and 'gt_root'.")
        return _run_sportppg(cfg, encoder, device, sportppg_dir,
                             hz, batch_size, n_folds, seed)

    else:
        raise NotImplementedError(f"split_mode='{split_mode}' is not yet implemented.")


def run_all_tasks(
    encoder: nn.Module,
    device: str,
    data_dirs: Dict[str, Any],
    tasks: Optional[Sequence[str]] = None,
    hz: int = 50,
    batch_size: int = 256,
    n_folds: int = 5,
    seed: int = 42,
    skip_missing: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run a battery of downstream linear-probe tasks.

    Parameters
    ----------
    encoder     : frozen backbone
    device      : "cuda" | "cpu" | "cuda:N"
    data_dirs   : dict mapping dataset key → local path (see run_task)
    tasks       : list of task names to run; None = all registered tasks
    hz          : sampling rate (must match preprocessed .npy files)
    batch_size  : embedding extraction batch size
    n_folds     : folds for subject-wise k-fold tasks
    seed        : random seed for fold shuffles
    skip_missing: if True, skip tasks whose data_dirs key is absent
                  (logs a warning instead of raising)

    Returns
    -------
    dict  {task_name: metric_dict}
    """
    task_list = list(tasks) if tasks is not None else list(TASK_REGISTRY)
    results: Dict[str, Dict[str, Any]] = {}
    _emb_cache: Dict = {}   # shared embedding cache across tasks

    if max_samples is not None:
        print(f"[downstream_eval] max_samples={max_samples} per task", flush=True)

    for task in task_list:
        cfg = TASK_REGISTRY.get(task)
        if cfg is None:
            warnings.warn(f"Unknown task '{task}' — skipping.")
            continue

        dataset_key = cfg["dataset"]
        if dataset_key not in data_dirs:
            if skip_missing:
                print(f"  [skip] '{task}': data_dirs['{dataset_key}'] not provided.")
                continue
            else:
                raise KeyError(f"Task '{task}' needs data_dirs['{dataset_key}']")

        print(f"\n── {task} ──────────────────────────────", flush=True)
        try:
            m = run_task(task, encoder, device, data_dirs,
                         hz=hz, batch_size=batch_size, n_folds=n_folds, seed=seed,
                         max_samples=max_samples, _emb_cache=_emb_cache)
            pm = m["primary_metric"]
            print(f"  {pm} = {m['primary']:.4f}", flush=True)
            results[task] = m
        except FileNotFoundError as exc:
            if skip_missing:
                warnings.warn(f"Skipping '{task}': {exc}")
            else:
                raise
        except Exception as exc:
            warnings.warn(f"Error in task '{task}': {exc}")
            if not skip_missing:
                raise

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: pretty-print results table
# ──────────────────────────────────────────────────────────────────────────────

def print_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a compact text table of run_all_tasks() output."""
    print(f"\n{'Task':<28}  {'Metric':<12}  {'Value':>8}  {'Secondary':>10}")
    print("─" * 65)
    for task, m in results.items():
        pm = m["primary_metric"]
        pv = m["primary"]
        # pick one secondary
        if pm == "f1_macro":
            sec_k, sec_v = "auroc", m.get("auroc", float("nan"))
        else:
            sec_k, sec_v = "r2", m.get("r2", float("nan"))
        print(f"{task:<28}  {pm:<12}  {pv:>8.4f}  {sec_k}={sec_v:.4f}")

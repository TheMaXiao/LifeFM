#!/usr/bin/env python3
"""Stage-2 same-domain multi-task adaptation for compressed PulsePPG students.

This script implements the deployment setting described in
research_plan_compressed_fm_deployment.md:

    one same-domain dataset -> multiple tasks -> one compressed MCU backbone

Default experiment:

    Dataset: PPGBP
    Tasks:   HR, systolic BP, diastolic BP, hypertension
    Backbones: BF20_NB8 and BF48_NB8 NS-KD encoders

Methods:

    frozen_heads
        Freeze the KD encoder and train one small head per task.

    independent_finetune
        Fine-tune a separate encoder+head per task. This is a strong
        non-deployable upper bound because deployment would need one encoder
        copy per task.

    joint_finetune
        Train one shared encoder with one head per task. The encoder is updated
        directly with a small LR multiplier; no adapters are used.

    sa_mtft
        Same lightweight direct joint fine-tuning, plus a Stage-1 subspace
        anchor and null-space suppression through the frozen KD projection head.

Outputs:

    kd_sweep_results/stage2_multitask_ppg/<dataset>/stage2_results.csv
    kd_sweep_results/stage2_multitask_ppg/<dataset>/stage2_summary.csv

Example:

    CUDA_VISIBLE_DEVICES=0 /home/xiaoma/anaconda3/envs/pulseppg/bin/python \
      run_stage2_multitask_ppg.py --students BF20_NB8 --seeds 0 --epochs 40
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

from run_lifefm_kd import PP_STUDENT_CONFIGS  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "kd_sweep_results/stage2_multitask_ppg"
DEFAULT_SVD_CACHE = ROOT / "kd_sweep_results/pulseppg_preliminary/exp_ns_kd_adaptive_k40"
SPORTPPG_DATA_ROOT = Path("/home/xiaoma/datasets/ppg/PPG-Sport-F5B0/dataset")
SPORTPPG_GT_ROOT = Path("/home/xiaoma/datasets/ppg/PPG-Sport-F5B0/cleaned_gt")
SPORTPPG_CACHE_ROOT = ROOT / "kd_sweep_results/stage2_sportppg_cache"
TEMPORAL_POLICIES = (
    "shared_window",
    "task_specific_crop",
    "shared_multi_scale",
    "task_routed_multiscale",
)
ENCODER_UPDATE_SCOPES = ("none", "full", "top1", "top2")
TEMPORAL_DEFAULT_SUPPORTS_SEC: Dict[str, Dict[str, float]] = {
    "bidmc": {
        "hr": 6.0,
        "spo2": 6.0,
        "rr": 10.0,
    },
    "ppgbp": {
        "hr": 8.0,
        "sysbp": 6.0,
        "diasbp": 6.0,
        "hypertension": 10.0,
    },
    "dalia": {
        "hr": 8.0,
        "activity": 2.0,
    },
    "sportppg": {
        "hr": 10.0,
        "activity": 2.5,
    },
    "wesad": {
        "binary": 10.0,
        "multiclass": 10.0,
    },
}
TEMPORAL_DEFAULT_SCALES_SEC: Dict[str, List[float]] = {
    "bidmc": [4.0, 8.0, 10.0],
    "ppgbp": [4.0, 8.0, 10.0],
    "dalia": [2.0, 4.0, 8.0],
    "sportppg": [2.5, 5.0, 10.0],
    "wesad": [5.0, 10.0],
    "wesad_cross_interface": [10.0, 60.0],
}
DEFAULT_CKPT_DIRS = {
    "BF20_NB8": ROOT / "kd_sweep_results/pulseppg_preliminary/bf20_ckpts",
    "BF48_NB8": ROOT / "kd_sweep_results/pulseppg_preliminary/bf48_ckpts",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str  # "reg" or "cls"
    y_template: str
    metric: str
    label_transform: Optional[str] = None


@dataclass
class TaskRuntime:
    spec: TaskSpec
    n_out: int
    y_mean: float = 0.0
    y_std: float = 1.0


@dataclass
class TaskData:
    runtime: TaskRuntime
    split_X: Dict[str, np.ndarray]
    split_y: Dict[str, np.ndarray]


DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "bidmc": {
        "mode": "subject_split_aligned",
        "root": ROOT / "pulseppg/data/datasets/bidmc",
        "x_template": "X_ppg_{hz}Hz.npy",
        "subject_template": "subject_{hz}Hz.npy",
        "tasks": [
            TaskSpec("hr", "reg", "y_hr_{hz}Hz.npy", "mae"),
            TaskSpec("spo2", "reg", "y_spo2_{hz}Hz.npy", "mae"),
            TaskSpec("rr", "reg", "y_rr_{hz}Hz.npy", "mae"),
        ],
    },
    "wfppg": {
        "mode": "subject_split_taskwise",
        "root": ROOT / "pulseppg/data/datasets/wfppg",
        "x_template": "{subdir}/X_ppg_{hz}Hz.npy",
        "subject_template": "{subdir}/subject_{hz}Hz.npy",
        "tasks": [
            TaskSpec("wrist", "reg", "{subdir}/y_hr_{hz}Hz.npy", "mae"),
            TaskSpec("finger", "reg", "{subdir}/y_hr_{hz}Hz.npy", "mae"),
        ],
    },
    "sdb": {
        "mode": "presplit_aligned",
        "root": ROOT / "pulseppg/data/datasets/sdb",
        "x_template": "{split}_X_ppg_{hz}Hz.npy",
        "tasks": [
            TaskSpec("binary", "cls", "{split}_y_sdb.npy", "f1_macro", "binarize"),
            TaskSpec("multiclass", "cls", "{split}_y_sdb.npy", "f1_macro"),
        ],
    },
    "ppgbp": {
        "mode": "presplit_aligned",
        "root": ROOT / "pulseppg/data/datasets/ppgbp",
        "x_template": "{split}_X_ppg_{hz}Hz.npy",
        "tasks": [
            TaskSpec("hr", "reg", "{split}_y_hr_{hz}Hz.npy", "mae"),
            TaskSpec("sysbp", "reg", "{split}_y_sysbp_{hz}Hz.npy", "mae"),
            TaskSpec("diasbp", "reg", "{split}_y_diasbp_{hz}Hz.npy", "mae"),
            TaskSpec("hypertension", "cls", "{split}_y_ht_{hz}Hz.npy", "f1_macro", "binarize"),
        ],
    },
    "dalia": {
        "mode": "presplit_aligned",
        "root": ROOT / "pulseppg/data/datasets/dalia",
        "x_template": "{split}_X_ppg_{hz}Hz.npy",
        "tasks": [
            TaskSpec("hr", "reg", "{split}_y_hr.npy", "mae"),
            TaskSpec("activity", "cls", "{split}_y_act.npy", "f1_macro", "to_int"),
        ],
    },
    "wesad": {
        "mode": "subject_split_taskwise",
        "root": ROOT / "pulseppg/data/datasets/wesad",
        "x_template": "{subdir}/X_ppg_{hz}Hz.npy",
        "subject_template": "{subdir}/subject_{hz}Hz.npy",
        "tasks": [
            TaskSpec("binary", "cls", "{subdir}/y_stress_{hz}Hz.npy", "f1_macro"),
            TaskSpec("multiclass", "cls", "{subdir}/y_stress_{hz}Hz.npy", "f1_macro"),
        ],
    },
    "wesad_cross_interface": {
        "mode": "subject_split_taskwise",
        "root": ROOT / "pulseppg/data/datasets/wesad_cross_interface",
        "x_template": "{subdir}/X_ppg_{hz}Hz.npy",
        "subject_template": "{subdir}/subject_{hz}Hz.npy",
        "tasks": [
            TaskSpec("hr_regression", "reg", "{subdir}/y_hr_{hz}Hz.npy", "mae"),
            TaskSpec("rr_regression", "reg", "{subdir}/y_rr_{hz}Hz.npy", "mae"),
            TaskSpec("stress_binary", "cls", "{subdir}/y_stress_binary_{hz}Hz.npy", "f1_macro"),
            TaskSpec("affect_multiclass", "cls", "{subdir}/y_affect_multiclass_{hz}Hz.npy", "f1_macro"),
        ],
    },
    "wesad_cross_interface_shifted30": {
        "mode": "subject_split_taskwise",
        "root": ROOT / "pulseppg/data/datasets/wesad_cross_interface_shifted30",
        "x_template": "{subdir}/X_ppg_{hz}Hz.npy",
        "subject_template": "{subdir}/subject_{hz}Hz.npy",
        "tasks": [
            TaskSpec("hr_regression", "reg", "{subdir}/y_hr_{hz}Hz.npy", "mae"),
            TaskSpec("rr_regression", "reg", "{subdir}/y_rr_{hz}Hz.npy", "mae"),
            TaskSpec("stress_binary", "cls", "{subdir}/y_stress_binary_{hz}Hz.npy", "f1_macro"),
            TaskSpec("affect_multiclass", "cls", "{subdir}/y_affect_multiclass_{hz}Hz.npy", "f1_macro"),
        ],
    },
    "sportppg": {
        "mode": "sportppg_subject_split",
        "data_root": SPORTPPG_DATA_ROOT,
        "gt_root": SPORTPPG_GT_ROOT,
        "subjects": [f"S{sid:02d}" for sid in range(1, 31)],
        "acts": ["stationary", "walking", "running", "badminton", "tabletennis", "basketball"],
        "hand": "L",
        "tasks": [
            TaskSpec("hr", "reg", "", "mae"),
            TaskSpec("activity", "cls", "", "f1_macro"),
        ],
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def build_encoder(student: str, seed: int) -> Tuple[nn.Module, int]:
    """Build the same ResNet1D encoder family used by PulsePPG KD."""
    from pulseppg.nets.ResNet1D.ResNet1D_Net import Net as ResNet1DNet

    if student not in PP_STUDENT_CONFIGS:
        raise KeyError(f"Unknown student {student}; choices={sorted(PP_STUDENT_CONFIGS)}")
    cfg = PP_STUDENT_CONFIGS[student]
    set_seed(seed)
    enc = ResNet1DNet(
        in_channels=1,
        base_filters=cfg["base_filters"],
        kernel_size=11,
        stride=2,
        groups=1,
        n_block=cfg["n_block"],
        finalpool="max",
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        verbose=False,
    )
    with torch.inference_mode():
        ed = enc(torch.randn(2, 1, 500)).shape[-1]
    return enc, int(ed)


def disable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d)):
            m.p = 0.0


def load_stage1_full(
    student: str,
    ckpt_dir: Path,
    method: str,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, nn.Linear, Dict[str, Any]]:
    """Load a KD checkpoint and return (encoder, frozen_projection, metadata)."""
    ckpt_path = ckpt_dir / f"{method}_seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing Stage-I checkpoint: {ckpt_path}\n"
            f"Run `bash scripts/setup_checkpoints.sh` to install the released checkpoints, "
            f"or pass --student_encoder_ckpt with a checkpoint produced by run_lifefm_kd.py."
        )

    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    d_full = int(obj.get("d_full", 512))
    enc, enc_dim = build_encoder(student, seed)
    proj = nn.Linear(enc_dim, d_full, bias=False)
    full = nn.Sequential(enc, proj)
    full.load_state_dict(obj["state_dict"], strict=True)
    enc = full[0].to(device)
    proj = full[1].to(device)
    proj.requires_grad_(False)
    proj.eval()
    disable_dropout(enc)
    meta = dict(obj)
    meta.pop("state_dict", None)
    meta["ckpt_path"] = str(ckpt_path)
    meta["enc_dim"] = enc_dim
    return enc, proj, meta


def topk_basis_from_cache(cache_dir: Path, k: int) -> torch.Tensor:
    z_path = cache_dir / "Z_mix.npy"
    if not z_path.exists():
        raise FileNotFoundError(f"Missing SVD cache: {z_path}")
    Z = np.load(str(z_path))
    Zc = Z.astype(np.float32) - Z.astype(np.float32).mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return torch.from_numpy(Vt[:k].astype(np.float32))


def load_y(path: Path, transform: Optional[str]) -> np.ndarray:
    y = np.asarray(np.load(str(path)))
    if transform == "binarize":
        y = (y > 0).astype(np.int64)
    elif transform == "to_int":
        y = y.astype(np.int64)
    return y


def maybe_limit_split(
    X: np.ndarray,
    ys: Dict[str, np.ndarray],
    limit: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if limit <= 0 or len(X) <= limit:
        return X, ys
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), limit, replace=False))
    return np.asarray(X[idx]), {k: np.asarray(v[idx]) for k, v in ys.items()}


def split_subjects(subjects: Iterable[str], seed: int) -> Tuple[set[str], set[str], set[str]]:
    unique = np.array(sorted(set(map(str, subjects))))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = max(1, int(round(0.6 * n)))
    n_val = max(1, int(round(0.2 * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train = set(unique[:n_train].tolist())
    val = set(unique[n_train:n_train + n_val].tolist())
    test = set(unique[n_train + n_val:].tolist())
    return train, val, test


def load_or_build_sportppg_arrays(
    hz: int,
    data_root: Path,
    gt_root: Path,
    subjects: List[str],
    acts: List[str],
    hand: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_dir = SPORTPPG_CACHE_ROOT / f"hz{hz}_{hand.lower()}_ppg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path = cache_dir / "X.npy"
    y_hr_path = cache_dir / "y_hr.npy"
    y_act_path = cache_dir / "y_activity.npy"
    subj_path = cache_dir / "subject.npy"

    if all(p.exists() for p in (x_path, y_hr_path, y_act_path, subj_path)):
        X = np.asarray(np.load(str(x_path), mmap_mode="r"), dtype=np.float32)
        y_hr = np.asarray(np.load(str(y_hr_path)), dtype=np.float32)
        y_act = np.asarray(np.load(str(y_act_path)), dtype=np.int64)
        subj = np.asarray(np.load(str(subj_path), allow_pickle=True), dtype=str)
        return X, y_hr, y_act, subj

    from sportppg import SportPPGConfig, SportPPGWindowDataset

    sp_cfg = SportPPGConfig(
        data_root=data_root,
        gt_root=gt_root,
        subjects=subjects,
        acts=acts,
        hand=hand,
        use_acc=False,
        orig_fs=64,
        target_fs=hz,
        win_sec=10.0,
        hop_sec=2.0,
    )
    ds = SportPPGWindowDataset(sp_cfg, task="activity", mmap=True)
    n = len(ds)
    win_len = int(round(sp_cfg.win_sec * sp_cfg.target_fs))

    X = np.empty((n, 1, win_len), dtype=np.float32)
    y_hr = np.empty((n,), dtype=np.float32)
    y_act = np.empty((n,), dtype=np.int64)
    subj = np.empty((n,), dtype=object)

    for idx in range(n):
        x, y, meta = ds[idx]
        X[idx] = x.numpy().astype(np.float32, copy=False)
        y_act[idx] = int(y.item())
        subj[idx] = str(meta["subject"])

        ri, w = ds.master_index[idx]
        rec = ds.records[ri]
        gt = ds._get_cached(rec.gt_path, kind="gt")
        y_hr[idx] = float(ds._hr_label_for_window(gt, w))

    np.save(str(x_path), X)
    np.save(str(y_hr_path), y_hr)
    np.save(str(y_act_path), y_act)
    np.save(str(subj_path), subj)
    return X, y_hr, y_act, np.asarray(subj, dtype=str)


def load_multitask_dataset(
    dataset: str,
    hz: int,
    limit_train: int,
    limit_val: int,
    limit_test: int,
    seed: int,
) -> List[TaskData]:
    if dataset not in DATASET_SPECS:
        raise KeyError(f"Unsupported dataset {dataset}; choices={sorted(DATASET_SPECS)}")

    cfg = DATASET_SPECS[dataset]
    mode = cfg["mode"]
    task_data: List[TaskData] = []

    if mode == "presplit_aligned":
        root = Path(cfg["root"])
        x_template = cfg["x_template"]
        shared_X: Dict[str, np.ndarray] = {}
        split_ys: Dict[str, Dict[str, np.ndarray]] = {}
        for split in ("train", "val", "test"):
            x_path = root / x_template.format(split=split, hz=hz)
            if not x_path.exists():
                raise FileNotFoundError(f"Missing X file: {x_path}")
            X = np.asarray(np.load(str(x_path)), dtype=np.float32)
            if X.ndim == 2:
                X = X[:, None, :]

            ys: Dict[str, np.ndarray] = {}
            for task in cfg["tasks"]:
                y_path = root / task.y_template.format(split=split, hz=hz)
                if not y_path.exists():
                    raise FileNotFoundError(f"Missing y file for {task.name}: {y_path}")
                ys[task.name] = load_y(y_path, task.label_transform)

            limit = {"train": limit_train, "val": limit_val, "test": limit_test}[split]
            X, ys = maybe_limit_split(X, ys, limit, seed + {"train": 0, "val": 1, "test": 2}[split])
            shared_X[split] = X
            split_ys[split] = ys

        for task in cfg["tasks"]:
            if task.kind == "cls":
                y_all = np.concatenate([split_ys[s][task.name] for s in ("train", "val", "test")])
                runtime = TaskRuntime(task, n_out=int(np.max(y_all)) + 1)
            else:
                y_train = split_ys["train"][task.name].astype(np.float32)
                mean = float(y_train.mean())
                std = float(y_train.std())
                if not math.isfinite(std) or std < 1e-6:
                    std = 1.0
                runtime = TaskRuntime(task, n_out=1, y_mean=mean, y_std=std)
            task_data.append(
                TaskData(
                    runtime=runtime,
                    split_X={split: shared_X[split] for split in ("train", "val", "test")},
                    split_y={split: split_ys[split][task.name] for split in ("train", "val", "test")},
                )
            )
        return task_data

    if mode == "subject_split_aligned":
        root = Path(cfg["root"])
        x_path = root / cfg["x_template"].format(hz=hz)
        s_path = root / cfg["subject_template"].format(hz=hz)
        for p in (x_path, s_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing aligned subject-split file: {p}")
        X = np.asarray(np.load(str(x_path)), dtype=np.float32)
        if X.ndim == 2:
            X = X[:, None, :]
        subjects = np.asarray(np.load(str(s_path), allow_pickle=True), dtype=str)
        train_subj, val_subj, test_subj = split_subjects(subjects, seed)
        masks = {
            "train": np.isin(subjects, list(train_subj)),
            "val": np.isin(subjects, list(val_subj)),
            "test": np.isin(subjects, list(test_subj)),
        }

        split_X_shared: Dict[str, np.ndarray] = {}
        split_ys_shared: Dict[str, Dict[str, np.ndarray]] = {}
        for split in ("train", "val", "test"):
            Xi = np.asarray(X[masks[split]], dtype=np.float32)
            ys: Dict[str, np.ndarray] = {}
            for task in cfg["tasks"]:
                y_path = root / task.y_template.format(hz=hz)
                if not y_path.exists():
                    raise FileNotFoundError(f"Missing y file for {task.name}: {y_path}")
                y_all = load_y(y_path, task.label_transform)
                ys[task.name] = np.asarray(y_all[masks[split]])
            limit = {"train": limit_train, "val": limit_val, "test": limit_test}[split]
            Xi, ys = maybe_limit_split(
                Xi,
                ys,
                limit,
                seed + {"train": 0, "val": 1, "test": 2}[split],
            )
            split_X_shared[split] = Xi
            split_ys_shared[split] = ys

        for task in cfg["tasks"]:
            y_train = split_ys_shared["train"][task.name].astype(np.float32)
            mean = float(y_train.mean())
            std = float(y_train.std())
            if not math.isfinite(std) or std < 1e-6:
                std = 1.0
            runtime = TaskRuntime(task, n_out=1, y_mean=mean, y_std=std)
            task_data.append(
                TaskData(
                    runtime=runtime,
                    split_X={split: split_X_shared[split] for split in ("train", "val", "test")},
                    split_y={split: split_ys_shared[split][task.name] for split in ("train", "val", "test")},
                )
            )
        return task_data

    if mode == "subject_split_taskwise":
        root = Path(cfg["root"])
        subject_anchor = cfg["tasks"][0].name
        train_subj, val_subj, test_subj = split_subjects(
            np.load(str(root / cfg["subject_template"].format(subdir=subject_anchor, hz=hz)), allow_pickle=True),
            seed,
        )

        for task in cfg["tasks"]:
            subdir = task.name
            x_path = root / cfg["x_template"].format(subdir=subdir, hz=hz)
            y_path = root / task.y_template.format(subdir=subdir, hz=hz)
            s_path = root / cfg["subject_template"].format(subdir=subdir, hz=hz)
            for p in (x_path, y_path, s_path):
                if not p.exists():
                    raise FileNotFoundError(f"Missing WESAD file: {p}")
            X = np.asarray(np.load(str(x_path)), dtype=np.float32)
            y = load_y(y_path, task.label_transform)
            subjects = np.asarray(np.load(str(s_path), allow_pickle=True), dtype=str)

            masks = {
                "train": np.isin(subjects, list(train_subj)),
                "val": np.isin(subjects, list(val_subj)),
                "test": np.isin(subjects, list(test_subj)),
            }
            split_X: Dict[str, np.ndarray] = {}
            split_y: Dict[str, np.ndarray] = {}
            for split in ("train", "val", "test"):
                Xi = X[masks[split]]
                yi = y[masks[split]]
                limit = {"train": limit_train, "val": limit_val, "test": limit_test}[split]
                Xi, one_y = maybe_limit_split(Xi, {task.name: yi}, limit, seed + {"train": 0, "val": 1, "test": 2}[split])
                split_X[split] = Xi
                split_y[split] = one_y[task.name]

            if task.kind == "cls":
                runtime = TaskRuntime(task, n_out=int(np.max(y)) + 1)
            else:
                y_train = split_y["train"].astype(np.float32)
                mean = float(y_train.mean())
                std = float(y_train.std())
                if not math.isfinite(std) or std < 1e-6:
                    std = 1.0
                runtime = TaskRuntime(task, n_out=1, y_mean=mean, y_std=std)
            task_data.append(TaskData(runtime=runtime, split_X=split_X, split_y=split_y))
        return task_data

    if mode == "sportppg_subject_split":
        X, y_hr, y_act, subjects = load_or_build_sportppg_arrays(
            hz=hz,
            data_root=Path(cfg["data_root"]),
            gt_root=Path(cfg["gt_root"]),
            subjects=list(cfg["subjects"]),
            acts=list(cfg["acts"]),
            hand=str(cfg.get("hand", "L")),
        )
        train_subj, val_subj, test_subj = split_subjects(subjects, seed)
        masks = {
            "train": np.isin(subjects, list(train_subj)),
            "val": np.isin(subjects, list(val_subj)),
            "test": np.isin(subjects, list(test_subj)),
        }

        split_X_shared: Dict[str, np.ndarray] = {}
        split_hr: Dict[str, np.ndarray] = {}
        split_act: Dict[str, np.ndarray] = {}
        for split in ("train", "val", "test"):
            Xi = np.asarray(X[masks[split]], dtype=np.float32)
            ys = {
                "hr": np.asarray(y_hr[masks[split]], dtype=np.float32),
                "activity": np.asarray(y_act[masks[split]], dtype=np.int64),
            }
            limit = {"train": limit_train, "val": limit_val, "test": limit_test}[split]
            Xi, ys = maybe_limit_split(Xi, ys, limit, seed + {"train": 0, "val": 1, "test": 2}[split])
            split_X_shared[split] = Xi
            split_hr[split] = ys["hr"]
            split_act[split] = ys["activity"]

        hr_train = split_hr["train"].astype(np.float32)
        hr_mean = float(hr_train.mean())
        hr_std = float(hr_train.std())
        if not math.isfinite(hr_std) or hr_std < 1e-6:
            hr_std = 1.0
        task_data.append(
            TaskData(
                runtime=TaskRuntime(TaskSpec("hr", "reg", "", "mae"), n_out=1, y_mean=hr_mean, y_std=hr_std),
                split_X={split: split_X_shared[split] for split in ("train", "val", "test")},
                split_y={split: split_hr[split] for split in ("train", "val", "test")},
            )
        )
        task_data.append(
            TaskData(
                runtime=TaskRuntime(
                    TaskSpec("activity", "cls", "", "f1_macro"),
                    n_out=int(np.max(y_act)) + 1,
                ),
                split_X={split: split_X_shared[split] for split in ("train", "val", "test")},
                split_y={split: split_act[split] for split in ("train", "val", "test")},
            )
        )
        return task_data

    raise NotImplementedError(f"Unsupported dataset mode: {mode}")


class SingleTaskArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, runtime: TaskRuntime) -> None:
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if self.X.ndim == 2:
            self.X = self.X[:, None, :]
        if runtime.spec.kind == "reg":
            y = (np.asarray(y).astype(np.float32) - runtime.y_mean) / runtime.y_std
            self.y = torch.as_tensor(y, dtype=torch.float32)
        else:
            self.y = torch.as_tensor(np.asarray(y).astype(np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def _parse_jsonish(text: str) -> Any:
    text = str(text).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON string: {text}") from e


def _seconds_to_len(sec: float, hz: int, native_len: int) -> int:
    target = int(round(float(sec) * float(hz)))
    target = max(1, target)
    return min(int(native_len), target)


def resolve_task_supports_sec(
    dataset: str,
    task_names: Iterable[str],
    *,
    native_sec: float,
    override_json: str = "",
) -> Dict[str, float]:
    defaults = TEMPORAL_DEFAULT_SUPPORTS_SEC.get(dataset, {})
    overrides = _parse_jsonish(override_json) if override_json else None
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("--temporal-task-supports must decode to a JSON object")
    out: Dict[str, float] = {}
    for name in task_names:
        sec = float(defaults.get(name, native_sec))
        if overrides is not None and name in overrides:
            sec = float(overrides[name])
        out[str(name)] = min(float(native_sec), max(0.25, sec))
    return out


def resolve_scale_grid_sec(
    dataset: str,
    *,
    native_sec: float,
    override_json: str = "",
) -> List[float]:
    defaults = TEMPORAL_DEFAULT_SCALES_SEC.get(dataset, [native_sec])
    if override_json:
        parsed = _parse_jsonish(override_json)
        if not isinstance(parsed, list):
            raise ValueError("--temporal-scales must decode to a JSON list")
        vals = [float(v) for v in parsed]
    else:
        vals = [float(v) for v in defaults]
    vals = [min(float(native_sec), max(0.25, v)) for v in vals]
    vals = sorted(set(vals))
    if not vals:
        vals = [float(native_sec)]
    if float(native_sec) not in vals:
        vals.append(float(native_sec))
        vals = sorted(set(vals))
    return vals


def resolve_input_window_sec_map(
    task_names: Iterable[str],
    *,
    override_json: str = "",
) -> Dict[str, float]:
    overrides = _parse_jsonish(override_json) if override_json else None
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("--input-window-sec-map must decode to a JSON object")
    out: Dict[str, float] = {}
    if overrides is None:
        return out
    for name in task_names:
        if name in overrides:
            out[str(name)] = max(0.25, float(overrides[name]))
    return out


def _center_crop_np(X: np.ndarray, target_len: int) -> np.ndarray:
    X = np.asarray(X)
    if target_len <= 0 or X.shape[-1] <= target_len:
        return np.asarray(X)
    start = max(0, int((X.shape[-1] - target_len) // 2))
    end = start + int(target_len)
    return np.ascontiguousarray(X[..., start:end])


def apply_task_specific_input_windows(
    task_data_list: Iterable[TaskData],
    hz: int,
    sec_map: Dict[str, float],
) -> List[TaskData]:
    cropped: List[TaskData] = []
    for td in task_data_list:
        name = td.runtime.spec.name
        target_sec = sec_map.get(name)
        if target_sec is None:
            cropped.append(td)
            continue
        native_len = int(next(iter(td.split_X.values())).shape[-1])
        target_len = min(native_len, max(1, int(round(float(target_sec) * float(hz)))))
        split_X = {
            split: _center_crop_np(X, target_len)
            for split, X in td.split_X.items()
        }
        split_y = {split: np.asarray(y) for split, y in td.split_y.items()}
        cropped.append(
            TaskData(
                runtime=copy.deepcopy(td.runtime),
                split_X=split_X,
                split_y=split_y,
            )
        )
    return cropped


class TaskHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        enc_dim: int,
        tasks: List[TaskRuntime],
        dropout: float,
        *,
        dataset_name: str = "",
        hz: int = 50,
        input_len: Optional[int] = None,
        temporal_policy: str = "shared_window",
        temporal_task_supports_sec: Optional[Dict[str, float]] = None,
        temporal_scales_sec: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if temporal_policy not in TEMPORAL_POLICIES:
            raise ValueError(f"Unsupported temporal_policy={temporal_policy}; choices={TEMPORAL_POLICIES}")
        self.encoder = encoder
        self.dataset_name = str(dataset_name)
        self.hz = int(hz)
        self.input_len = int(input_len or 0)
        self.temporal_policy = str(temporal_policy)
        self.heads = nn.ModuleDict({
            rt.spec.name: TaskHead(enc_dim, rt.n_out, dropout) for rt in tasks
        })
        task_names = [rt.spec.name for rt in tasks]
        native_sec = float(self.input_len) / float(self.hz) if self.input_len > 0 else 0.0
        support_sec = temporal_task_supports_sec or resolve_task_supports_sec(
            self.dataset_name,
            task_names,
            native_sec=native_sec,
        )
        self.temporal_task_supports_sec = {str(k): float(v) for k, v in support_sec.items()}
        self.temporal_task_support_lens = {
            str(k): _seconds_to_len(v, self.hz, self.input_len) if self.input_len > 0 else 0
            for k, v in self.temporal_task_supports_sec.items()
        }
        scale_sec = temporal_scales_sec or resolve_scale_grid_sec(self.dataset_name, native_sec=native_sec)
        self.temporal_scales_sec = [float(v) for v in scale_sec]
        self.temporal_scale_lens = [
            _seconds_to_len(v, self.hz, self.input_len) if self.input_len > 0 else 0
            for v in self.temporal_scales_sec
        ]
        if self.temporal_policy == "shared_multi_scale":
            self.temporal_shared_logits = nn.Parameter(torch.zeros(len(self.temporal_scale_lens), dtype=torch.float32))
        elif self.temporal_policy == "task_routed_multiscale":
            self.temporal_router_logits = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(len(self.temporal_scale_lens), dtype=torch.float32))
                for name in task_names
            })

    @staticmethod
    def _pool_encoded(z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 3:
            z = z.mean(-1)
        elif z.ndim > 2:
            z = z.flatten(1)
        return z

    @staticmethod
    def _center_crop(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if target_len <= 0 or x.shape[-1] <= target_len:
            return x
        start = max(0, int((x.shape[-1] - target_len) // 2))
        end = start + int(target_len)
        return x[..., start:end]

    def _task_or_single(self, task_name: Optional[str]) -> str:
        if task_name is not None:
            return str(task_name)
        if len(self.heads) == 1:
            return next(iter(self.heads.keys()))
        raise ValueError(f"temporal policy '{self.temporal_policy}' requires task_name for multi-task batches")

    def _encode_backbone(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self._pool_encoded(encoder(x))

    def encode_with_encoder(self, encoder: nn.Module, x: torch.Tensor, task_name: Optional[str] = None) -> torch.Tensor:
        if self.temporal_policy == "shared_window":
            return self._encode_backbone(encoder, x)
        if self.temporal_policy == "task_specific_crop":
            name = self._task_or_single(task_name)
            target_len = int(self.temporal_task_support_lens.get(name, x.shape[-1]))
            return self._encode_backbone(encoder, self._center_crop(x, target_len))

        zs = [
            self._encode_backbone(encoder, self._center_crop(x, int(scale_len)))
            for scale_len in self.temporal_scale_lens
        ]
        z_stack = torch.stack(zs, dim=0)
        if self.temporal_policy == "shared_multi_scale":
            weights = torch.softmax(self.temporal_shared_logits, dim=0)
        elif self.temporal_policy == "task_routed_multiscale":
            name = self._task_or_single(task_name)
            weights = torch.softmax(self.temporal_router_logits[name], dim=0)
        else:
            raise ValueError(f"Unsupported temporal policy: {self.temporal_policy}")
        return (weights[:, None, None] * z_stack).sum(0)

    def temporal_config_json(self) -> str:
        payload = {
            "policy": self.temporal_policy,
            "task_supports_sec": self.temporal_task_supports_sec,
            "scales_sec": self.temporal_scales_sec,
        }
        return json.dumps(payload, sort_keys=True)

    def encode(self, x: torch.Tensor, task_name: Optional[str] = None) -> torch.Tensor:
        return self.encode_with_encoder(self.encoder, x, task_name=task_name)

    def forward(self, x: torch.Tensor, task_name: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.encode(x, task_name=task_name)
        return z, {name: head(z) for name, head in self.heads.items()}


def init_regression_biases(model: MultiTaskModel, tasks: List[TaskRuntime]) -> None:
    """Regression targets are standardized, so the best initial bias is zero."""
    for rt in tasks:
        if rt.spec.kind == "reg":
            last = model.heads[rt.spec.name].net[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                with torch.no_grad():
                    last.bias.zero_()


def set_encoder_trainable_scope(model: MultiTaskModel, scope: str) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False
    if scope == "none":
        return
    if scope == "full":
        for p in model.encoder.parameters():
            p.requires_grad = True
        return
    if scope not in {"top1", "top2"}:
        raise ValueError(f"Unsupported encoder scope: {scope}")
    n_blocks = 1 if scope == "top1" else 2
    block_list = getattr(model.encoder, "basicblock_list", None)
    if block_list is None:
        raise ValueError(f"Encoder scope '{scope}' requires encoder.basicblock_list")
    n_blocks = min(int(n_blocks), len(block_list))
    for block in list(block_list)[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


def set_trainable_for_method(model: MultiTaskModel, method: str, encoder_scope: str) -> None:
    requires_encoder = method in {"joint_finetune", "sa_mtft", "independent_finetune"}
    set_encoder_trainable_scope(model, encoder_scope if requires_encoder else "none")
    for p in model.heads.parameters():
        p.requires_grad = True


def make_optimizer(
    model: MultiTaskModel,
    method: str,
    head_lr: float,
    encoder_lr_mult: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if method in {"joint_finetune", "sa_mtft", "independent_finetune"}:
        return torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.encoder.parameters() if p.requires_grad],
                    "lr": head_lr * encoder_lr_mult,
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for p in model.heads.parameters() if p.requires_grad],
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                },
            ]
        )
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=head_lr,
        weight_decay=weight_decay,
    )


def task_losses(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    tasks: List[TaskRuntime],
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    for rt in tasks:
        name = rt.spec.name
        if name not in outputs:
            continue
        if rt.spec.kind == "cls":
            losses[name] = F.cross_entropy(outputs[name], labels[name].long())
        else:
            losses[name] = F.smooth_l1_loss(outputs[name].squeeze(-1), labels[name].float())
    return losses


def anchor_losses(
    model: MultiTaskModel,
    z: torch.Tensor,
    x: torch.Tensor,
    current_proj: nn.Linear,
    anchor_encoder: nn.Module,
    V: torch.Tensor,
    task_name: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    z_full = current_proj(z).float()
    with torch.no_grad():
        z0 = model.encode_with_encoder(anchor_encoder, x, task_name=task_name)
        z0_full = current_proj(z0).float()

    z_proj = z_full @ V.T
    z0_proj = z0_full @ V.T
    # Normalize the anchor by the frozen representation energy. The raw MSE in
    # teacher-512 space can be hundreds of times larger than standardized task
    # losses, which makes the anchor a hard freeze rather than a gentle prior.
    sub_scale = z0_proj.detach().pow(2).mean().clamp_min(1e-6)
    sub_anchor = F.mse_loss(z_proj, z0_proj) / sub_scale
    z_recon = z_proj @ V
    null_scale = z_full.detach().pow(2).mean().clamp_min(1e-6)
    null_loss = F.mse_loss(z_full, z_recon) / null_scale
    return sub_anchor, null_loss


@torch.no_grad()
def evaluate(
    model: MultiTaskModel,
    test_loaders: Dict[str, DataLoader],
    tasks: List[TaskRuntime],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    preds: Dict[str, List[np.ndarray]] = {rt.spec.name: [] for rt in tasks}
    trues: Dict[str, List[np.ndarray]] = {rt.spec.name: [] for rt in tasks}
    logits: Dict[str, List[np.ndarray]] = {rt.spec.name: [] for rt in tasks}

    for rt in tasks:
        name = rt.spec.name
        loader = test_loaders[name]
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            _, out = model(xb, task_name=name)
            if rt.spec.kind == "cls":
                logit = out[name].detach().cpu().numpy()
                yp = logit.argmax(1)
                yt = yb.cpu().numpy()
                logits[name].append(logit)
            else:
                yp = out[name].squeeze(-1).detach().cpu().numpy()
                yt = yb.cpu().numpy()
                yp = yp * rt.y_std + rt.y_mean
                yt = yt * rt.y_std + rt.y_mean
            preds[name].append(yp)
            trues[name].append(yt)

    metrics: Dict[str, Dict[str, float]] = {}
    for rt in tasks:
        name = rt.spec.name
        if not preds[name]:
            continue
        yp = np.concatenate(preds[name])
        yt = np.concatenate(trues[name])
        if rt.spec.kind == "cls":
            val = float(f1_score(yt, yp, average="macro", zero_division=0))
            balanced_acc = float(balanced_accuracy_score(yt, yp))
            labels = np.unique(np.concatenate([yt.astype(np.int64), yp.astype(np.int64)]))
            _, _, per_class_f1, _ = precision_recall_fscore_support(
                yt,
                yp,
                labels=labels,
                zero_division=0,
            )
            per_class_f1_json = json.dumps(
                {str(int(label)): float(score) for label, score in zip(labels, per_class_f1)},
                sort_keys=True,
            )
            auroc = np.nan
            auprc = np.nan
            if logits[name]:
                logit = np.concatenate(logits[name], axis=0)
                if logit.ndim == 2 and logit.shape[1] == 2 and np.unique(yt).size == 2:
                    prob_pos = torch.softmax(torch.from_numpy(logit), dim=1).numpy()[:, 1]
                    try:
                        auroc = float(roc_auc_score(yt.astype(np.int64), prob_pos.astype(np.float64)))
                    except Exception:
                        auroc = np.nan
                    try:
                        auprc = float(average_precision_score(yt.astype(np.int64), prob_pos.astype(np.float64)))
                    except Exception:
                        auprc = np.nan
        else:
            val = float(mean_absolute_error(yt.astype(np.float32), yp.astype(np.float32)))
            balanced_acc = np.nan
            per_class_f1_json = "{}"
            auroc = np.nan
            auprc = np.nan
        metrics[name] = {
            "value": val,
            "metric": rt.spec.metric,
            "higher_better": bool(rt.spec.kind == "cls"),
            "macro_f1": float(val) if rt.spec.kind == "cls" else np.nan,
            "mae": float(val) if rt.spec.kind != "cls" else np.nan,
            "balanced_acc": balanced_acc,
            "auroc": auroc,
            "auprc": auprc,
            "per_class_f1_json": per_class_f1_json,
            "n_eval": int(len(yt)),
        }
    return metrics


@torch.no_grad()
def validation_loss(
    model: MultiTaskModel,
    val_loaders: Dict[str, DataLoader],
    tasks: List[TaskRuntime],
    device: torch.device,
) -> float:
    model.eval()
    vals: List[float] = []
    for rt in tasks:
        loader = val_loaders[rt.spec.name]
        task_vals: List[float] = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            _, out = model(xb, task_name=rt.spec.name)
            losses = task_losses(out, {rt.spec.name: yb}, [rt])
            if losses:
                task_vals.append(float(losses[rt.spec.name].item()))
        if task_vals:
            vals.append(float(np.mean(task_vals)))
    return float(np.mean(vals)) if vals else float("inf")


def train_joint_or_frozen(
    model: MultiTaskModel,
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, DataLoader],
    val_loaders: Dict[str, DataLoader],
    test_loaders: Dict[str, DataLoader],
    method: str,
    device: torch.device,
    epochs: int,
    patience: int,
    head_lr: float,
    encoder_lr_mult: float,
    weight_decay: float,
    anchor_encoder: Optional[nn.Module],
    projector: Optional[nn.Linear],
    V: Optional[torch.Tensor],
    alpha_anchor: float,
    beta_null: float,
    encoder_update_scope: str,
    log_prefix: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    model = model.to(device)
    set_trainable_for_method(model, method, encoder_update_scope)
    init_regression_biases(model, tasks)

    if anchor_encoder is not None:
        anchor_encoder = anchor_encoder.to(device).eval()
        anchor_encoder.requires_grad_(False)
    if projector is not None:
        projector = projector.to(device).eval()
        projector.requires_grad_(False)
    if V is not None:
        V = V.to(device)

    opt = make_optimizer(model, method, head_lr, encoder_lr_mult, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    last_anchor = 0.0
    last_null = 0.0

    for ep in range(epochs):
        model.train()
        train_losses: List[float] = []
        task_iters = {rt.spec.name: iter(train_loaders[rt.spec.name]) for rt in tasks}
        n_steps = max(len(train_loaders[rt.spec.name]) for rt in tasks)
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            per_task_losses: List[torch.Tensor] = []
            per_anchor: List[torch.Tensor] = []
            per_null: List[torch.Tensor] = []

            for rt in tasks:
                name = rt.spec.name
                try:
                    xb, yb = next(task_iters[name])
                except StopIteration:
                    task_iters[name] = iter(train_loaders[name])
                    xb, yb = next(task_iters[name])
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                z, out = model(xb, task_name=name)
                losses = task_losses(out, {name: yb}, [rt])
                if name not in losses:
                    continue
                per_task_losses.append(losses[name])

                if method == "sa_mtft":
                    if anchor_encoder is None or projector is None or V is None:
                        raise RuntimeError("sa_mtft requires anchor_encoder, projector, and V")
                    sub_anchor, null_loss = anchor_losses(model, z, xb, projector, anchor_encoder, V, task_name=name)
                    per_anchor.append(sub_anchor)
                    per_null.append(null_loss)

            if not per_task_losses:
                continue
            task_loss = torch.stack(per_task_losses).mean()
            loss = task_loss

            if method == "sa_mtft":
                sub_anchor = torch.stack(per_anchor).mean()
                null_loss = torch.stack(per_null).mean()
                loss = loss + alpha_anchor * sub_anchor + beta_null * null_loss
                last_anchor = float(sub_anchor.detach().item())
                last_null = float(null_loss.detach().item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            train_losses.append(float(loss.detach().item()))
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
            extra = f" anchor={last_anchor:.4g} null={last_null:.4g}" if method == "sa_mtft" else ""
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
    meta = {
        "val_loss_best": float(best_val),
        "epochs_trained": int(ep + 1),
    }
    return metrics, meta


def make_task_loaders(
    task_data: List[TaskData],
    batch_size: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, DataLoader]]:
    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    test_loaders: Dict[str, DataLoader] = {}
    for td in task_data:
        name = td.runtime.spec.name
        train_ds = SingleTaskArrayDataset(td.split_X["train"], td.split_y["train"], td.runtime)
        val_ds = SingleTaskArrayDataset(td.split_X["val"], td.split_y["val"], td.runtime)
        test_ds = SingleTaskArrayDataset(td.split_X["test"], td.split_y["test"], td.runtime)
        train_loaders[name] = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loaders[name] = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loaders[name] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loaders, val_loaders, test_loaders


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in module.parameters()))


def records_from_metrics(
    metrics: Dict[str, Dict[str, float]],
    base_record: Dict[str, Any],
    task_runtimes: List[TaskRuntime],
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    task_kind = {rt.spec.name: rt.spec.kind for rt in task_runtimes}
    out: List[Dict[str, Any]] = []
    for task_name, m in metrics.items():
        rec = dict(base_record)
        rec.update(meta)
        rec.update(
            {
                "task": task_name,
                "task_kind": task_kind[task_name],
                "metric": m["metric"],
                "value": float(m["value"]),
                "higher_better": bool(m["higher_better"]),
            }
        )
        out.append(rec)
    return out


def run_method(
    method: str,
    student: str,
    seed: int,
    kd_method: str,
    ckpt_dir: Path,
    task_data: List[TaskData],
    V: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, Any]]:
    tasks = [td.runtime for td in task_data]
    input_window_sec_map = resolve_input_window_sec_map(
        [rt.spec.name for rt in tasks],
        override_json=args.input_window_sec_map,
    )
    input_window_policy = "task_specific_input" if input_window_sec_map else "shared_native"
    input_window_sec_map_json = json.dumps(input_window_sec_map, sort_keys=True)
    input_len = max(int(td.split_X["train"].shape[-1]) for td in task_data)
    task_supports_sec = resolve_task_supports_sec(
        args.dataset,
        [rt.spec.name for rt in tasks],
        native_sec=float(input_len) / float(args.hz),
        override_json=args.temporal_task_supports,
    )
    scale_grid_sec = resolve_scale_grid_sec(
        args.dataset,
        native_sec=float(input_len) / float(args.hz),
        override_json=args.temporal_scales,
    )
    temporal_payload = {
        "policy": args.temporal_policy,
        "task_supports_sec": task_supports_sec,
        "scales_sec": scale_grid_sec,
    }
    effective_encoder_scope = args.encoder_update_scope if method in {"joint_finetune", "sa_mtft", "independent_finetune"} else "none"
    base = {
        "dataset": args.dataset,
        "student": student,
        "kd_method": kd_method,
        "seed": seed,
        "method": method,
        "encoder_update_scope": effective_encoder_scope,
        "temporal_policy": args.temporal_policy,
        "input_window_policy": input_window_policy,
        "input_window_sec_map": input_window_sec_map_json,
        "temporal_config": json.dumps(temporal_payload, sort_keys=True),
        "head_lr": args.head_lr,
        "encoder_lr_mult": args.encoder_lr_mult,
        "alpha_anchor": args.alpha_anchor if method == "sa_mtft" else 0.0,
        "beta_null": args.beta_null if method == "sa_mtft" else 0.0,
    }

    if method == "independent_finetune":
        all_records: List[Dict[str, Any]] = []
        for td in task_data:
            rt = td.runtime
            enc, proj, ck_meta = load_stage1_full(student, ckpt_dir, kd_method, seed, device)
            model = MultiTaskModel(
                enc,
                int(ck_meta["enc_dim"]),
                [rt],
                args.dropout,
                dataset_name=args.dataset,
                hz=args.hz,
                input_len=int(td.split_X["train"].shape[-1]),
                temporal_policy=args.temporal_policy,
                temporal_task_supports_sec={rt.spec.name: float(task_supports_sec[rt.spec.name])},
                temporal_scales_sec=scale_grid_sec,
            )
            train_loaders, val_loaders, test_loaders = make_task_loaders([td], args.batch_size)
            log_prefix = f"[{student}/s{seed}/{method}/{rt.spec.name}] "
            t0 = time.time()
            metrics, meta = train_joint_or_frozen(
                model=model,
                tasks=[rt],
                train_loaders=train_loaders,
                val_loaders=val_loaders,
                test_loaders=test_loaders,
                method=method,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                head_lr=args.head_lr,
                encoder_lr_mult=args.encoder_lr_mult,
                weight_decay=args.weight_decay,
                anchor_encoder=None,
                projector=None,
                V=None,
                alpha_anchor=0.0,
                beta_null=0.0,
                encoder_update_scope=effective_encoder_scope,
                log_prefix=log_prefix,
            )
            deploy_encoders = len(tasks)
            base_task = dict(base)
            base_task.update(
                {
                    "deploy_encoders": deploy_encoders,
                    "encoder_params": count_params(model.encoder),
                    "total_params": deploy_encoders * count_params(model.encoder)
                    + sum(count_params(TaskHead(int(ck_meta["enc_dim"]), t.n_out, args.dropout)) for t in tasks),
                    "trainable_params": count_params(model, trainable_only=True),
                    "stage1_ckpt": ck_meta["ckpt_path"],
                    "time_s": float(time.time() - t0),
                }
            )
            all_records.extend(records_from_metrics(metrics, base_task, [rt], meta))
        return all_records

    enc, proj, ck_meta = load_stage1_full(student, ckpt_dir, kd_method, seed, device)
    model = MultiTaskModel(
        enc,
        int(ck_meta["enc_dim"]),
        tasks,
        args.dropout,
        dataset_name=args.dataset,
        hz=args.hz,
        input_len=input_len,
        temporal_policy=args.temporal_policy,
        temporal_task_supports_sec=task_supports_sec,
        temporal_scales_sec=scale_grid_sec,
    )
    train_loaders, val_loaders, test_loaders = make_task_loaders(task_data, args.batch_size)

    anchor_encoder = None
    if method == "sa_mtft":
        anchor_encoder, _, _ = load_stage1_full(student, ckpt_dir, kd_method, seed, device)

    log_prefix = f"[{student}/s{seed}/{method}] "
    t0 = time.time()
    metrics, meta = train_joint_or_frozen(
        model=model,
        tasks=tasks,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        method=method,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        head_lr=args.head_lr,
        encoder_lr_mult=args.encoder_lr_mult,
        weight_decay=args.weight_decay,
        anchor_encoder=anchor_encoder,
        projector=proj,
        V=V,
        alpha_anchor=args.alpha_anchor,
        beta_null=args.beta_null,
        encoder_update_scope=effective_encoder_scope,
        log_prefix=log_prefix,
    )
    base.update(
        {
            "deploy_encoders": 1,
            "encoder_params": count_params(model.encoder),
            "total_params": count_params(model),
            "trainable_params": count_params(model, trainable_only=True),
            "temporal_config": model.temporal_config_json(),
            "stage1_ckpt": ck_meta["ckpt_path"],
            "time_s": float(time.time() - t0),
        }
    )
    return records_from_metrics(metrics, base, tasks, meta)


def ensure_temporal_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "encoder_update_scope" not in df.columns:
        needs_enc = ~df["method"].eq("frozen_heads")
        df["encoder_update_scope"] = np.where(needs_enc, "full", "none")
    else:
        needs_enc = ~df["method"].eq("frozen_heads")
        missing_scope = df["encoder_update_scope"].isna() | df["encoder_update_scope"].astype(str).str.strip().eq("")
        df.loc[missing_scope, "encoder_update_scope"] = np.where(needs_enc[missing_scope], "full", "none")
    if "temporal_policy" not in df.columns:
        df["temporal_policy"] = "shared_window"
    else:
        missing_policy = df["temporal_policy"].isna() | df["temporal_policy"].astype(str).str.strip().eq("")
        df.loc[missing_policy, "temporal_policy"] = "shared_window"
    if "temporal_config" not in df.columns:
        df["temporal_config"] = df["temporal_policy"].map(lambda p: json.dumps({"policy": str(p)}, sort_keys=True))
    else:
        missing_config = df["temporal_config"].isna() | df["temporal_config"].astype(str).str.strip().eq("")
        df.loc[missing_config, "temporal_config"] = df.loc[missing_config, "temporal_policy"].map(
            lambda p: json.dumps({"policy": str(p)}, sort_keys=True)
        )
    if "input_window_policy" not in df.columns:
        df["input_window_policy"] = "shared_native"
    else:
        missing_input_policy = df["input_window_policy"].isna() | df["input_window_policy"].astype(str).str.strip().eq("")
        df.loc[missing_input_policy, "input_window_policy"] = "shared_native"
    if "input_window_sec_map" not in df.columns:
        df["input_window_sec_map"] = "{}"
    else:
        missing_input_map = df["input_window_sec_map"].isna() | df["input_window_sec_map"].astype(str).str.strip().eq("")
        df.loc[missing_input_map, "input_window_sec_map"] = "{}"

    if {"dataset", "method", "task", "temporal_policy", "temporal_config"}.issubset(df.columns):
        normalized_configs: List[str] = []
        for _, row in df.iterrows():
            cfg_text = str(row.get("temporal_config", "") or "")
            dataset = str(row.get("dataset", "") or "")
            method = str(row.get("method", "") or "")
            task_name = str(row.get("task", "") or "")
            policy = str(row.get("temporal_policy", "") or "")
            if policy != "shared_window" or dataset not in DATASET_SPECS:
                normalized_configs.append(cfg_text)
                continue
            try:
                cfg = json.loads(cfg_text) if cfg_text else {}
            except json.JSONDecodeError:
                normalized_configs.append(cfg_text)
                continue
            if not isinstance(cfg, dict):
                normalized_configs.append(cfg_text)
                continue
            task_supports = cfg.get("task_supports_sec", {})
            expected_tasks = [spec.name for spec in DATASET_SPECS[dataset]["tasks"]]
            support_keys = {str(k) for k in task_supports.keys()} if isinstance(task_supports, dict) else set()
            has_full_supports = support_keys == set(expected_tasks)
            has_scales = isinstance(cfg.get("scales_sec"), list) and len(cfg.get("scales_sec")) > 0
            needs_canonical = (not has_scales) or (not has_full_supports)
            if not needs_canonical:
                normalized_configs.append(cfg_text)
                continue
            if method == "independent_finetune" and support_keys and support_keys != {task_name} and not has_full_supports:
                normalized_configs.append(cfg_text)
                continue
            input_map_text = str(row.get("input_window_sec_map", "{}") or "{}")
            try:
                input_map = _parse_jsonish(input_map_text) if input_map_text else {}
            except ValueError:
                input_map = {}
            native_candidates: List[float] = []
            if isinstance(input_map, dict):
                native_candidates.extend(float(v) for k, v in input_map.items() if str(k) in expected_tasks)
            defaults = TEMPORAL_DEFAULT_SUPPORTS_SEC.get(dataset, {})
            native_candidates.extend(float(defaults[name]) for name in expected_tasks if name in defaults)
            native_candidates.extend(float(v) for v in TEMPORAL_DEFAULT_SCALES_SEC.get(dataset, []))
            native_sec = max(native_candidates) if native_candidates else max(float(v) for v in task_supports.values())
            normalized_configs.append(
                json.dumps(
                    {
                        "policy": "shared_window",
                        "task_supports_sec": resolve_task_supports_sec(dataset, expected_tasks, native_sec=native_sec),
                        "scales_sec": resolve_scale_grid_sec(dataset, native_sec=native_sec),
                    },
                    sort_keys=True,
                )
            )
        df["temporal_config"] = normalized_configs
    return df


def add_relative_scores(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, pd.DataFrame()

    df = ensure_temporal_cols(df)
    df["rel_vs_frozen"] = np.nan
    key_cols = [
        "dataset",
        "student",
        "kd_method",
        "seed",
        "temporal_policy",
        "temporal_config",
        "input_window_policy",
        "input_window_sec_map",
        "task",
    ]
    frozen = df[df["method"] == "frozen_heads"].set_index(key_cols)
    for idx, row in df.iterrows():
        key = tuple(row[c] for c in key_cols)
        if key not in frozen.index:
            continue
        base_val = float(frozen.loc[key, "value"])
        val = float(row["value"])
        if not np.isfinite(base_val) or not np.isfinite(val) or val == 0 or base_val == 0:
            continue
        if bool(row["higher_better"]):
            df.loc[idx, "rel_vs_frozen"] = val / base_val
        else:
            df.loc[idx, "rel_vs_frozen"] = base_val / val

    summary = (
        df.groupby(
            [
                "dataset",
                "student",
                "kd_method",
                "seed",
                "encoder_update_scope",
                "temporal_policy",
                "temporal_config",
                "input_window_policy",
                "input_window_sec_map",
                "method",
            ],
            dropna=False,
        )
        .agg(
            avg_rel_vs_frozen=("rel_vs_frozen", "mean"),
            floor_rel_vs_frozen=("rel_vs_frozen", "min"),
            avg_raw_value=("value", "mean"),
            n_tasks=("task", "nunique"),
            deploy_encoders=("deploy_encoders", "first"),
            encoder_params=("encoder_params", "first"),
            total_params=("total_params", "first"),
            trainable_params=("trainable_params", "first"),
            time_s=("time_s", "sum"),
        )
        .reset_index()
    )
    return df, summary


def aggregate_over_seeds(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return (
        summary.groupby(
            [
                "dataset",
                "student",
                "kd_method",
                "encoder_update_scope",
                "temporal_policy",
                "temporal_config",
                "input_window_policy",
                "input_window_sec_map",
                "method",
            ],
            dropna=False,
        )
        .agg(
            avg_rel_mean=("avg_rel_vs_frozen", "mean"),
            avg_rel_std=("avg_rel_vs_frozen", "std"),
            floor_rel_mean=("floor_rel_vs_frozen", "mean"),
            floor_rel_std=("floor_rel_vs_frozen", "std"),
            n_seeds=("seed", "nunique"),
            deploy_encoders=("deploy_encoders", "first"),
            encoder_params=("encoder_params", "first"),
            total_params=("total_params", "first"),
            trainable_params=("trainable_params", "first"),
        )
        .reset_index()
    )


def method_completed(
    existing: pd.DataFrame,
    dataset: str,
    student: str,
    kd_method: str,
    seed: int,
    method: str,
    task_data: List[TaskData],
    encoder_update_scope: str = "none",
    temporal_policy: str = "shared_window",
    temporal_config: str = "",
    input_window_policy: str = "shared_native",
    input_window_sec_map: str = "{}",
) -> bool:
    if existing.empty:
        return False
    existing = ensure_temporal_cols(existing)
    if not temporal_config:
        temporal_config = json.dumps({"policy": str(temporal_policy)}, sort_keys=True)
    sub = existing[
        (existing["dataset"] == dataset)
        & (existing["student"] == student)
        & (existing["kd_method"] == kd_method)
        & (existing["seed"] == seed)
        & (existing["encoder_update_scope"] == encoder_update_scope)
        & (existing["temporal_policy"] == temporal_policy)
        & (existing["temporal_config"] == temporal_config)
        & (existing["input_window_policy"] == input_window_policy)
        & (existing["input_window_sec_map"] == input_window_sec_map)
        & (existing["method"] == method)
    ]
    return set(sub["task"].astype(str).tolist()) >= {td.runtime.spec.name for td in task_data}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="ppgbp")
    ap.add_argument("--students", nargs="+", default=["BF20_NB8", "BF48_NB8"])
    ap.add_argument("--kd-method", default="lifefm_kd", choices=["lifefm_kd", "subspace_mse"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["frozen_heads", "independent_finetune", "joint_finetune", "sa_mtft"],
        choices=["frozen_heads", "independent_finetune", "joint_finetune", "sa_mtft"],
    )
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument("--k", type=int, default=None, help="default: k stored in checkpoint, else 40")
    ap.add_argument("--svd-cache-dir", type=Path, default=DEFAULT_SVD_CACHE)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--encoder-update-scope", choices=ENCODER_UPDATE_SCOPES, default="full")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--temporal-policy", choices=TEMPORAL_POLICIES, default="shared_window")
    ap.add_argument(
        "--temporal-task-supports",
        type=str,
        default="",
        help='JSON dict of task-specific effective support in seconds, e.g. {"activity":2.5,"hr":10}',
    )
    ap.add_argument(
        "--temporal-scales",
        type=str,
        default="",
        help='JSON list of multi-scale supports in seconds, e.g. [2.5,5,10]',
    )
    ap.add_argument(
        "--input-window-sec-map",
        type=str,
        default="",
        help='JSON dict of task-specific raw input windows in seconds, e.g. {"activity":2.5,"hr":10}',
    )
    ap.add_argument("--alpha-anchor", type=float, default=0.1)
    ap.add_argument("--beta-null", type=float, default=0.01)
    ap.add_argument("--limit-train", type=int, default=0, help="0 means full split")
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="load existing CSV, rebuild summary, and exit")
    ap.add_argument("--fresh", action="store_true", help="ignore existing CSV and rerun requested settings")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[device] {device}", flush=True)
    print(f"[dataset] {args.dataset}", flush=True)
    print(f"[temporal] policy={args.temporal_policy}", flush=True)
    print(f"[encoder-scope] {args.encoder_update_scope}", flush=True)

    out_dir = args.out_root / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "stage2_results.csv"
    summary_csv = out_dir / "stage2_summary.csv"
    summary_by_student_csv = out_dir / "stage2_summary_by_student.csv"
    config_json = out_dir / "stage2_config.json"
    config_json.write_text(json.dumps(vars(args), indent=2, default=str))

    if args.resume and out_csv.exists():
        print(f"[resume] loading existing {out_csv}", flush=True)
        df = ensure_temporal_cols(pd.read_csv(out_csv))
        df, summary = add_relative_scores(df)
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
                native_sec = float(max(int(td.split_X["train"].shape[-1]) for td in task_data)) / float(args.hz)
                print(
                    f"[temporal-defaults] supports={resolve_task_supports_sec(args.dataset, [td.runtime.spec.name for td in task_data], native_sec=native_sec, override_json=args.temporal_task_supports)} "
                    f"scales={resolve_scale_grid_sec(args.dataset, native_sec=native_sec, override_json=args.temporal_scales)}",
                    flush=True,
                )
                if input_window_sec_map:
                    print(f"[input-window-sec-map] {json.dumps(input_window_sec_map, sort_keys=True)}", flush=True)
                printed_data = True
            try:
                _, _, ck_meta = load_stage1_full(student, ckpt_dir, args.kd_method, seed, device)
            except Exception as e:
                warnings.warn(str(e))
                continue
            k = int(args.k if args.k is not None else ck_meta.get("k", 40))
            V = topk_basis_from_cache(args.svd_cache_dir, k)
            print(
                f"\n{'=' * 80}\n"
                f"student={student} seed={seed} kd={args.kd_method} k={k}\n"
                f"ckpt={ck_meta['ckpt_path']}\n"
                f"{'=' * 80}",
                flush=True,
            )
            for method in args.methods:
                temporal_config = json.dumps(
                    {
                        "policy": args.temporal_policy,
                        "task_supports_sec": resolve_task_supports_sec(
                            args.dataset,
                            [td.runtime.spec.name for td in task_data],
                            native_sec=float(max(int(td.split_X["train"].shape[-1]) for td in task_data)) / float(args.hz),
                            override_json=args.temporal_task_supports,
                        ),
                        "scales_sec": resolve_scale_grid_sec(
                            args.dataset,
                            native_sec=float(max(int(td.split_X["train"].shape[-1]) for td in task_data)) / float(args.hz),
                            override_json=args.temporal_scales,
                        ),
                    },
                    sort_keys=True,
                )
                input_window_policy = "task_specific_input" if input_window_sec_map else "shared_native"
                input_window_sec_map_json = json.dumps(input_window_sec_map, sort_keys=True)
                if method_completed(
                    existing_df,
                    args.dataset,
                    student,
                    args.kd_method,
                    seed,
                    method,
                    task_data,
                    args.encoder_update_scope if method in {"joint_finetune", "sa_mtft", "independent_finetune"} else "none",
                    args.temporal_policy,
                    temporal_config,
                    input_window_policy,
                    input_window_sec_map_json,
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
                        V=V,
                        args=args,
                        device=device,
                    )
                except Exception:
                    import traceback

                    traceback.print_exc()
                    continue
                all_rows.extend(rows)
                df_tmp, summary_tmp = add_relative_scores(pd.DataFrame(all_rows))
                df_tmp.to_csv(out_csv, index=False)
                summary_tmp.to_csv(summary_csv, index=False)
                aggregate_over_seeds(summary_tmp).to_csv(summary_by_student_csv, index=False)

    if not all_rows:
        raise RuntimeError("No rows were produced. Check checkpoint paths and dataset files.")

    df, summary = add_relative_scores(pd.DataFrame(all_rows))
    df.to_csv(out_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_by_student = aggregate_over_seeds(summary)
    summary_by_student.to_csv(summary_by_student_csv, index=False)
    print(f"\n[done] wrote {out_csv}", flush=True)
    print(f"[done] wrote {summary_csv}", flush=True)
    print(f"[done] wrote {summary_by_student_csv}", flush=True)
    print("\n=== Stage-2 summary: retention relative to frozen_heads ===")
    show_cols = [
        "dataset",
        "student",
        "seed",
        "encoder_update_scope",
        "temporal_policy",
        "method",
        "avg_rel_vs_frozen",
        "floor_rel_vs_frozen",
        "deploy_encoders",
        "total_params",
        "trainable_params",
    ]
    print(summary[show_cols].sort_values(["student", "seed", "avg_rel_vs_frozen"], ascending=[True, True, False]).to_string(index=False))
    if not summary_by_student.empty:
        print("\n=== Across-seed summary ===")
        show_seed_cols = [
            "dataset",
            "student",
            "encoder_update_scope",
            "temporal_policy",
            "method",
            "avg_rel_mean",
            "avg_rel_std",
            "floor_rel_mean",
            "n_seeds",
            "deploy_encoders",
        ]
        print(summary_by_student[show_seed_cols].sort_values(["student", "avg_rel_mean"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()

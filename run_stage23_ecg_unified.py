#!/usr/bin/env python3
"""Unified Stage-2 -> Stage-3 evaluation for compressed ECG encoders.

First-pass ECG unified protocol:

1. Stage 2: train a shared multitask model on an initial PTB-XL task pack.
2. Stage 3: add held-out PTB-XL tasks sequentially using only new-task data.
3. Measure:
   - Stage-2 current-task performance
   - old-task retention after Stage 3
   - future-task learnability vs a single-task oracle

This mirrors the PPG unified protocol so the two modalities are directly
comparable.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import least_squares
from torch.utils.data import DataLoader, Dataset

from run_stage2_multitask_ppg import (
    MultiTaskModel,
    TaskRuntime,
    TaskSpec,
    count_params,
    evaluate,
    init_regression_biases,
    set_encoder_trainable_scope,
    task_losses,
    validation_loss,
)
from run_stage23_ppg_unified import (
    _build_care_auto_anchor_basis,
    _cpca_ts_basis,
    _care_auto_reuse_terms,
    _care_auto_trainable_scope,
    _care_v3_candidate_param_names,
    _resolve_care_auto_reuse_loss_type,
    _lpr_preconditionable_layers,
    _lpr_build_preconditioners,
    _lpr_apply_preconditioners,
    _lpr_replay_loss,
    _dwdp_loss,
    _core_dwdp_loss,
    _old_task_head_basis,
    _set_encoder_trainable_names,
    compare_metric,
)


def _lpr_build_replay_ecg(old_task_data: List[Any], buffer_size: int, rng: np.random.Generator):
    per_task = max(1, int(buffer_size) // max(1, len(old_task_data)))
    X_list, y_list, runtimes = [], [], []
    for td in old_task_data:
        idx = np.asarray(td.split_idx["train"])
        n = len(idx)
        k = min(per_task, n)
        sel = rng.choice(n, size=k, replace=False)
        Xs = []
        for j in sel:
            x = np.asarray(td.signals[int(idx[j])], dtype=np.float32)
            if td.input_len > 0 and x.shape[-1] > td.input_len:
                st = max(0, int((x.shape[-1] - td.input_len) // 2))
                x = np.ascontiguousarray(x[..., st : st + td.input_len])
            Xs.append(zscore_batch(x[None])[0].astype(np.float32, copy=False))
        X = np.stack(Xs)
        y = np.asarray(td.split_y["train"])[sel]
        if td.runtime.spec.kind == "reg":
            y = (y.astype(np.float32) - td.runtime.y_mean) / td.runtime.y_std
        else:
            y = y.astype(np.int64)
        X_list.append(X)
        y_list.append(y)
        runtimes.append(td.runtime)
    return X_list, y_list, runtimes


ROOT = Path(__file__).resolve().parent
ECG_ROOT = ROOT / "ecg" if (ROOT / "ecg").is_dir() else ROOT.parent / "ecg"
ECG_DATA_ROOT = ECG_ROOT / "data"
CHAP_DIR = ECG_DATA_ROOT / "chapman_processed"
CHAP_RAW = ECG_DATA_ROOT / "chapman_extract" / "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
if str(ECG_ROOT) not in sys.path:
    sys.path.insert(0, str(ECG_ROOT))
from run_ecgfounder_kd import build_student, load_task, split_by_fold, zscore_batch

DEFAULT_OUT_ROOT = ROOT / "kd_sweep_results/stage23_ecg_unified"
DEFAULT_CKPT_DIRS = {
    "BF6_NB7": ROOT / "kd_sweep_results/ecg_preliminary/exp_ecgfounder_BF6_NB7_10K_nskd_tune_lam005_e40/exp_ecgfounder_BF6_NB7/checkpoints",
    "BF8_NB7": ROOT / "kd_sweep_results/ecg_preliminary/exp_ecgfounder_BF8_NB7_10K_13tasks/checkpoints",
    "BF10_NB7": ROOT / "kd_sweep_results/ecg_preliminary/exp_ecgfounder_BF10_NB7_10K_13tasks/checkpoints",
    "BF12_NB7": ROOT / "kd_sweep_results/ecg_preliminary/exp_ecgfounder_BF12_NB7_10K_13tasks/checkpoints",
}
STAGE2_METHODS = ("lifefm_adapt",)
CARE_AUTO_STAGE2_METHODS: frozenset = frozenset({"lifefm_adapt"})
STAGE3_POLICIES = ("lifefm_evolve",)


PACKS: Dict[str, Dict[str, Any]] = {
    "ptbxl_core": {
        "dataset": "ptbxl",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "super5": 8.0,
            "rhythm12": 10.0,
            "mi_bin": 5.0,
            "afib_bin": 10.0,
            "sttc_bin": 5.0,
        },
        "initial_tasks": ["super5", "rhythm12", "mi_bin"],
        "future_tasks": ["afib_bin", "sttc_bin"],
        "description": "Task-specific PTB-XL raw windows: 8s super-class diagnosis, 10s rhythm/AF, and 5s MI/STTC morphology tasks.",
    },
    "chapman_core": {
        "dataset": "chapman",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "chap_rhythm4": 10.0,
            "chap_abnormal_bin": 8.0,
            "chap_stachy_bin": 4.0,
            "chap_afib_bin": 10.0,
            "chap_sbrady_bin": 10.0,
        },
        "initial_tasks": ["chap_rhythm4", "chap_abnormal_bin", "chap_stachy_bin"],
        "future_tasks": ["chap_afib_bin", "chap_sbrady_bin"],
        "description": "Task-specific Chapman raw windows: 10s rhythm/AF/brady, 8s abnormality, and 4s tachycardia detection.",
    },
    "chapman2_to_ptbxl_sttc": {
        "dataset": "chapman_to_ptbxl",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "chap_rhythm4": 10.0,
            "chap_abnormal_bin": 8.0,
            "sttc_bin": 5.0,
        },
        "task_defs": {
            "chap_rhythm4": {"dataset": "chapman"},
            "chap_abnormal_bin": {"dataset": "chapman"},
            "sttc_bin": {"dataset": "ptbxl"},
        },
        "initial_tasks": ["chap_rhythm4", "chap_abnormal_bin"],
        "future_tasks": ["sttc_bin"],
        "description": "Cross-modal Stage-III ECG setting: adapt on the paper's two Chapman rhythm/abnormality tasks, then add one PTB-XL ST-T morphology task.",
    },
    "chapman2_to_ptbxl_morph": {
        "dataset": "chapman_to_ptbxl",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "chap_rhythm4": 10.0,
            "chap_abnormal_bin": 8.0,
            "sttc_bin": 5.0,
            "cd_bin": 5.0,
        },
        "task_defs": {
            "chap_rhythm4": {"dataset": "chapman"},
            "chap_abnormal_bin": {"dataset": "chapman"},
            "sttc_bin": {"dataset": "ptbxl"},
            "cd_bin": {"dataset": "ptbxl"},
        },
        "initial_tasks": ["chap_rhythm4", "chap_abnormal_bin"],
        "future_tasks": ["sttc_bin", "cd_bin"],
        "description": "Two-step Stage-III ECG: adapt on Chapman rhythm/abnormality, add PTB-XL ST-T change (step 1), then add PTB-XL conduction disturbance (step 2).",
    },
    "chapman2_to_ptbxl_morph_seq": {
        "dataset": "chapman_to_ptbxl",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "chap_rhythm4": 10.0,
            "chap_abnormal_bin": 8.0,
            "sttc_bin": 5.0,
            "cd_bin": 5.0,
        },
        "task_defs": {
            "chap_rhythm4": {"dataset": "chapman"},
            "chap_abnormal_bin": {"dataset": "chapman"},
            "sttc_bin": {"dataset": "ptbxl"},
            "cd_bin": {"dataset": "ptbxl"},
        },
        "initial_tasks": ["chap_rhythm4"],
        "future_tasks": ["chap_abnormal_bin", "sttc_bin", "cd_bin"],
        "description": "Pipeline A (pure CL) ECG: start from one initial task, then sequentially add chap_abnormal_bin, sttc_bin, cd_bin.",
    },
    "ptbxl_to_chapman_shift": {
        "dataset": "ptbxl_to_chapman",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "super5": 8.0,
            "rhythm12": 10.0,
            "mi_bin": 5.0,
            "chap_afib_bin": 10.0,
            "chap_sbrady_bin": 10.0,
        },
        "task_defs": {
            "super5": {"dataset": "ptbxl"},
            "rhythm12": {"dataset": "ptbxl"},
            "mi_bin": {"dataset": "ptbxl"},
            "chap_afib_bin": {"dataset": "chapman"},
            "chap_sbrady_bin": {"dataset": "chapman"},
        },
        "initial_tasks": ["super5", "rhythm12", "mi_bin"],
        "future_tasks": ["chap_afib_bin", "chap_sbrady_bin"],
        "description": "Primary ECG shifted candidate: adapt on PTB-XL diagnosis and rhythm structure, then probe Chapman AF/brady rhythm tasks under a dataset shift.",
    },
    "chapman_to_ptbxl_shift": {
        "dataset": "chapman_to_ptbxl",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "chap_rhythm4": 10.0,
            "chap_abnormal_bin": 8.0,
            "chap_stachy_bin": 4.0,
            "afib_bin": 10.0,
            "sttc_bin": 5.0,
        },
        "task_defs": {
            "chap_rhythm4": {"dataset": "chapman"},
            "chap_abnormal_bin": {"dataset": "chapman"},
            "chap_stachy_bin": {"dataset": "chapman"},
            "afib_bin": {"dataset": "ptbxl"},
            "sttc_bin": {"dataset": "ptbxl"},
        },
        "initial_tasks": ["chap_rhythm4", "chap_abnormal_bin", "chap_stachy_bin"],
        "future_tasks": ["afib_bin", "sttc_bin"],
        "description": "Reverse ECG shift candidate: adapt on Chapman rhythm/abnormality structure, then probe PTB-XL AF and ST-T change tasks.",
    },
    "ptbxl_to_cinc_af_shift": {
        "dataset": "ptbxl_to_cinc",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "super5": 8.0,
            "rhythm12": 10.0,
            "mi_bin": 5.0,
            "cinc_af_bin": 10.0,
        },
        "task_defs": {
            "super5": {"dataset": "ptbxl"},
            "rhythm12": {"dataset": "ptbxl"},
            "mi_bin": {"dataset": "ptbxl"},
            "cinc_af_bin": {"dataset": "cinc"},
        },
        "initial_tasks": ["super5", "rhythm12", "mi_bin"],
        "future_tasks": ["cinc_af_bin"],
        "description": "Auxiliary high-shift candidate: adapt on PTB-XL, then probe single-lead CinC2017 AF detection with a new head only.",
    },
}


@dataclass
class ECGTaskData:
    runtime: TaskRuntime
    signals: Any
    split_idx: Dict[str, np.ndarray]
    split_y: Dict[str, np.ndarray]
    input_len: int


class ECGIndexedDataset(Dataset):
    def __init__(self, sigs, idx: np.ndarray, y: np.ndarray, runtime: TaskRuntime, input_len: int | None = None) -> None:
        self.sigs = sigs
        self.idx = np.asarray(idx)
        self.runtime = runtime
        self.input_len = int(input_len or 0)
        if runtime.spec.kind == "reg":
            y = (np.asarray(y).astype(np.float32) - runtime.y_mean) / runtime.y_std
            self.y = torch.as_tensor(y, dtype=torch.float32)
        else:
            self.y = torch.as_tensor(np.asarray(y).astype(np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self.idx))

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = np.asarray(self.sigs[int(self.idx[i])], dtype=np.float32)
        if self.input_len > 0 and x.shape[-1] > self.input_len:
            start = max(0, int((x.shape[-1] - self.input_len) // 2))
            x = np.ascontiguousarray(x[..., start : start + self.input_len])
        x = zscore_batch(x[None])[0].astype(np.float32, copy=False)
        return torch.from_numpy(x), self.y[i]


class ECGStudentFeatureEncoder(nn.Module):
    """Expose the feature vector before the Stage-1 KD projection head."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, feat = self.backbone(x)
        return feat


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+", default=["ptbxl_core"], choices=sorted(PACKS))
    ap.add_argument("--students", nargs="+", default=["BF10_NB7"])
    ap.add_argument("--stage1-method", default="lifefm_kd", choices=["lifefm_kd", "subspace_mse", "full_mse", "lower_bound"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--stage2-methods", nargs="+", default=list(STAGE2_METHODS), choices=STAGE2_METHODS)
    ap.add_argument("--stage3-policy", default="naive_ft", choices=STAGE3_POLICIES)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--stage2-epochs", type=int, default=30)
    ap.add_argument("--stage2-patience", type=int, default=6)
    ap.add_argument("--stage2-head-lr", type=float, default=1e-3)
    ap.add_argument("--stage2-encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--stage3-epochs", type=int, default=20)
    ap.add_argument("--stage3-patience", type=int, default=5)
    ap.add_argument("--stage3-head-lr", type=float, default=1e-3)
    ap.add_argument("--stage3-encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--stage2-scope", "--care-stage2-scope", dest="care_stage2_scope", choices=["none", "full", "top1", "top2"], default="top2")
    ap.add_argument("--stage3-scope", "--care-stage3-scope", dest="care_stage3_scope", choices=["none", "full", "top1", "top2"], default="top1")
    ap.add_argument("--lwf-old-distill-lambda", type=float, default=0.25)
    ap.add_argument("--care-stage3-anchor-lambda", type=float, default=0.1)
    ap.add_argument("--lambda-evo-core", "--core-lambda", dest="core_lambda", type=float, default=0.1)
    ap.add_argument("--core-mode", type=str, default="refresh")
    ap.add_argument("--core-budget", type=int, default=0)
    ap.add_argument("--ewc-lambda", type=float, default=1000.0)
    ap.add_argument("--cpca-core-k", type=int, default=0)
    ap.add_argument("--stage1-ckpt-seed", type=int, default=None)
    ap.add_argument("--stage2-cache-dir", type=Path, default=None)
    ap.add_argument("--oracle-cache-dir", type=Path, default=None)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--limit-train", type=int, default=20_000)
    ap.add_argument("--limit-val", type=int, default=5_000)
    ap.add_argument("--limit-test", type=int, default=5_000)
    ap.add_argument("--fresh", action="store_true")
    return ap.parse_args()


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


def maybe_limit(idx: np.ndarray, y: np.ndarray, limit: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if limit <= 0 or len(idx) <= limit:
        return idx, y
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(idx), limit, replace=False))
    return idx[keep], y[keep]


def resolve_input_len(sec_map: Dict[str, float], task_name: str, native_len: int, hz: int) -> int:
    target_sec = sec_map.get(task_name)
    if target_sec is None:
        return int(native_len)
    return min(int(native_len), max(1, int(round(float(target_sec) * float(hz)))))


def ensure_chapman_extra_labels() -> None:
    required = [
        CHAP_DIR / "labels_afib_bin.npy",
        CHAP_DIR / "labels_sbrady_bin.npy",
        CHAP_DIR / "labels_stachy_bin.npy",
    ]
    if all(p.exists() for p in required):
        return

    import wfdb

    print("[chapman] deriving extra Stage2/3 labels ...", flush=True)
    CHAP_DIR.mkdir(parents=True, exist_ok=True)
    signals = np.load(CHAP_DIR / "signals_500hz.npy", mmap_mode="r")
    cond = pd.read_csv(CHAP_RAW / "ConditionNames_SNOMED-CT.csv")
    snomed_to_acro = dict(zip(cond["Snomed_CT"].astype(str), cond["Acronym Name"].astype(str)))
    afib_set = {"AF", "AFIB"}
    sb_set = {"SB"}
    st_set = {"ST"}

    y_afib: List[int] = []
    y_sb: List[int] = []
    y_st: List[int] = []
    rec_paths = sorted((CHAP_RAW / "WFDBRecords").rglob("*.hea"))
    for i, hp in enumerate(rec_paths):
        if i % 5000 == 0:
            print(f"  [chapman] {i}/{len(rec_paths)}", flush=True)
        try:
            sig, meta = wfdb.rdsamp(str(hp)[:-4])
        except Exception:
            continue
        if sig.shape[1] < 12 or sig.shape[0] < 5000:
            continue
        dx = ""
        for c in meta.get("comments", []):
            if c.startswith("Dx:"):
                dx = c[3:].strip()
                break
        if not dx:
            continue
        codes = [c.strip() for c in dx.split(",") if c.strip()]
        acros = {snomed_to_acro.get(c, "UNK") for c in codes}
        y_afib.append(int(bool(acros & afib_set)))
        y_sb.append(int(bool(acros & sb_set)))
        y_st.append(int(bool(acros & st_set)))

    n_expected = int(signals.shape[0])
    for name, arr in {
        "labels_afib_bin.npy": np.asarray(y_afib, dtype=np.int64),
        "labels_sbrady_bin.npy": np.asarray(y_sb, dtype=np.int64),
        "labels_stachy_bin.npy": np.asarray(y_st, dtype=np.int64),
    }.items():
        if len(arr) != n_expected:
            raise RuntimeError(f"Chapman label mismatch for {name}: got {len(arr)} expected {n_expected}")
        np.save(CHAP_DIR / name, arr)
        print(f"  [chapman] saved {name} pos_rate={float(arr.mean()):.3f}", flush=True)


def load_chapman_task(task_name: str) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, str]:
    ensure_chapman_extra_labels()
    mapping = {
        "chap_rhythm4": ("labels_rhythm4.npy", "cls"),
        "chap_abnormal_bin": ("labels_abnormal_bin.npy", "bin"),
        "chap_afib_bin": ("labels_afib_bin.npy", "bin"),
        "chap_sbrady_bin": ("labels_sbrady_bin.npy", "bin"),
        "chap_stachy_bin": ("labels_stachy_bin.npy", "bin"),
    }
    if task_name not in mapping:
        raise ValueError(f"Unknown Chapman task: {task_name}")
    label_fn, kind = mapping[task_name]
    sigs = np.load(CHAP_DIR / "signals_500hz.npy", mmap_mode="r")
    fold = np.load(CHAP_DIR / "strat_fold.npy")
    y = np.load(CHAP_DIR / label_fn)
    valid = np.isfinite(y) if kind == "reg" else (y >= 0)
    sig_valid_path = CHAP_DIR / "_sig_valid.npy"
    if sig_valid_path.exists():
        valid = valid & np.load(sig_valid_path).astype(bool)
    return sigs, y, fold, valid, kind


def resolve_pack_task_defs(cfg: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    task_defs_raw = cfg.get("task_defs")
    if not task_defs_raw:
        dataset = str(cfg["dataset"])
        return {
            str(task_name): {"dataset": dataset}
            for task_name in list(cfg["initial_tasks"]) + list(cfg["future_tasks"])
        }

    task_defs: Dict[str, Dict[str, str]] = {}
    for task_name, raw in dict(task_defs_raw).items():
        task_defs[str(task_name)] = {"dataset": str(raw["dataset"])}
    return task_defs


def load_ecg_task(task_name: str, dataset: str) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, str]:
    if dataset == "chapman":
        return load_chapman_task(task_name)
    return load_task(task_name)


def load_pack_data(pack_name: str, args: argparse.Namespace) -> Tuple[List[ECGTaskData], List[ECGTaskData]]:
    cfg = PACKS[pack_name]
    all_names = list(cfg["initial_tasks"]) + list(cfg["future_tasks"])
    sec_map = {str(k): float(v) for k, v in (cfg.get("input_window_sec_map") or {}).items()}
    task_defs = resolve_pack_task_defs(cfg)
    task_data_all: Dict[str, ECGTaskData] = {}
    for task_name in all_names:
        task_dataset = str(task_defs[task_name]["dataset"])
        sigs, y, fold, valid, kind = load_ecg_task(task_name, task_dataset)
        idx_pool = np.where(valid)[0]
        tr_i, va_i, te_i = split_by_fold(idx_pool, fold)
        y_tr = np.asarray(y[tr_i])
        y_va = np.asarray(y[va_i])
        y_te = np.asarray(y[te_i])
        tr_i, y_tr = maybe_limit(tr_i, y_tr, args.limit_train, 11)
        va_i, y_va = maybe_limit(va_i, y_va, args.limit_val, 17)
        te_i, y_te = maybe_limit(te_i, y_te, args.limit_test, 23)

        if kind == "reg":
            y_mean = float(np.mean(y_tr))
            y_std = float(np.std(y_tr))
            if not np.isfinite(y_std) or y_std < 1e-6:
                y_std = 1.0
            runtime = TaskRuntime(TaskSpec(task_name, "reg", "", "mae"), n_out=1, y_mean=y_mean, y_std=y_std)
        else:
            n_out = int(np.max(np.concatenate([y_tr, y_va, y_te])) + 1)
            runtime = TaskRuntime(TaskSpec(task_name, "cls", "", "f1_macro"), n_out=n_out)

        task_data_all[task_name] = ECGTaskData(
            runtime=runtime,
            signals=sigs,
            split_idx={"train": tr_i, "val": va_i, "test": te_i},
            split_y={"train": y_tr, "val": y_va, "test": y_te},
            input_len=resolve_input_len(sec_map, task_name, int(sigs.shape[-1]), 500),
        )

    return [task_data_all[name] for name in cfg["initial_tasks"]], [task_data_all[name] for name in cfg["future_tasks"]]


def make_task_loaders_ecg(task_data: List[ECGTaskData], batch_size: int) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, DataLoader]]:
    train, val, test = {}, {}, {}
    for td in task_data:
        name = td.runtime.spec.name
        train[name] = DataLoader(
            ECGIndexedDataset(td.signals, td.split_idx["train"], td.split_y["train"], td.runtime, input_len=td.input_len),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val[name] = DataLoader(
            ECGIndexedDataset(td.signals, td.split_idx["val"], td.split_y["val"], td.runtime, input_len=td.input_len),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        test[name] = DataLoader(
            ECGIndexedDataset(td.signals, td.split_idx["test"], td.split_y["test"], td.runtime, input_len=td.input_len),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    return train, val, test


def load_stage1_ecgfounder(student: str, method: str, seed: int, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt_dir = DEFAULT_CKPT_DIRS[student]
    ckpt_path = ckpt_dir / f"{method}_seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing Stage-1 ECGFounder checkpoint: {ckpt_path}")
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    full, feat_dim = build_student(student, seed)
    full.load_state_dict(obj["state_dict"], strict=True)
    enc = ECGStudentFeatureEncoder(full.enc).to(device)
    meta = dict(obj)
    meta.pop("state_dict", None)
    meta["ckpt_path"] = str(ckpt_path)
    meta["enc_dim"] = int(obj.get("feat_dim", feat_dim))
    return enc, meta


def _encoder_and_other_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    enc_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            enc_params.append(p)
        else:
            other_params.append(p)
    return enc_params, other_params


def make_optimizer(model: nn.Module, head_lr: float, encoder_lr_mult: float, weight_decay: float) -> torch.optim.Optimizer:
    enc_params, other_params = _encoder_and_other_params(model)
    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": head_lr * encoder_lr_mult, "weight_decay": weight_decay})
    if other_params:
        groups.append({"params": other_params, "lr": head_lr, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups)


def set_stage2_trainable(model: MultiTaskModel, method: str, care_stage2_scope: str = "top2", care_auto_stage2_scope: str = "full_encoder") -> None:
    if method in CARE_AUTO_STAGE2_METHODS:
        set_encoder_trainable_scope(model, _care_auto_trainable_scope(care_auto_stage2_scope))
    elif method == "scope_matched_naive":
        _set_encoder_trainable_names(model, _care_v3_candidate_param_names(model, care_stage2_scope))
    else:
        for p in model.encoder.parameters():
            p.requires_grad = method != "frozen_heads"
    for p in model.heads.parameters():
        p.requires_grad = True


def fit_stage2(
    model: MultiTaskModel,
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
    log_prefix: str,
    care_stage2_scope: str = "top2",
    care_auto_stage2_scope: str = "full_encoder",
    care_auto_anchor_source: str = "source_union",
    care_auto_reuse_k: int = 0,
    care_auto_reuse_lambda: float = 0.05,
    care_auto_reuse_loss_type: str = "mse",
    care_auto_anchor_max_batches: int = 8,
    care_auto_anchor_basis_samples: int = 0,
    care_auto_energy_threshold: float = 0.95,
    care_auto_max_rank_ratio: float = 0.5,
    care_auto_min_rank: int = 4,
    enable_early_stop: bool = True,
    restore_best_checkpoint: bool = True,
    task_loss_reduction: str = "mean",
    fairgrad_alpha: float = 1.0,
) -> Tuple[MultiTaskModel, Dict[str, Dict[str, float]], Dict[str, Any]]:
    if task_loss_reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported task_loss_reduction: {task_loss_reduction}")
    model = model.to(device)
    set_stage2_trainable(model, method, care_stage2_scope=care_stage2_scope, care_auto_stage2_scope=care_auto_stage2_scope)
    init_regression_biases(model, tasks)
    opt = make_optimizer(model, head_lr, encoder_lr_mult, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    task_logits: nn.Parameter | None = None
    task_opt: torch.optim.Optimizer | None = None
    init_losses: torch.Tensor | None = None
    if method == "gradnorm_joint":
        task_logits = nn.Parameter(torch.zeros(len(tasks), device=device))
        task_opt = torch.optim.Adam([task_logits], lr=2.5e-2)

    anchor_encoder: Optional[nn.Module] = None
    care_auto_basis: Optional[torch.Tensor] = None
    care_auto_anchor_mean: Optional[torch.Tensor] = None
    care_auto_anchor_std: Optional[torch.Tensor] = None
    care_auto_active_loss_type = _resolve_care_auto_reuse_loss_type(method, care_auto_reuse_loss_type)
    if method in CARE_AUTO_STAGE2_METHODS:
        anchor_encoder = copy.deepcopy(model.encoder).to(device).eval()
        anchor_encoder.requires_grad_(False)
        care_auto_basis_info = _build_care_auto_anchor_basis(
            model=model,
            anchor_encoder=anchor_encoder,
            tasks=tasks,
            train_loaders=train_loaders,
            device=device,
            feature_space="encoder_pooled",
            anchor_source=care_auto_anchor_source,
            requested_k=int(care_auto_reuse_k),
            max_batches=int(care_auto_anchor_max_batches),
            requested_samples=int(care_auto_anchor_basis_samples),
            public_pool_path="",
            energy_threshold=float(care_auto_energy_threshold),
            max_rank_ratio=float(care_auto_max_rank_ratio),
            min_rank=int(care_auto_min_rank),
        )
        care_auto_basis = care_auto_basis_info["basis"]
        care_auto_anchor_mean = care_auto_basis_info["anchor_mean"]
        care_auto_anchor_std = care_auto_basis_info["anchor_std"]
        _auto_tag = "(auto_energy)" if care_auto_basis_info.get("auto_energy_selected") else "(fixed)"
        print(
            f"    {log_prefix}CARE-v1 anchor basis: k={care_auto_basis_info['reuse_k']}{_auto_tag}"
            f" samples={care_auto_basis_info['num_anchor_samples']}"
            f" lambda={care_auto_reuse_lambda} loss={care_auto_active_loss_type}",
            flush=True,
        )

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    last_aux = 0.0
    last_fair_weights: np.ndarray | None = None
    last_fair_gtg_trace = float("nan")

    for ep in range(epochs):
        model.train()
        train_losses: List[float] = []
        task_iters = {rt.spec.name: iter(train_loaders[rt.spec.name]) for rt in tasks}
        n_steps = max(len(train_loaders[rt.spec.name]) for rt in tasks)
        for _ in range(n_steps):
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
                from run_stage2_student_subspace import assign_flat_grad, flatten_grads, pcgrad_merge

                opt.zero_grad(set_to_none=True)
                params = [p for p in model.parameters() if p.requires_grad]
                task_grads: List[torch.Tensor] = []
                task_scalars: List[float] = []
                for i, (rt, xb, yb) in enumerate(batches):
                    _, out = model(xb, task_name=rt.spec.name)
                    loss_i = task_losses(out, {rt.spec.name: yb}, [rt])[rt.spec.name]
                    grads = torch.autograd.grad(loss_i, params, retain_graph=i < len(batches) - 1, allow_unused=True)
                    task_grads.append(flatten_grads(grads, params))
                    task_scalars.append(float(loss_i.detach().item()))
                assign_flat_grad(params, pcgrad_merge(torch.stack(task_grads)))
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                train_losses.append(float(np.mean(task_scalars)))
                continue

            per_task_losses: List[torch.Tensor] = []
            care_anchor_terms: List[torch.Tensor] = []
            for rt, xb, yb in batches:
                z, out = model(xb, task_name=rt.spec.name)
                per_task_losses.append(task_losses(out, {rt.spec.name: yb}, [rt])[rt.spec.name])
                if method in CARE_AUTO_STAGE2_METHODS and anchor_encoder is not None:
                    with torch.no_grad():
                        z_anchor = model.encode_with_encoder(anchor_encoder, xb, task_name=rt.spec.name)
                    assert care_auto_basis is not None
                    reuse_term, _, _, _ = _care_auto_reuse_terms(
                        current_features=z,
                        anchor_features=z_anchor,
                        basis=care_auto_basis,
                        loss_type=care_auto_active_loss_type,
                        anchor_mean=care_auto_anchor_mean,
                        anchor_std=care_auto_anchor_std,
                    )
                    care_anchor_terms.append(reuse_term)
            if not per_task_losses:
                continue

            if method == "gradnorm_joint":
                assert task_logits is not None and task_opt is not None
                losses_tensor = torch.stack(per_task_losses)
                if init_losses is None:
                    init_losses = losses_tensor.detach().clamp_min(1e-6)
                weights = len(tasks) * torch.softmax(task_logits, dim=0)
                enc_params = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("encoder.")]
                from run_stage2_student_subspace import flatten_grads
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
                target = G.detach().mean() * inv_rate.pow(1.5)
                gradnorm_loss = torch.abs(G - target).sum()
                total_loss = torch.sum(weights.detach() * losses_tensor)

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

            if method == "fairgrad_joint":
                from run_stage2_student_subspace import assign_flat_grad, flatten_grads

                losses_tensor = torch.stack(per_task_losses)
                encoder_params = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("encoder.")]
                head_params = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("encoder.")]
                if encoder_params:
                    task_grads: List[torch.Tensor] = []
                    for loss_i in per_task_losses:
                        grads = torch.autograd.grad(
                            loss_i,
                            encoder_params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        task_grads.append(flatten_grads(grads, encoder_params))
                    grad_stack = torch.stack(task_grads)
                    fair_gram = torch.mm(grad_stack, grad_stack.t())
                    gram_np = fair_gram.detach().cpu().numpy()
                    x_start = np.full(len(per_task_losses), 1.0 / max(len(per_task_losses), 1), dtype=np.float64)

                    def _fairgrad_residual(weights: np.ndarray) -> np.ndarray:
                        safe_weights = np.clip(weights, 1e-8, None)
                        return gram_np.dot(safe_weights) - np.power(1.0 / safe_weights, 1.0 / max(float(fairgrad_alpha), 1e-8))

                    try:
                        res = least_squares(_fairgrad_residual, x_start, bounds=(0.0, np.inf))
                        weights_np = np.asarray(res.x, dtype=np.float32)
                    except Exception:
                        weights_np = x_start.astype(np.float32)
                    if not np.all(np.isfinite(weights_np)) or float(weights_np.sum()) <= 0.0:
                        weights_np = x_start.astype(np.float32)

                    fair_weights = torch.as_tensor(weights_np, device=device, dtype=grad_stack.dtype)
                    merged_grad = torch.sum(grad_stack * fair_weights.unsqueeze(1), dim=0)
                    head_grads = list(torch.autograd.grad(losses_tensor.sum(), head_params, allow_unused=True)) if head_params else []

                    opt.zero_grad(set_to_none=True)
                    for p, g in zip(head_params, head_grads):
                        p.grad = None if g is None else g.detach().clone()
                    assign_flat_grad(encoder_params, merged_grad)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                    opt.step()

                    last_fair_weights = weights_np.copy()
                    last_fair_gtg_trace = float(np.trace(gram_np)) if gram_np.size else float("nan")
                    last_aux = float(np.linalg.norm(weights_np, ord=1))
                    train_losses.append(float(losses_tensor.sum().detach().item()))
                    continue

                loss = losses_tensor.sum() if task_loss_reduction == "sum" else losses_tensor.mean()
            elif method in CARE_AUTO_STAGE2_METHODS:
                losses_tensor = torch.stack(per_task_losses)
                source_loss = losses_tensor.sum() if task_loss_reduction == "sum" else losses_tensor.mean()
                anchor_loss = torch.stack(care_anchor_terms).mean() if care_anchor_terms else torch.zeros((), device=device)
                loss = source_loss + float(care_auto_reuse_lambda) * anchor_loss
                last_aux = float(anchor_loss.detach().item())
            else:
                losses_tensor = torch.stack(per_task_losses)
                loss = losses_tensor.sum() if task_loss_reduction == "sum" else losses_tensor.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            train_losses.append(float(loss.detach().item()))

        sched.step()
        val = validation_loss(model, val_loaders, tasks, device)
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if val <= best_val or (ep + 1) % 10 == 0:
            extra = ""
            if method in CARE_AUTO_STAGE2_METHODS:
                extra = f" anchor={last_aux:.4g}"
            elif method == "gradnorm_joint":
                extra = f" gradnorm={last_aux:.4g}"
            elif method == "fairgrad_joint":
                if last_fair_weights is not None and len(last_fair_weights):
                    extra = (
                        f" fair_alpha={fairgrad_alpha:.3f}"
                        f" wmin={float(np.min(last_fair_weights)):.4f}"
                        f" wmax={float(np.max(last_fair_weights)):.4f}"
                        f" gtg_trace={last_fair_gtg_trace:.4g}"
                    )
                else:
                    extra = f" fair_alpha={fairgrad_alpha:.3f}"
            print(
                f"    {log_prefix}ep{ep:03d} train={np.mean(train_losses):.4f} "
                f"val_loss={val:.4f} best={best_val:.4f} bad={bad}{extra}",
                flush=True,
            )
        if enable_early_stop and bad >= patience:
            print(f"    {log_prefix}early-stop @ ep{ep}", flush=True)
            break

    if restore_best_checkpoint:
        model.load_state_dict(best_state)
    metrics = evaluate(model, test_loaders, tasks, device)
    return model, metrics, {
        "val_loss_best": float(best_val),
        "epochs_trained": int(ep + 1),
        "early_stop_enabled": bool(enable_early_stop),
        "stopped_early": bool(enable_early_stop and bad >= patience),
        "checkpoint_used": "best" if restore_best_checkpoint else "final",
        "fairgrad_alpha": float(fairgrad_alpha),
        "fairgrad_last_weights": last_fair_weights.tolist() if last_fair_weights is not None else [],
        "fairgrad_last_gtg_trace": float(last_fair_gtg_trace),
        "care_auto_last_anchor_loss": float(last_aux) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "care_auto_reuse_lambda": float(care_auto_reuse_lambda) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "care_auto_basis_k": int(care_auto_basis.shape[1]) if care_auto_basis is not None else 0,
    }


def set_stage3_trainable(model: MultiTaskModel, new_task_name: str, stage3_policy: str, care_stage3_scope: str = "top1") -> None:
    if stage3_policy == "scope_matched_naive":
        _set_encoder_trainable_names(model, _care_v3_candidate_param_names(model, care_stage3_scope))
    else:
        for p in model.encoder.parameters():
            p.requires_grad = stage3_policy in {"naive_ft", "lwf_ft", "anchor_only", "lpr", "cpca_core", "cpca_ts", "cpca_ca", "cpca_tsca", "lwf_core", "lifefm_evolve", "lpr_core", "core_dwdp", "core_affinity", "core_affinity_hybrid", "core_task", "lwp_task", "ewc", "sgp", "sgp_core"}
    for name, head in model.heads.items():
        req = name == new_task_name
        for p in head.parameters():
            p.requires_grad = req



_SGP_SCALE = 0.0
_SGP_BASIS_SIZE = 32
_SGP_GRAD_BATCHES = 16


def _sgp_trainable_encoder_params(model):
    return [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]


def _sgp_flat_grads(params):
    gs = [p.grad.detach().reshape(-1) for _, p in params if p.grad is not None]
    return torch.cat(gs) if gs else None


def _sgp_project_encoder_grads(model, basis, scale: float = 0.0) -> None:
    if basis is None or basis.numel() == 0:
        return
    params = _sgp_trainable_encoder_params(model)
    g = _sgp_flat_grads(params)
    if g is None:
        return
    B = basis.to(device=g.device, dtype=g.dtype)
    coeff = B @ g
    g = g - (1.0 + float(scale)) * (coeff @ B)
    off = 0
    for _, p in params:
        if p.grad is None:
            continue
        n = p.numel()
        p.grad.copy_(g[off:off + n].view_as(p))
        off += n


def _sgp_append_basis(model, loader, device, task, size: int, n_batches: int) -> None:
    was_training = model.training
    model.eval()
    rows = []
    for i, (xb, yb) in enumerate(loader):
        if i >= n_batches:
            break
        xb = xb.to(device); yb = yb.to(device)
        model.zero_grad(set_to_none=True)
        _, out = model(xb, task_name=task.spec.name)
        loss = task_losses(out, {task.spec.name: yb}, [task])[task.spec.name]
        loss.backward()
        g = _sgp_flat_grads(_sgp_trainable_encoder_params(model))
        if g is not None:
            rows.append(g.detach())
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    if not rows:
        return
    M = torch.stack(rows, dim=0)
    old = getattr(model, "_sgp_basis", None)
    if old is not None and old.numel() > 0:
        M = torch.cat([old.to(device=M.device, dtype=M.dtype), M], dim=0)
    Q, _ = torch.linalg.qr(M.T)
    Q = Q.T.contiguous()
    if Q.shape[0] > size:
        Q = Q[:size]
    model._sgp_basis = Q.detach()


def fit_new_task_extension(
    model: MultiTaskModel,
    old_tasks: List[TaskRuntime],
    new_task_data: ECGTaskData,
    device: torch.device,
    epochs: int,
    patience: int,
    head_lr: float,
    encoder_lr_mult: float,
    weight_decay: float,
    stage3_policy: str,
    batch_size: int,
    log_prefix: str,
    care_stage3_scope: str = "top1",
    lwf_old_distill_lambda: float = 0.25,
    care_stage3_anchor_lambda: float = 0.1,
    old_task_data: Optional[List[Any]] = None,
    cpca_core_k: int = 0,
    lpr_buffer_size: int = 256,
    lpr_alpha: float = 1.0,
    lpr_omega0: float = 1.0,
    lpr_refresh_interval: int = 10,
    lpr_lr: float = 1e-3,
    seed: int = 0,
    lwp_lam_old: float = 1.0,
    lwp_lam_dwdp: float = 0.01,
    core_lambda: float = 0.1,
    core_mode: str = "refresh",
    core_budget: int = 0,
    ewc_lambda: float = 1000.0,
) -> Tuple[MultiTaskModel, Dict[str, Dict[str, float]], Dict[str, Any]]:
    model = model.to(device)
    stage2_anchor_model: MultiTaskModel | None = None
    if stage3_policy in {"lwf_ft", "anchor_only", "cpca_core", "lwp", "cpca_ts", "cpca_ca", "cpca_tsca", "lwf_core", "lifefm_evolve", "lpr_core", "core_dwdp", "core_affinity", "core_affinity_hybrid", "core_task", "lwp_task", "ewc", "sgp_core"}:
        stage2_anchor_model = copy.deepcopy(model).to(device)
        stage2_anchor_model.eval()
        stage2_anchor_model.requires_grad_(False)
    set_stage3_trainable(model, new_task_data.runtime.spec.name, stage3_policy, care_stage3_scope=care_stage3_scope)
    init_regression_biases(model, [new_task_data.runtime])
    train_loaders, val_loaders, _ = make_task_loaders_ecg([new_task_data], batch_size)
    if stage3_policy in {"lpr", "lpr_core"}:
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lpr_lr, momentum=0.0, weight_decay=0.0)
    else:
        opt = make_optimizer(model, head_lr, encoder_lr_mult, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    lpr_layers = []
    lpr_X_list = []
    lpr_y_list = []
    lpr_runtimes = []
    lpr_precond = {}
    lpr_step = 0
    lpr_step_rng = None
    if stage3_policy in {"lpr", "lpr_core"}:
        lpr_layers = _lpr_preconditionable_layers(model.encoder)
        _rng = np.random.default_rng(int(seed))
        lpr_X_list, lpr_y_list, lpr_runtimes = _lpr_build_replay_ecg(
            list(old_task_data) if old_task_data else list(old_tasks), lpr_buffer_size, _rng
        )
        lpr_step_rng = np.random.default_rng(int(seed) + 9999)
        lpr_omega = lpr_omega0 / max(1, sum(int(X.shape[0]) for X in lpr_X_list))
        _nt_xb, _ = next(iter(train_loaders[new_task_data.runtime.spec.name]))
        print(
            f"[ecg-lpr] new-task xb={tuple(_nt_xb.shape)} {_nt_xb.dtype} | replay_shapes={[tuple(X.shape) for X in lpr_X_list]} "
            f"replay_dtypes={[str(X.dtype) for X in lpr_X_list]} input_len={new_task_data.input_len} layers={len(lpr_layers)}",
            flush=True,
        )
        lpr_precond = _lpr_build_preconditioners(model.encoder, lpr_layers, lpr_X_list, lpr_omega, device)

    cpca_basis = None
    cpca_k = 0
    if stage3_policy in {"cpca_core", "cpca_ts", "cpca_ca", "cpca_tsca", "lwf_core", "lifefm_evolve", "lpr_core", "core_dwdp", "core_affinity", "core_affinity_hybrid", "core_task", "lwp_task", "ewc", "sgp_core"} and stage2_anchor_model is not None:
        _old_td = list(old_task_data) if old_task_data else list(old_tasks)
        _old_train = make_task_loaders_ecg(_old_td, batch_size)[0]
        if stage3_policy in {"core_task", "lwp_task"}:
            _bi = _old_task_head_basis(model, _old_td, device=device)
        else:
            _bi = _build_care_auto_anchor_basis(
            model=model,
            anchor_encoder=stage2_anchor_model.encoder,
            tasks=[td.runtime for td in _old_td],
            train_loaders=_old_train,
            device=device,
            feature_space="encoder_pooled",
            anchor_source="source_union",
            requested_k=int(cpca_core_k),
            max_batches=8,
            requested_samples=0,
            energy_threshold=0.95,
        )
        if stage3_policy in {"cpca_ts", "cpca_tsca"}:
            _k = int(cpca_core_k) if int(cpca_core_k) > 0 else int(_bi["reuse_k"])
            _bi = _cpca_ts_basis(model, stage2_anchor_model.encoder, [td.runtime for td in _old_td], _old_train, device, _k)
        cpca_basis = _bi["basis"]
        cpca_k = int(_bi["reuse_k"])
        if stage3_policy == "lifefm_evolve":
            _prev = getattr(model, "_carried_core", None)
            if _prev is not None:
                _prev = _prev.to(device=device, dtype=torch.float32)
                if core_mode == "fixed":
                    cpca_basis = _prev; cpca_k = int(_prev.shape[1])
                elif core_mode in {"fuse_grow", "fuse_budget"}:
                    _q = torch.linalg.qr(torch.cat([_prev, cpca_basis], dim=1))[0]
                    if core_mode == "fuse_budget" and int(core_budget) > 0 and _q.shape[1] > int(core_budget):
                        _q = _q[:, :int(core_budget)]
                    cpca_basis = _q.contiguous(); cpca_k = int(cpca_basis.shape[1])
            model._carried_core = cpca_basis.detach().clone().cpu()
        print(
            f"    {log_prefix}[{stage3_policy}] core basis k={cpca_k}",
            flush=True,
        )

    sema_wrapper = None
    if stage3_policy == "sema":
        from sema_stage3 import SEMAWrapper
        model.train()
        sema_wrapper = SEMAWrapper(model, None, n_layers=3, r=16, rd_dim=128, threshold=2.0).to(device)
        for p in model.parameters():
            p.requires_grad = False
        for p in sema_wrapper.parameters():
            p.requires_grad = True
        _head_params = [p for n, p in model.named_parameters() if n.startswith("heads.")]
        for p in _head_params:
            p.requires_grad = True
        opt = torch.optim.SGD(list(sema_wrapper.parameters()) + _head_params, lr=0.005, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    new_task = new_task_data.runtime
    old_task_map = {rt.spec.name: rt for rt in old_tasks}
    last_distill = 0.0
    last_anchor = 0.0

    for ep in range(epochs):
        model.train()
        losses_ep: List[float] = []
        for xb, yb in train_loaders[new_task.spec.name]:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if stage3_policy in {"lpr", "lpr_core"}:
                z, out = model(xb, task_name=new_task.spec.name)
                new_loss = task_losses(out, {new_task.spec.name: yb}, [new_task])[new_task.spec.name]
                replay_loss = _lpr_replay_loss(model, lpr_X_list, lpr_y_list, lpr_runtimes, device, lpr_step_rng, 32)
                if stage3_policy == "lpr_core":
                    with torch.no_grad():
                        z_anchor, _ = stage2_anchor_model(xb, task_name=new_task.spec.name)
                    core_loss = F.mse_loss(z @ cpca_basis, z_anchor @ cpca_basis)
                    loss = new_loss + lpr_alpha * replay_loss + core_lambda * core_loss
                else:
                    loss = new_loss + lpr_alpha * replay_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                _lpr_apply_preconditioners(model.encoder, lpr_layers, lpr_precond)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                opt.step()
                losses_ep.append(float(loss.detach().item()))
                lpr_step += 1
                if lpr_step % max(1, lpr_refresh_interval) == 0:
                    lpr_omega = lpr_omega0 / max(1, sum(int(X.shape[0]) for X in lpr_X_list))
                    lpr_precond = _lpr_build_preconditioners(model.encoder, lpr_layers, lpr_X_list, lpr_omega, device)
                last_anchor = float(replay_loss.detach().item())
                continue
            z, out = model(xb, task_name=new_task.spec.name)
            new_loss = task_losses(out, {new_task.spec.name: yb}, [new_task])[new_task.spec.name]
            if stage3_policy in {"sgp", "sgp_core"}:
                loss = new_loss
                if stage3_policy == "sgp_core":
                    with torch.no_grad():
                        z_anchor_sgp, _ = stage2_anchor_model(xb, task_name=new_task.spec.name)
                    core_term = F.mse_loss(z @ cpca_basis, z_anchor_sgp @ cpca_basis)
                    loss = loss + core_lambda * core_term
                    last_anchor = float(core_term.detach().item())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                _sgp_project_encoder_grads(model, getattr(model, "_sgp_basis", None), _SGP_SCALE)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                opt.step()
                losses_ep.append(float(loss.detach().item()))
                continue
            loss = new_loss
            if stage3_policy == "ewc":
                _fisher = getattr(model, "_ewc_fisher", None)
                _prev = getattr(model, "_ewc_params", None)
                _pen = torch.zeros((), device=device)
                if _fisher is not None and _prev is not None:
                    for n, p in model.encoder.named_parameters():
                        if n in _fisher and n in _prev:
                            _pen = _pen + (_fisher[n].to(device) * (p - _prev[n].to(device)) ** 2).sum()
                loss = new_loss + float(ewc_lambda) * _pen
                last_anchor = float(_pen.detach().item())
            elif stage3_policy == "lwf_ft":
                assert stage2_anchor_model is not None
                with torch.no_grad():
                    _, out_anchor = stage2_anchor_model(xb, task_name=new_task.spec.name)
                distill_terms: List[torch.Tensor] = []
                for old_name, old_rt in old_task_map.items():
                    if old_name not in out or old_name not in out_anchor:
                        continue
                    if old_rt.spec.kind == "cls":
                        distill_terms.append(
                            F.kl_div(
                                F.log_softmax(out[old_name], dim=-1),
                                F.softmax(out_anchor[old_name], dim=-1),
                                reduction="batchmean",
                            )
                        )
                    else:
                        distill_terms.append(F.mse_loss(out[old_name], out_anchor[old_name]))
                distill_loss = torch.stack(distill_terms).mean() if distill_terms else torch.zeros((), device=device)
                loss = new_loss + lwf_old_distill_lambda * distill_loss
                last_distill = float(distill_loss.detach().item())
            elif stage3_policy == "lwf_core":
                assert stage2_anchor_model is not None and cpca_basis is not None
                with torch.no_grad():
                    z_anchor, out_anchor = stage2_anchor_model(xb, task_name=new_task.spec.name)
                distill_terms = []
                for old_name, old_rt in old_task_map.items():
                    if old_name not in out or old_name not in out_anchor:
                        continue
                    if old_rt.spec.kind == "cls":
                        distill_terms.append(F.kl_div(F.log_softmax(out[old_name], dim=-1), F.softmax(out_anchor[old_name], dim=-1), reduction="batchmean"))
                    else:
                        distill_terms.append(F.mse_loss(out[old_name], out_anchor[old_name]))
                distill_loss = torch.stack(distill_terms).mean() if distill_terms else torch.zeros((), device=device)
                anchor_loss = F.mse_loss(z @ cpca_basis, z_anchor @ cpca_basis)
                loss = new_loss + lwf_old_distill_lambda * distill_loss + core_lambda * anchor_loss
                last_distill = float(distill_loss.detach().item()); last_anchor = float(anchor_loss.detach().item())
            elif stage3_policy == "anchor_only":
                assert stage2_anchor_model is not None
                with torch.no_grad():
                    z_anchor, _ = stage2_anchor_model(xb, task_name=new_task.spec.name)
                anchor_loss = F.mse_loss(z, z_anchor)
                loss = new_loss + care_stage3_anchor_lambda * anchor_loss
                last_anchor = float(anchor_loss.detach().item())
            elif stage3_policy in {"cpca_core", "cpca_ts", "cpca_ca", "cpca_tsca", "core_task"}:
                assert stage2_anchor_model is not None and cpca_basis is not None
                with torch.no_grad():
                    z_anchor, _ = stage2_anchor_model(xb, task_name=new_task.spec.name)
                anchor_loss = F.mse_loss(z @ cpca_basis, z_anchor @ cpca_basis)
                if stage3_policy in {"cpca_ca", "cpca_tsca"}:
                    _params = [p for p in model.parameters() if p.requires_grad]
                    _gn = torch.autograd.grad(new_loss, _params, retain_graph=True, allow_unused=True)
                    _gc = torch.autograd.grad(anchor_loss, _params, retain_graph=True, allow_unused=True)
                    _vn = torch.cat([(t if t is not None else torch.zeros(p.numel(), device=device)).reshape(-1) for t, p in zip(_gn, _params)])
                    _vc = torch.cat([(t if t is not None else torch.zeros(p.numel(), device=device)).reshape(-1) for t, p in zip(_gc, _params)])
                    _c = float((_vn @ _vc) / (_vn.norm() * _vc.norm() + 1e-8))
                    _lam = float(care_stage3_anchor_lambda) * (1.0 + max(0.0, -_c))
                    loss = new_loss + _lam * anchor_loss
                else:
                    loss = new_loss + care_stage3_anchor_lambda * anchor_loss
                last_anchor = float(anchor_loss.detach().item())
            elif stage3_policy == "sema":
                assert sema_wrapper is not None
                rd_loss = sema_wrapper.adapter_rd_loss()
                loss = new_loss + 0.1 * rd_loss
                last_anchor = float(rd_loss.detach().item())
            elif stage3_policy == "lwp_task":
                assert stage2_anchor_model is not None and cpca_basis is not None
                with torch.no_grad():
                    z_anchor, out_anchor = stage2_anchor_model(xb, task_name=new_task.spec.name)
                distill_terms = []
                for old_name, old_rt in old_task_map.items():
                    if old_name not in out or old_name not in out_anchor:
                        continue
                    if old_rt.spec.kind == "cls":
                        distill_terms.append(F.kl_div(F.log_softmax(out[old_name], dim=-1), F.softmax(out_anchor[old_name], dim=-1), reduction="batchmean"))
                    else:
                        distill_terms.append(F.mse_loss(out[old_name], out_anchor[old_name]))
                distill_loss = torch.stack(distill_terms).mean() if distill_terms else torch.zeros((), device=device)
                dwdp_loss = _dwdp_loss(z_anchor, z, yb, new_task.spec.kind)
                anchor_loss = F.mse_loss(z @ cpca_basis, z_anchor @ cpca_basis)
                loss = new_loss + lwp_lam_old * distill_loss + lwp_lam_dwdp * dwdp_loss + care_stage3_anchor_lambda * anchor_loss
                last_distill = float(distill_loss.detach().item()); last_anchor = float(dwdp_loss.detach().item())
            elif stage3_policy == "lwp":
                assert stage2_anchor_model is not None
                with torch.no_grad():
                    z_anchor, out_anchor = stage2_anchor_model(xb, task_name=new_task.spec.name)
                distill_terms: List[torch.Tensor] = []
                for old_name, old_rt in old_task_map.items():
                    if old_name not in out or old_name not in out_anchor:
                        continue
                    if old_rt.spec.kind == "cls":
                        distill_terms.append(
                            F.kl_div(F.log_softmax(out[old_name], dim=-1), F.softmax(out_anchor[old_name], dim=-1), reduction="batchmean")
                        )
                    else:
                        distill_terms.append(F.mse_loss(out[old_name], out_anchor[old_name]))
                distill_loss = torch.stack(distill_terms).mean() if distill_terms else torch.zeros((), device=device)
                dwdp_loss = _dwdp_loss(z_anchor, z, yb, new_task.spec.kind)
                loss = new_loss + lwp_lam_old * distill_loss + lwp_lam_dwdp * dwdp_loss
                last_distill = float(distill_loss.detach().item())
                last_anchor = float(dwdp_loss.detach().item())
            elif stage3_policy == "lifefm_evolve":
                assert stage2_anchor_model is not None and cpca_basis is not None
                with torch.no_grad():
                    z_anchor, out_anchor = stage2_anchor_model(xb, task_name=new_task.spec.name)
                distill_terms = []
                for old_name, old_rt in old_task_map.items():
                    if old_name not in out or old_name not in out_anchor:
                        continue
                    if old_rt.spec.kind == "cls":
                        distill_terms.append(F.kl_div(F.log_softmax(out[old_name], dim=-1), F.softmax(out_anchor[old_name], dim=-1), reduction="batchmean"))
                    else:
                        distill_terms.append(F.mse_loss(out[old_name], out_anchor[old_name]))
                distill_loss = torch.stack(distill_terms).mean() if distill_terms else torch.zeros((), device=device)
                dwdp_loss = _dwdp_loss(z_anchor, z, yb, new_task.spec.kind)
                anchor_loss = F.mse_loss(z @ cpca_basis, z_anchor @ cpca_basis)
                loss = new_loss + lwp_lam_old * distill_loss + lwp_lam_dwdp * dwdp_loss + core_lambda * anchor_loss
                last_distill = float(distill_loss.detach().item()); last_anchor = float(dwdp_loss.detach().item())
            elif stage3_policy in {"core_dwdp", "core_affinity", "core_affinity_hybrid"}:
                assert stage2_anchor_model is not None and cpca_basis is not None
                with torch.no_grad():
                    z_anchor, out_anchor = stage2_anchor_model(xb, task_name=new_task.spec.name)
                distill_terms = []
                for old_name, old_rt in old_task_map.items():
                    if old_name not in out or old_name not in out_anchor:
                        continue
                    if old_rt.spec.kind == "cls":
                        distill_terms.append(F.kl_div(F.log_softmax(out[old_name], dim=-1), F.softmax(out_anchor[old_name], dim=-1), reduction="batchmean"))
                    else:
                        distill_terms.append(F.mse_loss(out[old_name], out_anchor[old_name]))
                distill_loss = torch.stack(distill_terms).mean() if distill_terms else torch.zeros((), device=device)
                dwdp_loss = _core_dwdp_loss(z_anchor, z, yb, new_task.spec.kind, cpca_basis,
                                            affinity=(stage3_policy in {"core_affinity", "core_affinity_hybrid"}),
                                            hybrid=(stage3_policy == "core_affinity_hybrid"))
                loss = new_loss + lwp_lam_old * distill_loss + lwp_lam_dwdp * dwdp_loss
                last_distill = float(distill_loss.detach().item()); last_anchor = float(dwdp_loss.detach().item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            losses_ep.append(float(loss.detach().item()))
        sched.step()

        val = validation_loss(model, val_loaders, [new_task], device)
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if val <= best_val or (ep + 1) % 10 == 0:
            if stage3_policy == "lwf_ft":
                extra = f" distill={last_distill:.4g}"
            elif stage3_policy in {"anchor_only", "cpca_core"}:
                extra = f" anchor={last_anchor:.4g}"
            else:
                extra = ""
            print(
                f"    {log_prefix}ep{ep:03d} train={np.mean(losses_ep):.4f} "
                f"val_loss={val:.4f} best={best_val:.4f} bad={bad}{extra}",
                flush=True,
            )
        if bad >= patience:
            print(f"    {log_prefix}early-stop @ ep{ep}", flush=True)
            break

    model.load_state_dict(best_state)
    if stage3_policy in {"sgp", "sgp_core"}:
        _sgp_append_basis(model, train_loaders[new_task.spec.name], device, new_task, _SGP_BASIS_SIZE, _SGP_GRAD_BATCHES)
    if stage3_policy == "ewc":
        model.eval()
        _F = dict(getattr(model, "_ewc_fisher", {}) or {})
        _nb = 0
        for _i, (_xb, _yb) in enumerate(train_loaders[new_task_data.runtime.spec.name]):
            if _i >= 4:
                break
            _xb = _xb.to(device); _yb = _yb.to(device)
            _, _out = model(_xb, task_name=new_task_data.runtime.spec.name)
            _l = task_losses(_out, {new_task_data.runtime.spec.name: _yb}, [new_task_data.runtime])[new_task_data.runtime.spec.name]
            _gs = torch.autograd.grad(_l, [p for p in model.encoder.parameters()], allow_unused=True)
            for (n, p), gg in zip(model.encoder.named_parameters(), _gs):
                if gg is None:
                    continue
                _F[n] = (_F[n].to(device) if n in _F else 0) + gg.detach() ** 2
            _nb += 1
        model._ewc_fisher = {k: (v / max(1, _nb)).cpu() for k, v in _F.items()}
        model._ewc_params = {n: p.detach().clone().cpu() for n, p in model.encoder.named_parameters()}
        model.train()
    _, _, test_loaders = make_task_loaders_ecg([new_task_data], batch_size)
    metrics = evaluate(model, test_loaders, [new_task], device)
    return model, metrics, {"val_loss_best": float(best_val), "epochs_trained": int(ep + 1)}


def evaluate_all_tasks(model: MultiTaskModel, task_data_all: List[ECGTaskData], device: torch.device, batch_size: int) -> Dict[str, Dict[str, float]]:
    _, _, test_loaders = make_task_loaders_ecg(task_data_all, batch_size)
    return evaluate(model, test_loaders, [td.runtime for td in task_data_all], device)


def fit_stage1_single_task_oracle(
    student: str,
    seed: int,
    stage1_method: str,
    task_data: ECGTaskData,
    dataset_name: str,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    enc, ck_meta = load_stage1_ecgfounder(
        student, stage1_method,
        int(args.stage1_ckpt_seed) if getattr(args, "stage1_ckpt_seed", None) is not None else int(seed),
        device,
    )
    model = MultiTaskModel(
        enc,
        int(ck_meta["enc_dim"]),
        [task_data.runtime],
        args.dropout,
        dataset_name=dataset_name,
        hz=500,
        input_len=int(task_data.input_len),
        temporal_policy="shared_window",
    )
    train_loaders, val_loaders, test_loaders = make_task_loaders_ecg([task_data], args.batch_size)
    model, metrics, _ = fit_stage2(
        model=model,
        method="joint_finetune",
        tasks=[task_data.runtime],
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        device=device,
        epochs=args.stage3_epochs,
        patience=args.stage3_patience,
        head_lr=args.stage3_head_lr,
        encoder_lr_mult=args.stage3_encoder_lr_mult,
        weight_decay=args.weight_decay,
        log_prefix=f"[oracle/{student}/s{seed}/{task_data.runtime.spec.name}] ",
    )
    return metrics


def combo_completed(existing: pd.DataFrame, pack: str, student: str, seed: int, stage2_method: str, stage3_policy: str) -> bool:
    if existing.empty:
        return False
    cfg = PACKS[pack]
    sub = existing[
        (existing["pack"] == pack)
        & (existing["student"] == student)
        & (existing["seed"] == seed)
        & (existing["stage2_method"] == stage2_method)
        & (existing["stage3_policy"] == stage3_policy)
    ]
    expected = set()
    for task in cfg["initial_tasks"]:
        expected.add(("stage2", "", task))
    old = list(cfg["initial_tasks"])
    for fut in cfg["future_tasks"]:
        for task in old + [fut]:
            expected.add(("stage3", fut, task))
        expected.add(("oracle", fut, fut))
        old = old + [fut]
    got = set(zip(sub["phase"].astype(str), sub["added_task"].fillna("").astype(str), sub["task"].astype(str)))
    return expected.issubset(got)


def ensure_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "input_window_policy" not in df.columns:
        df["input_window_policy"] = "shared_native"
    if "input_window_sec_map" not in df.columns:
        df["input_window_sec_map"] = "{}"
    return df


def metric_row_payload(metric: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "macro_f1": metric.get("macro_f1", np.nan),
        "mae": metric.get("mae", np.nan),
        "balanced_acc": metric.get("balanced_acc", np.nan),
        "auroc": metric.get("auroc", np.nan),
        "auprc": metric.get("auprc", np.nan),
        "per_class_f1_json": metric.get("per_class_f1_json", "{}"),
        "n_eval": metric.get("n_eval", np.nan),
    }


@torch.no_grad()
def feature_drift_cosine(
    encoder_ref: nn.Module,
    encoder_cur: nn.Module,
    task_data: ECGTaskData,
    device: torch.device,
    batch_size: int,
) -> float:
    ref = copy.deepcopy(encoder_ref).to(device).eval()
    cur = copy.deepcopy(encoder_cur).to(device).eval()
    loader = DataLoader(
        ECGIndexedDataset(task_data.signals, task_data.split_idx["test"], task_data.split_y["test"], task_data.runtime, input_len=task_data.input_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    cos_vals: List[torch.Tensor] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        z_ref = ref(xb).reshape(xb.shape[0], -1)
        z_cur = cur(xb).reshape(xb.shape[0], -1)
        cos_vals.append(torch.sum(F.normalize(z_ref, dim=1) * F.normalize(z_cur, dim=1), dim=1).detach().cpu())
    if not cos_vals:
        return float("nan")
    return float(1.0 - torch.cat(cos_vals).mean().item())


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary_rows: List[Dict[str, Any]] = []
    keys = ["pack", "dataset", "student", "seed", "stage1_method", "stage2_method", "stage3_policy"]
    for group_key, g in df.groupby(keys, dropna=False):
        base = dict(zip(keys, group_key))
        stage2 = g[g["phase"] == "stage2"]
        stage3 = g[g["phase"] == "stage3"]
        oracle = g[g["phase"] == "oracle"]
        if stage2.empty or stage3.empty:
            continue
        for added_task, g_add in stage3.groupby("added_task", dropna=False):
            old_rows = g_add[g_add["task_role"] == "old"]
            new_rows = g_add[g_add["task_role"] == "future"]
            rec = dict(base)
            rec["added_task"] = str(added_task)
            rec["stage2_avg_value"] = float(stage2["value"].mean())
            rec["stage2_n_tasks"] = int(stage2["task"].nunique())
            rec["old_avg_ret_from_stage2"] = float(old_rows["rel_vs_stage2"].mean()) if not old_rows.empty else np.nan
            rec["old_floor_ret_from_stage2"] = float(old_rows["rel_vs_stage2"].min()) if not old_rows.empty else np.nan
            rec["new_task_value"] = float(new_rows["value"].iloc[0]) if len(new_rows) == 1 else np.nan
            oracle_row = oracle[oracle["added_task"] == added_task]
            rec["new_task_rel_vs_oracle"] = float(new_rows["rel_vs_oracle"].iloc[0]) if len(new_rows) == 1 else np.nan
            rec["oracle_value"] = float(oracle_row["value"].iloc[0]) if len(oracle_row) == 1 else np.nan
            summary_rows.append(rec)
    by_seed = pd.DataFrame(summary_rows)
    if by_seed.empty:
        return by_seed, pd.DataFrame()
    agg = (
        by_seed.groupby(["pack", "dataset", "student", "stage1_method", "stage2_method", "stage3_policy"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            old_avg_ret_mean=("old_avg_ret_from_stage2", "mean"),
            old_avg_ret_std=("old_avg_ret_from_stage2", "std"),
            old_floor_ret_mean=("old_floor_ret_from_stage2", "mean"),
            old_floor_ret_std=("old_floor_ret_from_stage2", "std"),
            new_task_rel_mean=("new_task_rel_vs_oracle", "mean"),
            new_task_rel_std=("new_task_rel_vs_oracle", "std"),
        )
        .reset_index()
    )
    return by_seed, agg


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = resolve_device(args.device)
    print(f"[device] {device}", flush=True)

    for pack in args.packs:
        pack_cfg = PACKS[pack]
        out_dir = args.out_root / pack
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "stage23_results.csv"
        by_seed_csv = out_dir / "stage23_summary_by_seed.csv"
        summary_csv = out_dir / "stage23_summary.csv"
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

        existing = pd.DataFrame()
        if out_csv.exists() and not args.fresh:
            existing = ensure_result_columns(pd.read_csv(out_csv))
            print(f"[append:{pack}] loaded {len(existing)} rows", flush=True)

        all_rows: List[Dict[str, Any]] = existing.to_dict("records") if not existing.empty else []
        for student in args.students:
            for seed in args.seeds:
                initial_task_data, future_task_data = load_pack_data(pack, args)
                if not existing.empty:
                    done_methods = [m for m in args.stage2_methods if combo_completed(existing, pack, student, seed, m, args.stage3_policy)]
                else:
                    done_methods = []
                oracle_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
                for stage2_method in args.stage2_methods:
                    if stage2_method in done_methods:
                        print(f"[skip:{pack}] {student} seed={seed} {stage2_method} [cached]", flush=True)
                        continue
                    set_seed(seed)
                    print(
                        f"\n{'='*88}\n"
                        f"[pack={pack}] student={student} seed={seed} stage2={stage2_method}\n"
                        f"initial={pack_cfg['initial_tasks']} future={pack_cfg['future_tasks']}\n"
                        f"{'='*88}",
                        flush=True,
                    )
                    ckpt_seed = int(args.stage1_ckpt_seed) if args.stage1_ckpt_seed is not None else int(seed)
                    enc, ck_meta = load_stage1_ecgfounder(student, args.stage1_method, ckpt_seed, device)
                    # load_stage1_ecgfounder -> build_student() calls torch.manual_seed(ckpt_seed),
                    # which overrides the per-seed seeding. Re-seed so each run is independent.
                    set_seed(int(seed))
                    enc_ref = copy.deepcopy(enc).cpu()
                    all_tasks = [td.runtime for td in initial_task_data + future_task_data]
                    input_len = max(int(td.input_len) for td in initial_task_data)
                    model = MultiTaskModel(
                        enc,
                        int(ck_meta["enc_dim"]),
                        all_tasks,
                        args.dropout,
                        dataset_name=pack_cfg["dataset"],
                        hz=500,
                        input_len=input_len,
                        temporal_policy=str(pack_cfg.get("temporal_policy", "shared_window")),
                    )
                    train_loaders, val_loaders, test_loaders = make_task_loaders_ecg(initial_task_data, args.batch_size)
                    stage2_cache_path = None
                    if args.stage2_cache_dir is not None:
                        stage2_cache_path = args.stage2_cache_dir / f"{pack}_{student}_s{seed}_{stage2_method}.pt"
                    if stage2_cache_path is not None and stage2_cache_path.exists():
                        cached = torch.load(str(stage2_cache_path), map_location="cpu", weights_only=False)
                        model.load_state_dict(cached["state_dict"])
                        stage2_metrics = cached["stage2_metrics"]
                        stage2_meta = cached["stage2_meta"]
                        t0 = time.time()
                        print(f"    [{pack}/{student}/s{seed}/{stage2_method}/stage2] loaded cached Stage-II checkpoint", flush=True)
                    else:
                        t0 = time.time()
                        model, stage2_metrics, stage2_meta = fit_stage2(
                            model=model,
                            method=stage2_method,
                            tasks=[td.runtime for td in initial_task_data],
                            train_loaders=train_loaders,
                            val_loaders=val_loaders,
                            test_loaders=test_loaders,
                            device=device,
                            epochs=args.stage2_epochs,
                            patience=args.stage2_patience,
                            head_lr=args.stage2_head_lr,
                            encoder_lr_mult=args.stage2_encoder_lr_mult,
                            weight_decay=args.weight_decay,
                            care_stage2_scope=args.care_stage2_scope,
                            log_prefix=f"[{pack}/{student}/s{seed}/{stage2_method}/stage2] ",
                        )
                        if stage2_cache_path is not None:
                            stage2_cache_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(
                                {"state_dict": model.state_dict(), "stage2_metrics": stage2_metrics, "stage2_meta": stage2_meta},
                                str(stage2_cache_path),
                            )
                            print(f"    [{pack}/{student}/s{seed}/{stage2_method}/stage2] saved Stage-II checkpoint to {stage2_cache_path}", flush=True)
                    base = {
                        "pack": pack,
                        "dataset": pack_cfg["dataset"],
                        "student": student,
                        "seed": seed,
                        "stage1_method": args.stage1_method,
                        "stage2_method": stage2_method,
                        "stage3_policy": args.stage3_policy,
                        "stage1_ckpt": ck_meta["ckpt_path"],
                        "time_s_stage2": float(time.time() - t0),
                        "encoder_params": count_params(model.encoder),
                        "total_params": count_params(model),
                        "trainable_params_stage2": count_params(model, trainable_only=True),
                        "input_window_policy": str(pack_cfg.get("input_window_policy", "shared_native")),
                        "input_window_sec_map": json.dumps(pack_cfg.get("input_window_sec_map", {}), sort_keys=True),
                    }
                    for td in initial_task_data:
                        m = stage2_metrics[td.runtime.spec.name]
                        all_rows.append(
                            {
                                **base,
                                **stage2_meta,
                                "phase": "stage2",
                                "added_task": "",
                                "task": td.runtime.spec.name,
                                "task_role": "initial",
                                "metric": m["metric"],
                                "higher_better": bool(m["higher_better"]),
                                "value": float(m["value"]),
                                **metric_row_payload(m),
                                "rel_vs_stage2": 1.0,
                                "rel_vs_oracle": np.nan,
                            }
                        )

                    stage2_snapshot = {k: dict(v) for k, v in stage2_metrics.items()}
                    current_old = list(initial_task_data)
                    current_model = copy.deepcopy(model).cpu()
                    for added_task_data in future_task_data:
                        new_name = added_task_data.runtime.spec.name
                        oracle_cache_path = None
                        if args.oracle_cache_dir is not None:
                            oracle_cache_path = args.oracle_cache_dir / f"{pack}_{student}_s{seed}_{new_name}.pt"
                        if new_name not in oracle_cache:
                            if oracle_cache_path is not None and oracle_cache_path.exists():
                                oracle_cache[new_name] = torch.load(str(oracle_cache_path), map_location="cpu", weights_only=False)
                            else:
                                oracle_cache[new_name] = fit_stage1_single_task_oracle(
                                    student=student,
                                    seed=seed,
                                    stage1_method=args.stage1_method,
                                    task_data=added_task_data,
                                    dataset_name=pack_cfg["dataset"],
                                    device=device,
                                    args=args,
                                )
                                if oracle_cache_path is not None:
                                    oracle_cache_path.parent.mkdir(parents=True, exist_ok=True)
                                    torch.save(oracle_cache[new_name], str(oracle_cache_path))
                        oracle_m = oracle_cache[new_name][new_name]
                        all_rows.append(
                            {
                                **base,
                                "phase": "oracle",
                                "added_task": new_name,
                                "task": new_name,
                                "task_role": "future",
                                "metric": oracle_m["metric"],
                                "higher_better": bool(oracle_m["higher_better"]),
                                "value": float(oracle_m["value"]),
                                **metric_row_payload(oracle_m),
                                "rel_vs_stage2": np.nan,
                                "rel_vs_oracle": 1.0,
                            }
                        )

                        ext_model = copy.deepcopy(current_model).to(device)
                        t1 = time.time()
                        ext_model, _, stage3_meta = fit_new_task_extension(
                            model=ext_model,
                            old_tasks=[td.runtime for td in current_old],
                            new_task_data=added_task_data,
                            device=device,
                            epochs=args.stage3_epochs,
                            patience=args.stage3_patience,
                            head_lr=args.stage3_head_lr,
                            encoder_lr_mult=args.stage3_encoder_lr_mult,
                            weight_decay=args.weight_decay,
                            stage3_policy=args.stage3_policy,
                            batch_size=args.batch_size,
                            log_prefix=f"[{pack}/{student}/s{seed}/{stage2_method}/add:{new_name}] ",
                            care_stage3_scope=args.care_stage3_scope,
                            lwf_old_distill_lambda=args.lwf_old_distill_lambda,
                            care_stage3_anchor_lambda=args.care_stage3_anchor_lambda,
                            cpca_core_k=args.cpca_core_k,
                            old_task_data=current_old,
                            seed=int(seed),
                            core_lambda=args.core_lambda,
                            core_mode=args.core_mode,
                            core_budget=args.core_budget,
                            ewc_lambda=args.ewc_lambda,
                        )
                        current_model = copy.deepcopy(ext_model).cpu()
                        eval_task_data = current_old + [added_task_data]
                        stage3_metrics = evaluate_all_tasks(ext_model, eval_task_data, device, args.batch_size)
                        drift_cosine = feature_drift_cosine(enc_ref, ext_model.encoder, added_task_data, device, args.batch_size)
                        for td in eval_task_data:
                            task_name = td.runtime.spec.name
                            m = stage3_metrics[task_name]
                            higher_better = bool(m["higher_better"])
                            rel_stage2 = (
                                compare_metric(float(m["value"]), float(stage2_snapshot[task_name]["value"]), higher_better)
                                if task_name in stage2_snapshot
                                else np.nan
                            )
                            rel_oracle = (
                                compare_metric(float(m["value"]), float(oracle_cache[new_name][new_name]["value"]), higher_better)
                                if task_name == new_name
                                else np.nan
                            )
                            all_rows.append(
                                {
                                    **base,
                                    **stage3_meta,
                                    "phase": "stage3",
                                    "added_task": new_name,
                                    "task": task_name,
                                    "task_role": "future" if task_name == new_name else "old",
                                    "metric": m["metric"],
                                    "higher_better": higher_better,
                                    "value": float(m["value"]),
                                    **metric_row_payload(m),
                                    "rel_vs_stage2": rel_stage2,
                                    "rel_vs_oracle": rel_oracle,
                                    "feature_drift_cosine": drift_cosine,
                                    "time_s_stage3": float(time.time() - t1),
                                    "trainable_params_stage3": count_params(ext_model, trainable_only=True),
                                }
                            )
                        stage2_snapshot.update({k: dict(v) for k, v in stage3_metrics.items()})
                        current_old = current_old + [added_task_data]

                    df = pd.DataFrame(all_rows)
                    df = ensure_result_columns(df)
                    df.to_csv(out_csv, index=False)
                    by_seed, agg = summarize(df)
                    by_seed.to_csv(by_seed_csv, index=False)
                    agg.to_csv(summary_csv, index=False)
                    if not agg.empty:
                        print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

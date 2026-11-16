#!/usr/bin/env python3
"""Stage-2 MTL adaptation with frozen future-task probes for PPG case studies."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_ecg._preprocessors import Normalize

from downstream_eval import TASK_REGISTRY
from downstream_eval import _kfold_arrays, _presplit_arrays
from sdb import SDB_TEST_SUBJECTS, SDB_TRAIN_SUBJECTS, SDB_VAL_SUBJECTS, _process_subject_ppg

from run_specialization_sweep_ppg import (
    DEFAULT_DATA_DIRS,
    _auto_cap_large_kfold,
    _auto_cap_large_presplit,
    build_ppg_task_data,
    fit_frozen_probe,
    metric_payload,
    metric_value,
    task_dataset,
)
from run_stage23_ppg_unified import fit_stage2, make_task_loaders_for_device
from run_stage2_multitask_ppg import (
    MultiTaskModel,
    TaskData,
    TaskRuntime,
    TaskSpec,
    build_encoder,
    disable_dropout,
    load_stage1_full,
    maybe_limit_split,
    resolve_device,
    set_seed,
    split_subjects,
)
from run_stage2_reusability_ppg import LOCAL_CKPT_DIRS


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "kd_sweep_results/stage2_mtl_future_probe"
DEFAULT_SAME10_BASELINE_CSV = ROOT / "kd_sweep_results/stage2_mtl_future_probe_aggregated_final" / "overall_summary.csv"
DEFAULT_SAME10_TEACHER_CSV = ROOT / "kd_sweep_results/stage2_mtl_future_probe_aggregated_teacher_final" / "overall_summary.csv"
ALL_METHODS = ("pulseppg_teacher_e0", "e0_probe", "joint_mtl", "gradnorm_mtl", "pcgrad_mtl", "fairgrad_mtl", "lifefm_adapt", "care_v2", "care_v3", "care_v2_normmse", "care_v2_cosine", "care_v2_statnorm", "care_v2_tasknorm", "l2sp_mtl")
METHOD_TO_STAGE2 = {
    "joint_mtl": "joint_finetune",
    "gradnorm_mtl": "gradnorm_joint",
    "pcgrad_mtl": "pcgrad_joint",
    "fairgrad_mtl": "fairgrad_joint",
    "lifefm_adapt": "lifefm_adapt",
    "care_v2": "care_v2",
    "care_v3": "care_v3",
    "care_v2_normmse": "care_v2_normmse",
    "care_v2_cosine": "care_v2_cosine",
    "care_v2_statnorm": "care_v2_statnorm",
    "care_v2_tasknorm": "care_v2_tasknorm",
    "l2sp_mtl": "l2sp",
}
NON_STAGE2_METHODS = {"pulseppg_teacher_e0", "e0_probe"}
CASE_DEFS: Dict[str, Dict[str, Any]] = {
    "clinical": {
        "case_name": "clinical",
        "description": "Clinical cardiopulmonary monitoring",
        "source_tasks": ["sdb_binary", "sdb_multiclass"],
        "target_tasks": ["bidmc_hr"],
    },
    "wearable": {
        "case_name": "wearable",
        "description": "Wearable stress/wellness monitoring",
        "source_tasks": ["wesad_binary", "wesad_multiclass"],
        "target_tasks": ["wfppg_finger", "wfppg_wrist"],
    },
    "vitalsign": {
        "case_name": "vitalsign",
        "description": "Vital-sign PPG specialization",
        "source_tasks": ["ppgbp_sysbp", "ppgbp_diasbp"],
        "target_tasks": ["wfppg_finger", "wfppg_wrist"],
    },
}
TASK_LENGTH_AUDIT_TASKS = [
    "sdb_binary",
    "sdb_multiclass",
    "bidmc_hr",
    "wesad_binary",
    "wesad_multiclass",
    "wfppg_finger",
    "wfppg_wrist",
    "ppgbp_sysbp",
    "ppgbp_diasbp",
]
TASK_LENGTH_PROFILE_PRESETS: Dict[str, Dict[str, float]] = {
    "same10": {task_name: 10.0 for task_name in TASK_LENGTH_AUDIT_TASKS},
    "ppg_task_specific_v1": {
        "sdb_binary": 30.0,
        "sdb_multiclass": 30.0,
        "bidmc_hr": 10.0,
        "wesad_binary": 60.0,
        "wesad_multiclass": 60.0,
        "wfppg_finger": 10.0,
        "wfppg_wrist": 10.0,
        "ppgbp_sysbp": 10.0,
        "ppgbp_diasbp": 10.0,
    },
    "ppg_task_specific_uniform": {
        "sdb_binary": 30.0,
        "sdb_multiclass": 30.0,
        "bidmc_hr": 10.0,
        "wesad_binary": 60.0,
        "wesad_multiclass": 60.0,
        "wfppg_finger": 10.0,
        "wfppg_wrist": 10.0,
        "ppgbp_sysbp": 10.0,
        "ppgbp_diasbp": 10.0,
    },
    "ppg_task_specific_mixed": {
        "sdb_binary": 20.0,
        "sdb_multiclass": 30.0,
        "bidmc_hr": 10.0,
        "wesad_binary": 20.0,
        "wesad_multiclass": 60.0,
        "wfppg_finger": 10.0,
        "wfppg_wrist": 10.0,
        "ppgbp_sysbp": 10.0,
        "ppgbp_diasbp": 10.0,
    },
}
TASK_LENGTH_REASONS: Dict[str, str] = {
    "sdb_binary": "Sleep / breathing screening typically benefits from longer temporal context than 10s.",
    "sdb_multiclass": "Sleep / breathing severity grading typically benefits from longer temporal context than 10s.",
    "bidmc_hr": "The current BIDMC HR preprocessing is natively 10s and already matches short-support HR estimation.",
    "wesad_binary": "Stress recognition usually benefits from longer physiological context than 10s.",
    "wesad_multiclass": "Affective-state recognition usually benefits from longer physiological context than 10s.",
    "wfppg_finger": "The current WFPPG HR cache is already 10s and treated as the native monitoring window.",
    "wfppg_wrist": "The current WFPPG HR cache is already 10s and treated as the native monitoring window.",
    "ppgbp_sysbp": "The current PPG-BP cache stores fixed native segments and is kept at 10s.",
    "ppgbp_diasbp": "The current PPG-BP cache stores fixed native segments and is kept at 10s.",
}
CURRENT_VARIABLE_LENGTH_SUPPORT = {
    "sdb_binary": True,
    "sdb_multiclass": True,
    "bidmc_hr": False,
    "wesad_binary": True,
    "wesad_multiclass": True,
    "wfppg_finger": False,
    "wfppg_wrist": False,
    "ppgbp_sysbp": False,
    "ppgbp_diasbp": False,
}
SDB_RAW_FS_SOURCE = 62.5
SDB_SPLIT_SUBJECTS = {
    "train": list(SDB_TRAIN_SUBJECTS),
    "val": list(SDB_VAL_SUBJECTS),
    "test": list(SDB_TEST_SUBJECTS),
}
_SDB_RAW_SPLIT_CACHE: Dict[tuple[str, int, float], tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}


class TeacherEncoderAdapter(nn.Module):
    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        self.teacher = teacher

    @staticmethod
    def _unwrap_output(out: Any) -> torch.Tensor:
        if isinstance(out, dict):
            for key in ("feat", "features", "embedding", "emb", "rep", "z", "proj", "logits"):
                if key in out and torch.is_tensor(out[key]):
                    out = out[key]
                    break
            else:
                out = next(v for v in out.values() if torch.is_tensor(v))
        elif isinstance(out, (tuple, list)):
            out = next(v for v in out if torch.is_tensor(v))
        if not torch.is_tensor(out):
            raise TypeError(f"Unexpected teacher output type: {type(out)}")
        if out.ndim == 3:
            out = out.mean(-1)
        elif out.ndim > 2:
            out = out.flatten(1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._unwrap_output(self.teacher(x))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", default="", help="Checkpoint file, checkpoint directory, or template path with {seed}/{method} placeholders.")
    ap.add_argument("--student_encoder_ckpt", dest="encoder_ckpt", help="Alias for --encoder_ckpt.")
    ap.add_argument("--student", default="BF32_NB8")
    ap.add_argument("--stage1_method", default="lifefm_kd")
    ap.add_argument("--cases", nargs="+", default=["all"])
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--stage2_epochs", type=int, default=100)
    ap.add_argument("--stage2_patience", type=int, default=10)
    ap.add_argument("--stage2_head_lr", type=float, default=1e-3)
    ap.add_argument("--stage2_encoder_lr_mult", type=float, default=0.1)
    ap.add_argument("--probe_epochs", type=int, default=20)
    ap.add_argument("--probe_patience", type=int, default=5)
    ap.add_argument("--probe_head_lr", type=float, default=1e-3)
    ap.add_argument("--probe_encoder_lr_mult", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--fairgrad_alpha", type=float, default=1.0)
    ap.add_argument("--l2sp_lambda", type=float, default=1e-2)
    ap.add_argument("--care_anchor_source", type=str, default="source_union", choices=["source_union", "stage1_public_pool", "mixed_source_public"])
    ap.add_argument("--care_reuse_feature_space", type=str, default="encoder_pooled", choices=["encoder_pooled"])
    ap.add_argument("--care_reuse_k", type=int, default=0, help="0 => auto=min(32, feature_dim)")
    ap.add_argument("--lambda-core", "--care_reuse_lambda", dest="care_reuse_lambda", type=float, default=0.05)
    ap.add_argument("--care_train_scope", type=str, default="full_encoder", choices=["full_encoder", "upper_blocks_only", "late_blocks_plus_heads", "heads_only", "top1", "top2", "none"])
    ap.add_argument("--care_reuse_loss_type", type=str, default="mse", choices=["mse", "full_mse", "cosine", "huber", "norm_mse", "statnorm_mse"])
    ap.add_argument("--care_stat_lambda", type=float, default=0.01)
    ap.add_argument("--care_anchor_max_batches", type=int, default=8)
    ap.add_argument("--care_anchor_basis_samples", type=int, default=128)
    ap.add_argument("--care_drift_budget", type=float, default=0.02)
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--stage1-ckpt-seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--large_task_threshold", type=int, default=50000)
    ap.add_argument("--large_task_cap", type=int, default=20000)
    ap.add_argument("--length_profile", default="same10", choices=sorted(TASK_LENGTH_PROFILE_PRESETS))
    ap.add_argument("--same10_baseline_csv", type=Path, default=DEFAULT_SAME10_BASELINE_CSV)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_teacher_baselines", action="store_true")
    ap.add_argument("--no_early_stop_stage2", dest="no_early_stop_stage2", action="store_true")
    ap.add_argument("--allow_early_stop_stage2", dest="no_early_stop_stage2", action="store_false")
    ap.add_argument("--use_final_checkpoint", dest="use_final_checkpoint", action="store_true")
    ap.add_argument("--use_best_checkpoint", dest="use_final_checkpoint", action="store_false")
    ap.set_defaults(no_early_stop_stage2=True, use_final_checkpoint=True)
    return ap.parse_args()


def _parse_name_list(raw_values: Sequence[str], valid: Iterable[str], allow_all: bool = False) -> List[str]:
    items: List[str] = []
    for raw in raw_values:
        for token in str(raw).split(","):
            token = token.strip()
            if token:
                items.append(token)
    if allow_all and any(token == "all" for token in items):
        return list(valid)
    valid_set = set(valid)
    unknown = sorted(set(items) - valid_set)
    if unknown:
        raise ValueError(f"Unknown values {unknown}; choices={sorted(valid_set)}")
    return items


def _task_x_path(task_name: str, hz: int) -> Path:
    cfg = TASK_REGISTRY[str(task_name)]
    dataset = str(cfg["dataset"])
    root = Path(DEFAULT_DATA_DIRS[dataset])
    subdir = str(cfg.get("subdir", "") or "")
    if subdir:
        root = root / subdir
    if cfg["split_mode"] == "presplit":
        return root / str(cfg["x_file"]).format(split="train", hz=int(hz))
    if cfg["split_mode"] == "kfold":
        return root / str(cfg["x_file"]).format(hz=int(hz))
    raise NotImplementedError(f"Unsupported split_mode for length audit: {cfg['split_mode']}")


def _task_current_default_length_sec(task_name: str, hz: int) -> float:
    x_path = _task_x_path(str(task_name), int(hz))
    if not x_path.exists():
        raise FileNotFoundError(f"Missing X file for {task_name}: {x_path}")
    X = np.load(str(x_path), mmap_mode="r")
    return float(int(X.shape[-1])) / float(hz)


def _length_root_token(length_sec: float) -> str:
    rounded = int(round(float(length_sec)))
    if abs(float(length_sec) - float(rounded)) > 1e-6:
        raise ValueError(f"Only integer-second task lengths are supported in cache naming, got {length_sec}")
    return f"len{rounded}"


def _task_length_cache_root(task_name: str, target_length_sec: float) -> Path | None:
    token = _length_root_token(float(target_length_sec))
    dataset_root = ROOT / "pulseppg" / "data" / "datasets"
    if str(task_name).startswith("sdb_"):
        return dataset_root / f"sdb_{token}"
    if str(task_name).startswith("wesad_"):
        return dataset_root / f"wesad_{token}"
    return None


def _build_ppg_task_data_from_root(task_name: str, args: argparse.Namespace, seed: int, root_override: Path) -> TaskData:
    cfg = dict(TASK_REGISTRY[str(task_name)])
    if not root_override.exists():
        raise FileNotFoundError(f"Missing variable-length cache for {task_name}: {root_override}")

    if cfg["split_mode"] == "presplit":
        X_tr, y_tr, X_va, y_va, X_te, y_te = _presplit_arrays(str(root_override), cfg, int(args.hz))
        splits_X = {
            "train": np.asarray(X_tr, dtype=np.float32),
            "val": np.asarray(X_va, dtype=np.float32),
            "test": np.asarray(X_te, dtype=np.float32),
        }
        splits_y = {
            "train": np.asarray(y_tr),
            "val": np.asarray(y_va),
            "test": np.asarray(y_te),
        }
        splits_X, splits_y = _auto_cap_large_presplit(
            splits_X,
            splits_y,
            threshold=int(args.large_task_threshold),
            cap=int(args.large_task_cap),
            seed=int(seed),
            task_name=str(task_name),
        )
        for split in ("train", "val", "test"):
            limit = {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test}[split]
            if int(limit) > 0 and len(splits_X[split]) > int(limit):
                Xi, ys = maybe_limit_split(
                    splits_X[split],
                    {str(task_name): splits_y[split]},
                    int(limit),
                    int(seed) + {"train": 0, "val": 1, "test": 2}[split],
                )
                splits_X[split] = np.asarray(Xi, dtype=np.float32)
                splits_y[split] = np.asarray(ys[str(task_name)])
    elif cfg["split_mode"] == "kfold":
        X_all, y_all, subjects = _kfold_arrays(str(root_override), cfg, int(args.hz))
        if cfg.get("label_transform") == "binarize":
            y_all = (np.asarray(y_all) > 0).astype(np.int64)
        elif cfg.get("label_transform") == "to_int":
            y_all = np.asarray(y_all).astype(np.int64)
        else:
            y_all = np.asarray(y_all)
        X_all, y_all, subjects = _auto_cap_large_kfold(
            np.asarray(X_all, dtype=np.float32),
            np.asarray(y_all),
            np.asarray(subjects),
            threshold=int(args.large_task_threshold),
            cap=int(args.large_task_cap),
            seed=int(seed),
            task_name=str(task_name),
        )
        X_all = np.asarray(X_all, dtype=np.float32)
        y_all = np.asarray(y_all)
        subjects = np.asarray(subjects, dtype=str)
        train_subj, val_subj, test_subj = split_subjects(np.asarray(subjects, dtype=str), int(seed))
        splits_X = {}
        splits_y = {}
        for split, split_subj in (("train", train_subj), ("val", val_subj), ("test", test_subj)):
            mask = np.isin(np.asarray(subjects, dtype=str), list(split_subj))
            Xi = np.asarray(X_all[mask], dtype=np.float32)
            yi = np.asarray(y_all[mask])
            limit = {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test}[split]
            if int(limit) > 0 and len(Xi) > int(limit):
                Xi, ys = maybe_limit_split(
                    Xi,
                    {str(task_name): yi},
                    int(limit),
                    int(seed) + {"train": 0, "val": 1, "test": 2}[split],
                )
                yi = np.asarray(ys[str(task_name)])
            splits_X[split] = Xi
            splits_y[split] = yi
    else:
        raise NotImplementedError(f"Unsupported split_mode for override root: {cfg['split_mode']}")

    if cfg["type"] == "cls":
        y_cat = np.concatenate([np.asarray(splits_y[s], dtype=np.int64) for s in ("train", "val", "test") if len(splits_y[s]) > 0], axis=0)
        n_out = int(np.max(y_cat)) + 1 if len(y_cat) > 0 else 2
        runtime = TaskRuntime(TaskSpec(str(task_name), "cls", "", str(cfg["primary_metric"])), n_out=n_out)
    else:
        y_train = np.asarray(splits_y["train"], dtype=np.float32)
        y_mean = float(y_train.mean()) if len(y_train) > 0 else 0.0
        y_std = float(y_train.std()) if len(y_train) > 0 else 1.0
        if not np.isfinite(y_std) or y_std < 1e-6:
            y_std = 1.0
        runtime = TaskRuntime(TaskSpec(str(task_name), "reg", "", str(cfg["primary_metric"])), n_out=1, y_mean=y_mean, y_std=y_std)
    return TaskData(runtime=runtime, split_X=splits_X, split_y=splits_y)


def _json_float_map(values: Dict[str, float]) -> str:
    return json.dumps({str(k): float(v) for k, v in values.items()}, sort_keys=True)


def _approx_equal_sec(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-6


def _load_same10_reference_rows(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing same-10 baseline summary: {path}")
    frames = [pd.read_csv(path)]
    teacher_path = DEFAULT_SAME10_TEACHER_CSV
    if Path(path).resolve() != teacher_path.resolve() and teacher_path.exists():
        frames.append(pd.read_csv(teacher_path))
    df = pd.concat(frames, ignore_index=True)
    df = pd.DataFrame(_dedupe_rows(df.to_dict(orient="records")))
    if df.empty:
        raise ValueError(f"Same-10 baseline summary is empty: {path}")
    df = df.copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(-1).astype(int)
    return df


def _student_e0_lookup(rows: pd.DataFrame) -> Dict[tuple[str, str, int], float]:
    lookup: Dict[tuple[str, str, int], float] = {}
    subset = rows[rows["method"] == "e0_probe"].copy()
    if subset.empty:
        return lookup
    for row in subset.to_dict(orient="records"):
        key = (str(row["case_name"]), str(row["target_task"]), int(row["seed"]))
        value = row.get("target_after_stage2_probe", row.get("target_E0_probe", np.nan))
        lookup[key] = float(value)
    return lookup


def _teacher_probe_lookup(rows: pd.DataFrame) -> Dict[tuple[str, str, int], float]:
    lookup: Dict[tuple[str, str, int], float] = {}
    subset = rows[rows["method"] == "pulseppg_teacher_e0"].copy()
    if subset.empty:
        return lookup
    for row in subset.to_dict(orient="records"):
        key = (str(row["case_name"]), str(row["target_task"]), int(row["seed"]))
        value = row.get("target_after_stage2_probe", row.get("target_E0_probe", np.nan))
        lookup[key] = float(value)
    return lookup


def _resolve_task_length_profile(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
    requested = TASK_LENGTH_PROFILE_PRESETS[str(args.length_profile)]
    current_default_sec: Dict[str, float] = {}
    chosen_sec: Dict[str, float] = {}
    audit_rows: List[Dict[str, Any]] = []
    for task_name in TASK_LENGTH_AUDIT_TASKS:
        current_sec = _task_current_default_length_sec(task_name, int(args.hz))
        current_default_sec[task_name] = current_sec
        recommended_sec = float(requested.get(task_name, current_sec))
        chosen = float(recommended_sec)
        caveat = ""
        if not _approx_equal_sec(float(recommended_sec), float(current_sec)):
            cache_root = _task_length_cache_root(str(task_name), float(recommended_sec))
            if cache_root is not None:
                caveat = f"Requires explicit variable-length cache at {cache_root}"
            else:
                caveat = "Requires explicit non-native cache or builder support; no fallback to 10s is allowed."
        chosen_sec[task_name] = float(chosen)
        audit_rows.append(
            {
                "task_name": str(task_name),
                "dataset": str(task_dataset(task_name)),
                "task_type": "classification" if str(TASK_REGISTRY[task_name]["type"]) == "cls" else "regression",
                "current_default_length_sec": float(current_sec),
                "supports_variable_length": bool(CURRENT_VARIABLE_LENGTH_SUPPORT.get(task_name, False)),
                "native_or_recommended_length_sec": float(recommended_sec),
                "chosen_length_sec": float(chosen),
                "reason": str(TASK_LENGTH_REASONS[task_name]),
                "caveat": str(caveat),
            }
        )
    return audit_rows, current_default_sec, chosen_sec


def _write_task_length_artifacts(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    audit_rows: Sequence[Dict[str, Any]],
    current_default_sec: Dict[str, float],
    chosen_sec: Dict[str, float],
) -> None:
    pd.DataFrame(list(audit_rows)).to_csv(output_dir / "task_length_audit.csv", index=False)
    profile_payload = {
        "length_profile": str(args.length_profile),
        "hz": int(args.hz),
        "current_default_length_sec": {k: float(v) for k, v in current_default_sec.items()},
        "task_lengths_sec": {k: float(v) for k, v in chosen_sec.items()},
    }
    (output_dir / "task_length_profile.json").write_text(json.dumps(profile_payload, indent=2, sort_keys=True), encoding="utf-8")


def _sdb_raw_splits(length_sec: float, hz: int) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    cache_key = (str(DEFAULT_DATA_DIRS["sdb"]), int(hz), float(length_sec))
    cached = _SDB_RAW_SPLIT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    root = Path(DEFAULT_DATA_DIRS["sdb"])
    labels = pd.read_csv(root / "AHI.csv")
    label_by_subject = {
        int(row["subjectNumber"]): int(row["AHI"])
        for row in labels.to_dict(orient="records")
    }
    norm = Normalize(method="z-score")
    split_X: Dict[str, List[np.ndarray]] = {split: [] for split in ("train", "val", "test")}
    split_y: Dict[str, List[np.ndarray]] = {split: [] for split in ("train", "val", "test")}
    for split, subject_ids in SDB_SPLIT_SUBJECTS.items():
        for subject_id in subject_ids:
            chunks = _process_subject_ppg(
                root_dir=str(root),
                subject_id=int(subject_id),
                hz=int(hz),
                window_sec=int(round(float(length_sec))),
                fs_source=float(SDB_RAW_FS_SOURCE),
                norm=norm,
            )
            if len(chunks) == 0:
                continue
            label_value = int(label_by_subject[int(subject_id)])
            split_X[split].append(np.asarray(chunks, dtype=np.float32))
            split_y[split].append(np.full((len(chunks),), label_value, dtype=np.int64))
    packed_X = {
        split: np.concatenate(values, axis=0) if values else np.zeros((0, 1, int(round(float(length_sec) * float(hz)))), dtype=np.float32)
        for split, values in split_X.items()
    }
    packed_y = {
        split: np.concatenate(values, axis=0) if values else np.zeros((0,), dtype=np.int64)
        for split, values in split_y.items()
    }
    _SDB_RAW_SPLIT_CACHE[cache_key] = (packed_X, packed_y)
    return packed_X, packed_y


def _build_sdb_task_data_from_raw(task_name: str, args: argparse.Namespace, seed: int, length_sec: float) -> TaskData:
    base_split_X, base_split_y = _sdb_raw_splits(float(length_sec), int(args.hz))
    split_X = {split: np.asarray(X, dtype=np.float32).copy() for split, X in base_split_X.items()}
    if task_name == "sdb_binary":
        split_y = {split: (np.asarray(y, dtype=np.int64) > 0).astype(np.int64) for split, y in base_split_y.items()}
    else:
        split_y = {split: np.asarray(y, dtype=np.int64).copy() for split, y in base_split_y.items()}

    split_X, split_y = _auto_cap_large_presplit(
        split_X,
        split_y,
        threshold=int(args.large_task_threshold),
        cap=int(args.large_task_cap),
        seed=int(seed),
        task_name=str(task_name),
    )
    for split in ("train", "val", "test"):
        limit = {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test}[split]
        if int(limit) > 0 and len(split_X[split]) > int(limit):
            Xi, ys = maybe_limit_split(
                split_X[split],
                {str(task_name): split_y[split]},
                int(limit),
                int(seed) + {"train": 0, "val": 1, "test": 2}[split],
            )
            split_X[split] = np.asarray(Xi, dtype=np.float32)
            split_y[split] = np.asarray(ys[str(task_name)], dtype=np.int64)

    y_all = np.concatenate([np.asarray(split_y[split], dtype=np.int64) for split in ("train", "val", "test") if len(split_y[split]) > 0], axis=0)
    runtime = TaskRuntime(
        TaskSpec(str(task_name), "cls", "", str(TASK_REGISTRY[task_name]["primary_metric"])),
        n_out=int(np.max(y_all)) + 1 if len(y_all) > 0 else (2 if task_name == "sdb_binary" else 4),
    )
    return TaskData(runtime=runtime, split_X=split_X, split_y=split_y)


def _center_crop_np(X: np.ndarray, target_len: int) -> np.ndarray:
    if int(target_len) <= 0 or int(X.shape[-1]) <= int(target_len):
        return np.asarray(X, dtype=np.float32)
    start = max(0, int((int(X.shape[-1]) - int(target_len)) // 2))
    end = start + int(target_len)
    return np.ascontiguousarray(np.asarray(X, dtype=np.float32)[..., start:end])


def _build_ppg_task_data_with_length(task_name: str, args: argparse.Namespace, seed: int, target_length_sec: float) -> TaskData:
    native_sec = _task_current_default_length_sec(task_name, int(args.hz))
    if _approx_equal_sec(float(target_length_sec), float(native_sec)):
        return build_ppg_task_data(str(task_name), args, int(seed))
    cache_root = _task_length_cache_root(str(task_name), float(target_length_sec))
    if cache_root is not None:
        return _build_ppg_task_data_from_root(str(task_name), args, int(seed), cache_root)
    if float(target_length_sec) > float(native_sec):
        if str(task_name).startswith("sdb_"):
            return _build_sdb_task_data_from_raw(str(task_name), args, int(seed), float(target_length_sec))
        raise ValueError(f"Unsupported longer-window build for {task_name}: requested {target_length_sec}s > native {native_sec}s")

    task_data = build_ppg_task_data(str(task_name), args, int(seed))
    target_len = max(1, int(round(float(target_length_sec) * float(args.hz))))
    split_X = {
        split: _center_crop_np(X, target_len)
        for split, X in task_data.split_X.items()
    }
    split_y = {split: np.asarray(y).copy() for split, y in task_data.split_y.items()}
    return TaskData(runtime=copy.deepcopy(task_data.runtime), split_X=split_X, split_y=split_y)


def _case_is_same_as_same10(case_cfg: Dict[str, Any], chosen_sec: Dict[str, float], current_default_sec: Dict[str, float]) -> bool:
    for task_name in list(case_cfg["source_tasks"]) + list(case_cfg["target_tasks"]):
        if not _approx_equal_sec(chosen_sec[str(task_name)], current_default_sec[str(task_name)]):
            return False
    return True


def _reuse_same10_rows_for_case(
    *,
    case_name: str,
    case_cfg: Dict[str, Any],
    length_profile: str,
    methods: Sequence[str],
    seeds: Sequence[int],
    chosen_sec: Dict[str, float],
    same10_rows: pd.DataFrame,
) -> List[Dict[str, Any]]:
    student_lookup = _student_e0_lookup(same10_rows)
    teacher_lookup = _teacher_probe_lookup(same10_rows)
    source_len_map = {str(task): float(chosen_sec[str(task)]) for task in case_cfg["source_tasks"]}
    reused_rows: List[Dict[str, Any]] = []
    for method in methods:
        for target_task in case_cfg["target_tasks"]:
            for seed in seeds:
                subset = same10_rows[
                    (same10_rows["case_name"] == str(case_name))
                    & (same10_rows["method"] == str(method))
                    & (same10_rows["target_task"] == str(target_task))
                    & (same10_rows["seed"] == int(seed))
                ]
                if subset.empty:
                    raise KeyError(f"Missing same-10 reference row for case={case_name} method={method} target={target_task} seed={seed}")
                row = dict(subset.iloc[-1].to_dict())
                higher_is_better = bool(row.get("higher_is_better", False))
                key = (str(case_name), str(target_task), int(seed))
                student_e0 = float(student_lookup[key])
                teacher_e0 = float(teacher_lookup[key])
                after = float(row.get("target_after_stage2_probe", np.nan))
                retention_student, degradation_student = _retention_and_degradation(student_e0, after, higher_is_better)
                retention_teacher, degradation_teacher = _retention_and_degradation(teacher_e0, after, higher_is_better)
                row.update(
                    {
                        "length_profile": str(length_profile),
                        "source_task_lengths_sec": _json_float_map(source_len_map),
                        "target_length_sec": float(chosen_sec[str(target_task)]),
                        "target_E0_probe": float(student_e0),
                        "target_teacher_probe": float(teacher_e0),
                        "retention": float(retention_student),
                        "degradation": float(degradation_student),
                        "retention_vs_student_e0": float(retention_student),
                        "retention_vs_teacher_e0": float(retention_teacher),
                        "degradation_vs_student_e0": float(degradation_student),
                        "degradation_vs_teacher_e0": float(degradation_teacher),
                        "result_origin": "same10_reused",
                        "same10_reference_result_dir": str(row.get("result_dir", "")),
                        "same10_reference_source_file": str(row.get("source_file", "")),
                    }
                )
                reused_rows.append(row)
    return reused_rows


def _load_stage1_from_path(student: str, ckpt_path: Path, seed: int, device: torch.device) -> tuple[nn.Module, Dict[str, Any]]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    d_full = int(obj.get("d_full", 512))
    enc, enc_dim = build_encoder(student, int(seed))
    proj = nn.Linear(int(enc_dim), d_full, bias=False)
    full = nn.Sequential(enc, proj)
    full.load_state_dict(obj["state_dict"], strict=True)
    enc = full[0].to(device)
    disable_dropout(enc)
    meta = dict(obj)
    meta.pop("state_dict", None)
    meta["ckpt_path"] = str(ckpt_path)
    meta["enc_dim"] = int(enc_dim)
    return enc, meta


def load_stage1_encoder(args: argparse.Namespace, seed: int, device: torch.device) -> tuple[nn.Module, Dict[str, Any]]:
    ckpt_seed = int(args.stage1_ckpt_seed) if args.stage1_ckpt_seed is not None else int(seed)
    ckpt_spec = str(args.encoder_ckpt).strip()
    if ckpt_spec:
        if "{" in ckpt_spec:
            ckpt_path = Path(ckpt_spec.format(seed=int(ckpt_seed), method=str(args.stage1_method)))
            return _load_stage1_from_path(str(args.student), ckpt_path, int(ckpt_seed), device)
        ckpt_path = Path(ckpt_spec)
        if ckpt_path.is_dir():
            enc, _, meta = load_stage1_full(str(args.student), ckpt_path, str(args.stage1_method), int(ckpt_seed), device)
            return enc, meta
        return _load_stage1_from_path(str(args.student), ckpt_path, int(ckpt_seed), device)

    ckpt_dir = LOCAL_CKPT_DIRS.get(str(args.student))
    if ckpt_dir is None:
        raise KeyError(f"No default PPG checkpoint dir for student {args.student}")
    enc, _, meta = load_stage1_full(str(args.student), Path(ckpt_dir), str(args.stage1_method), int(ckpt_seed), device)
    return enc, meta


def _resolve_stage1_public_pool_path(args: argparse.Namespace, stage1_meta: Dict[str, Any]) -> str:
    candidates: List[Path] = []
    ckpt_path_str = str(stage1_meta.get("ckpt_path", "")).strip()
    if ckpt_path_str and ckpt_path_str != "random_init":
        ckpt_path = Path(ckpt_path_str)
        if ckpt_path.exists():
            if ckpt_path.is_dir():
                candidates.append(ckpt_path)
            else:
                candidates.extend([ckpt_path.parent, ckpt_path.parent.parent])

    local_ckpt_dir = LOCAL_CKPT_DIRS.get(str(args.student))
    if local_ckpt_dir is not None:
        candidates.append(Path(local_ckpt_dir))

    candidates.extend(
        [
            ROOT / f"kd_sweep_results/pulseppg_preliminary/exp_ns_kd_{str(args.student)}",
            ROOT / "kd_sweep_results/pulseppg_preliminary/exp_ns_kd",
        ]
    )

    seen: set[str] = set()
    for base in candidates:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        pool_path = base / "X_mix.pt"
        if pool_path.exists():
            return str(pool_path)

    raise FileNotFoundError(
        "Unable to resolve X_mix.pt for care anchor source stage1_public_pool/mixed_source_public; "
        f"student={args.student} ckpt_path={ckpt_path_str}"
    )


def load_pulseppg_teacher_encoder(device: torch.device) -> tuple[nn.Module, Dict[str, Any]]:
    from run_k_probe_pulseppg import load_pulseppg_teacher

    teacher, teacher_name = load_pulseppg_teacher(device)
    wrapped = TeacherEncoderAdapter(teacher).to(device).eval()
    with torch.no_grad():
        dummy = torch.randn(2, 1, 500, device=device)
        enc_dim = int(wrapped(dummy).shape[-1])
    ckpt_path = ROOT / "pulseppg/experiments/out" / str(teacher_name) / "checkpoint_best.pkl"
    return wrapped, {
        "teacher_name": str(teacher_name),
        "ckpt_path": str(ckpt_path),
        "enc_dim": int(enc_dim),
    }


def build_stage2_multitask_model(encoder: nn.Module, enc_dim: int, task_data_all: List[TaskData], dropout: float, hz: int) -> MultiTaskModel:
    dataset_names = {task_dataset(td.runtime.spec.name) for td in task_data_all}
    if len(dataset_names) != 1:
        raise ValueError(f"Stage2 source tasks must share one dataset, got {sorted(dataset_names)}")
    input_lens = [int(td.split_X["train"].shape[-1]) for td in task_data_all]
    return MultiTaskModel(
        copy.deepcopy(encoder),
        int(enc_dim),
        [td.runtime for td in task_data_all],
        float(dropout),
        dataset_name=next(iter(dataset_names)),
        hz=int(hz),
        input_len=max(input_lens),
        temporal_policy="shared_window",
    )


def _metric_value_from_payload(metric: Dict[str, Any]) -> float:
    return float(metric.get("value", np.nan))


def _normalized_gain(before_metric: Dict[str, Any], after_metric: Dict[str, Any]) -> float:
    before = _metric_value_from_payload(before_metric)
    after = _metric_value_from_payload(after_metric)
    higher_better = bool(before_metric.get("higher_better", after_metric.get("higher_better", False)))
    if not np.isfinite(before) or not np.isfinite(after):
        return float("nan")
    if higher_better:
        return float(after / max(before, 1e-12))
    return float(before / max(after, 1e-12))


def _retention_and_degradation(target_e0_probe: float, target_after_stage2_probe: float, higher_is_better: bool) -> tuple[float, float]:
    if not np.isfinite(target_e0_probe) or not np.isfinite(target_after_stage2_probe):
        return float("nan"), float("nan")
    if higher_is_better:
        retention = float(target_after_stage2_probe / max(target_e0_probe, 1e-12))
        degradation = float(target_e0_probe - target_after_stage2_probe)
        return retention, degradation
    retention = float(target_e0_probe / max(target_after_stage2_probe, 1e-12))
    degradation = float(target_after_stage2_probe - target_e0_probe)
    return retention, degradation


def _metrics_json(metric_map: Dict[str, Dict[str, Any]]) -> str:
    return json.dumps(metric_map, sort_keys=True)


def _format_metric(metric: Dict[str, Any]) -> str:
    return f"{metric.get('metric', '')}={_metric_value_from_payload(metric):.6f}"


def _source_metric_map_from_baselines(probes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {task_name: metric_payload(probe["test_metric"]) for task_name, probe in probes.items()}


def _source_metric_map_from_stage2(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {task_name: metric_payload(metric) for task_name, metric in metrics.items()}


def _row_key(row: Dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("case_name", "")),
        str(row.get("method", "")),
        str(row.get("target_task", "")),
        int(row.get("seed", -1)),
    )


def _dedupe_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for row in rows:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped


def _load_existing_rows(output_dir: Path, case_names: Sequence[str]) -> tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], set[tuple[str, str, str, int]]]:
    overall_path = output_dir / "overall_summary.csv"
    if overall_path.exists():
        overall_df = pd.read_csv(overall_path)
        all_rows = _dedupe_rows(overall_df.to_dict(orient="records"))
    else:
        all_rows = []
        for case_name in case_names:
            case_path = output_dir / f"{case_name}_detailed.csv"
            if case_path.exists():
                case_df = pd.read_csv(case_path)
                all_rows.extend(case_df.to_dict(orient="records"))
        all_rows = _dedupe_rows(all_rows)
    case_rows = {
        case_name: [dict(row) for row in all_rows if str(row.get("case_name", "")) == str(case_name)]
        for case_name in case_names
    }
    return case_rows, list(all_rows), {_row_key(row) for row in all_rows}


def _ensure_safe_output_dir(output_dir: Path, resume: bool) -> None:
    if resume:
        return
    candidate_files = [output_dir / "overall_summary.csv", output_dir / "source_gain_vs_future_retention.csv"]
    candidate_files.extend(output_dir.glob("*_detailed.csv"))
    if any(path.exists() for path in candidate_files):
        raise FileExistsError(
            f"Refusing to write into existing result directory without --resume: {output_dir}"
        )


def _stage2_checkpoint_path(output_dir: Path, case_name: str, method: str, seed: int) -> Path:
    return output_dir / "artifacts" / str(case_name) / str(method) / f"seed{int(seed)}" / "stage2_final.pt"


def _save_stage2_checkpoint(
    *,
    checkpoint_path: Path,
    stage2_model: MultiTaskModel,
    seed: int,
    case_name: str,
    method: str,
    stage2_meta: Dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": stage2_model.state_dict(),
            "seed": int(seed),
            "case_name": str(case_name),
            "method": str(method),
            "stage2_meta": dict(stage2_meta),
        },
        checkpoint_path,
    )


def _write_outputs(output_dir: Path, case_rows: Dict[str, List[Dict[str, Any]]], all_rows: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = _dedupe_rows(all_rows)
    for case_name, rows in case_rows.items():
        pd.DataFrame(_dedupe_rows(rows)).to_csv(output_dir / f"{case_name}_detailed.csv", index=False)
    overall_df = pd.DataFrame(all_rows)
    overall_df.to_csv(output_dir / "overall_summary.csv", index=False)
    tradeoff_cols = [
        "case_name",
        "method",
        "target_task",
        "seed",
        "length_profile",
        "source_task_lengths_sec",
        "target_length_sec",
        "source_gain_normalized",
        "retention",
        "degradation",
        "higher_is_better",
        "target_metric",
        "result_origin",
    ]
    tradeoff_df = overall_df[tradeoff_cols].copy() if not overall_df.empty else pd.DataFrame(columns=tradeoff_cols)
    if not tradeoff_df.empty:
        tradeoff_df["retention_below_one"] = tradeoff_df["retention"] < 1.0
    else:
        tradeoff_df["retention_below_one"] = pd.Series(dtype=bool)
    tradeoff_df.to_csv(output_dir / "source_gain_vs_future_retention.csv", index=False)


def run_case(
    *,
    case_name: str,
    case_cfg: Dict[str, Any],
    task_length_sec_map: Dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
    case_rows: Dict[str, List[Dict[str, Any]]],
    all_rows: List[Dict[str, Any]],
    completed_keys: set[tuple[str, str, str, int]],
) -> None:
    source_tasks = list(map(str, case_cfg["source_tasks"]))
    target_tasks = list(map(str, case_cfg["target_tasks"]))
    source_length_map = {str(task_name): float(task_length_sec_map[str(task_name)]) for task_name in source_tasks}
    target_length_map = {str(task_name): float(task_length_sec_map[str(task_name)]) for task_name in target_tasks}

    for seed in args.seeds:
        case_seed_expected = [
            (str(case_name), str(method), str(target_task), int(seed))
            for method in args.methods
            for target_task in target_tasks
        ]
        if case_seed_expected and all(key in completed_keys for key in case_seed_expected):
            print(f"[resume] skip case={case_name} seed={seed} (all targets already complete)", flush=True)
            continue

        set_seed(int(seed))
        task_bank: Dict[str, TaskData] = {}
        for task_name in list(dict.fromkeys(source_tasks + target_tasks)):
            task_bank[str(task_name)] = _build_ppg_task_data_with_length(
                str(task_name),
                args,
                int(seed),
                float(task_length_sec_map[str(task_name)]),
            )
        print(
            f"[case-lengths] case={case_name} seed={seed} source={_json_float_map(source_length_map)} targets={_json_float_map(target_length_map)} profile={args.length_profile}",
            flush=True,
        )
        source_train_counts = {str(task_name): int(len(task_bank[str(task_name)].split_X["train"])) for task_name in source_tasks}
        source_total_counts = {
            str(task_name): int(sum(len(task_bank[str(task_name)].split_X[split]) for split in ("train", "val", "test")))
            for task_name in source_tasks
        }
        target_total_counts = {
            str(task_name): int(sum(len(task_bank[str(task_name)].split_X[split]) for split in ("train", "val", "test")))
            for task_name in target_tasks
        }

        enc_e0: nn.Module | None = None
        ck_meta: Dict[str, Any] | None = None
        enc_dim: int | None = None
        source_baselines: Dict[str, Dict[str, Any]] | None = None
        target_baselines: Dict[str, Dict[str, Any]] | None = None
        student_source_before: Dict[str, Dict[str, Any]] | None = None

        def ensure_student_baselines() -> None:
            nonlocal enc_e0, ck_meta, enc_dim, source_baselines, target_baselines, student_source_before
            if enc_e0 is not None and ck_meta is not None and enc_dim is not None and source_baselines is not None and target_baselines is not None and student_source_before is not None:
                return

            enc_e0, ck_meta = load_stage1_encoder(args, int(seed), device)
            enc_dim = int(ck_meta["enc_dim"])
            print(
                f"[case] name={case_name} seed={seed} desc={case_cfg['description']} ckpt={ck_meta['ckpt_path']}",
                flush=True,
            )

            source_baselines = {}
            target_baselines = {}
            for source_task in source_tasks:
                print(f"[baseline source] case={case_name} seed={seed} task={source_task}", flush=True)
                source_baselines[source_task] = fit_frozen_probe(
                    encoder=enc_e0,
                    enc_dim=enc_dim,
                    task_data=task_bank[source_task],
                    device=device,
                    args=args,
                    log_prefix=f"[E0src/{case_name}/{source_task}/s{seed}] ",
                )
            for target_task in target_tasks:
                print(f"[baseline target] case={case_name} seed={seed} task={target_task}", flush=True)
                target_baselines[target_task] = fit_frozen_probe(
                    encoder=enc_e0,
                    enc_dim=enc_dim,
                    task_data=task_bank[target_task],
                    device=device,
                    args=args,
                    log_prefix=f"[A/{case_name}/{target_task}/s{seed}] ",
                )

            student_source_before = _source_metric_map_from_baselines(source_baselines)
        teacher_encoder: nn.Module | None = None
        teacher_ck_meta: Dict[str, Any] | None = None
        teacher_enc_dim: int | None = None
        teacher_source_baselines: Dict[str, Dict[str, Any]] | None = None
        teacher_target_baselines: Dict[str, Dict[str, Any]] | None = None
        teacher_source_before: Dict[str, Dict[str, Any]] | None = None

        def ensure_teacher_encoder_loaded() -> None:
            nonlocal teacher_encoder, teacher_ck_meta, teacher_enc_dim
            if teacher_encoder is not None and teacher_ck_meta is not None and teacher_enc_dim is not None:
                return
            teacher_encoder, teacher_ck_meta = load_pulseppg_teacher_encoder(device)
            teacher_enc_dim = int(teacher_ck_meta["enc_dim"])
            print(
                f"[case] name={case_name} seed={seed} desc={case_cfg['description']} teacher={teacher_ck_meta['teacher_name']} ckpt={teacher_ck_meta['ckpt_path']}",
                flush=True,
            )

        def ensure_teacher_target_baselines() -> None:
            nonlocal teacher_target_baselines
            if teacher_target_baselines is not None:
                return
            if bool(args.skip_teacher_baselines):
                teacher_target_baselines = {}
                return
            ensure_teacher_encoder_loaded()
            assert teacher_encoder is not None
            assert teacher_enc_dim is not None
            teacher_target_baselines = {}
            for target_task in target_tasks:
                print(f"[teacher baseline target] case={case_name} seed={seed} task={target_task}", flush=True)
                teacher_target_baselines[target_task] = fit_frozen_probe(
                    encoder=teacher_encoder,
                    enc_dim=teacher_enc_dim,
                    task_data=task_bank[target_task],
                    device=device,
                    args=args,
                    log_prefix=f"[TeacherA/{case_name}/{target_task}/s{seed}] ",
                )

        def ensure_teacher_source_baselines() -> None:
            nonlocal teacher_source_baselines, teacher_source_before
            if teacher_source_baselines is not None and teacher_source_before is not None:
                return
            if bool(args.skip_teacher_baselines):
                teacher_source_baselines = {}
                teacher_source_before = {}
                return
            ensure_teacher_encoder_loaded()
            assert teacher_encoder is not None
            assert teacher_enc_dim is not None
            teacher_source_baselines = {}
            for source_task in source_tasks:
                print(f"[teacher baseline source] case={case_name} seed={seed} task={source_task}", flush=True)
                teacher_source_baselines[source_task] = fit_frozen_probe(
                    encoder=teacher_encoder,
                    enc_dim=teacher_enc_dim,
                    task_data=task_bank[source_task],
                    device=device,
                    args=args,
                    log_prefix=f"[TeacherE0src/{case_name}/{source_task}/s{seed}] ",
                )
            teacher_source_before = _source_metric_map_from_baselines(teacher_source_baselines)

        for method in args.methods:
            method_expected = [
                (str(case_name), str(method), str(target_task), int(seed))
                for target_task in target_tasks
            ]
            if method_expected and all(key in completed_keys for key in method_expected):
                print(f"[resume] skip case={case_name} seed={seed} method={method} (all targets already complete)", flush=True)
                continue

            print(
                f"[stage2-config] case={case_name} seed={seed} method={method} early_stop={not args.no_early_stop_stage2 if method not in NON_STAGE2_METHODS else False} checkpoint={'final' if args.use_final_checkpoint or method in NON_STAGE2_METHODS else 'best'} batch_size={args.batch_size if method not in NON_STAGE2_METHODS else 0} epochs={args.stage2_epochs if method not in NON_STAGE2_METHODS else 0} fairgrad_alpha={args.fairgrad_alpha if method == 'fairgrad_mtl' else 'na'}",
                flush=True,
            )

            if bool(args.skip_teacher_baselines) and method == "pulseppg_teacher_e0":
                raise ValueError("--skip_teacher_baselines cannot be used with method pulseppg_teacher_e0")

            stage2_meta: Dict[str, Any] = {}
            if method == "pulseppg_teacher_e0":
                ensure_teacher_source_baselines()
                ensure_teacher_target_baselines()
                assert teacher_encoder is not None
                assert teacher_enc_dim is not None
                assert teacher_source_before is not None
                source_before = teacher_source_before
                source_final = {task_name: dict(metric) for task_name, metric in source_before.items()}
                adapted_encoder = teacher_encoder
                probe_enc_dim = int(teacher_enc_dim)
                checkpoint_used = "final"
                stage2_early_stop = False
                stage2_checkpoint_path = ""
            elif method == "e0_probe":
                ensure_student_baselines()
                assert student_source_before is not None
                assert enc_e0 is not None
                assert enc_dim is not None
                source_before = student_source_before
                source_final = {task_name: dict(metric) for task_name, metric in source_before.items()}
                adapted_encoder = enc_e0
                probe_enc_dim = int(enc_dim)
                checkpoint_used = "final"
                stage2_early_stop = False
                stage2_checkpoint_path = ""
            else:
                ensure_student_baselines()
                assert student_source_before is not None
                assert enc_e0 is not None
                assert enc_dim is not None
                source_before = student_source_before
                care_public_pool_path = ""
                if str(args.care_anchor_source) in {"stage1_public_pool", "mixed_source_public"}:
                    care_public_pool_path = _resolve_stage1_public_pool_path(args, ck_meta)
                source_task_data = [task_bank[task_name] for task_name in source_tasks]
                stage2_model = build_stage2_multitask_model(enc_e0, enc_dim, source_task_data, float(args.dropout), int(args.hz)).to(device)
                train_loaders, val_loaders, test_loaders = make_task_loaders_for_device(source_task_data, int(args.batch_size), device)
                stage2_model, stage2_test_metrics, stage2_meta = fit_stage2(
                    model=stage2_model,
                    method=str(METHOD_TO_STAGE2[method]),
                    tasks=[td.runtime for td in source_task_data],
                    train_loaders=train_loaders,
                    val_loaders=val_loaders,
                    test_loaders=test_loaders,
                    device=device,
                    epochs=int(args.stage2_epochs),
                    patience=int(args.stage2_patience),
                    head_lr=float(args.stage2_head_lr),
                    encoder_lr_mult=float(args.stage2_encoder_lr_mult),
                    weight_decay=float(args.weight_decay),
                    log_prefix=f"[stage2/{case_name}/{method}/s{seed}] ",
                    enable_early_stop=not bool(args.no_early_stop_stage2),
                    restore_best_checkpoint=not bool(args.use_final_checkpoint),
                    task_loss_reduction="sum",
                    fairgrad_alpha=float(args.fairgrad_alpha),
                    l2sp_lambda=float(args.l2sp_lambda),
                    care_auto_anchor_source=str(args.care_anchor_source),
                    care_auto_reuse_feature_space=str(args.care_reuse_feature_space),
                    care_auto_reuse_k=int(args.care_reuse_k),
                    care_auto_reuse_lambda=float(args.care_reuse_lambda),
                    care_auto_stage2_scope=str(args.care_train_scope),
                    care_auto_reuse_loss_type=str(args.care_reuse_loss_type),
                    care_auto_anchor_max_batches=int(args.care_anchor_max_batches),
                    care_auto_anchor_basis_samples=int(args.care_anchor_basis_samples),
                    care_auto_public_pool_path=str(care_public_pool_path),
                    care_auto_drift_budget=float(args.care_drift_budget),
                    care_auto_stat_lambda=float(args.care_stat_lambda),
                )
                source_final = _source_metric_map_from_stage2(stage2_test_metrics)
                adapted_encoder = stage2_model.encoder
                probe_enc_dim = int(enc_dim)
                checkpoint_used = str(stage2_meta.get("checkpoint_used", "final"))
                stage2_early_stop = bool(stage2_meta.get("early_stop_enabled", False))
                stage2_ckpt = _stage2_checkpoint_path(Path(args.output_dir), str(case_name), str(method), int(seed))
                _save_stage2_checkpoint(
                    checkpoint_path=stage2_ckpt,
                    stage2_model=stage2_model,
                    seed=int(seed),
                    case_name=str(case_name),
                    method=str(method),
                    stage2_meta=stage2_meta,
                )
                stage2_checkpoint_path = str(stage2_ckpt)

            source_gains: List[float] = []
            for source_task in source_tasks:
                gain = _normalized_gain(source_before[source_task], source_final[source_task])
                source_gains.append(gain)
                print(
                    f"[source-metric] case={case_name} seed={seed} method={method} task={source_task} before={_format_metric(source_before[source_task])} final={_format_metric(source_final[source_task])} normalized_gain={gain:.6f}",
                    flush=True,
                )
            source_gain_normalized = float(np.nanmean(np.asarray(source_gains, dtype=np.float64))) if source_gains else float("nan")
            ensure_student_baselines()
            assert target_baselines is not None
            if not bool(args.skip_teacher_baselines):
                ensure_teacher_target_baselines()
                assert teacher_target_baselines is not None

            for target_task in target_tasks:
                key = (str(case_name), str(method), str(target_task), int(seed))
                if key in completed_keys:
                    print(f"[resume] skip case={case_name} seed={seed} method={method} target={target_task}", flush=True)
                    continue

                assert target_baselines is not None
                student_target_baseline = target_baselines[target_task]
                target_e0_probe = metric_value(student_target_baseline["test_metric"])
                if bool(args.skip_teacher_baselines):
                    target_teacher_probe = float("nan")
                else:
                    assert teacher_target_baselines is not None
                    teacher_target_baseline = teacher_target_baselines[target_task]
                    target_teacher_probe = metric_value(teacher_target_baseline["test_metric"])
                higher_is_better = bool(student_target_baseline["test_metric"]["higher_better"])
                target_metric = str(student_target_baseline["test_metric"]["metric"])

                if method == "pulseppg_teacher_e0":
                    target_after_stage2_probe = float(target_teacher_probe)
                elif method in NON_STAGE2_METHODS:
                    target_after_stage2_probe = float(target_e0_probe)
                else:
                    print(f"[probe] case={case_name} seed={seed} method={method} target={target_task}", flush=True)
                    probe = fit_frozen_probe(
                        encoder=adapted_encoder,
                        enc_dim=probe_enc_dim,
                        task_data=task_bank[target_task],
                        device=device,
                        args=args,
                        log_prefix=f"[probe/{case_name}/{method}/{target_task}/s{seed}] ",
                    )
                    target_after_stage2_probe = metric_value(probe["test_metric"])

                retention, degradation = _retention_and_degradation(target_e0_probe, target_after_stage2_probe, higher_is_better)
                if bool(args.skip_teacher_baselines):
                    retention_vs_teacher = float("nan")
                    degradation_vs_teacher = float("nan")
                else:
                    retention_vs_teacher, degradation_vs_teacher = _retention_and_degradation(target_teacher_probe, target_after_stage2_probe, higher_is_better)
                print(
                    f"[target-probe] case={case_name} seed={seed} method={method} target={target_task} target_len_sec={target_length_map[target_task]:.3f} metric={target_metric} e0={target_e0_probe:.6f} teacher={target_teacher_probe:.6f} after={target_after_stage2_probe:.6f} retention={retention:.6f} degradation={degradation:.6f} retention_lt_1={bool(np.isfinite(retention) and retention < 1.0)}",
                    flush=True,
                )

                row = {
                    "case_name": str(case_name),
                    "method": str(method),
                    "source_tasks": json.dumps(source_tasks),
                    "target_task": str(target_task),
                    "seed": int(seed),
                    "length_profile": str(args.length_profile),
                    "source_task_lengths_sec": _json_float_map(source_length_map),
                    "target_length_sec": float(target_length_map[target_task]),
                    "source_train_window_counts_json": json.dumps(source_train_counts, sort_keys=True),
                    "source_total_window_counts_json": json.dumps(source_total_counts, sort_keys=True),
                    "target_total_window_counts_json": json.dumps(target_total_counts, sort_keys=True),
                    "target_total_window_count": int(target_total_counts[str(target_task)]),
                    "stage2_batch_size": int(args.batch_size) if method not in NON_STAGE2_METHODS else 0,
                    "stage2_epochs": int(args.stage2_epochs) if method not in NON_STAGE2_METHODS else 0,
                    "stage2_early_stop": bool(stage2_early_stop),
                    "checkpoint_used": str(checkpoint_used),
                    "stage2_checkpoint_path": str(stage2_checkpoint_path),
                    "stage2_final_checkpoint_exists": bool(stage2_checkpoint_path and Path(stage2_checkpoint_path).exists()),
                    "source_task_metrics_before": _metrics_json(source_before),
                    "source_task_metrics_final": _metrics_json(source_final),
                    "source_gain_normalized": float(source_gain_normalized),
                    "target_metric": str(target_metric),
                    "target_E0_probe": float(target_e0_probe),
                    "target_teacher_probe": float(target_teacher_probe),
                    "target_after_stage2_probe": float(target_after_stage2_probe),
                    "retention": float(retention),
                    "degradation": float(degradation),
                    "retention_vs_student_e0": float(retention),
                    "retention_vs_teacher_e0": float(retention_vs_teacher),
                    "degradation_vs_student_e0": float(degradation),
                    "degradation_vs_teacher_e0": float(degradation_vs_teacher),
                    "higher_is_better": bool(higher_is_better),
                    "fairgrad_alpha": float(args.fairgrad_alpha) if method == "fairgrad_mtl" else float("nan"),
                    "reuse_loss_last": float(stage2_meta.get("reuse_loss_last", np.nan)),
                    "reuse_loss_mean": float(stage2_meta.get("reuse_loss_mean", np.nan)),
                    "anchor_feature_drift_mse": float(stage2_meta.get("anchor_feature_drift_mse", np.nan)),
                    "anchor_feature_drift_cosine": float(stage2_meta.get("anchor_feature_drift_cosine", np.nan)),
                    "param_drift_rel": float(stage2_meta.get("param_drift_rel", np.nan)),
                    "feature_norm_before": float(stage2_meta.get("feature_norm_before", np.nan)),
                    "feature_norm_after": float(stage2_meta.get("feature_norm_after", np.nan)),
                    "feature_norm_ratio": float(stage2_meta.get("feature_norm_ratio", np.nan)),
                    "source_loss_mean": float(stage2_meta.get("source_loss_mean", np.nan)),
                    "reuse_loss_scale_relative_to_source": float(stage2_meta.get("reuse_loss_scale_relative_to_source", np.nan)),
                    "stat_loss_last": float(stage2_meta.get("stat_loss_last", np.nan)),
                    "anchor_feature_norm_mean": float(stage2_meta.get("anchor_feature_norm_mean", np.nan)),
                    "trainable_param_count": int(stage2_meta.get("trainable_param_count", 0)),
                    "encoder_trainable_param_count": int(stage2_meta.get("encoder_trainable_param_count", 0)),
                    "updated_layer_names_json": json.dumps(stage2_meta.get("updated_layer_names", [])),
                    "updated_param_names_json": json.dumps(stage2_meta.get("updated_param_names", [])),
                    "reuse_feature_space": str(stage2_meta.get("reuse_feature_space", "")),
                    "reuse_k": int(stage2_meta.get("reuse_k", 0)),
                    "reuse_k_candidates_json": json.dumps(stage2_meta.get("reuse_k_candidates", [])),
                    "reuse_lambda": float(stage2_meta.get("reuse_lambda", np.nan)),
                    "reuse_lambda_effective_last": float(stage2_meta.get("reuse_lambda_effective_last", np.nan)),
                    "anchor_source": str(stage2_meta.get("anchor_source", "")),
                    "care_variant": str(stage2_meta.get("care_variant", "")),
                    "train_scope": str(stage2_meta.get("train_scope", "")),
                    "feature_dim": int(stage2_meta.get("feature_dim", 0)),
                    "anchor_basis_batches": int(stage2_meta.get("anchor_basis_batches", 0)),
                    "anchor_basis_samples": int(stage2_meta.get("anchor_basis_samples", 0)),
                    "result_origin": "fresh_run" if str(args.length_profile) == "same10" else "tasklen_rerun",
                }
                case_rows[case_name].append(row)
                all_rows.append(row)
                completed_keys.add(key)
                _write_outputs(Path(args.output_dir), case_rows, all_rows)


def main() -> None:
    args = parse_args()
    args.cases = _parse_name_list(args.cases, CASE_DEFS.keys(), allow_all=True)
    args.methods = _parse_name_list(args.methods, ALL_METHODS, allow_all=False)
    device = resolve_device(str(args.device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows, current_default_sec, chosen_sec = _resolve_task_length_profile(args)
    _write_task_length_artifacts(
        output_dir,
        args=args,
        audit_rows=audit_rows,
        current_default_sec=current_default_sec,
        chosen_sec=chosen_sec,
    )
    config = {
        "encoder_ckpt": str(args.encoder_ckpt),
        "student": str(args.student),
        "stage1_method": str(args.stage1_method),
        "cases": list(args.cases),
        "methods": list(args.methods),
        "batch_size": int(args.batch_size),
        "stage2_epochs": int(args.stage2_epochs),
        "probe_epochs": int(args.probe_epochs),
        "fairgrad_alpha": float(args.fairgrad_alpha),
        "l2sp_lambda": float(args.l2sp_lambda),
        "care_anchor_source": str(args.care_anchor_source),
        "care_reuse_feature_space": str(args.care_reuse_feature_space),
        "care_reuse_k": int(args.care_reuse_k),
        "care_reuse_lambda": float(args.care_reuse_lambda),
        "care_train_scope": str(args.care_train_scope),
        "care_reuse_loss_type": str(args.care_reuse_loss_type),
        "care_stat_lambda": float(args.care_stat_lambda),
        "care_anchor_max_batches": int(args.care_anchor_max_batches),
        "care_anchor_basis_samples": int(args.care_anchor_basis_samples),
        "care_drift_budget": float(args.care_drift_budget),
        "seeds": [int(seed) for seed in args.seeds],
        "length_profile": str(args.length_profile),
        "same10_baseline_csv": str(args.same10_baseline_csv),
        "task_lengths_sec": {k: float(v) for k, v in chosen_sec.items()},
        "resume": bool(args.resume),
        "skip_teacher_baselines": bool(args.skip_teacher_baselines),
        "no_early_stop_stage2": bool(args.no_early_stop_stage2),
        "use_final_checkpoint": bool(args.use_final_checkpoint),
        "device": str(device),
    }
    _ensure_safe_output_dir(output_dir, bool(args.resume))
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.resume:
        case_rows, all_rows, completed_keys = _load_existing_rows(output_dir, args.cases)
        print(f"[resume] loaded rows={len(all_rows)} from {output_dir}", flush=True)
    else:
        case_rows = {case_name: [] for case_name in args.cases}
        all_rows = []
        completed_keys: set[tuple[str, str, str, int]] = set()
    same10_rows = _load_same10_reference_rows(Path(args.same10_baseline_csv)) if str(args.length_profile) != "same10" else pd.DataFrame()
    for case_name in args.cases:
        if str(args.length_profile) != "same10" and _case_is_same_as_same10(CASE_DEFS[case_name], chosen_sec, current_default_sec):
            reused_rows = _reuse_same10_rows_for_case(
                case_name=str(case_name),
                case_cfg=CASE_DEFS[case_name],
                length_profile=str(args.length_profile),
                methods=list(args.methods),
                seeds=list(args.seeds),
                chosen_sec=chosen_sec,
                same10_rows=same10_rows,
            )
            added = 0
            for row in reused_rows:
                key = _row_key(row)
                if key in completed_keys:
                    continue
                case_rows[str(case_name)].append(row)
                all_rows.append(row)
                completed_keys.add(key)
                added += 1
            _write_outputs(output_dir, case_rows, all_rows)
            print(f"[same10-reuse] case={case_name} reused_rows={added} profile={args.length_profile}", flush=True)
            continue
        run_case(
            case_name=case_name,
            case_cfg=CASE_DEFS[case_name],
            task_length_sec_map=chosen_sec,
            args=args,
            device=device,
            case_rows=case_rows,
            all_rows=all_rows,
            completed_keys=completed_keys,
        )
    _write_outputs(output_dir, case_rows, all_rows)
    print(f"[done] wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
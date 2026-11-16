#!/usr/bin/env python3
"""Single-source specialization sweep for PPG Stage-2 stress-test search."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from downstream_eval import TASK_REGISTRY, _kfold_arrays, _presplit_arrays
from run_stage2_multitask_ppg import (
    ROOT,
    TaskData,
    TaskRuntime,
    TaskSpec,
    MultiTaskModel,
    evaluate,
    init_regression_biases,
    load_multitask_dataset,
    load_stage1_full,
    maybe_limit_split,
    resolve_device,
    set_seed,
    split_subjects,
    task_losses,
    validation_loss,
)
from run_stage23_ppg_unified import fit_stage2 as fit_stage2_probe, make_task_loaders_for_device
from run_stage2_reusability_ppg import LOCAL_CKPT_DIRS


DEFAULT_OUT_ROOT = ROOT / "kd_sweep_results/specialization_sweep_ppg"
DEFAULT_SOURCE_TASKS = [
    "bidmc_hr",
    "ppgarrhythmia_6class",
    "ppgbp_hr",
    "ppgbp_hypertension",
    "ppgbp_sysbp",
    "ppgbp_diasbp",
    "ppgdalia_hr",
    "ppgdalia_activity",
    "sdb_binary",
    "sdb_multiclass",
    "wesad_multiclass",
    "wesad_binary",
    "wfppg_wrist",
    "wfppg_finger",
]
DEFAULT_TARGET_TASKS = [
    "bidmc_hr",
    "ppgarrhythmia_6class",
    "ppgbp_hr",
    "ppgbp_hypertension",
    "ppgbp_sysbp",
    "ppgbp_diasbp",
    "ppgdalia_hr",
    "ppgdalia_activity",
    "sdb_binary",
    "sdb_multiclass",
    "wesad_multiclass",
    "wesad_binary",
    "wfppg_wrist",
    "wfppg_finger",
]
DEFAULT_DATA_DIRS = {
    "bidmc": ROOT / "pulseppg/data/datasets/bidmc",
    "sdb": ROOT / "pulseppg/data/datasets/sdb",
    "dalia": ROOT / "pulseppg/data/datasets/dalia",
    "ppgbp": ROOT / "pulseppg/data/datasets/ppgbp",
    "wesad": ROOT / "pulseppg/data/datasets/wesad",
    "wfppg": ROOT / "pulseppg/data/datasets/wfppg",
    "ppgarrhythmia": ROOT / "pulseppg/data/datasets/ppgarrhythmia",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="BF32_NB8")
    ap.add_argument("--stage1-method", default="ns_kd")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--source-tasks", nargs="+", default=list(DEFAULT_SOURCE_TASKS))
    ap.add_argument("--target-tasks", nargs="+", default=list(DEFAULT_TARGET_TASKS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--source-epochs", type=int, default=30)
    ap.add_argument("--source-head-lr", type=float, default=1e-3)
    ap.add_argument("--source-encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--probe-patience", type=int, default=5)
    ap.add_argument("--probe-head-lr", type=float, default=1e-3)
    ap.add_argument("--probe-encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--large-task-threshold", type=int, default=50000)
    ap.add_argument("--large-task-cap", type=int, default=20000)
    ap.add_argument("--include-sportppg-activity", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    return ap.parse_args()


def task_dataset(task_name: str) -> str:
    if task_name == "sportppg_activity":
        return "sportppg"
    if task_name not in TASK_REGISTRY:
        raise KeyError(f"Unsupported PPG task {task_name}; choices={sorted(TASK_REGISTRY)}")
    return str(TASK_REGISTRY[task_name]["dataset"])


def _normalize_ppg_input(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    return arr


def _auto_cap_large_presplit(
    splits_X: Dict[str, np.ndarray],
    splits_y: Dict[str, np.ndarray],
    *,
    threshold: int,
    cap: int,
    seed: int,
    task_name: str,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    total = sum(len(splits_X[split]) for split in ("train", "val", "test"))
    if int(cap) <= 0 or total <= int(threshold) or total <= int(cap):
        return splits_X, splits_y
    ratio = float(cap) / float(total)
    rng = np.random.default_rng(int(seed))
    for split in ("train", "val", "test"):
        target_n = max(1, int(round(len(splits_X[split]) * ratio)))
        if len(splits_X[split]) > target_n:
            idx = np.sort(rng.choice(len(splits_X[split]), target_n, replace=False))
            splits_X[split] = np.asarray(splits_X[split][idx], dtype=np.float32)
            splits_y[split] = np.asarray(splits_y[split][idx])
    capped_total = sum(len(splits_X[split]) for split in ("train", "val", "test"))
    print(f"[cap] task={task_name} total={total} -> {capped_total} (threshold={threshold}, cap={cap})", flush=True)
    return splits_X, splits_y


def _auto_cap_large_kfold(
    X_all: np.ndarray,
    y_all: np.ndarray,
    subjects: np.ndarray,
    *,
    threshold: int,
    cap: int,
    seed: int,
    task_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = len(X_all)
    if int(cap) <= 0 or total <= int(threshold) or total <= int(cap):
        return X_all, y_all, subjects
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(total, int(cap), replace=False))
    print(f"[cap] task={task_name} total={total} -> {len(idx)} (threshold={threshold}, cap={cap})", flush=True)
    return X_all[idx], y_all[idx], subjects[idx]


def build_ppg_task_data(task_name: str, args: argparse.Namespace, seed: int) -> TaskData:
    if task_name == "sportppg_activity":
        task_map = {td.runtime.spec.name: td for td in load_multitask_dataset("sportppg", args.hz, args.limit_train, args.limit_val, args.limit_test, seed)}
        td = task_map["activity"]
        runtime = TaskRuntime(TaskSpec("sportppg_activity", "cls", "", "f1_macro"), n_out=int(td.runtime.n_out))
        return TaskData(runtime=runtime, split_X={k: np.asarray(v).copy() for k, v in td.split_X.items()}, split_y={k: np.asarray(v).copy() for k, v in td.split_y.items()})

    cfg = TASK_REGISTRY[str(task_name)]
    dataset = str(cfg["dataset"])
    root = DEFAULT_DATA_DIRS.get(dataset)
    if root is None:
        raise KeyError(f"No default data root configured for dataset {dataset}")
    if not Path(root).exists():
        raise FileNotFoundError(f"Missing dataset root for {task_name}: {root}")

    if cfg["split_mode"] == "presplit":
        X_tr, y_tr, X_va, y_va, X_te, y_te = _presplit_arrays(str(root), cfg, int(args.hz))
        splits_X = {
            "train": _normalize_ppg_input(X_tr),
            "val": _normalize_ppg_input(X_va),
            "test": _normalize_ppg_input(X_te),
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
        X_all, y_all, subjects = _kfold_arrays(str(root), cfg, int(args.hz))
        if cfg.get("label_transform") == "binarize":
            y_all = (np.asarray(y_all) > 0).astype(np.int64)
        elif cfg.get("label_transform") == "to_int":
            y_all = np.asarray(y_all).astype(np.int64)
        else:
            y_all = np.asarray(y_all)
        X_all, y_all, subjects = _auto_cap_large_kfold(
            X_all,
            y_all,
            np.asarray(subjects),
            threshold=int(args.large_task_threshold),
            cap=int(args.large_task_cap),
            seed=int(seed),
            task_name=str(task_name),
        )
        X_all = _normalize_ppg_input(X_all)
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
        raise NotImplementedError(f"PPG split_mode={cfg['split_mode']} is not supported in specialization sweep")

    if cfg["type"] == "cls":
        y_cat = np.concatenate([np.asarray(splits_y[s], dtype=np.int64) for s in ("train", "val", "test")])
        runtime = TaskRuntime(TaskSpec(str(task_name), "cls", "", str(cfg["primary_metric"])), n_out=int(np.max(y_cat)) + 1)
    else:
        y_train = np.asarray(splits_y["train"], dtype=np.float32)
        y_mean = float(y_train.mean())
        y_std = float(y_train.std())
        if not np.isfinite(y_std) or y_std < 1e-6:
            y_std = 1.0
        runtime = TaskRuntime(TaskSpec(str(task_name), "reg", "", str(cfg["primary_metric"])), n_out=1, y_mean=y_mean, y_std=y_std)
    return TaskData(runtime=runtime, split_X=splits_X, split_y=splits_y)


def clone_task_data(task_data: TaskData) -> TaskData:
    return TaskData(
        runtime=copy.deepcopy(task_data.runtime),
        split_X={k: np.asarray(v).copy() for k, v in task_data.split_X.items()},
        split_y={k: np.asarray(v).copy() for k, v in task_data.split_y.items()},
    )


def build_model_for_task(encoder: nn.Module, enc_dim: int, task_data: TaskData, dropout: float, hz: int) -> MultiTaskModel:
    return MultiTaskModel(
        encoder,
        int(enc_dim),
        [task_data.runtime],
        float(dropout),
        dataset_name=task_dataset(task_data.runtime.spec.name),
        hz=int(hz),
        input_len=int(task_data.split_X["train"].shape[-1]),
        temporal_policy="shared_window",
    )


def evaluate_split(model: MultiTaskModel, loader, runtime: TaskRuntime, device: torch.device) -> Dict[str, Any]:
    return evaluate(model, {runtime.spec.name: loader}, [runtime], device)[runtime.spec.name]


def metric_value(metric: Dict[str, Any]) -> float:
    return float(metric.get("value", np.nan))


def metric_payload(metric: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "value": float(metric.get("value", np.nan)),
        "metric": str(metric.get("metric", "")),
        "higher_better": bool(metric.get("higher_better", False)),
        "macro_f1": metric.get("macro_f1", np.nan),
        "mae": metric.get("mae", np.nan),
        "balanced_acc": metric.get("balanced_acc", np.nan),
        "auroc": metric.get("auroc", np.nan),
        "auprc": metric.get("auprc", np.nan),
        "n_eval": metric.get("n_eval", np.nan),
    }


def metric_improves(new_value: float, ref_value: float, higher_better: bool) -> bool:
    if not np.isfinite(new_value) or not np.isfinite(ref_value):
        return False
    return bool(new_value > ref_value) if higher_better else bool(new_value < ref_value)


def metric_delta(new_value: float, ref_value: float, higher_better: bool) -> float:
    return float(new_value - ref_value) if higher_better else float(ref_value - new_value)


def parameter_delta_ratio(encoder_ref: nn.Module, encoder_cur: nn.Module) -> float:
    ref_vec = torch.cat([p.detach().reshape(-1).cpu() for p in encoder_ref.parameters()])
    cur_vec = torch.cat([p.detach().reshape(-1).cpu() for p in encoder_cur.parameters()])
    denom = ref_vec.norm(2).clamp_min(1e-12)
    return float((cur_vec - ref_vec).norm(2).div(denom).item())


def feature_drift_cosine(encoder_ref: nn.Module, encoder_cur: nn.Module, task_data: TaskData, device: torch.device, batch_size: int) -> float:
    ref = copy.deepcopy(encoder_ref).to(device).eval()
    cur = copy.deepcopy(encoder_cur).to(device).eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(task_data.split_X["test"], dtype=np.float32))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type != "cpu",
    )
    cos_vals: List[torch.Tensor] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            z_ref = ref(xb).reshape(xb.shape[0], -1)
            z_cur = cur(xb).reshape(xb.shape[0], -1)
            cos_vals.append(torch.sum(torch.nn.functional.normalize(z_ref, dim=1) * torch.nn.functional.normalize(z_cur, dim=1), dim=1).detach().cpu())
    if not cos_vals:
        return float("nan")
    return float(1.0 - torch.cat(cos_vals).mean().item())


def _encoder_and_other_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    enc_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            enc_params.append(param)
        else:
            other_params.append(param)
    return enc_params, other_params


def make_optimizer(model: nn.Module, head_lr: float, encoder_lr_mult: float, weight_decay: float) -> torch.optim.Optimizer:
    enc_params, other_params = _encoder_and_other_params(model)
    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": float(head_lr) * float(encoder_lr_mult), "weight_decay": float(weight_decay)})
    if other_params:
        groups.append({"params": other_params, "lr": float(head_lr), "weight_decay": float(weight_decay)})
    return torch.optim.AdamW(groups)


def fit_frozen_probe(*, encoder: nn.Module, enc_dim: int, task_data: TaskData, device: torch.device, args: argparse.Namespace, log_prefix: str) -> Dict[str, Any]:
    td = clone_task_data(task_data)
    model = build_model_for_task(copy.deepcopy(encoder).to(device), int(enc_dim), td, float(args.dropout), int(args.hz))
    train_loaders, val_loaders, test_loaders = make_task_loaders_for_device([td], int(args.batch_size), device)
    model, test_metrics, meta = fit_stage2_probe(
        model=model,
        method="frozen_heads",
        tasks=[td.runtime],
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        device=device,
        epochs=int(args.probe_epochs),
        patience=int(args.probe_patience),
        head_lr=float(args.probe_head_lr),
        encoder_lr_mult=float(args.probe_encoder_lr_mult),
        weight_decay=float(args.weight_decay),
        log_prefix=log_prefix,
    )
    val_metric = evaluate_split(model, val_loaders[td.runtime.spec.name], td.runtime, device)
    train_metric = evaluate_split(model, train_loaders[td.runtime.spec.name], td.runtime, device)
    return {
        "model": model,
        "test_metric": metric_payload(test_metrics[td.runtime.spec.name]),
        "val_metric": metric_payload(val_metric),
        "train_metric": metric_payload(train_metric),
        "meta": meta,
    }


def fit_source_specialization(*, encoder: nn.Module, enc_dim: int, task_data: TaskData, device: torch.device, args: argparse.Namespace, log_prefix: str) -> Dict[str, Any]:
    td = clone_task_data(task_data)
    model = build_model_for_task(copy.deepcopy(encoder).to(device), int(enc_dim), td, float(args.dropout), int(args.hz)).to(device)
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.heads.parameters():
        param.requires_grad = True
    init_regression_biases(model, [td.runtime])
    train_loaders, val_loaders, test_loaders = make_task_loaders_for_device([td], int(args.batch_size), device)
    opt = make_optimizer(model, float(args.source_head_lr), float(args.source_encoder_lr_mult), float(args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(args.source_epochs)))
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history_rows: List[Dict[str, Any]] = []
    lr_encoder = float(args.source_head_lr) * float(args.source_encoder_lr_mult)
    lr_head = float(args.source_head_lr)

    for ep in range(int(args.source_epochs)):
        model.train()
        losses_ep: List[float] = []
        for xb, yb in train_loaders[td.runtime.spec.name]:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            _, out = model(xb, task_name=td.runtime.spec.name)
            loss = task_losses(out, {td.runtime.spec.name: yb}, [td.runtime])[td.runtime.spec.name]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            losses_ep.append(float(loss.detach().item()))
        sched.step()
        val_loss = validation_loss(model, val_loaders, [td.runtime], device)
        val_metric = evaluate_split(model, val_loaders[td.runtime.spec.name], td.runtime, device)
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = int(ep)
        history_rows.append(
            {
                "epoch": int(ep),
                "train_loss": float(np.mean(losses_ep)) if losses_ep else np.nan,
                "val_loss": float(val_loss),
                "val_value": float(val_metric.get("value", np.nan)),
                "metric": str(val_metric.get("metric", "")),
                "higher_better": bool(val_metric.get("higher_better", False)),
                "lr_encoder": lr_encoder,
                "lr_head": lr_head,
            }
        )
        print(f"    {log_prefix}ep{ep:03d} train={np.mean(losses_ep):.4f} val_loss={val_loss:.4f} best={best_val_loss:.4f}", flush=True)

    final_train_metric = evaluate_split(model, train_loaders[td.runtime.spec.name], td.runtime, device)
    final_val_metric = evaluate_split(model, val_loaders[td.runtime.spec.name], td.runtime, device)
    final_test_metric = evaluate_split(model, test_loaders[td.runtime.spec.name], td.runtime, device)

    best_model = build_model_for_task(copy.deepcopy(encoder).to(device), int(enc_dim), td, float(args.dropout), int(args.hz)).to(device)
    best_model.load_state_dict(best_state)
    best_val_metric = evaluate_split(best_model, val_loaders[td.runtime.spec.name], td.runtime, device)
    best_test_metric = evaluate_split(best_model, test_loaders[td.runtime.spec.name], td.runtime, device)
    return {
        "model": model,
        "final_train_metric": metric_payload(final_train_metric),
        "final_val_metric": metric_payload(final_val_metric),
        "final_test_metric": metric_payload(final_test_metric),
        "best_val_metric": metric_payload(best_val_metric),
        "best_test_metric": metric_payload(best_test_metric),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs": int(args.source_epochs),
        "lr_encoder": float(lr_encoder),
        "lr_head": float(lr_head),
        "history": pd.DataFrame(history_rows),
    }


def source_overfit(*, baseline_val: float, final_train: float, final_val: float, best_val: float, higher_better: bool) -> bool:
    if not all(np.isfinite(v) for v in [baseline_val, final_train, final_val, best_val]):
        return False
    if higher_better:
        return bool(final_train > baseline_val + 0.03 and best_val - final_val >= 0.03 and final_val < baseline_val)
    return bool(final_train < baseline_val * 0.97 and final_val - best_val >= max(0.02 * abs(best_val), 0.05) and final_val > baseline_val)


def pair_degradation_and_retention(*, baseline: float, probe: float, higher_better: bool) -> tuple[float, float]:
    if higher_better:
        degradation = float(baseline - probe)
        retention = float(probe / baseline) if baseline else float("nan")
    else:
        degradation = float(probe - baseline)
        retention = float(baseline / probe) if probe else float("nan")
    return degradation, retention


def pair_verdict(*, source_improves_final: bool, source_overfit_final: bool, target_strong: bool, baseline: float, higher_better: bool, degradation: float, retention: float) -> str:
    if not target_strong:
        return "weak_target_probe"
    if source_overfit_final:
        return "overfit_source"
    if not source_improves_final:
        return "source_not_learned"
    if not np.isfinite(degradation) or not np.isfinite(retention):
        return "neutral"
    if higher_better:
        strong_drop = degradation >= 0.03 or retention < 0.95
        mild_drop = degradation >= 0.01 or retention < 0.98
    else:
        strong_drop = (degradation / max(abs(float(baseline)), 1e-12)) >= 0.05 or retention < 0.95
        mild_drop = (degradation / max(abs(float(baseline)), 1e-12)) >= 0.02 or retention < 0.98
    if strong_drop and degradation > 0:
        return "strong_candidate_degradation"
    if mild_drop and degradation > 0:
        return "candidate_degradation"
    if degradation < 0:
        return "positive_transfer"
    return "neutral"


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cand = df[df["pair_verdict"].isin(["strong_candidate_degradation", "candidate_degradation"])].copy()
    if cand.empty:
        return cand
    cand["is_strong"] = (cand["pair_verdict"] == "strong_candidate_degradation").astype(int)
    cand = cand.sort_values(by=["is_strong", "target_strength", "degradation", "source_improvement", "feature_drift"], ascending=[False, False, False, False, False])
    cand.insert(0, "rank", np.arange(1, len(cand) + 1))
    return cand


def write_outputs(out_root: Path, baseline_rows: List[Dict[str, Any]], source_rows: List[Dict[str, Any]], matrix_rows: List[Dict[str, Any]]) -> None:
    baseline_df = pd.DataFrame(baseline_rows)
    source_df = pd.DataFrame(source_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    ranked_df = rank_candidates(matrix_df)
    baseline_df.to_csv(out_root / "baseline_probes.csv", index=False)
    source_df.to_csv(out_root / "source_adaptation_summary.csv", index=False)
    matrix_df.to_csv(out_root / "degradation_matrix.csv", index=False)
    ranked_df.to_csv(out_root / "top_candidates.csv", index=False)
    progress = {
        "n_baselines": int(len(baseline_df)),
        "n_sources": int(len(source_df)),
        "n_matrix_rows": int(len(matrix_df)),
        "n_candidate_pairs": int(matrix_df["pair_verdict"].isin(["candidate_degradation", "strong_candidate_degradation"]).sum()) if not matrix_df.empty else 0,
    }
    (out_root / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.include_sportppg_activity and "sportppg_activity" not in args.target_tasks:
        args.target_tasks = list(args.target_tasks) + ["sportppg_activity"]

    device = resolve_device(str(args.device))
    ckpt_dir = args.ckpt_dir or LOCAL_CKPT_DIRS.get(str(args.student))
    if ckpt_dir is None:
        raise KeyError(f"No default PPG checkpoint dir for student {args.student}")
    out_root = Path(args.out_root)
    if args.fresh and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    matrix_rows: List[Dict[str, Any]] = []

    for seed in args.seeds:
        set_seed(int(seed))
        task_bank: Dict[str, TaskData] = {}
        for task_name in list(dict.fromkeys(list(args.source_tasks) + list(args.target_tasks))):
            task_bank[str(task_name)] = build_ppg_task_data(str(task_name), args, int(seed))

        enc_e0, _, ck_meta = load_stage1_full(str(args.student), Path(ckpt_dir), str(args.stage1_method), int(seed), device)
        enc_dim = int(ck_meta["enc_dim"])
        print(f"[stage1] student={args.student} method={args.stage1_method} ckpt={ck_meta['ckpt_path']}", flush=True)

        target_baselines: Dict[str, Dict[str, Any]] = {}
        source_baselines: Dict[str, Dict[str, Any]] = {}
        for task_name in sorted(set(map(str, args.target_tasks))):
            td = task_bank[task_name]
            print(f"[baseline target] seed={seed} task={task_name}", flush=True)
            probe = fit_frozen_probe(encoder=enc_e0, enc_dim=enc_dim, task_data=td, device=device, args=args, log_prefix=f"[A/{task_name}/s{seed}] ")
            target_baselines[task_name] = probe
            baseline_rows.append({"seed": int(seed), "task": task_name, "role": "target", "dataset": task_dataset(task_name), "A_test_value": metric_value(probe["test_metric"]), "A_val_value": metric_value(probe["val_metric"]), "metric": str(probe["test_metric"]["metric"]), "higher_better": bool(probe["test_metric"]["higher_better"])})
            write_outputs(out_root, baseline_rows, source_rows, matrix_rows)

        for task_name in sorted(set(map(str, args.source_tasks))):
            td = task_bank[task_name]
            print(f"[baseline source] seed={seed} task={task_name}", flush=True)
            probe = fit_frozen_probe(encoder=enc_e0, enc_dim=enc_dim, task_data=td, device=device, args=args, log_prefix=f"[E0src/{task_name}/s{seed}] ")
            source_baselines[task_name] = probe
            baseline_rows.append({"seed": int(seed), "task": task_name, "role": "source", "dataset": task_dataset(task_name), "A_test_value": metric_value(probe["test_metric"]), "A_val_value": metric_value(probe["val_metric"]), "metric": str(probe["test_metric"]["metric"]), "higher_better": bool(probe["test_metric"]["higher_better"])})
            write_outputs(out_root, baseline_rows, source_rows, matrix_rows)

        for source_task in map(str, args.source_tasks):
            td_source = task_bank[source_task]
            print(f"\n========================================================================================\n[source={source_task}] seed={seed} dataset={task_dataset(source_task)}\n========================================================================================", flush=True)
            t0 = time.time()
            source_result = fit_source_specialization(encoder=enc_e0, enc_dim=enc_dim, task_data=td_source, device=device, args=args, log_prefix=f"[source/{source_task}/s{seed}] ")
            param_delta = parameter_delta_ratio(enc_e0, source_result["model"].encoder)
            feat_drift = feature_drift_cosine(enc_e0, source_result["model"].encoder, td_source, device, int(args.batch_size))
            source_e0_val = metric_value(source_baselines[source_task]["val_metric"])
            source_final_val = metric_value(source_result["final_val_metric"])
            source_best_val = metric_value(source_result["best_val_metric"])
            source_final_train = metric_value(source_result["final_train_metric"])
            higher_better = bool(source_result["final_val_metric"]["higher_better"])
            source_improves_final = metric_improves(source_final_val, source_e0_val, higher_better)
            source_improvement = metric_delta(source_final_val, source_e0_val, higher_better)
            overfit_final = source_overfit(baseline_val=source_e0_val, final_train=source_final_train, final_val=source_final_val, best_val=source_best_val, higher_better=higher_better)
            history_path = out_root / f"source_history_seed{seed}_{source_task}.csv"
            source_result["history"].to_csv(history_path, index=False)
            source_rows.append({
                "seed": int(seed),
                "source_task": source_task,
                "source_dataset": task_dataset(source_task),
                "source_metric": str(source_result["final_val_metric"]["metric"]),
                "higher_better": higher_better,
                "source_e0_val": source_e0_val,
                "source_final_val": source_final_val,
                "source_best_val": source_best_val,
                "source_final_train": source_final_train,
                "source_final_test": metric_value(source_result["final_test_metric"]),
                "source_improves_final": bool(source_improves_final),
                "source_overfit": bool(overfit_final),
                "source_improvement": float(source_improvement),
                "lr_encoder": float(source_result["lr_encoder"]),
                "lr_head": float(source_result["lr_head"]),
                "epochs": int(source_result["epochs"]),
                "early_stopping": "No",
                "best_epoch": int(source_result["best_epoch"]),
                "best_val_loss": float(source_result["best_val_loss"]),
                "param_delta": float(param_delta),
                "feature_drift": float(feat_drift),
                "history_path": str(history_path),
                "time_s": float(time.time() - t0),
            })
            write_outputs(out_root, baseline_rows, source_rows, matrix_rows)

            for target_task in map(str, args.target_tasks):
                if str(source_task) == str(target_task):
                    matrix_rows.append({"modality": "ppg", "seed": int(seed), "source_task": source_task, "target_task": target_task, "source_dataset": task_dataset(source_task), "target_dataset": task_dataset(target_task), "source_e0_metric": source_e0_val, "source_final_metric": source_final_val, "source_best_metric": source_best_val, "source_improves_final": bool(source_improves_final), "source_overfit": bool(overfit_final), "A_target": metric_value(target_baselines[target_task]["test_metric"]), "P_final": np.nan, "retention": np.nan, "degradation": np.nan, "pair_verdict": "invalid_pair", "invalid_reason": "same_exact_task", "target_metric": str(target_baselines[target_task]["test_metric"]["metric"]), "target_higher_better": bool(target_baselines[target_task]["test_metric"]["higher_better"]), "param_delta": float(param_delta), "feature_drift": float(feat_drift), "target_strength": float(metric_value(target_baselines[target_task]["test_metric"]) if bool(target_baselines[target_task]["test_metric"]["higher_better"]) else 1.0 / max(metric_value(target_baselines[target_task]["test_metric"]), 1e-12)), "source_improvement": float(source_improvement)})
                    write_outputs(out_root, baseline_rows, source_rows, matrix_rows)
                    continue

                td_target = task_bank[target_task]
                baseline_t = target_baselines[target_task]
                A_t = metric_value(baseline_t["test_metric"])
                target_higher_better = bool(baseline_t["test_metric"]["higher_better"])
                target_strength = float(A_t if target_higher_better else (1.0 / max(A_t, 1e-12)))
                print(f"[probe] seed={seed} source={source_task} -> target={target_task}", flush=True)
                probe = fit_frozen_probe(encoder=source_result["model"].encoder, enc_dim=enc_dim, task_data=td_target, device=device, args=args, log_prefix=f"[probe/{source_task}->{target_task}/s{seed}] ")
                probe_value = metric_value(probe["test_metric"])
                degradation, retention = pair_degradation_and_retention(baseline=A_t, probe=probe_value, higher_better=target_higher_better)
                verdict = pair_verdict(source_improves_final=bool(source_improves_final), source_overfit_final=bool(overfit_final), target_strong=True, baseline=A_t, higher_better=target_higher_better, degradation=degradation, retention=retention)
                matrix_rows.append({
                    "modality": "ppg",
                    "seed": int(seed),
                    "source_task": source_task,
                    "target_task": target_task,
                    "source_dataset": task_dataset(source_task),
                    "target_dataset": task_dataset(target_task),
                    "source_e0_metric": source_e0_val,
                    "source_final_metric": source_final_val,
                    "source_best_metric": source_best_val,
                    "source_improves_final": bool(source_improves_final),
                    "source_overfit": bool(overfit_final),
                    "A_target": A_t,
                    "P_final": probe_value,
                    "retention": retention,
                    "degradation": degradation,
                    "pair_verdict": verdict,
                    "invalid_reason": "",
                    "target_metric": str(probe["test_metric"]["metric"]),
                    "target_higher_better": target_higher_better,
                    "param_delta": float(param_delta),
                    "feature_drift": float(feat_drift),
                    "target_strength": float(target_strength),
                    "source_improvement": float(source_improvement),
                })
                write_outputs(out_root, baseline_rows, source_rows, matrix_rows)

    write_outputs(out_root, baseline_rows, source_rows, matrix_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    summary = {
        "student": str(args.student),
        "stage1_method": str(args.stage1_method),
        "seeds": [int(x) for x in args.seeds],
        "source_tasks": list(map(str, args.source_tasks)),
        "target_tasks": list(map(str, args.target_tasks)),
        "source_epochs": int(args.source_epochs),
        "source_head_lr": float(args.source_head_lr),
        "source_encoder_lr": float(args.source_head_lr) * float(args.source_encoder_lr_mult),
        "probe_epochs": int(args.probe_epochs),
        "probe_patience": int(args.probe_patience),
        "batch_size": int(args.batch_size),
        "n_rows_matrix": int(len(matrix_df)),
        "n_candidate_pairs": int(matrix_df["pair_verdict"].isin(["candidate_degradation", "strong_candidate_degradation"]).sum()) if not matrix_df.empty else 0,
        "ckpt_dir": str(ckpt_dir),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[done] specialization sweep PPG outputs:", flush=True)
    print(f"  baseline: {out_root / 'baseline_probes.csv'}", flush=True)
    print(f"  sources:  {out_root / 'source_adaptation_summary.csv'}", flush=True)
    print(f"  matrix:   {out_root / 'degradation_matrix.csv'}", flush=True)
    print(f"  ranked:   {out_root / 'top_candidates.csv'}", flush=True)


if __name__ == "__main__":
    main()
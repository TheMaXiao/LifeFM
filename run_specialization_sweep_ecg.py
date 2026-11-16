#!/usr/bin/env python3
"""Single-source specialization sweep for ECG Stage-2 stress-test search.

This runner simulates realistic Stage-2-style full-encoder source adaptation:

1. start from the Stage-1 compressed student encoder E0,
2. adapt on exactly one source task with full encoder updates,
3. train for the full epoch budget with no early stopping,
4. use the final checkpoint as the primary adapted encoder E_s,
5. freeze E_s and probe target tasks with fresh linear heads only,
6. compare target probes against frozen-E0 probe baselines A_t.

The primary intent is candidate discovery, not final MTL evaluation.
"""

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
from pandas.errors import EmptyDataError
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_stage2_multitask_ppg import MultiTaskModel, TaskRuntime, TaskSpec, evaluate, init_regression_biases, task_losses, validation_loss
from run_stage23_ecg_unified import (
    ECGTaskData,
    feature_drift_cosine,
    fit_stage2 as fit_stage2_probe,
    load_ecg_task,
    load_stage1_ecgfounder,
    make_optimizer,
    make_task_loaders_ecg,
    resolve_device,
    set_seed,
    set_stage2_trainable,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "kd_sweep_results/specialization_sweep_ecg"
DEFAULT_SOURCE_TASKS = [
    "super5",
    "rhythm12",
    "mi_bin",
    "sttc_bin",
    "cd_bin",
    "chap_rhythm4",
    "chap_abnormal_bin",
]
DEFAULT_TARGET_TASKS = [
    "cinc_af_bin",
    "chap_rhythm4",
    "chap_abnormal_bin",
    "afib_bin",
    "sttc_bin",
    "cd_bin",
]
TASK_INPUT_SEC = {
    "super5": 8.0,
    "rhythm12": 10.0,
    "afib_bin": 10.0,
    "sttc_bin": 5.0,
    "cd_bin": 5.0,
    "mi_bin": 5.0,
    "norm_bin": 5.0,
    "chap_rhythm4": 10.0,
    "chap_abnormal_bin": 8.0,
    "chap_afib_bin": 10.0,
    "chap_sbrady_bin": 10.0,
    "chap_stachy_bin": 4.0,
    "cinc_af_bin": 10.0,
}
ALLOWED_DATASET_PAIRS = {
    ("ptbxl", "chapman"),
    ("ptbxl", "cinc"),
    ("chapman", "ptbxl"),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="BF10_NB7")
    ap.add_argument("--stage1-method", default="lifefm_kd", choices=["lifefm_kd", "subspace_mse", "full_mse", "lower_bound"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--source-tasks", nargs="+", default=list(DEFAULT_SOURCE_TASKS))
    ap.add_argument("--target-tasks", nargs="+", default=list(DEFAULT_TARGET_TASKS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--batch-size", type=int, default=16)
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
    ap.add_argument("--include-same-dataset-controls", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def task_dataset(task_name: str) -> str:
    if str(task_name).startswith("chap_"):
        return "chapman"
    if str(task_name).startswith("cinc_"):
        return "cinc"
    return "ptbxl"


def is_valid_pair(source_task: str, target_task: str, include_same_dataset_controls: bool) -> tuple[bool, str]:
    if str(source_task) == str(target_task):
        return False, "same_exact_task"
    src_ds = task_dataset(source_task)
    tgt_ds = task_dataset(target_task)
    if src_ds == tgt_ds and not include_same_dataset_controls:
        return False, "same_dataset_control"
    if src_ds != tgt_ds and (src_ds, tgt_ds) not in ALLOWED_DATASET_PAIRS:
        return False, "invalid_domain_pair"
    return True, ""


def maybe_limit(idx: np.ndarray, y: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if int(limit) <= 0 or len(idx) <= int(limit):
        return np.asarray(idx), np.asarray(y)
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(len(idx), int(limit), replace=False))
    return np.asarray(idx)[keep], np.asarray(y)[keep]


def resolve_input_len(task_name: str, native_len: int, hz: int = 500) -> int:
    target_sec = float(TASK_INPUT_SEC.get(str(task_name), 10.0))
    return min(int(native_len), max(1, int(round(target_sec * float(hz)))))


def clone_ecg_task_data(task_data: ECGTaskData) -> ECGTaskData:
    return ECGTaskData(
        runtime=copy.deepcopy(task_data.runtime),
        signals=task_data.signals,
        split_idx={k: np.asarray(v).copy() for k, v in task_data.split_idx.items()},
        split_y={k: np.asarray(v).copy() for k, v in task_data.split_y.items()},
        input_len=int(task_data.input_len),
    )


def build_ecg_task_data(task_name: str, args: argparse.Namespace) -> ECGTaskData:
    dataset = task_dataset(task_name)
    sigs, y, fold, valid, kind = load_ecg_task(task_name, dataset)
    idx_pool = np.where(valid)[0]
    from run_ecgfounder_kd import split_by_fold

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
        task_runtime = TaskRuntime(
            TaskSpec(task_name, "reg", "", "mae"),
            n_out=1,
            y_mean=y_mean,
            y_std=y_std,
        )
    else:
        n_out = int(np.max(np.concatenate([y_tr, y_va, y_te])) + 1)
        task_runtime = TaskRuntime(
            TaskSpec(task_name, "cls", "", "f1_macro"),
            n_out=n_out,
        )

    return ECGTaskData(
        runtime=task_runtime,
        signals=sigs,
        split_idx={"train": tr_i, "val": va_i, "test": te_i},
        split_y={"train": y_tr, "val": y_va, "test": y_te},
        input_len=resolve_input_len(task_name, int(sigs.shape[-1])),
    )


def build_model_for_task(
    encoder: nn.Module,
    enc_dim: int,
    task_data: ECGTaskData,
    dropout: float,
) -> MultiTaskModel:
    return MultiTaskModel(
        encoder,
        int(enc_dim),
        [task_data.runtime],
        float(dropout),
        dataset_name=task_dataset(task_data.runtime.spec.name),
        hz=500,
        input_len=int(task_data.input_len),
        temporal_policy="shared_window",
    )


def metric_value(metric: Dict[str, float]) -> float:
    return float(metric.get("value", np.nan))


def metric_improves(new_value: float, ref_value: float, higher_better: bool) -> bool:
    if not np.isfinite(new_value) or not np.isfinite(ref_value):
        return False
    return bool(new_value > ref_value) if higher_better else bool(new_value < ref_value)


def metric_delta(new_value: float, ref_value: float, higher_better: bool) -> float:
    if higher_better:
        return float(new_value - ref_value)
    return float(ref_value - new_value)


def parameter_delta_ratio(encoder_ref: nn.Module, encoder_cur: nn.Module) -> float:
    ref_vec = torch.cat([p.detach().reshape(-1).cpu() for p in encoder_ref.parameters()])
    cur_vec = torch.cat([p.detach().reshape(-1).cpu() for p in encoder_cur.parameters()])
    denom = ref_vec.norm(2).clamp_min(1e-12)
    return float((cur_vec - ref_vec).norm(2).div(denom).item())


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


def evaluate_split(
    model: MultiTaskModel,
    loader: DataLoader,
    runtime,
    device: torch.device,
) -> Dict[str, Any]:
    return evaluate(model, {runtime.spec.name: loader}, [runtime], device)[runtime.spec.name]


def fit_frozen_probe(
    *,
    encoder: nn.Module,
    enc_dim: int,
    task_data: ECGTaskData,
    device: torch.device,
    args: argparse.Namespace,
    log_prefix: str,
) -> Dict[str, Any]:
    td = clone_ecg_task_data(task_data)
    model = build_model_for_task(copy.deepcopy(encoder).to(device), int(enc_dim), td, float(args.dropout))
    train_loaders, val_loaders, test_loaders = make_task_loaders_ecg([td], int(args.batch_size))
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


def fit_source_specialization(
    *,
    encoder: nn.Module,
    enc_dim: int,
    task_data: ECGTaskData,
    device: torch.device,
    args: argparse.Namespace,
    log_prefix: str,
) -> Dict[str, Any]:
    td = clone_ecg_task_data(task_data)
    model = build_model_for_task(copy.deepcopy(encoder).to(device), int(enc_dim), td, float(args.dropout))
    model = model.to(device)
    set_stage2_trainable(model, "joint_finetune")
    init_regression_biases(model, [td.runtime])
    train_loaders, val_loaders, test_loaders = make_task_loaders_ecg([td], int(args.batch_size))
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
        print(
            f"    {log_prefix}ep{ep:03d} train={np.mean(losses_ep):.4f} val_loss={val_loss:.4f} best={best_val_loss:.4f}",
            flush=True,
        )

    final_train_metric = evaluate_split(model, train_loaders[td.runtime.spec.name], td.runtime, device)
    final_val_metric = evaluate_split(model, val_loaders[td.runtime.spec.name], td.runtime, device)
    final_test_metric = evaluate_split(model, test_loaders[td.runtime.spec.name], td.runtime, device)

    best_model = build_model_for_task(copy.deepcopy(encoder).to(device), int(enc_dim), td, float(args.dropout))
    best_model.load_state_dict(best_state)
    best_model = best_model.to(device)
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


def source_overfit(
    *,
    baseline_val: float,
    final_train: float,
    final_val: float,
    best_val: float,
    higher_better: bool,
) -> bool:
    if not all(np.isfinite(v) for v in [baseline_val, final_train, final_val, best_val]):
        return False
    if higher_better:
        return bool(final_train > baseline_val + 0.03 and best_val - final_val >= 0.03 and final_val < baseline_val)
    return bool(final_train < baseline_val * 0.97 and final_val - best_val >= max(0.02 * abs(best_val), 0.05) and final_val > baseline_val)


def pair_degradation_and_retention(
    *,
    baseline: float,
    probe: float,
    higher_better: bool,
) -> tuple[float, float]:
    if higher_better:
        degradation = float(baseline - probe)
        retention = float(probe / baseline) if baseline else float("nan")
    else:
        degradation = float(probe - baseline)
        retention = float(baseline / probe) if probe else float("nan")
    return degradation, retention


def pair_verdict(
    *,
    source_improves_final: bool,
    source_overfit_final: bool,
    target_strong: bool,
    baseline: float,
    higher_better: bool,
    degradation: float,
    retention: float,
) -> str:
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


def write_outputs(
    out_root: Path,
    baseline_rows: List[Dict[str, Any]],
    source_rows: List[Dict[str, Any]],
    matrix_rows: List[Dict[str, Any]],
) -> None:
    baseline_df = pd.DataFrame(baseline_rows)
    source_df = pd.DataFrame(source_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    ranked_df = rank_candidates(matrix_df)

    baseline_df.to_csv(out_root / "baseline_probes.csv", index=False)
    source_df.to_csv(out_root / "source_adaptation_summary.csv", index=False)
    matrix_df.to_csv(out_root / "degradation_matrix.csv", index=False)
    ranked_df.to_csv(out_root / "top_candidates.csv", index=False)

    summary = {
        "n_baselines": int(len(baseline_df)),
        "n_sources": int(len(source_df)),
        "n_matrix_rows": int(len(matrix_df)),
        "n_candidate_pairs": int(matrix_df["pair_verdict"].isin(["candidate_degradation", "strong_candidate_degradation"]).sum()) if not matrix_df.empty else 0,
    }
    (out_root / "progress.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cand = df[df["pair_verdict"].isin(["strong_candidate_degradation", "candidate_degradation"])].copy()
    if cand.empty:
        return cand
    cand["is_strong"] = (cand["pair_verdict"] == "strong_candidate_degradation").astype(int)
    cand = cand.sort_values(
        by=["is_strong", "target_strength", "degradation", "source_improvement", "feature_drift"],
        ascending=[False, False, False, False, False],
    )
    cand.insert(0, "rank", np.arange(1, len(cand) + 1))
    return cand


def _read_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def load_existing_state(out_root: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    baseline_df = _read_existing_csv(out_root / "baseline_probes.csv")
    source_df = _read_existing_csv(out_root / "source_adaptation_summary.csv")
    matrix_df = _read_existing_csv(out_root / "degradation_matrix.csv")
    return baseline_df.to_dict("records"), source_df.to_dict("records"), matrix_df.to_dict("records")


def baseline_probe_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    metric = str(row.get("metric", ""))
    higher_better = _as_bool(row.get("higher_better", False))
    return {
        "val_metric": {
            "value": float(row.get("A_val_value", np.nan)),
            "metric": metric,
            "higher_better": higher_better,
        },
        "test_metric": {
            "value": float(row.get("A_test_value", np.nan)),
            "metric": metric,
            "higher_better": higher_better,
        },
    }


def drop_source_state(
    source_rows: List[Dict[str, Any]],
    matrix_rows: List[Dict[str, Any]],
    *,
    seed: int,
    source_task: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    source_rows = [
        row
        for row in source_rows
        if not (int(row.get("seed", -1)) == int(seed) and str(row.get("source_task", "")) == str(source_task))
    ]
    matrix_rows = [
        row
        for row in matrix_rows
        if not (int(row.get("seed", -1)) == int(seed) and str(row.get("source_task", "")) == str(source_task))
    ]
    return source_rows, matrix_rows


def main() -> None:
    args = parse_args()
    if args.fresh and args.resume:
        raise ValueError("--fresh and --resume are mutually exclusive")
    device = resolve_device(str(args.device))
    out_root = Path(args.out_root)
    if args.fresh and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_tasks = list(map(str, args.source_tasks))
    target_tasks = list(map(str, args.target_tasks))

    task_bank: Dict[str, ECGTaskData] = {}
    for task_name in list(dict.fromkeys(source_tasks + target_tasks)):
        task_bank[str(task_name)] = build_ecg_task_data(str(task_name), args)

    baseline_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    matrix_rows: List[Dict[str, Any]] = []
    if args.resume:
        baseline_rows, source_rows, matrix_rows = load_existing_state(out_root)
        print(
            f"[resume] loaded baselines={len(baseline_rows)} sources={len(source_rows)} matrix={len(matrix_rows)} from {out_root}",
            flush=True,
        )

    for seed in args.seeds:
        set_seed(int(seed))
        enc_e0, ck_meta = load_stage1_ecgfounder(str(args.student), str(args.stage1_method), int(seed), device)
        enc_dim = int(ck_meta["enc_dim"])

        target_baselines: Dict[str, Dict[str, Any]] = {}
        source_baselines: Dict[str, Dict[str, Any]] = {}
        baseline_index = {
            (int(row.get("seed", -1)), str(row.get("task", "")), str(row.get("role", ""))): row
            for row in baseline_rows
        }

        for task_name in sorted(set(target_tasks)):
            existing = baseline_index.get((int(seed), str(task_name), "target"))
            if existing is not None:
                target_baselines[task_name] = baseline_probe_from_row(existing)
                print(f"[resume] skip baseline target seed={seed} task={task_name}", flush=True)
                continue
            td = task_bank[task_name]
            print(f"[baseline target] seed={seed} task={task_name}", flush=True)
            probe = fit_frozen_probe(
                encoder=enc_e0,
                enc_dim=enc_dim,
                task_data=td,
                device=device,
                args=args,
                log_prefix=f"[A/{task_name}/s{seed}] ",
            )
            target_baselines[task_name] = probe
            baseline_rows.append(
                {
                    "seed": int(seed),
                    "task": task_name,
                    "role": "target",
                    "dataset": task_dataset(task_name),
                    "A_test_value": metric_value(probe["test_metric"]),
                    "A_val_value": metric_value(probe["val_metric"]),
                    "metric": str(probe["test_metric"]["metric"]),
                    "higher_better": bool(probe["test_metric"]["higher_better"]),
                }
            )
            baseline_index[(int(seed), str(task_name), "target")] = baseline_rows[-1]
            write_outputs(out_root, baseline_rows, source_rows, matrix_rows)

        for task_name in sorted(set(source_tasks)):
            existing = baseline_index.get((int(seed), str(task_name), "source"))
            if existing is not None:
                source_baselines[task_name] = baseline_probe_from_row(existing)
                print(f"[resume] skip baseline source seed={seed} task={task_name}", flush=True)
                continue
            td = task_bank[task_name]
            print(f"[baseline source] seed={seed} task={task_name}", flush=True)
            probe = fit_frozen_probe(
                encoder=enc_e0,
                enc_dim=enc_dim,
                task_data=td,
                device=device,
                args=args,
                log_prefix=f"[E0src/{task_name}/s{seed}] ",
            )
            source_baselines[task_name] = probe
            baseline_rows.append(
                {
                    "seed": int(seed),
                    "task": task_name,
                    "role": "source",
                    "dataset": task_dataset(task_name),
                    "A_test_value": metric_value(probe["test_metric"]),
                    "A_val_value": metric_value(probe["val_metric"]),
                    "metric": str(probe["test_metric"]["metric"]),
                    "higher_better": bool(probe["test_metric"]["higher_better"]),
                }
            )
            baseline_index[(int(seed), str(task_name), "source")] = baseline_rows[-1]
            write_outputs(out_root, baseline_rows, source_rows, matrix_rows)

        for source_task in source_tasks:
            existing_source = [
                row
                for row in source_rows
                if int(row.get("seed", -1)) == int(seed) and str(row.get("source_task", "")) == str(source_task)
            ]
            existing_pairs = [
                row
                for row in matrix_rows
                if int(row.get("seed", -1)) == int(seed) and str(row.get("source_task", "")) == str(source_task)
            ]
            if existing_source and len(existing_pairs) >= len(target_tasks):
                print(f"[resume] skip completed source seed={seed} task={source_task}", flush=True)
                continue
            if existing_source or existing_pairs:
                print(
                    f"[resume] rerun incomplete source seed={seed} task={source_task} "
                    f"(source_rows={len(existing_source)} pair_rows={len(existing_pairs)})",
                    flush=True,
                )
                source_rows, matrix_rows = drop_source_state(
                    source_rows,
                    matrix_rows,
                    seed=int(seed),
                    source_task=str(source_task),
                )
                write_outputs(out_root, baseline_rows, source_rows, matrix_rows)
            td_source = task_bank[source_task]
            print(
                f"\n========================================================================================\n"
                f"[source={source_task}] seed={seed} dataset={task_dataset(source_task)}\n"
                f"========================================================================================",
                flush=True,
            )
            t0 = time.time()
            source_result = fit_source_specialization(
                encoder=enc_e0,
                enc_dim=enc_dim,
                task_data=td_source,
                device=device,
                args=args,
                log_prefix=f"[source/{source_task}/s{seed}] ",
            )
            param_delta = parameter_delta_ratio(enc_e0, source_result["model"].encoder)
            feat_drift = feature_drift_cosine(enc_e0, source_result["model"].encoder, td_source, device, int(args.batch_size))
            source_e0_val = metric_value(source_baselines[source_task]["val_metric"])
            source_final_val = metric_value(source_result["final_val_metric"])
            source_best_val = metric_value(source_result["best_val_metric"])
            source_final_train = metric_value(source_result["final_train_metric"])
            higher_better = bool(source_result["final_val_metric"]["higher_better"])
            source_improves_final = metric_improves(source_final_val, source_e0_val, higher_better)
            source_improvement = metric_delta(source_final_val, source_e0_val, higher_better)
            overfit_final = source_overfit(
                baseline_val=source_e0_val,
                final_train=source_final_train,
                final_val=source_final_val,
                best_val=source_best_val,
                higher_better=higher_better,
            )
            history_path = out_root / f"source_history_seed{seed}_{source_task}.csv"
            source_result["history"].to_csv(history_path, index=False)
            source_rows.append(
                {
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
                }
            )
            write_outputs(out_root, baseline_rows, source_rows, matrix_rows)

            for target_task in target_tasks:
                valid, invalid_reason = is_valid_pair(source_task, target_task, bool(args.include_same_dataset_controls))
                baseline_t = target_baselines[target_task]
                A_t = metric_value(baseline_t["test_metric"])
                target_higher_better = bool(baseline_t["test_metric"]["higher_better"])
                target_strength = float(A_t if target_higher_better else (1.0 / max(A_t, 1e-12)))
                if not valid:
                    matrix_rows.append(
                        {
                            "modality": "ecg",
                            "seed": int(seed),
                            "source_task": source_task,
                            "target_task": target_task,
                            "source_dataset": task_dataset(source_task),
                            "target_dataset": task_dataset(target_task),
                            "source_e0_metric": source_e0_val,
                            "source_final_metric": source_final_val,
                            "source_best_metric": source_best_val,
                            "source_improves_final": bool(source_improves_final),
                            "A_target": A_t,
                            "P_final": np.nan,
                            "retention": np.nan,
                            "degradation": np.nan,
                            "pair_verdict": "invalid_pair",
                            "invalid_reason": invalid_reason,
                            "param_delta": float(param_delta),
                            "feature_drift": float(feat_drift),
                            "target_strength": float(target_strength),
                            "source_improvement": float(source_improvement),
                        }
                    )
                    write_outputs(out_root, baseline_rows, source_rows, matrix_rows)
                    continue

                td_target = task_bank[target_task]
                print(f"[probe] seed={seed} source={source_task} -> target={target_task}", flush=True)
                probe = fit_frozen_probe(
                    encoder=source_result["model"].encoder,
                    enc_dim=enc_dim,
                    task_data=td_target,
                    device=device,
                    args=args,
                    log_prefix=f"[probe/{source_task}->{target_task}/s{seed}] ",
                )
                probe_value = metric_value(probe["test_metric"])
                degradation, retention = pair_degradation_and_retention(
                    baseline=A_t,
                    probe=probe_value,
                    higher_better=target_higher_better,
                )
                verdict = pair_verdict(
                    source_improves_final=bool(source_improves_final),
                    source_overfit_final=bool(overfit_final),
                    target_strong=True,
                    baseline=A_t,
                    higher_better=target_higher_better,
                    degradation=degradation,
                    retention=retention,
                )
                matrix_rows.append(
                    {
                        "modality": "ecg",
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
                    }
                )
                write_outputs(out_root, baseline_rows, source_rows, matrix_rows)

    write_outputs(out_root, baseline_rows, source_rows, matrix_rows)

    baseline_df = pd.DataFrame(baseline_rows)
    source_df = pd.DataFrame(source_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    ranked_df = rank_candidates(matrix_df)

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
        "include_same_dataset_controls": bool(args.include_same_dataset_controls),
        "n_rows_matrix": int(len(matrix_df)),
        "n_candidate_pairs": int(matrix_df["pair_verdict"].isin(["candidate_degradation", "strong_candidate_degradation"]).sum()) if not matrix_df.empty else 0,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[done] specialization sweep ECG outputs:", flush=True)
    print(f"  baseline: {out_root / 'baseline_probes.csv'}", flush=True)
    print(f"  sources:  {out_root / 'source_adaptation_summary.csv'}", flush=True)
    print(f"  matrix:   {out_root / 'degradation_matrix.csv'}", flush=True)
    print(f"  ranked:   {out_root / 'top_candidates.csv'}", flush=True)


if __name__ == "__main__":
    main()
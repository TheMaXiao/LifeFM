#!/usr/bin/env python3
"""Stage-2 MTL adaptation with frozen future-task probes for IMU case studies."""

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

from run_stage2_mtl_future_probe_imu_heterogeneous_search import (
    TASK_CATALOG,
    PreparedTask,
    build_imu_task_data,
    build_stage2_multitask_model,
    fit_frozen_probe,
    metric_payload,
    metric_value,
    resolve_stage1_method,
    task_dataset,
    task_meta,
    task_type,
    _metrics_json,
    _normalized_gain,
    _retention_and_degradation,
    _source_metric_map_from_baselines,
    _source_metric_map_from_stage2,
)
from normwear_kd_experiment import NormWearTeacherEncoder
from run_stage23_imu_unified import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CKPT_DIRS,
    load_stage1_imu_encoder,
    split_subjects,
)
from run_stage23_ppg_unified import fit_stage2
from run_stage2_multitask_ppg import (
    MultiTaskModel,
    TaskData,
    TaskRuntime,
    TaskSpec,
    make_task_loaders,
    resolve_device,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "results/stage2_mtl_future_probe_imu_opportunity_to_mhealth_hr"

ALL_METHODS = ("e0_probe", "teacher_probe", "gradnorm_mtl", "pcgrad_mtl", "fairgrad_mtl", "care_v1")
METHOD_TO_STAGE2 = {
    "gradnorm_mtl": "gradnorm_joint",
    "pcgrad_mtl": "pcgrad_joint",
    "fairgrad_mtl": "fairgrad_joint",
    "care_v1": "care_v1",
}
NON_STAGE2_METHODS = {"e0_probe", "teacher_probe"}

CASE_DEFS: Dict[str, Dict[str, Any]] = {
    "imu_opportunity_to_mhealth_hr": {
        "case_name": "imu_opportunity_to_mhealth_hr",
        "description": "Opportunity activity adaptation (locomotion4+gesture17) to MHEALTH HR regression.",
        "source_tasks": ["opportunity_locomotion4", "opportunity_gesture17"],
        "target_tasks": ["mhealth_hr_reg"],
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_encoder_ckpt", default="", help="Checkpoint file or directory override.")
    ap.add_argument("--student", default="imu2m")
    ap.add_argument("--stage1_method", default="ns_kd")
    ap.add_argument("--cases", nargs="+", default=["all"])
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--stage2_epochs", type=int, default=80)
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
    ap.add_argument("--care_auto_reuse_lambda", type=float, default=0.05)
    ap.add_argument("--care_auto_anchor_source", type=str, default="source_union", choices=["source_union"])
    ap.add_argument("--care_auto_reuse_k", type=int, default=0)
    ap.add_argument("--care_auto_anchor_max_batches", type=int, default=8)
    ap.add_argument("--care_auto_anchor_basis_samples", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--ckpt_dir", type=Path, default=None)
    ap.add_argument("--target_len", type=int, default=500)
    ap.add_argument("--default_win_sec", type=float, default=10.0)
    ap.add_argument("--limit_train", type=int, default=20000)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--teacher_ckpt", type=str, default="NormWear/checkpoints/normwear_pretrain_ckpt.pth", help="Path to NormWear teacher checkpoint for teacher_probe baseline.")
    ap.add_argument("--resume", action="store_true")
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


def _resolve_run_device(args: argparse.Namespace) -> torch.device:
    if args.gpu is not None:
        return torch.device(f"cuda:{int(args.gpu)}")
    return resolve_device(str(args.device))


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
    candidate_files = [output_dir / "overall_summary.csv"]
    candidate_files.extend(output_dir.glob("*_detailed.csv"))
    if any(path.exists() for path in candidate_files):
        raise FileExistsError(f"Refusing to write into existing result directory without --resume: {output_dir}")


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


def _format_metric(metric: Dict[str, Any]) -> str:
    return f"{metric.get('metric', '')}={float(metric.get('value', float('nan'))):.6f}"


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
        "source_gain_normalized",
        "retention",
        "degradation",
        "higher_is_better",
        "target_metric",
    ]
    tradeoff_df = overall_df[[c for c in tradeoff_cols if c in overall_df.columns]].copy() if not overall_df.empty else pd.DataFrame(columns=tradeoff_cols)
    tradeoff_df.to_csv(output_dir / "source_gain_vs_future_retention.csv", index=False)


def load_stage1_encoder(args: argparse.Namespace, seed: int, device: torch.device) -> tuple[nn.Module, Dict[str, Any]]:
    ckpt_spec = str(args.student_encoder_ckpt).strip()
    if ckpt_spec:
        ckpt_path = Path(ckpt_spec)
        if ckpt_path.is_dir():
            ckpt_dir = ckpt_path
        else:
            # Single checkpoint file: derive ckpt_dir and method from the path
            ckpt_dir = ckpt_path.parent.parent  # .../exp_xxx/checkpoints/ -> .../exp_xxx/
            resolved_method = ckpt_path.stem  # e.g. ns_kd_lam0.1_seed0 -> use as-is with split
            # Strip _seed{N} suffix to get method name
            import re
            m = re.match(r"^(.+)_seed\d+$", ckpt_path.stem)
            resolved_method = m.group(1) if m else ckpt_path.stem
            encoder, meta = load_stage1_imu_encoder(
                student=str(args.student),
                ckpt_dir=ckpt_dir,
                method=resolved_method,
                seed=int(seed),
                device=device,
            )
            return encoder, meta
    else:
        ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir is not None else None
        if ckpt_dir is None:
            if str(args.student) in DEFAULT_CKPT_DIRS:
                ckpt_dir = Path(DEFAULT_CKPT_DIRS[str(args.student)])
            else:
                raise ValueError(f"No checkpoint directory for student={args.student}; pass --ckpt_dir or --student_encoder_ckpt")

    resolved_method = resolve_stage1_method(ckpt_dir, str(args.stage1_method), int(seed))
    encoder, meta = load_stage1_imu_encoder(
        student=str(args.student),
        ckpt_dir=ckpt_dir,
        method=resolved_method,
        seed=int(seed),
        device=device,
    )
    meta["resolved_method"] = str(resolved_method)
    return encoder, meta


def run_case(
    *,
    case_name: str,
    case_cfg: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    case_rows: Dict[str, List[Dict[str, Any]]],
    all_rows: List[Dict[str, Any]],
    completed_keys: set[tuple[str, str, str, int]],
) -> None:
    source_tasks = list(map(str, case_cfg["source_tasks"]))
    target_tasks = list(map(str, case_cfg["target_tasks"]))

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
        bundle_cache: Dict[str, Any] = {}
        task_bank: Dict[str, PreparedTask] = {}
        for task_name in list(dict.fromkeys(source_tasks + target_tasks)):
            print(f"[data] loading task={task_name} seed={seed}", flush=True)
            task_bank[str(task_name)] = build_imu_task_data(str(task_name), args, int(seed), bundle_cache)

        enc_e0, ck_meta = load_stage1_encoder(args, int(seed), device)
        enc_dim = int(ck_meta["enc_dim"])
        print(
            f"[case] name={case_name} seed={seed} desc={case_cfg['description']} student_ckpt={ck_meta['ckpt_path']} enc_dim={enc_dim}",
            flush=True,
        )

        source_baselines: Dict[str, Dict[str, Any]] = {}
        target_baselines: Dict[str, Dict[str, Any]] = {}
        for source_task in source_tasks:
            print(f"[baseline source] case={case_name} seed={seed} task={source_task}", flush=True)
            source_baselines[source_task] = fit_frozen_probe(
                encoder=enc_e0,
                enc_dim=enc_dim,
                task_data=task_bank[source_task].task_data,
                device=device,
                args=args,
                log_prefix=f"[E0src/{case_name}/{source_task}/s{seed}] ",
            )
        for target_task in target_tasks:
            print(f"[baseline target] case={case_name} seed={seed} task={target_task}", flush=True)
            target_baselines[target_task] = fit_frozen_probe(
                encoder=enc_e0,
                enc_dim=enc_dim,
                task_data=task_bank[target_task].task_data,
                device=device,
                args=args,
                log_prefix=f"[A/{case_name}/{target_task}/s{seed}] ",
            )
        student_source_before = _source_metric_map_from_baselines(source_baselines)

        for method in args.methods:
            method_expected = [(str(case_name), str(method), str(target_task), int(seed)) for target_task in target_tasks]
            if method_expected and all(key in completed_keys for key in method_expected):
                print(f"[resume] skip case={case_name} seed={seed} method={method} (all targets already complete)", flush=True)
                continue

            print(
                f"[stage2-config] case={case_name} seed={seed} method={method} "
                f"early_stop={not args.no_early_stop_stage2 if method not in NON_STAGE2_METHODS else False} "
                f"checkpoint={'final' if args.use_final_checkpoint or method in NON_STAGE2_METHODS else 'best'} "
                f"batch_size={args.batch_size if method not in NON_STAGE2_METHODS else 0} "
                f"epochs={args.stage2_epochs if method not in NON_STAGE2_METHODS else 0}",
                flush=True,
            )

            _row_encoder_ckpt = str(ck_meta.get("ckpt_path", ""))
            if method == "e0_probe":
                source_before = student_source_before
                source_final = {task_name: dict(metric) for task_name, metric in source_before.items()}
                adapted_encoder = enc_e0
                probe_enc_dim = int(enc_dim)
                checkpoint_used = "student_e0"
                stage2_early_stop = False
                stage2_checkpoint_path = ""
                source_gain_normalized = 1.0
            elif method == "teacher_probe":
                _teacher_ckpt_path = str(getattr(args, "teacher_ckpt", "NormWear/checkpoints/normwear_pretrain_ckpt.pth"))
                teacher_enc = NormWearTeacherEncoder(_teacher_ckpt_path, device=device)
                with torch.inference_mode():
                    teacher_enc_dim = int(teacher_enc(torch.zeros(2, 6, 500, device=device)).shape[-1])
                _teacher_src_baselines: Dict[str, Any] = {}
                for _tsrc in source_tasks:
                    print(f"[baseline source/teacher] case={case_name} seed={seed} task={_tsrc}", flush=True)
                    _teacher_src_baselines[_tsrc] = fit_frozen_probe(
                        encoder=teacher_enc,
                        enc_dim=teacher_enc_dim,
                        task_data=task_bank[_tsrc].task_data,
                        device=device,
                        args=args,
                        log_prefix=f"[Tsrc/{case_name}/{_tsrc}/s{seed}] ",
                    )
                source_before = _source_metric_map_from_baselines(_teacher_src_baselines)
                source_final = {t: dict(m) for t, m in source_before.items()}
                adapted_encoder = teacher_enc
                probe_enc_dim = int(teacher_enc_dim)
                checkpoint_used = "normwear_teacher"
                stage2_early_stop = False
                stage2_checkpoint_path = ""
                source_gain_normalized = 1.0
                _row_encoder_ckpt = _teacher_ckpt_path
                for _tsrc in source_tasks:
                    _gain = _normalized_gain(source_before[_tsrc], source_final[_tsrc])
                    print(
                        f"[source-metric] case={case_name} seed={seed} method={method} task={_tsrc} "
                        f"before={_format_metric(source_before[_tsrc])} "
                        f"final={_format_metric(source_final[_tsrc])} normalized_gain={_gain:.6f}",
                        flush=True,
                    )
            else:
                source_before = student_source_before
                source_task_data = [task_bank[task_name].task_data for task_name in source_tasks]
                stage2_model = build_stage2_multitask_model(enc_e0, enc_dim, source_task_data, float(args.dropout)).to(device)
                train_loaders, val_loaders, test_loaders = make_task_loaders(source_task_data, int(args.batch_size))
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
                    care_auto_reuse_lambda=float(args.care_auto_reuse_lambda),
                    care_auto_anchor_source=str(args.care_auto_anchor_source),
                    care_auto_reuse_k=int(args.care_auto_reuse_k),
                    care_auto_anchor_max_batches=int(args.care_auto_anchor_max_batches),
                    care_auto_anchor_basis_samples=int(args.care_auto_anchor_basis_samples),
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
                        f"[source-metric] case={case_name} seed={seed} method={method} task={source_task} "
                        f"before={_format_metric(source_before[source_task])} "
                        f"final={_format_metric(source_final[source_task])} normalized_gain={gain:.6f}",
                        flush=True,
                    )
                source_gain_normalized = float(np.nanmean(np.asarray(source_gains, dtype=np.float64))) if source_gains else float("nan")

            for target_task in target_tasks:
                key = (str(case_name), str(method), str(target_task), int(seed))
                if key in completed_keys:
                    print(f"[resume] skip case={case_name} seed={seed} method={method} target={target_task}", flush=True)
                    continue

                target_student_baseline = target_baselines[target_task]
                target_e0_probe = metric_value(target_student_baseline["test_metric"])
                higher_is_better = bool(target_student_baseline["test_metric"]["higher_better"])
                target_metric = str(target_student_baseline["test_metric"]["metric"])

                if method == "e0_probe":
                    target_after_stage2_probe = float(target_e0_probe)
                else:
                    print(f"[probe] case={case_name} seed={seed} method={method} target={target_task}", flush=True)
                    probe = fit_frozen_probe(
                        encoder=adapted_encoder,
                        enc_dim=probe_enc_dim,
                        task_data=task_bank[target_task].task_data,
                        device=device,
                        args=args,
                        log_prefix=f"[probe/{case_name}/{method}/{target_task}/s{seed}] ",
                    )
                    target_after_stage2_probe = metric_value(probe["test_metric"])

                retention, degradation = _retention_and_degradation(target_e0_probe, target_after_stage2_probe, higher_is_better)
                print(
                    f"[target-probe] case={case_name} seed={seed} method={method} target={target_task} "
                    f"metric={target_metric} e0={target_e0_probe:.6f} after={target_after_stage2_probe:.6f} "
                    f"retention={retention:.6f} degradation={degradation:.6f}",
                    flush=True,
                )

                row = {
                    "case_name": str(case_name),
                    "method": str(method),
                    "source_tasks": json.dumps(source_tasks),
                    "target_task": str(target_task),
                    "seed": int(seed),
                    "student_encoder_ckpt": _row_encoder_ckpt,
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
                    "target_after_stage2_probe": float(target_after_stage2_probe),
                    "retention": float(retention),
                    "degradation": float(degradation),
                    "higher_is_better": bool(higher_is_better),
                    "fairgrad_alpha": float(args.fairgrad_alpha) if method == "fairgrad_mtl" else float("nan"),
                }
                case_rows[case_name].append(row)
                all_rows.append(row)
                completed_keys.add(key)
                _write_outputs(Path(args.output_dir), case_rows, all_rows)


def main() -> None:
    args = parse_args()
    args.cases = _parse_name_list(args.cases, CASE_DEFS.keys(), allow_all=True)
    args.methods = _parse_name_list(args.methods, ALL_METHODS, allow_all=False)
    device = _resolve_run_device(args)
    # Disable mem_efficient_sdp (its backward is not implemented for retain_graph).
    # Keep flash_sdp (proper backward, O(n) memory) and math_sdp as fallback.
    _cuda_be = getattr(torch.backends, "cuda", None)
    if _cuda_be is not None:
        if hasattr(_cuda_be, "enable_mem_efficient_sdp"):
            _cuda_be.enable_mem_efficient_sdp(False)
        if hasattr(_cuda_be, "enable_flash_sdp"):
            _cuda_be.enable_flash_sdp(True)
        if hasattr(_cuda_be, "enable_math_sdp"):
            _cuda_be.enable_math_sdp(True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "student": str(args.student),
        "stage1_method": str(args.stage1_method),
        "cases": list(args.cases),
        "methods": list(args.methods),
        "batch_size": int(args.batch_size),
        "stage2_epochs": int(args.stage2_epochs),
        "probe_epochs": int(args.probe_epochs),
        "fairgrad_alpha": float(args.fairgrad_alpha),
        "seeds": [int(seed) for seed in args.seeds],
        "resume": bool(args.resume),
        "no_early_stop_stage2": bool(args.no_early_stop_stage2),
        "use_final_checkpoint": bool(args.use_final_checkpoint),
        "device": str(device),
        "gpu": int(args.gpu) if args.gpu is not None else None,
        "limit_train": int(args.limit_train),
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

    for case_name in args.cases:
        run_case(
            case_name=case_name,
            case_cfg=CASE_DEFS[case_name],
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

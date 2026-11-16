#!/usr/bin/env python3
"""Stage-2 MTL adaptation with frozen future-task probes for ECG case studies."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_specialization_sweep_ecg import (
    build_ecg_task_data,
    fit_frozen_probe,
    metric_payload,
    metric_value,
    task_dataset,
)
from run_stage2_multitask_ppg import MultiTaskModel
from run_stage23_ecg_unified import (
    DEFAULT_CKPT_DIRS,
    ECGStudentFeatureEncoder,
    build_student,
    fit_stage2,
    load_stage1_ecgfounder,
    make_task_loaders_ecg,
    resolve_device,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "results/stage2_mtl_future_probe_ecg"
ALL_METHODS = ("lifefm_adapt",)
METHOD_TO_STAGE2 = {
    "joint_mtl": "joint_finetune",
    "gradnorm_mtl": "gradnorm_joint",
    "pcgrad_mtl": "pcgrad_joint",
    "fairgrad_mtl": "fairgrad_joint",
    "lifefm_adapt": "lifefm_adapt",
}
NON_STAGE2_METHODS = frozenset()
CASE_DEFS: Dict[str, Dict[str, Any]] = {
    "ecg_ptbxl_to_cinc_af": {
        "case_name": "ecg_ptbxl_to_cinc_af",
        "description": "PTB-XL diagnostic adaptation to external CinC AF detection.",
        "source_tasks": ["super5", "cd_bin"],
        "target_tasks": ["cinc_af_bin"],
    },
    "ecg_chapman_to_ptbxl_morphology": {
        "case_name": "ecg_chapman_to_ptbxl_morphology",
        "description": "Chapman rhythm/abnormality adaptation to PTB-XL morphology diagnosis.",
        "source_tasks": ["chap_rhythm4", "chap_abnormal_bin"],
        "target_tasks": ["sttc_bin", "cd_bin"],
    },
}


class ECGFounderEncoderAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, device: torch.device, half: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.runtime_device = torch.device(device)
        self.half = bool(half and self.runtime_device.type == "cuda")
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.half:
            self.backbone = self.backbone.half()
        self.backbone.to(self.runtime_device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.runtime_device, non_blocking=True)
        if self.half:
            x = x.half()
        _, feat = self.backbone(x)
        return feat.float()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_encoder_ckpt", default="", help="Checkpoint file, checkpoint directory, or template path with {seed}/{method} placeholders.")
    ap.add_argument("--teacher_encoder_ckpt", default="", help="Optional ECGFounder checkpoint override. Defaults to the canonical 12-lead ECGFounder checkpoint.")
    ap.add_argument("--student", default="BF10_NB7")
    ap.add_argument("--stage1_method", default="lifefm_kd")
    ap.add_argument("--cases", nargs="+", default=["all"])
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
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
    ap.add_argument("--lambda-core", "--care_auto_reuse_lambda", dest="care_auto_reuse_lambda", type=float, default=0.05)
    ap.add_argument("--care_auto_anchor_source", type=str, default="source_union", choices=["source_union"])
    ap.add_argument("--care_auto_reuse_k", type=int, default=0)
    ap.add_argument("--care_auto_anchor_max_batches", type=int, default=8)
    ap.add_argument("--care_auto_anchor_basis_samples", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_early_stop_stage2", dest="no_early_stop_stage2", action="store_true")
    ap.add_argument("--allow_early_stop_stage2", dest="no_early_stop_stage2", action="store_false")
    ap.add_argument("--use_final_checkpoint", dest="use_final_checkpoint", action="store_true")
    ap.add_argument("--use_best_checkpoint", dest="use_final_checkpoint", action="store_false")
    ap.add_argument(
        "--allow_early_stop_methods",
        nargs="+",
        default=[],
        help="Stage-2 methods that should override the global early-stop policy and allow early stopping.",
    )
    ap.add_argument(
        "--use_best_checkpoint_methods",
        nargs="+",
        default=[],
        help="Stage-2 methods that should override the global checkpoint policy and restore the best checkpoint.",
    )
    ap.set_defaults(no_early_stop_stage2=True, use_final_checkpoint=True)
    return ap.parse_args()


def _resolve_run_device(args: argparse.Namespace) -> torch.device:
    if args.gpu is not None:
        return resolve_device(f"cuda:{int(args.gpu)}")
    return resolve_device(str(args.device))


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


def _load_stage1_from_path(student: str, ckpt_path: Path, seed: int, device: torch.device) -> tuple[nn.Module, Dict[str, Any]]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    full, feat_dim = build_student(student, int(seed))
    full.load_state_dict(obj["state_dict"], strict=True)
    enc = ECGStudentFeatureEncoder(full.enc).to(device)
    meta = dict(obj)
    meta.pop("state_dict", None)
    meta["ckpt_path"] = str(ckpt_path)
    meta["enc_dim"] = int(obj.get("feat_dim", feat_dim))
    return enc, meta


def load_stage1_encoder(args: argparse.Namespace, seed: int, device: torch.device) -> tuple[nn.Module, Dict[str, Any]]:
    ckpt_spec = str(args.student_encoder_ckpt).strip()
    if ckpt_spec:
        if "{" in ckpt_spec:
            ckpt_path = Path(ckpt_spec.format(seed=int(seed), method=str(args.stage1_method)))
            return _load_stage1_from_path(str(args.student), ckpt_path, int(seed), device)
        ckpt_path = Path(ckpt_spec)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / f"{str(args.stage1_method)}_seed{int(seed)}.pt"
            return _load_stage1_from_path(str(args.student), ckpt_path, int(seed), device)
        if len(list(args.seeds)) > 1 and f"seed{int(seed)}" not in ckpt_path.name:
            raise ValueError("--student_encoder_ckpt file path must include the seed or use a {seed} template when running multiple seeds")
        return _load_stage1_from_path(str(args.student), ckpt_path, int(seed), device)

    return load_stage1_ecgfounder(str(args.student), str(args.stage1_method), int(seed), device)


def load_ecgfounder_teacher_encoder(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, Dict[str, Any]]:
    from ecgfounder_teacher import CKPT_12LEAD, ECGFOUNDER_DIM, _build_net1d

    ckpt_path = Path(str(args.teacher_encoder_ckpt).strip()) if str(args.teacher_encoder_ckpt).strip() else Path(CKPT_12LEAD)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing ECGFounder checkpoint: {ckpt_path}")

    backbone = _build_net1d(in_channels=12, n_classes=150)
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    state_dict = {str(k).replace("module.", ""): v for k, v in dict(state_dict).items()}
    backbone.load_state_dict(state_dict, strict=False)
    wrapped = ECGFounderEncoderAdapter(backbone, device=device, half=True)
    return wrapped, {
        "teacher_name": "ECGFounder_12lead",
        "ckpt_path": str(ckpt_path),
        "enc_dim": int(ECGFOUNDER_DIM),
    }


def build_stage2_multitask_model(encoder: nn.Module, enc_dim: int, task_data_all: List[Any], dropout: float) -> MultiTaskModel:
    dataset_names = {task_dataset(td.runtime.spec.name) for td in task_data_all}
    if len(dataset_names) != 1:
        raise ValueError(f"Stage2 source tasks must share one dataset, got {sorted(dataset_names)}")
    input_len = max(int(td.input_len) for td in task_data_all)
    return MultiTaskModel(
        copy.deepcopy(encoder),
        int(enc_dim),
        [td.runtime for td in task_data_all],
        float(dropout),
        dataset_name=next(iter(dataset_names)),
        hz=500,
        input_len=int(input_len),
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


def _retention_and_degradation(reference_probe: float, target_after_stage2_probe: float, higher_is_better: bool) -> tuple[float, float]:
    if not np.isfinite(reference_probe) or not np.isfinite(target_after_stage2_probe):
        return float("nan"), float("nan")
    if higher_is_better:
        return float(target_after_stage2_probe / max(reference_probe, 1e-12)), float(reference_probe - target_after_stage2_probe)
    return float(reference_probe / max(target_after_stage2_probe, 1e-12)), float(target_after_stage2_probe - reference_probe)


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


def _write_outputs(output_dir: Path, case_rows: Dict[str, List[Dict[str, Any]]], all_rows: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".write_lock"
    with open(lock_path, "a") as _lock_fh:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX)
        try:
            overall_path = output_dir / "overall_summary.csv"
            if overall_path.exists():
                try:
                    _disk_rows = pd.read_csv(overall_path).to_dict(orient="records")
                    all_rows = _dedupe_rows(_disk_rows + all_rows)
                except Exception:
                    pass
            all_rows = _dedupe_rows(all_rows)
            _merged_cr: Dict[str, List[Dict[str, Any]]] = {cn: [] for cn in case_rows}
            for _r in all_rows:
                _merged_cr.setdefault(str(_r.get("case_name", "")), []).append(_r)
            for _cn, _rows in _merged_cr.items():
                if _rows:
                    pd.DataFrame(_dedupe_rows(_rows)).to_csv(output_dir / f"{_cn}_detailed.csv", index=False)
            overall_df = pd.DataFrame(all_rows)
            overall_df.to_csv(output_dir / "overall_summary.csv", index=False)
        finally:
            fcntl.flock(_lock_fh, fcntl.LOCK_UN)
    tradeoff_cols = [
        "case_name",
        "method",
        "target_task",
        "seed",
        "source_gain_normalized",
        "retention_vs_student_e0",
        "retention_vs_teacher_e0",
        "degradation_vs_student_e0",
        "degradation_vs_teacher_e0",
        "higher_is_better",
        "target_metric",
    ]
    tradeoff_df = overall_df[tradeoff_cols].copy() if not overall_df.empty else pd.DataFrame(columns=tradeoff_cols)
    if not tradeoff_df.empty:
        tradeoff_df["retention_below_student_e0"] = tradeoff_df["retention_vs_student_e0"] < 1.0
        tradeoff_df["retention_below_teacher_e0"] = tradeoff_df["retention_vs_teacher_e0"] < 1.0
    else:
        tradeoff_df["retention_below_student_e0"] = pd.Series(dtype=bool)
        tradeoff_df["retention_below_teacher_e0"] = pd.Series(dtype=bool)
    tradeoff_df.to_csv(output_dir / "source_gain_vs_future_retention.csv", index=False)


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
        task_bank: Dict[str, Any] = {}
        for task_name in list(dict.fromkeys(source_tasks + target_tasks)):
            task_bank[str(task_name)] = build_ecg_task_data(str(task_name), args)

        enc_e0, ck_meta = load_stage1_encoder(args, int(seed), device)
        enc_dim = int(ck_meta["enc_dim"])
        print(
            f"[case] name={case_name} seed={seed} desc={case_cfg['description']} student_ckpt={ck_meta['ckpt_path']}",
            flush=True,
        )

        source_baselines: Dict[str, Dict[str, Any]] = {}
        target_baselines: Dict[str, Dict[str, Any]] = {}
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

        teacher_encoder, teacher_ck_meta = load_ecgfounder_teacher_encoder(args, device)
        teacher_enc_dim = int(teacher_ck_meta["enc_dim"])
        print(
            f"[teacher] case={case_name} seed={seed} teacher={teacher_ck_meta['teacher_name']} ckpt={teacher_ck_meta['ckpt_path']}",
            flush=True,
        )
        teacher_target_baselines: Dict[str, Dict[str, Any]] = {}
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

        teacher_source_baselines: Dict[str, Dict[str, Any]] | None = None
        teacher_source_before: Dict[str, Dict[str, Any]] | None = None
        if "ecgfounder_teacher_e0" in args.methods:
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
            _, _, _fresh_keys = _load_existing_rows(Path(args.output_dir), [case_name])
            completed_keys |= _fresh_keys
            method_expected = [(str(case_name), str(method), str(target_task), int(seed)) for target_task in target_tasks]
            if method_expected and all(key in completed_keys for key in method_expected):
                print(f"[resume] skip case={case_name} seed={seed} method={method} (all targets already complete)", flush=True)
                continue

            is_stage2_method = method not in NON_STAGE2_METHODS
            method_no_early_stop = bool(args.no_early_stop_stage2)
            method_use_final_checkpoint = bool(args.use_final_checkpoint)
            if method in args.allow_early_stop_methods:
                method_no_early_stop = False
            if method in args.use_best_checkpoint_methods:
                method_use_final_checkpoint = False
            checkpoint_label = "na" if not is_stage2_method else ("final" if method_use_final_checkpoint else "best")

            print(
                f"[stage2-config] case={case_name} seed={seed} method={method} early_stop={not method_no_early_stop if is_stage2_method else False} checkpoint={checkpoint_label} batch_size={args.batch_size if is_stage2_method else 0} epochs={args.stage2_epochs if is_stage2_method else 0} fairgrad_alpha={args.fairgrad_alpha if method == 'fairgrad_mtl' else 'na'}",
                flush=True,
            )

            if method == "ecgfounder_teacher_e0":
                assert teacher_source_before is not None
                source_before = teacher_source_before
                source_final = {task_name: dict(metric) for task_name, metric in source_before.items()}
                adapted_encoder = teacher_encoder
                probe_enc_dim = int(teacher_enc_dim)
                checkpoint_used = "teacher_e0"
                stage2_early_stop = False
                stage2_checkpoint_path = ""
                encoder_type = "teacher_e0"
                student_ckpt_path = ""
                teacher_ckpt_path = str(teacher_ck_meta["ckpt_path"])
                source_gain_normalized = 1.0
            elif method == "e0_probe":
                source_before = student_source_before
                source_final = {task_name: dict(metric) for task_name, metric in source_before.items()}
                adapted_encoder = enc_e0
                probe_enc_dim = int(enc_dim)
                checkpoint_used = "student_e0"
                stage2_early_stop = False
                stage2_checkpoint_path = ""
                encoder_type = "student_e0"
                student_ckpt_path = str(ck_meta["ckpt_path"])
                teacher_ckpt_path = str(teacher_ck_meta["ckpt_path"])
                source_gain_normalized = 1.0
            else:
                source_before = student_source_before
                source_task_data = [task_bank[task_name] for task_name in source_tasks]
                stage2_model = build_stage2_multitask_model(enc_e0, enc_dim, source_task_data, float(args.dropout)).to(device)
                train_loaders, val_loaders, test_loaders = make_task_loaders_ecg(source_task_data, int(args.batch_size))
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
                    enable_early_stop=not method_no_early_stop,
                    restore_best_checkpoint=not method_use_final_checkpoint,
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
                encoder_type = "student_stage2"
                student_ckpt_path = str(ck_meta["ckpt_path"])
                teacher_ckpt_path = str(teacher_ck_meta["ckpt_path"])

                source_gains: List[float] = []
                for source_task in source_tasks:
                    gain = _normalized_gain(source_before[source_task], source_final[source_task])
                    source_gains.append(gain)
                    print(
                        f"[source-metric] case={case_name} seed={seed} method={method} task={source_task} before={_format_metric(source_before[source_task])} final={_format_metric(source_final[source_task])} normalized_gain={gain:.6f}",
                        flush=True,
                    )
                source_gain_normalized = float(np.nanmean(np.asarray(source_gains, dtype=np.float64))) if source_gains else float("nan")

            for target_task in target_tasks:
                key = (str(case_name), str(method), str(target_task), int(seed))
                if key in completed_keys:
                    print(f"[resume] skip case={case_name} seed={seed} method={method} target={target_task}", flush=True)
                    continue

                target_student_baseline = target_baselines[target_task]
                target_teacher_baseline = teacher_target_baselines[target_task]
                target_e0_probe = metric_value(target_student_baseline["test_metric"])
                target_teacher_probe = metric_value(target_teacher_baseline["test_metric"])
                higher_is_better = bool(target_student_baseline["test_metric"]["higher_better"])
                target_metric = str(target_student_baseline["test_metric"]["metric"])

                if method == "ecgfounder_teacher_e0":
                    target_after_stage2_probe = float(target_teacher_probe)
                elif method == "e0_probe":
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

                retention_vs_student_e0, degradation_vs_student_e0 = _retention_and_degradation(target_e0_probe, target_after_stage2_probe, higher_is_better)
                retention_vs_teacher_e0, degradation_vs_teacher_e0 = _retention_and_degradation(target_teacher_probe, target_after_stage2_probe, higher_is_better)
                print(
                    f"[target-probe] case={case_name} seed={seed} method={method} target={target_task} metric={target_metric} student_e0={target_e0_probe:.6f} teacher_e0={target_teacher_probe:.6f} after={target_after_stage2_probe:.6f} ret_student={retention_vs_student_e0:.6f} ret_teacher={retention_vs_teacher_e0:.6f}",
                    flush=True,
                )

                row = {
                    "case_name": str(case_name),
                    "method": str(method),
                    "source_tasks": json.dumps(source_tasks),
                    "target_task": str(target_task),
                    "seed": int(seed),
                    "encoder_type": str(encoder_type),
                    "student_encoder_ckpt": str(student_ckpt_path),
                    "teacher_encoder_ckpt": str(teacher_ckpt_path),
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
                    "retention": float(retention_vs_student_e0),
                    "degradation": float(degradation_vs_student_e0),
                    "retention_vs_student_e0": float(retention_vs_student_e0),
                    "retention_vs_teacher_e0": float(retention_vs_teacher_e0),
                    "degradation_vs_student_e0": float(degradation_vs_student_e0),
                    "degradation_vs_teacher_e0": float(degradation_vs_teacher_e0),
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
    args.allow_early_stop_methods = _parse_name_list(args.allow_early_stop_methods, METHOD_TO_STAGE2.keys(), allow_all=False)
    args.use_best_checkpoint_methods = _parse_name_list(args.use_best_checkpoint_methods, METHOD_TO_STAGE2.keys(), allow_all=False)
    device = _resolve_run_device(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "student_encoder_ckpt": str(args.student_encoder_ckpt),
        "teacher_encoder_ckpt": str(args.teacher_encoder_ckpt),
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
        "allow_early_stop_methods": list(args.allow_early_stop_methods),
        "use_best_checkpoint_methods": list(args.use_best_checkpoint_methods),
        "device": str(device),
        "gpu": int(args.gpu) if args.gpu is not None else None,
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
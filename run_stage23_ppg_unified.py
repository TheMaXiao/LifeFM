#!/usr/bin/env python3
"""Unified Stage-2 -> Stage-3 evaluation for compressed PPG encoders.

This runner treats Stage 3 as a continuation test of the Stage-2 encoder state:

1. train a Stage-2 multi-task model on an initial task subset,
2. add a held-out future task using only new-task data,
3. evaluate both the newly added task and forgetting on the old tasks.

The first PPG loop focuses on two natural dataset-level task packs:

- `ppgbp_clinical`: `hr + sysbp + diasbp` -> add `hypertension`
- `bidmc_resp`: `hr + spo2` -> add `rr`

The goal is to compare Stage-2 policies by the tradeoff:

- current multi-task performance after Stage 2,
- old-task retention after Stage 3,
- new-task learnability after Stage 3.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_stage2_multitask_ppg import (
    ROOT,
    DATASET_SPECS,
    DEFAULT_CKPT_DIRS,
    TaskData,
    TaskRuntime,
    TaskSpec,
    TaskHead,
    MultiTaskModel,
    SingleTaskArrayDataset,
    count_params,
    build_encoder,
    disable_dropout,
    evaluate,
    init_regression_biases,
    load_multitask_dataset,
    load_stage1_full,
    make_task_loaders,
    maybe_limit_split,
    resolve_device,
    set_encoder_trainable_scope,
    set_seed,
    split_subjects,
    task_losses,
    validation_loss,
)
from run_stage2_student_subspace import assign_flat_grad, flatten_grads, pcgrad_merge


DEFAULT_OUT_ROOT = ROOT / "kd_sweep_results/stage23_ppg_unified"
STAGE2_METHODS = ("frozen_heads", "joint_finetune", "gradnorm_joint", "pcgrad_joint", "fairgrad_joint", "scope_matched_naive", "care_v1", "care_v2", "care_v3", "care_v2_normmse", "care_v2_cosine", "care_v2_statnorm", "care_v2_tasknorm", "care_direct", "care_direct_v2", "care_direct_v2b", "care_direct_v3", "care_direct_v4", "care_v5a", "care_v5b", "lora_tune")
STAGE3_POLICIES = ("naive_ft", "new_head_only", "scope_matched_naive", "care_direct", "care_direct_v2b", "care_direct_v3", "care_direct_v4", "care_v5a", "care_v5b", "lora_ft", "lwf_ft", "anchor_only", "lpr", "cpca_core", "lwp", "cpca_ts", "cpca_ca", "cpca_tsca", "sema", "lwf_core", "lwp_core", "lpr_core", "core_dwdp", "core_affinity", "core_affinity_hybrid", "core_task", "lwp_task", "head_mlp", "ewc", "sgp", "sgp_core")
CARE_AUTO_STAGE2_METHODS = {"care_v1", "care_v2", "care_v3", "care_v2_normmse", "care_v2_cosine", "care_v2_statnorm", "care_v2_tasknorm"}
CARE_STAGE2_METHODS = {"care_direct", "care_direct_v2", "care_direct_v2b", "care_direct_v3", "care_direct_v4", "care_v5a", "care_v5b"}
CARE_STAGE3_POLICIES = {"care_direct", "care_direct_v2b", "care_direct_v3", "care_direct_v4", "care_v5a", "care_v5b", "anchor_only", "cpca_core", "cpca_ts", "cpca_ca", "cpca_tsca"}
CARE_V5_STAGE2_METHODS = {"care_v5a", "care_v5b"}
CARE_V5_STAGE3_POLICIES = {"care_v5a", "care_v5b"}
ALL_CARE_STAGE2_METHODS = CARE_AUTO_STAGE2_METHODS | CARE_STAGE2_METHODS
SCOPE_MATCHED_STAGE2_METHODS = {"scope_matched_naive"}
SCOPE_MATCHED_STAGE3_POLICIES = {"scope_matched_naive"}
DISTILL_STAGE3_POLICIES = CARE_STAGE3_POLICIES | {"lwf_ft", "lwp", "lwf_core", "lwp_core", "core_dwdp", "core_affinity", "core_affinity_hybrid", "core_task", "lwp_task"}

# PPG-DaLiA activity IDs from the official readme:
# 0 transient, 1 sitting, 2 stairs, 3 table soccer, 4 cycling,
# 5 driving, 6 lunch break, 7 walking, 8 working.
DALIA_MOTION_POSITIVE = {0, 2, 3, 4, 7}

SPORTPPG_ACTIVITY_ORDER = ["stationary", "walking", "running", "badminton", "tabletennis", "basketball"]
SPORTPPG_DYNAMIC_POSITIVE = {
    idx for idx, name in enumerate(SPORTPPG_ACTIVITY_ORDER) if name != "stationary"
}


@dataclass(frozen=True)
class TaskInterface:
    task_name: str
    loss_name: str
    metric_name: str
    input_support_sec: float
    preserve_dist: str
    output_structure: str


PACKS: Dict[str, Dict[str, Any]] = {
    "ppgbp_clinical": {
        "dataset": "ppgbp",
        "temporal_policy": "shared_window",
        "input_window_policy": "shared_native",
        "initial_tasks": ["hr", "sysbp", "diasbp"],
        "future_tasks": ["hypertension"],
        "description": "Mixed clinical regression pack, then add hypertension.",
    },
    "ppgbp_cross_objective_bp_probe": {
        "dataset": "ppgbp",
        "temporal_policy": "shared_window",
        "input_window_policy": "shared_native",
        "support_policy_map": {
            "hr": "single_segment_native",
            "hypertension": "single_segment_native",
            "sysbp": "single_segment_native",
            "diasbp": "single_segment_native",
        },
        "initial_tasks": ["hr", "hypertension"],
        "future_tasks": ["sysbp", "diasbp"],
        "description": "PPG-BP cross-objective Stage2-only split: train on HR + hypertension, probe on SBP + DBP using native single-segment support.",
    },
    "ppgbp_regression_to_htn_probe": {
        "dataset": "ppgbp",
        "temporal_policy": "shared_window",
        "input_window_policy": "shared_native",
        "support_policy_map": {
            "hr": "single_segment_native",
            "sysbp": "single_segment_native",
            "diasbp": "single_segment_native",
            "hypertension": "single_segment_native",
        },
        "initial_tasks": ["hr", "sysbp", "diasbp"],
        "future_tasks": ["hypertension"],
        "description": "PPG-BP Stage2-only cross-interface split: train regression tasks HR + SBP + DBP, then probe held-out hypertension classification.",
    },
    "bidmc_resp": {
        "dataset": "bidmc",
        "temporal_policy": "task_specific_crop",
        "input_window_policy": "shared_native",
        "initial_tasks": ["hr", "spo2"],
        "future_tasks": ["rr"],
        "description": "Short-support cardiovascular pack, then add respiration.",
    },
    "bidmc_resp_input": {
        "dataset": "bidmc",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"hr": 6.0, "spo2": 6.0, "rr": 10.0},
        "initial_tasks": ["hr", "spo2"],
        "future_tasks": ["rr"],
        "description": "True task-specific raw input windows: 6s for HR/SpO2, 10s for RR.",
    },
    "dalia_motion": {
        "dataset": "dalia",
        "temporal_policy": "shared_window",
        "input_window_policy": "shared_native",
        "initial_tasks": ["hr", "activity"],
        "future_tasks": ["motion_binary"],
        "derived_tasks": {
            "motion_binary": {
                "source": "activity",
                "positive_ids": sorted(DALIA_MOTION_POSITIVE),
            },
        },
        "description": "PPG-DaLiA pack with HR + fine activity, then add a derived motion-vs-low-motion task.",
    },
    "dalia_motion_input": {
        "dataset": "dalia",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"hr": 8.0, "activity": 2.0, "motion_binary": 4.0},
        "initial_tasks": ["hr", "activity"],
        "future_tasks": ["motion_binary"],
        "derived_tasks": {
            "motion_binary": {
                "source": "activity",
                "positive_ids": sorted(DALIA_MOTION_POSITIVE),
            },
        },
        "description": "PPG-DaLiA Challenge2 branch with 8s HR, 2s activity, and 4s motion-binary raw windows.",
    },
    "sportppg_dynamic": {
        "dataset": "sportppg",
        "temporal_policy": "shared_window",
        "input_window_policy": "shared_native",
        "initial_tasks": ["hr", "activity"],
        "future_tasks": ["dynamic_binary"],
        "derived_tasks": {
            "dynamic_binary": {
                "source": "activity",
                "positive_ids": sorted(SPORTPPG_DYNAMIC_POSITIVE),
            },
        },
        "description": "SportPPG pack with HR + fine activity, then add a derived dynamic-vs-static task.",
    },
    "sportppg_dynamic_input": {
        "dataset": "sportppg",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"hr": 10.0, "activity": 2.5, "dynamic_binary": 5.0},
        "initial_tasks": ["hr", "activity"],
        "future_tasks": ["dynamic_binary"],
        "derived_tasks": {
            "dynamic_binary": {
                "source": "activity",
                "positive_ids": sorted(SPORTPPG_DYNAMIC_POSITIVE),
            },
        },
        "description": "SportPPG Challenge2 branch with 10s HR, 2.5s fine activity, and 5s dynamic-vs-static raw windows.",
    },
    "sportppg_dynamic_probe": {
        "dataset": "sportppg",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"hr": 10.0, "activity": 2.5, "dynamic_binary": 5.0},
        "initial_tasks": ["hr", "activity"],
        "future_tasks": ["dynamic_binary"],
        "derived_tasks": {
            "dynamic_binary": {
                "source": "activity",
                "positive_ids": sorted(SPORTPPG_DYNAMIC_POSITIVE),
            },
        },
        "description": "SportPPG leave-one-out reference split: train on HR + activity, probe on dynamic_binary with fixed 10s/2.5s/5s inputs.",
    },
    "sportppg_hr_probe": {
        "dataset": "sportppg",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"activity": 2.5, "dynamic_binary": 5.0, "hr": 10.0},
        "initial_tasks": ["activity", "dynamic_binary"],
        "future_tasks": ["hr"],
        "derived_tasks": {
            "dynamic_binary": {
                "source": "activity",
                "positive_ids": sorted(SPORTPPG_DYNAMIC_POSITIVE),
            },
        },
        "description": "SportPPG leave-one-out split: train on activity + dynamic_binary, probe on HR with fixed 2.5s/5s/10s inputs.",
    },
    "sportppg_hr_probe_cross_objective": {
        "dataset": "sportppg",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"activity": 2.5, "dynamic_binary": 5.0, "hr": 10.0},
        "initial_tasks": ["activity", "dynamic_binary"],
        "future_tasks": ["hr"],
        "derived_tasks": {
            "dynamic_binary": {
                "source": "activity",
                "positive_ids": sorted(SPORTPPG_DYNAMIC_POSITIVE),
            },
        },
        "description": "SportPPG cross-objective split: train on activity + dynamic_binary, probe on HR with fixed 2.5s/5s/10s inputs.",
    },
    "sportppg_activity_probe": {
        "dataset": "sportppg",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"hr": 10.0, "dynamic_binary": 5.0, "activity": 2.5},
        "initial_tasks": ["hr", "dynamic_binary"],
        "future_tasks": ["activity"],
        "derived_tasks": {
            "dynamic_binary": {
                "source": "activity",
                "positive_ids": sorted(SPORTPPG_DYNAMIC_POSITIVE),
            },
        },
        "description": "SportPPG leave-one-out split: train on HR + dynamic_binary, probe on activity with fixed 10s/5s/2.5s inputs.",
    },
    "wesad_regression_to_affect_probe": {
        "dataset": "wesad_cross_interface",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"hr_regression": 10.0, "rr_regression": 60.0, "stress_binary": 60.0, "affect_multiclass": 60.0},
        "support_policy_map": {
            "hr_regression": "10s_wrist_BVP_only_ecg_label",
            "rr_regression": "60s_wrist_BVP_only_resp_label",
            "stress_binary": "60s_wrist_BVP_only_baseline_vs_stress",
            "affect_multiclass": "60s_wrist_BVP_only_baseline_stress_amusement",
        },
        "input_modality": "wrist_BVP_only",
        "label_source_map": {"hr_regression": "chest_ECG_label_only", "rr_regression": "chest_respiration_label_only", "stress_binary": "WESAD_protocol_label", "affect_multiclass": "WESAD_protocol_label"},
        "initial_tasks": ["hr_regression", "rr_regression"],
        "future_tasks": ["stress_binary", "affect_multiclass"],
        "description": "WESAD regression-to-classification split: train HR/RR regression from wrist BVP input, then probe stress/affect classification.",
    },
    "wesad_subject_shift_seq_probe": {
        "dataset": "wesad_cross_interface_shifted30",
        "builder": "wesad_subject_shift",
        "outer_pool_ratio": 0.6,
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"hr_regression": 10.0, "rr_regression": 30.0, "stress_binary": 30.0, "affect_multiclass": 30.0},
        "support_policy_map": {
            "hr_regression": "10s_wrist_BVP_only_ecg_label",
            "rr_regression": "30s_wrist_BVP_only_resp_label",
            "stress_binary": "30s_wrist_BVP_only_baseline_vs_stress",
            "affect_multiclass": "30s_wrist_BVP_only_baseline_stress_amusement",
        },
        "input_modality": "wrist_BVP_only",
        "label_source_map": {"hr_regression": "chest_ECG_label_only", "rr_regression": "chest_respiration_label_only", "stress_binary": "WESAD_protocol_label", "affect_multiclass": "WESAD_protocol_label"},
        "initial_tasks": ["hr_regression"],
        "future_tasks": ["rr_regression", "stress_binary", "affect_multiclass"],
        "description": "Pipeline A (pure CL) WESAD subject-shift: one initial task, then sequentially add rr_regression, stress_binary, affect_multiclass.",
    },
    "sdb_to_wfppg_hr": {
        "dataset": "sdb",
        "train_dataset": "sdb",
        "probe_dataset": "wfppg",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"binary": 30.0, "multiclass": 30.0, "wrist": 10.0, "finger": 10.0},
        "initial_tasks": ["binary", "multiclass"],
        "future_tasks": ["wrist", "finger"],
        "description": "Paper-aligned PPG (SDB initial set -> WF-PPG wrist HR -> WF-PPG finger HR).",
    },
    "sdb_bidmc_finger_seq": {
        "dataset": "sdb", "train_dataset": "sdb", "probe_dataset": "sdb",
        "future_task_datasets": {"multiclass": "sdb", "hr": "bidmc", "finger": "wfppg"},
        "temporal_policy": "shared_window", "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"binary": 30.0, "multiclass": 30.0, "hr": 6.0, "finger": 10.0},
        "initial_tasks": ["binary"], "future_tasks": ["multiclass", "hr", "finger"],
        "description": "Pure-CL baseline for the aligned PPG setting: binary -> multiclass -> BIDMC HR -> WF-PPG finger, learned sequentially from Stage I.",
    },
    "sdb_to_bidmc_wrist": {
        "dataset": "sdb", "train_dataset": "sdb", "probe_dataset": "bidmc",
        "future_task_datasets": {"hr": "bidmc", "wrist": "wfppg"},
        "temporal_policy": "shared_window", "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"binary": 30.0, "multiclass": 30.0, "hr": 6.0, "wrist": 10.0},
        "initial_tasks": ["binary", "multiclass"], "future_tasks": ["hr", "wrist"],
        "description": "Paper-aligned PPG: SDB initial set -> BIDMC HR -> WF-PPG wrist HR.",
    },
    "sdb_to_bidmc_finger": {
        "dataset": "sdb", "train_dataset": "sdb", "probe_dataset": "bidmc",
        "future_task_datasets": {"hr": "bidmc", "finger": "wfppg"},
        "temporal_policy": "shared_window", "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"binary": 30.0, "multiclass": 30.0, "hr": 6.0, "finger": 10.0},
        "initial_tasks": ["binary", "multiclass"], "future_tasks": ["hr", "finger"],
        "description": "Paper-aligned PPG: SDB initial set -> BIDMC HR -> WF-PPG finger HR.",
    },
    "sdb_to_bidmc_hr": {
        "dataset": "sdb",
        "train_dataset": "sdb",
        "probe_dataset": "bidmc",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"binary": 30.0, "multiclass": 30.0, "hr": 6.0, "spo2": 6.0},
        "initial_tasks": ["binary", "multiclass"],
        "future_tasks": ["hr", "spo2"],
        "description": "Aligned with the paper's PPG Stage-II (SDB initial set): SDB binary+multiclass initial -> BIDMC HR (paper's newly added task) -> BIDMC SpO2 (WF-PPG tasks not implemented, BIDMC SpO2 used as the 2nd new task).",
    },
    "sdb_to_wesad_class": {
        "dataset": "sdb",
        "train_dataset": "sdb",
        "probe_dataset": "wesad_cross_interface",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"binary": 30.0, "multiclass": 30.0, "stress_binary": 60.0, "affect_multiclass": 60.0},
        "initial_tasks": ["binary", "multiclass"],
        "future_tasks": ["stress_binary", "affect_multiclass"],
        "description": "SDB classification initial set, then WESAD cross-interface classification new tasks (initial-set comparison, Table-4 style).",
    },
    "wesad_xif_class_first": {
        "dataset": "wesad_cross_interface",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"stress_binary": 60.0, "affect_multiclass": 60.0, "hr_regression": 10.0},
        "support_policy_map": {
            "stress_binary": "60s_wrist_BVP_only_baseline_vs_stress",
            "affect_multiclass": "60s_wrist_BVP_only_baseline_stress_amusement",
            "hr_regression": "10s_wrist_BVP_only_ecg_label",
        },
        "input_modality": "wrist_BVP_only",
        "label_source_map": {"stress_binary": "WESAD_protocol_label", "affect_multiclass": "WESAD_protocol_label", "hr_regression": "chest_ECG_label_only"},
        "initial_tasks": ["stress_binary"],
        "future_tasks": ["affect_multiclass", "hr_regression"],
        "description": "Classification-first split: stress-binary initial, then add affect-multiclass (classification), then HR regression.",
    },
    "wesad_affect_to_regression_probe": {
        "dataset": "wesad_cross_interface",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"stress_binary": 60.0, "affect_multiclass": 60.0, "hr_regression": 10.0, "rr_regression": 60.0},
        "support_policy_map": {
            "stress_binary": "60s_wrist_BVP_only_baseline_vs_stress",
            "affect_multiclass": "60s_wrist_BVP_only_baseline_stress_amusement",
            "hr_regression": "10s_wrist_BVP_only_ecg_label",
            "rr_regression": "60s_wrist_BVP_only_resp_label",
        },
        "input_modality": "wrist_BVP_only",
        "label_source_map": {"stress_binary": "WESAD_protocol_label", "affect_multiclass": "WESAD_protocol_label", "hr_regression": "chest_ECG_label_only", "rr_regression": "chest_respiration_label_only"},
        "initial_tasks": ["stress_binary", "affect_multiclass"],
        "future_tasks": ["hr_regression", "rr_regression"],
        "description": "WESAD classification-to-regression split: train stress/affect classification from wrist BVP input, then probe HR/RR regression.",
    },
    "wesad_subject_shift_regression_to_affect_probe": {
        "dataset": "wesad_cross_interface_shifted30",
        "builder": "wesad_subject_shift",
        "outer_pool_ratio": 0.6,
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"hr_regression": 10.0, "rr_regression": 30.0, "stress_binary": 30.0, "affect_multiclass": 30.0},
        "support_policy_map": {
            "hr_regression": "10s_wrist_BVP_only_ecg_label",
            "rr_regression": "30s_wrist_BVP_only_resp_label",
            "stress_binary": "30s_wrist_BVP_only_baseline_vs_stress",
            "affect_multiclass": "30s_wrist_BVP_only_baseline_stress_amusement",
        },
        "input_modality": "wrist_BVP_only",
        "label_source_map": {"hr_regression": "chest_ECG_label_only", "rr_regression": "chest_respiration_label_only", "stress_binary": "WESAD_protocol_label", "affect_multiclass": "WESAD_protocol_label"},
        "initial_tasks": ["hr_regression", "rr_regression"],
        "future_tasks": ["stress_binary", "affect_multiclass"],
        "description": "WESAD subject-pool shifted regression-to-classification split: Stage2 adapts on HR/RR from one subject pool and probes stress/affect on disjoint probe-pool subjects.",
    },
    "wesad_subject_shift_affect_to_regression_probe": {
        "dataset": "wesad_cross_interface_shifted30",
        "builder": "wesad_subject_shift",
        "outer_pool_ratio": 0.6,
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"stress_binary": 30.0, "affect_multiclass": 30.0, "hr_regression": 10.0, "rr_regression": 30.0},
        "support_policy_map": {
            "stress_binary": "30s_wrist_BVP_only_baseline_vs_stress",
            "affect_multiclass": "30s_wrist_BVP_only_baseline_stress_amusement",
            "hr_regression": "10s_wrist_BVP_only_ecg_label",
            "rr_regression": "30s_wrist_BVP_only_resp_label",
        },
        "input_modality": "wrist_BVP_only",
        "label_source_map": {"stress_binary": "WESAD_protocol_label", "affect_multiclass": "WESAD_protocol_label", "hr_regression": "chest_ECG_label_only", "rr_regression": "chest_respiration_label_only"},
        "initial_tasks": ["stress_binary", "affect_multiclass"],
        "future_tasks": ["hr_regression", "rr_regression"],
        "description": "WESAD subject-pool shifted classification-to-regression split: Stage2 adapts on stress/affect from one subject pool and probes HR/RR on disjoint probe-pool subjects.",
    },
    "ppgbp_to_wesad_affect_probe": {
        "dataset": "ppgbp_to_wesad_shifted30",
        "train_dataset": "ppgbp",
        "probe_dataset": "wesad_cross_interface_shifted30",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"stress_binary": 30.0, "affect_multiclass": 30.0},
        "support_policy_map": {
            "hr": "single_segment_native",
            "sysbp": "single_segment_native",
            "diasbp": "single_segment_native",
            "stress_binary": "30s_wrist_BVP_only_baseline_vs_stress",
            "affect_multiclass": "30s_wrist_BVP_only_baseline_stress_amusement",
        },
        "input_modality": "cross_dataset_ppg_only",
        "label_source_map": {
            "hr": "ppgbp_label",
            "sysbp": "ppgbp_label",
            "diasbp": "ppgbp_label",
            "stress_binary": "WESAD_protocol_label",
            "affect_multiclass": "WESAD_protocol_label",
        },
        "initial_tasks": ["hr", "sysbp", "diasbp"],
        "future_tasks": ["stress_binary", "affect_multiclass"],
        "description": "Cross-dataset enrollment: train on PPG-BP regression tasks and probe WESAD affect tasks on the shifted 30s cache.",
    },
    "ppgbp_to_wesad_hr_rr_probe": {
        "dataset": "ppgbp_to_wesad_shifted30",
        "train_dataset": "ppgbp",
        "probe_dataset": "wesad_cross_interface_shifted30",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {"hr_regression": 10.0, "rr_regression": 30.0},
        "support_policy_map": {
            "hr": "single_segment_native",
            "sysbp": "single_segment_native",
            "diasbp": "single_segment_native",
            "hr_regression": "10s_wrist_BVP_only_ecg_label",
            "rr_regression": "30s_wrist_BVP_only_resp_label",
        },
        "input_modality": "cross_dataset_ppg_only",
        "label_source_map": {
            "hr": "ppgbp_label",
            "sysbp": "ppgbp_label",
            "diasbp": "ppgbp_label",
            "hr_regression": "chest_ECG_label_only",
            "rr_regression": "chest_respiration_label_only",
        },
        "initial_tasks": ["hr", "sysbp", "diasbp"],
        "future_tasks": ["hr_regression", "rr_regression"],
        "description": "Cross-dataset enrollment: train on PPG-BP regression tasks and probe WESAD HR/RR tasks on the shifted 30s cache.",
    },
}


class LoRAConv1d(nn.Module):
    def __init__(self, base: nn.Conv1d, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if int(base.groups) != 1:
            raise ValueError("LoRAConv1d currently supports only groups=1 convs")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(max(1, rank))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.lora_down = nn.Conv1d(
            in_channels=base.in_channels,
            out_channels=int(rank),
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
        )
        self.lora_up = nn.Conv1d(
            in_channels=int(rank),
            out_channels=base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.base.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_up(self.lora_down(self.dropout(x)))


def encoder_has_lora(encoder: nn.Module) -> bool:
    return any(isinstance(module, LoRAConv1d) for module in encoder.modules())


def inject_lora_into_encoder(encoder: nn.Module, rank: int, alpha: float, dropout: float = 0.0) -> None:
    if encoder_has_lora(encoder):
        return
    for child_name, child in list(encoder.named_children()):
        if isinstance(child, nn.Conv1d):
            setattr(encoder, child_name, LoRAConv1d(child, rank=rank, alpha=alpha, dropout=dropout))
        else:
            inject_lora_into_encoder(child, rank=rank, alpha=alpha, dropout=dropout)


def set_lora_encoder_trainable(model: MultiTaskModel, enabled: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False
    for module in model.encoder.modules():
        if isinstance(module, LoRAConv1d):
            for p in module.lora_down.parameters():
                p.requires_grad = bool(enabled)
            for p in module.lora_up.parameters():
                p.requires_grad = bool(enabled)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+", default=["ppgbp_clinical", "bidmc_resp"], choices=sorted(PACKS))
    ap.add_argument("--students", nargs="+", default=["BF20_NB8"])
    ap.add_argument("--kd-method", default="ns_kd", choices=["ns_kd", "subspace_mse"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--stage2-methods", nargs="+", default=["frozen_heads", "joint_finetune", "gradnorm_joint", "pcgrad_joint"], choices=STAGE2_METHODS)
    ap.add_argument("--stage3-policy", default="naive_ft", choices=STAGE3_POLICIES)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--stage2-epochs", type=int, default=40)
    ap.add_argument("--stage2-patience", type=int, default=8)
    ap.add_argument("--stage2-head-lr", type=float, default=1e-3)
    ap.add_argument("--stage2-encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--stage3-epochs", type=int, default=30)
    ap.add_argument("--stage3-patience", type=int, default=6)
    ap.add_argument("--stage3-head-lr", type=float, default=1e-3)
    ap.add_argument("--stage3-encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--lpr-buffer-size", type=int, default=256)
    ap.add_argument("--lpr-alpha", type=float, default=1.0)
    ap.add_argument("--lpr-omega0", type=float, default=1.0)
    ap.add_argument("--lpr-refresh-interval", type=int, default=10)
    ap.add_argument("--lpr-lr", type=float, default=1e-3)
    ap.add_argument("--stage2-cache-dir", type=Path, default=None)
    ap.add_argument("--oracle-cache-dir", type=Path, default=None)
    ap.add_argument("--stage1-ckpt-seed", type=int, default=0)
    ap.add_argument("--care-stage2-scope", choices=["none", "full", "top1", "top2"], default="top2")
    ap.add_argument("--care-stage3-scope", choices=["none", "full", "top1", "top2"], default="top1")
    ap.add_argument("--care-stage2-anchor-lambda", type=float, default=0.1)
    ap.add_argument("--care-floor-lambda", type=float, default=0.25)
    ap.add_argument("--care-loss-ema", type=float, default=0.95)
    ap.add_argument("--care-stage3-anchor-lambda", type=float, default=0.1)
    ap.add_argument("--core-lambda", type=float, default=0.1)
    ap.add_argument("--core-energy", type=float, default=0.95)
    ap.add_argument("--core-mode", type=str, default="refresh")   # none|refresh|fixed|fuse_grow|fuse_budget
    ap.add_argument("--core-budget", type=int, default=0)
    ap.add_argument("--ewc-lambda", type=float, default=1000.0)
    ap.add_argument("--cpca-core-k", type=int, default=0)
    ap.add_argument("--lwp-lam-old", type=float, default=1.0)
    ap.add_argument("--lwp-lam-dwdp", type=float, default=0.01)
    ap.add_argument("--cpca-beta", type=float, default=1.0)
    ap.add_argument("--care-old-distill-lambda", type=float, default=0.25)
    ap.add_argument("--care-importance-lambda", type=float, default=1e-4)
    ap.add_argument("--care-importance-batches", type=int, default=8)
    ap.add_argument("--care-grad-mask-min", type=float, default=0.10)
    ap.add_argument("--care-grad-mask-power", type=float, default=1.5)
    ap.add_argument("--care-reg-floor-lambda", type=float, default=0.35)
    ap.add_argument("--care-cls-floor-lambda", type=float, default=0.15)
    ap.add_argument("--care-stage3-reg-distill-lambda", type=float, default=0.35)
    ap.add_argument("--care-stage3-cls-distill-lambda", type=float, default=0.15)
    ap.add_argument("--care-v4-extra-ratio", type=float, default=0.18)
    ap.add_argument("--care-v5-stage2-scope", choices=["full", "top1", "top2", "full_protected"], default="top2")
    ap.add_argument("--care-v5-stage3-scope", choices=["full", "top1", "top2", "masked_top2", "full_protected"], default="top1")
    ap.add_argument("--care-v5-anchor-lambda", type=float, default=0.1)
    ap.add_argument("--care-v5-lwf-lambda", type=float, default=0.25)
    ap.add_argument("--care-v5-importance-lambda", type=float, default=1e-4)
    ap.add_argument("--care-v5-support-anchor-lambda", type=float, default=0.0)
    ap.add_argument("--care-v5-loss-ema", type=float, default=0.95)
    ap.add_argument("--care-v5-floor-lambda", type=float, default=0.25)
    ap.add_argument("--care-v5-importance-batches", type=int, default=8)
    ap.add_argument("--lwf-old-distill-lambda", type=float, default=0.25)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=8.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--encoder-init", choices=["stage1", "random"], default="stage1")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def stage23_baseline_label(stage2_method: str, stage3_policy: str, encoder_init: str = "stage1") -> str:
    if stage2_method == "care_direct" and stage3_policy == "care_direct":
        label = "care_direct_v1"
    elif stage2_method == "care_direct_v2" and stage3_policy == "care_direct":
        label = "care_direct_v2a"
    elif stage2_method == "care_direct_v2b" and stage3_policy == "care_direct_v2b":
        label = "care_direct_v2b"
    elif stage2_method == "care_direct_v3" and stage3_policy == "care_direct_v3":
        label = "care_direct_v3"
    elif stage2_method == "care_direct_v4" and stage3_policy == "care_direct_v4":
        label = "care_direct_v4"
    elif stage2_method == "care_v5a" and stage3_policy == "care_v5a":
        label = "care_v5a"
    elif stage2_method == "care_v5b" and stage3_policy == "care_v5b":
        label = "care_v5b"
    elif stage2_method == "scope_matched_naive" and stage3_policy == "scope_matched_naive":
        label = "scope_matched_naive"
    elif stage2_method == "lora_tune" and stage3_policy == "lora_ft":
        label = "lora"
    elif stage2_method == "frozen_heads" and stage3_policy == "new_head_only":
        label = "head_only"
    elif stage2_method == "pcgrad_joint" and stage3_policy == "lwf_ft":
        label = "MTL + LwF"
    else:
        label = f"{stage2_method}+{stage3_policy}"
    if encoder_init == "random":
        return f"random_init + {label}"
    return label


def load_ppg_encoder_init(
    student: str,
    ckpt_dir: Path,
    kd_method: str,
    seed: int,
    device: torch.device,
    encoder_init: str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if encoder_init == "random":
        enc, enc_dim = build_encoder(student, seed)
        enc = enc.to(device)
        disable_dropout(enc)
        return enc, {
            "ckpt_path": "random_init",
            "enc_dim": int(enc_dim),
            "encoder_init": "random",
            "kd_method": "random_init",
        }

    enc, _, ck_meta = load_stage1_full(student, ckpt_dir, kd_method, seed, device)
    meta = dict(ck_meta)
    meta["encoder_init"] = "stage1"
    return enc, meta


def derive_binary_task(source_td: TaskData, task_name: str, positive_ids: Sequence[int]) -> TaskData:
    positive = {int(v) for v in positive_ids}
    split_y = {
        split: np.isin(np.asarray(y, dtype=np.int64), list(positive)).astype(np.int64)
        for split, y in source_td.split_y.items()
    }
    runtime = TaskRuntime(TaskSpec(task_name, "cls", "", "f1_macro"), n_out=2)
    return TaskData(
        runtime=runtime,
        split_X={split: np.asarray(X) for split, X in source_td.split_X.items()},
        split_y=split_y,
    )


def add_derived_tasks(task_map: Dict[str, TaskData], cfg: Dict[str, Any]) -> Dict[str, TaskData]:
    out = dict(task_map)
    for task_name, task_cfg in (cfg.get("derived_tasks") or {}).items():
        source_name = str(task_cfg["source"])
        if source_name not in out:
            raise KeyError(f"Missing source task '{source_name}' required for derived task '{task_name}'")
        out[task_name] = derive_binary_task(
            source_td=out[source_name],
            task_name=str(task_name),
            positive_ids=task_cfg.get("positive_ids", []),
        )
    return out


def _split_outer_values(values: Iterable[str], seed: int, first_ratio: float = 0.6) -> Tuple[set[str], set[str]]:
    unique = np.array(sorted(set(map(str, values))), dtype=object)
    if unique.size < 2:
        raise ValueError("Need at least two unique values for an outer shifted-pool split")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique)
    n_first = max(1, int(round(float(first_ratio) * float(unique.size))))
    n_first = min(int(unique.size) - 1, n_first)
    return set(unique[:n_first].tolist()), set(unique[n_first:].tolist())


def _task_spec_for_dataset(dataset: str, task_name: str) -> TaskSpec:
    cfg = DATASET_SPECS[dataset]
    for task in cfg["tasks"]:
        if task.name == task_name:
            return task
    raise KeyError(f"Task '{task_name}' not registered under dataset '{dataset}'")


def _load_taskwise_arrays(dataset: str, task_name: str, hz: int) -> Tuple[TaskSpec, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    cfg = DATASET_SPECS[dataset]
    if str(cfg["mode"]) != "subject_split_taskwise":
        raise ValueError(f"Custom taskwise array loader expects subject_split_taskwise dataset, got {dataset}")
    task_spec = _task_spec_for_dataset(dataset, task_name)
    root = Path(cfg["root"])
    x_path = root / cfg["x_template"].format(subdir=task_name, hz=hz)
    y_path = root / task_spec.y_template.format(subdir=task_name, hz=hz)
    s_path = root / cfg["subject_template"].format(subdir=task_name, hz=hz)
    meta_path = root / task_name / "window_metadata.csv"
    for path in (x_path, y_path, s_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(path)
    X = np.asarray(np.load(str(x_path)), dtype=np.float32)
    y = np.asarray(np.load(str(y_path)))
    if task_spec.kind == "cls":
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float32)
    subjects = np.asarray(np.load(str(s_path), allow_pickle=True), dtype=str)
    metadata = pd.read_csv(meta_path)
    if len(metadata) != len(X) or len(X) != len(y) or len(y) != len(subjects):
        raise ValueError(
            f"Task '{task_name}' under dataset '{dataset}' has misaligned arrays: "
            f"X={len(X)} y={len(y)} subjects={len(subjects)} meta={len(metadata)}"
        )
    return task_spec, X, y, subjects, metadata


def _build_taskdata_from_masks(
    task_spec: TaskSpec,
    X: np.ndarray,
    y: np.ndarray,
    masks: Dict[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> TaskData:
    split_X: Dict[str, np.ndarray] = {}
    split_y: Dict[str, np.ndarray] = {}
    split_offset = {"train": 0, "val": 1, "test": 2}
    split_limit = {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test}
    for split in ("train", "val", "test"):
        Xi = np.asarray(X[masks[split]], dtype=np.float32)
        yi = np.asarray(y[masks[split]])
        Xi, one_y = maybe_limit_split(Xi, {task_spec.name: yi}, int(split_limit[split]), int(seed) + split_offset[split])
        split_X[split] = Xi
        split_y[split] = np.asarray(one_y[task_spec.name])

    if task_spec.kind == "cls":
        runtime = TaskRuntime(task_spec, n_out=int(np.max(y)) + 1)
    else:
        y_train = split_y["train"].astype(np.float32)
        mean = float(y_train.mean())
        std = float(y_train.std())
        if not math.isfinite(std) or std < 1e-6:
            std = 1.0
        runtime = TaskRuntime(task_spec, n_out=1, y_mean=mean, y_std=std)
    return TaskData(runtime=runtime, split_X=split_X, split_y=split_y)


def _build_wesad_subject_shift_pack(cfg: Dict[str, Any], args: argparse.Namespace, seed: int) -> Tuple[List[TaskData], List[TaskData]]:
    dataset = str(cfg["dataset"])
    anchor_task = str(cfg["initial_tasks"][0])
    _, _, _, anchor_subjects, _ = _load_taskwise_arrays(dataset, anchor_task, args.hz)
    adapt_subjects, probe_subjects = _split_outer_values(
        anchor_subjects,
        seed=int(seed),
        first_ratio=float(cfg.get("outer_pool_ratio", 0.6)),
    )
    adapt_train, adapt_val, adapt_test = split_subjects(sorted(adapt_subjects), int(seed) + 101)
    probe_train, probe_val, probe_test = split_subjects(sorted(probe_subjects), int(seed) + 202)

    task_data_map: Dict[str, TaskData] = {}
    for task_name in list(cfg["initial_tasks"]) + list(cfg["future_tasks"]):
        task_spec, X, y, subjects, _ = _load_taskwise_arrays(dataset, str(task_name), args.hz)
        if task_name in cfg["initial_tasks"]:
            masks = {
                "train": np.isin(subjects, list(adapt_train)),
                "val": np.isin(subjects, list(adapt_val)),
                "test": np.isin(subjects, list(adapt_test)),
            }
        else:
            masks = {
                "train": np.isin(subjects, list(probe_train)),
                "val": np.isin(subjects, list(probe_val)),
                "test": np.isin(subjects, list(probe_test)),
            }
        task_data_map[str(task_name)] = _build_taskdata_from_masks(task_spec, X, y, masks, args, int(seed))

    initial = [task_data_map[name] for name in cfg["initial_tasks"]]
    future = [task_data_map[name] for name in cfg["future_tasks"]]
    sec_map = cfg.get("input_window_sec_map") or {}
    if sec_map:
        initial = apply_task_specific_input_windows(initial, args.hz, sec_map)
        future = apply_task_specific_input_windows(future, args.hz, sec_map)
    return initial, future


def build_pack_task_data(pack_name: str, args: argparse.Namespace, seed: int) -> Tuple[List[TaskData], List[TaskData]]:
    cfg = PACKS[pack_name]
    builder = str(cfg.get("builder", ""))
    if builder == "wesad_subject_shift":
        return _build_wesad_subject_shift_pack(cfg, args, seed)

    train_dataset = str(cfg.get("train_dataset", cfg["dataset"]))
    probe_dataset = str(cfg.get("probe_dataset", train_dataset))
    train_task_data = load_multitask_dataset(
        train_dataset,
        hz=args.hz,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
        seed=seed,
    )
    train_task_map = add_derived_tasks({td.runtime.spec.name: td for td in train_task_data}, cfg)
    if probe_dataset == train_dataset:
        probe_task_map = train_task_map
    else:
        probe_task_data = load_multitask_dataset(
            probe_dataset,
            hz=args.hz,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
            limit_test=args.limit_test,
            seed=seed,
        )
        probe_task_map = add_derived_tasks({td.runtime.spec.name: td for td in probe_task_data}, cfg)
    initial = [train_task_map[name] for name in cfg["initial_tasks"]]
    ftd = cfg.get("future_task_datasets")
    if ftd:
        _cache = {}
        future = []
        for name in cfg["future_tasks"]:
            ds = str(ftd.get(name, probe_dataset))
            if ds not in _cache:
                _cache[ds] = add_derived_tasks({td.runtime.spec.name: td for td in load_multitask_dataset(
                    ds, hz=args.hz, limit_train=args.limit_train, limit_val=args.limit_val, limit_test=args.limit_test, seed=seed)}, cfg)
            future.append(_cache[ds][name])
    else:
        future = [probe_task_map[name] for name in cfg["future_tasks"]]
    sec_map = cfg.get("input_window_sec_map") or {}
    if sec_map:
        initial = apply_task_specific_input_windows(initial, args.hz, sec_map)
        future = apply_task_specific_input_windows(future, args.hz, sec_map)
    return initial, future


def _center_crop_np(X: np.ndarray, target_len: int) -> np.ndarray:
    X = np.asarray(X)
    if target_len <= 0 or X.shape[-1] <= target_len:
        return np.asarray(X)
    start = max(0, int((X.shape[-1] - target_len) // 2))
    end = start + int(target_len)
    return np.ascontiguousarray(X[..., start:end])


def apply_task_specific_input_windows(
    task_data_list: Sequence[TaskData],
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


def _task_output_structure(rt: TaskRuntime) -> str:
    return "class_distribution" if rt.spec.kind == "cls" else "scalar"


def _task_default_loss_name(rt: TaskRuntime) -> str:
    return "cross_entropy" if rt.spec.kind == "cls" else "mse"


def _task_default_preserve_dist(rt: TaskRuntime, *, robust_scalar: bool = False) -> str:
    if _task_output_structure(rt) == "class_distribution":
        return "kl_distribution"
    return "huber_scalar" if robust_scalar else "mse_scalar"


def build_task_interfaces(task_data: Sequence[TaskData], hz: int, *, robust_scalar: bool = False) -> Dict[str, TaskInterface]:
    interfaces: Dict[str, TaskInterface] = {}
    for td in task_data:
        task_name = td.runtime.spec.name
        support_sec = float(td.split_X["train"].shape[-1]) / float(max(1, hz))
        interfaces[task_name] = TaskInterface(
            task_name=task_name,
            loss_name=_task_default_loss_name(td.runtime),
            metric_name=str(td.runtime.spec.metric),
            input_support_sec=float(support_sec),
            preserve_dist=_task_default_preserve_dist(td.runtime, robust_scalar=robust_scalar),
            output_structure=_task_output_structure(td.runtime),
        )
    return interfaces


def _care_v5_scope_alias(scope: str) -> str:
    if scope == "masked_top2":
        return "top2"
    if scope == "full_protected":
        return "full"
    return scope


def _task_preserve_distance(
    iface: TaskInterface,
    current_out: torch.Tensor,
    reference_out: torch.Tensor,
) -> torch.Tensor:
    if iface.preserve_dist == "kl_distribution":
        return F.kl_div(
            F.log_softmax(current_out, dim=-1),
            F.softmax(reference_out, dim=-1),
            reduction="batchmean",
        )
    if iface.preserve_dist == "huber_scalar":
        return F.huber_loss(current_out, reference_out, delta=1.0)
    if iface.preserve_dist == "mse_scalar":
        return F.mse_loss(current_out, reference_out)
    raise KeyError(f"Unsupported preserve distance '{iface.preserve_dist}'")


def _encoder_drift_metrics(
    current_snapshot: Dict[str, torch.Tensor],
    reference_snapshot: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    shared_names = sorted(set(current_snapshot) & set(reference_snapshot))
    if not shared_names:
        return {"l2": 0.0, "rmse": 0.0, "rel": 0.0}
    diff_sq = 0.0
    ref_sq = 0.0
    n = 0
    for name in shared_names:
        curr = current_snapshot[name].float().reshape(-1)
        ref = reference_snapshot[name].float().reshape(-1)
        diff = curr - ref
        diff_sq += float(diff.pow(2).sum().item())
        ref_sq += float(ref.pow(2).sum().item())
        n += int(diff.numel())
    l2 = math.sqrt(max(diff_sq, 0.0))
    rmse = math.sqrt(max(diff_sq / float(max(1, n)), 0.0))
    rel = l2 / math.sqrt(max(ref_sq, 1e-12))
    return {"l2": float(l2), "rmse": float(rmse), "rel": float(rel)}


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


def _lpr_preconditionable_layers(encoder: nn.Module) -> List[Tuple[str, nn.Module, str]]:
    layers: List[Tuple[str, nn.Module, str]] = []
    for name, mod in encoder.named_modules():
        if isinstance(mod, nn.Conv1d):
            layers.append((name, mod, "conv"))
        elif isinstance(mod, nn.BatchNorm1d):
            layers.append((name, mod, "bn"))
    return layers


def _lpr_unfold_conv1d(z: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    # z: (N, C_in, L) input to a padding=0 Conv1d; returns (N*L_out, C_in*K)
    n, c, _l = z.shape
    k = int(conv.kernel_size[0])
    s = int(conv.stride[0])
    unfolded = F.unfold(z.unsqueeze(2), kernel_size=(1, k), stride=(1, s), padding=(0, 0))
    n_u, ck, l_out = unfolded.shape
    return unfolded.transpose(1, 2).reshape(n_u * l_out, ck)


def _lpr_build_replay(old_tasks: List[Any], buffer_size: int, rng: np.random.Generator) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
    per_task = max(1, int(buffer_size) // max(1, len(old_tasks)))
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    runtimes: List[Any] = []
    for rt in old_tasks:
        X = np.asarray(rt.split_X["train"])
        y = np.asarray(rt.split_y["train"])
        if rt.runtime.spec.kind == "cls":
            y = y.astype(np.int64)
        else:
            y = y.astype(np.float32)
            y = (y - float(rt.runtime.y_mean)) / float(rt.runtime.y_std)
        n = X.shape[0]
        k = min(per_task, n)
        idx = rng.choice(n, size=k, replace=False)
        X_list.append(X[idx])
        y_list.append(y[idx])
        runtimes.append(rt.runtime)
    return X_list, y_list, runtimes


def _lpr_replay_loss(model: "MultiTaskModel", X_list: List[np.ndarray], y_list: List[np.ndarray], runtimes: List[Any], device: torch.device, rng: Any = None, batch_per_task: int = 32) -> torch.Tensor:
    # Replay buffer stays on CPU; sample a small per-step mini-batch per task, move to GPU, free immediately.
    terms: List[torch.Tensor] = []
    for X, y, rt in zip(X_list, y_list, runtimes):
        n = int(X.shape[0])
        if rng is not None and batch_per_task > 0 and n > batch_per_task:
            sel = rng.choice(n, size=int(batch_per_task), replace=False)
            xb = X[sel]
            yb = y[sel]
        else:
            xb = X
            yb = y
        xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
        yt = torch.as_tensor(yb, device=device)
        try:
            _, out = model(xt, task_name=rt.spec.name)
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"[lpr-debug] task={rt.spec.name} xt={tuple(xt.shape)} yt={tuple(yt.shape)} err={exc}", flush=True)
            raise
        terms.append(task_losses(out, {rt.spec.name: yt}, [rt])[rt.spec.name])
        del xt, yt, out
    return torch.stack(terms).mean() if terms else torch.zeros((), device=device)


def _lpr_build_preconditioners(
    encoder: nn.Module,
    layers: List[Tuple[str, nn.Module, str]],
    X_list: List[np.ndarray],
    omega: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    activations: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []

    def make_hook(name: str):
        def hook(mod: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            activations[name] = inp[0].detach()

        return hook

    for name, mod, kind in layers:
        handles.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        for X in X_list:
            encoder(torch.as_tensor(X, dtype=torch.float32, device=device))
    for h in handles:
        h.remove()

    precond: Dict[str, Any] = {}
    for name, mod, kind in layers:
        if name not in activations:
            continue
        z = activations[name]
        if kind == "conv":
            groups = int(getattr(mod, "groups", 1))
            if groups > 1:
                cg = int(z.shape[1]) // groups
                blocks = [_lpr_unfold_conv1d(z[:, i * cg : (i + 1) * cg, :], mod) for i in range(groups)]
            else:
                blocks = [_lpr_unfold_conv1d(z, mod)]
        else:
            blocks = [z.transpose(1, 2).reshape(-1, int(z.shape[1]))]
        lams: List[torch.Tensor] = []
        for Zf in blocks:
            d_in = int(Zf.shape[1])
            G = Zf.t().mm(Zf)
            lams.append(torch.inverse(torch.eye(d_in, device=device, dtype=Zf.dtype) + omega * G))
        precond[name] = lams[0] if len(lams) == 1 else lams
    return precond


def _dwdp_loss(z_old: torch.Tensor, z_new: torch.Tensor, labels: torch.Tensor, kind: str, enable_dynamic: bool = True) -> torch.Tensor:
    # Official LwP DWDP: mean((cdist(z)-cdist(z'))^2 with dynamic mask).
    d_old = torch.cdist(z_old, z_old, p=2)
    d_new = torch.cdist(z_new, z_new, p=2)
    diff = d_old - d_new
    if enable_dynamic:
        lab = labels.reshape(-1, 1) if labels.dim() < 2 else labels
        if kind == "cls":
            if lab.shape[-1] > 1:
                mask = (lab.unsqueeze(0) == lab.unsqueeze(1)).all(dim=-1).float()
            else:
                mask = (lab.unsqueeze(0) == lab.unsqueeze(1)).float()
        else:
            lf = lab.float()
            mask = torch.exp(-((lf - lf.T) ** 2))
        diff = diff * mask
    return torch.mean(diff ** 2)


def _core_dwdp_loss(z_old: torch.Tensor, z_new: torch.Tensor, labels: torch.Tensor, kind: str,
                    U_core: torch.Tensor, affinity: bool = False, hybrid: bool = False) -> torch.Tensor:
    """DWDP restricted to the transferable core subspace c = U_core^T z.

    - affinity=False: original LwP dynamic (current-label) mask, only the space changes.
    - affinity=True : weights from the FROZEN previous-stage core geometry (stop-grad), tau = median sq. dist.
    - hybrid=True   : affinity weights multiplied by the original LwP mask.
    Batch-mean distance normalization keeps the effective scale independent of k (documented).
    """
    c_old = (z_old @ U_core).detach()
    c_new = z_new @ U_core
    d_old = torch.cdist(c_old, c_old, p=2)
    d_new = torch.cdist(c_new, c_new, p=2)
    d_old = d_old / (d_old.mean() + 1e-8)
    d_new = d_new / (d_new.mean() + 1e-8)
    diff = d_old - d_new
    lab = labels.reshape(-1, 1) if labels.dim() < 2 else labels
    lf = lab.float()
    if kind == "cls":
        if lab.shape[-1] > 1:
            mask = (lab.unsqueeze(0) == lab.unsqueeze(1)).all(dim=-1).float()
        else:
            mask = (lab.unsqueeze(0) == lab.unsqueeze(1)).float()
    else:
        mask = torch.exp(-((lf - lf.T) ** 2))
    if affinity:
        with torch.no_grad():
            sq = (c_old.unsqueeze(0) - c_old.unsqueeze(1)).pow(2).sum(-1)
            tau = torch.median(sq.reshape(-1)) + 1e-8
            w = torch.exp(-sq / tau)
        if hybrid:
            w = w * mask
        return (w * diff.pow(2)).sum() / (w.sum() + 1e-8)
    return torch.mean((diff * mask) ** 2)


def _lpr_apply_preconditioners(encoder: nn.Module, layers: List[Tuple[str, nn.Module, str]], precond: Dict[str, Any]) -> None:
    for name, mod, kind in layers:
        if name not in precond:
            continue
        p = mod.weight
        if p.grad is None:
            continue
        lam = precond[name]
        if kind == "conv":
            if isinstance(lam, list):
                c_out = p.grad.size(0)
                g = p.grad.view(c_out, -1)
                co_g = c_out // len(lam)
                for i, li in enumerate(lam):
                    g[i * co_g : (i + 1) * co_g, :] = g[i * co_g : (i + 1) * co_g, :].mm(li)
                p.grad = g.view_as(p.grad)
            else:
                g = p.grad.view(p.grad.size(0), -1)
                p.grad = g.mm(lam).view_as(p.grad)
        else:
            p.grad = lam.mv(p.grad.reshape(-1)).reshape_as(p.grad)


def make_task_loaders_for_device(
    task_data: List[TaskData],
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, DataLoader]]:
    if device.type != "cpu":
        return make_task_loaders(task_data, batch_size)

    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    test_loaders: Dict[str, DataLoader] = {}
    for td in task_data:
        name = td.runtime.spec.name
        train_ds = SingleTaskArrayDataset(td.split_X["train"], td.split_y["train"], td.runtime)
        val_ds = SingleTaskArrayDataset(td.split_X["val"], td.split_y["val"], td.runtime)
        test_ds = SingleTaskArrayDataset(td.split_X["test"], td.split_y["test"], td.runtime)
        train_loaders[name] = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loaders[name] = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        test_loaders[name] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loaders, val_loaders, test_loaders


def _stage2_encoder_scope(method: str, care_stage2_scope: str, care_v5_stage2_scope: str) -> str:
    if method == "frozen_heads":
        return "none"
    if method in SCOPE_MATCHED_STAGE2_METHODS:
        return care_stage2_scope
    if method in CARE_V5_STAGE2_METHODS:
        return care_v5_stage2_scope
    if method == "care_direct_v4":
        return f"dynamic_{care_stage2_scope}"
    if method in CARE_STAGE2_METHODS:
        return care_stage2_scope
    if method == "lora_tune":
        return "lora"
    return "full"


def _stage3_encoder_scope(stage3_policy: str, care_stage3_scope: str, care_v5_stage3_scope: str) -> str:
    if stage3_policy in {"new_head_only", "head_mlp"}:
        return "none"
    if stage3_policy in SCOPE_MATCHED_STAGE3_POLICIES:
        return care_stage3_scope
    if stage3_policy in CARE_V5_STAGE3_POLICIES:
        return care_v5_stage3_scope
    if stage3_policy == "care_direct_v4":
        return f"dynamic_{care_stage3_scope}"
    if stage3_policy in CARE_STAGE3_POLICIES:
        return care_stage3_scope
    if stage3_policy == "lora_ft":
        return "lora"
    if stage3_policy in {"lwp", "lwp_core", "core_dwdp", "core_affinity", "core_affinity_hybrid", "lwp_task"}:
        return care_stage3_scope
    return "full"


def _is_norm_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm))


def _keep_frozen_encoder_norms_eval(model: MultiTaskModel) -> None:
    for module in model.encoder.modules():
        if _is_norm_module(module) and not any(p.requires_grad for p in module.parameters(recurse=False)):
            module.eval()


def _bn_affine_param_names(encoder: nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in encoder.named_modules():
        if not _is_norm_module(module):
            continue
        for param_name, _ in module.named_parameters(recurse=False):
            full = f"encoder.{module_name}.{param_name}" if module_name else f"encoder.{param_name}"
            names.add(full)
    return names


def _encoder_bias_param_names(encoder: nn.Module) -> set[str]:
    return {
        f"encoder.{name}"
        for name, _ in encoder.named_parameters()
        if name.endswith("bias")
    }


def _is_norm_param_name(name: str) -> bool:
    return (
        ".bn" in name
        or "first_block_bn" in name
        or "instnorm" in name
        or ".norm." in name
        or name.endswith("norm.weight")
        or name.endswith("norm.bias")
    )


def _tail_block_param_names(model: MultiTaskModel, scope: str) -> set[str]:
    scope = _care_v5_scope_alias(scope)
    if scope == "none":
        return set()
    if scope == "full":
        return {f"encoder.{name}" for name, _ in model.encoder.named_parameters()}
    if scope not in {"top1", "top2"}:
        raise ValueError(f"Unsupported tail scope for CARE-v2b: {scope}")
    block_list = getattr(model.encoder, "basicblock_list", None)
    if block_list is not None:
        n_blocks = min(1 if scope == "top1" else 2, len(block_list))
        names: set[str] = set()
        start_idx = len(block_list) - n_blocks
        for idx in range(start_idx, len(block_list)):
            block = block_list[idx]
            for param_name, _ in block.named_parameters():
                names.add(f"encoder.basicblock_list.{idx}.{param_name}")
        return names

    tail_modules: List[Tuple[str, List[str]]] = []
    for module_name, module in model.encoder.named_modules():
        direct_params = [param_name for param_name, _ in module.named_parameters(recurse=False)]
        if direct_params:
            tail_modules.append((module_name, direct_params))
    if tail_modules:
        n_modules = min(1 if scope == "top1" else 2, len(tail_modules))
        names = set()
        for module_name, param_names in tail_modules[-n_modules:]:
            prefix = f"encoder.{module_name}" if module_name else "encoder"
            for param_name in param_names:
                names.add(f"{prefix}.{param_name}" if module_name else f"encoder.{param_name}")
        if names:
            return names

    ordered_params = [f"encoder.{name}" for name, _ in model.encoder.named_parameters()]
    n_params = min(len(ordered_params), 2 if scope == "top1" else 4)
    return set(ordered_params[-n_params:])


def _care_v2b_candidate_param_names(model: MultiTaskModel, scope: str) -> set[str]:
    return _bn_affine_param_names(model.encoder) | _tail_block_param_names(model, scope)


def _care_v3_candidate_param_names(model: MultiTaskModel, scope: str) -> set[str]:
    return (
        _bn_affine_param_names(model.encoder)
        | _encoder_bias_param_names(model.encoder)
        | _tail_block_param_names(model, scope)
    )


def _care_v5_candidate_param_names(model: MultiTaskModel, scope: str) -> set[str]:
    resolved = _care_v5_scope_alias(scope)
    if resolved == "none":
        return set()
    if resolved == "full":
        return {f"encoder.{name}" for name, _ in model.encoder.named_parameters()}
    return _care_v3_candidate_param_names(model, resolved)


def _set_encoder_trainable_names(model: MultiTaskModel, trainable_names: set[str]) -> None:
    for name, p in model.named_parameters():
        if not name.startswith("encoder."):
            continue
        p.requires_grad = name in trainable_names


def _encoder_param_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if name.startswith("encoder.")
    }


def _care_auto_trainable_scope(scope: str) -> str:
    aliases = {
        "full": "full",
        "full_encoder": "full",
        "upper_blocks_only": "top2",
        "late_blocks_plus_heads": "top1",
        "heads_only": "none",
        "top1": "top1",
        "top2": "top2",
        "none": "none",
    }
    key = str(scope).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported care_auto scope '{scope}'")
    return aliases[key]


def _resolve_care_auto_reuse_k(feature_dim: int, requested_k: int) -> Tuple[int, List[int]]:
    feature_dim = int(max(1, feature_dim))
    candidate_ks: List[int] = []
    for cand in (16, 32, 48, 64, int(round(feature_dim * 0.25)), int(round(feature_dim * 0.5))):
        cand_i = int(cand)
        if 1 <= cand_i <= feature_dim and cand_i not in candidate_ks:
            candidate_ks.append(cand_i)
    candidate_ks = sorted(candidate_ks)
    if int(requested_k) > 0:
        reuse_k = min(int(requested_k), feature_dim)
    else:
        reuse_k = min(32, feature_dim)
    if reuse_k not in candidate_ks:
        candidate_ks.append(reuse_k)
        candidate_ks = sorted(candidate_ks)
    return int(reuse_k), candidate_ks


def _resolve_care_auto_reuse_loss_type(method: str, requested_loss_type: str) -> str:
    if method == "care_v2_normmse":
        return "norm_mse"
    if method == "care_v2_cosine":
        return "cosine"
    if method == "care_v2_statnorm":
        return "statnorm_mse"
    loss_type = str(requested_loss_type).strip().lower()
    if loss_type not in {"mse", "full_mse", "cosine", "huber", "norm_mse", "statnorm_mse"}:
        raise ValueError(f"Unsupported care_auto reuse loss type '{requested_loss_type}'")
    return loss_type


def _default_anchor_sample_budget(
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    max_batches: int,
) -> int:
    batch_size = 0
    for rt in tasks:
        loader = train_loaders.get(rt.spec.name)
        if loader is None:
            continue
        batch_size = int(getattr(loader, "batch_size", 0) or 0)
        if batch_size > 0:
            break
    if batch_size <= 0:
        batch_size = 8
    return max(1, int(max_batches)) * batch_size * max(1, len(tasks))


def _load_public_pool_inputs(public_pool_path: str) -> torch.Tensor:
    if not str(public_pool_path).strip():
        raise ValueError("stage1_public_pool anchor source requires a public pool path")
    pool_path = Path(public_pool_path)
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing public pool cache: {pool_path}")
    payload = torch.load(str(pool_path), map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("X_mix", "x", "X", "inputs", "data"):
            value = payload.get(key)
            if torch.is_tensor(value):
                payload = value
                break
    if not torch.is_tensor(payload):
        raise TypeError(f"Unsupported public pool cache type: {type(payload)!r}")
    x_pool = payload.detach().float().cpu().contiguous()
    if x_pool.ndim < 2:
        raise ValueError(f"Unexpected public pool tensor shape: {tuple(x_pool.shape)}")
    return x_pool


def _collect_source_union_anchor_features(
    model: MultiTaskModel,
    anchor_encoder: nn.Module,
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    sample_budget: int,
) -> Tuple[List[torch.Tensor], int]:
    features_cpu: List[torch.Tensor] = []
    total_batches = 0
    per_task_budget = max(1, int(math.ceil(float(sample_budget) / max(1, len(tasks)))))
    for rt in tasks:
        loader = train_loaders[rt.spec.name]
        seen = 0
        for xb, _ in loader:
            remaining = per_task_budget - seen
            if remaining <= 0:
                break
            xb = xb[:remaining].to(device, non_blocking=True)
            z_anchor = model.encode_with_encoder(anchor_encoder, xb, task_name=rt.spec.name)
            features_cpu.append(z_anchor.detach().float().cpu())
            total_batches += 1
            seen += int(xb.shape[0])
            if seen >= per_task_budget:
                break
    return features_cpu, total_batches


def _collect_public_pool_anchor_features(
    model: MultiTaskModel,
    anchor_encoder: nn.Module,
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    public_pool_path: str,
    sample_budget: int,
) -> Tuple[List[torch.Tensor], int]:
    x_pool = _load_public_pool_inputs(public_pool_path)
    if sample_budget > 0:
        x_pool = x_pool[:sample_budget]
    if x_pool.shape[0] == 0:
        raise ValueError(f"Public pool cache is empty: {public_pool_path}")
    batch_size = 0
    for rt in tasks:
        loader = train_loaders.get(rt.spec.name)
        if loader is None:
            continue
        batch_size = int(getattr(loader, "batch_size", 0) or 0)
        if batch_size > 0:
            break
    if batch_size <= 0:
        batch_size = 8
    task_name = tasks[0].spec.name
    features_cpu: List[torch.Tensor] = []
    total_batches = 0
    for start in range(0, int(x_pool.shape[0]), int(batch_size)):
        xb = x_pool[start : start + int(batch_size)].to(device, non_blocking=True)
        z_anchor = model.encode_with_encoder(anchor_encoder, xb, task_name=task_name)
        features_cpu.append(z_anchor.detach().float().cpu())
        total_batches += 1
    return features_cpu, total_batches


def _trainable_param_summary(model: MultiTaskModel) -> Dict[str, Any]:
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    encoder_names = [name for name in trainable_names if name.startswith("encoder.")]
    layer_names = sorted({name.rsplit(".", 1)[0] for name in encoder_names})
    return {
        "trainable_param_count": int(sum(int(p.numel()) for _, p in model.named_parameters() if p.requires_grad)),
        "encoder_trainable_param_count": int(sum(int(p.numel()) for name, p in model.named_parameters() if p.requires_grad and name.startswith("encoder."))),
        "updated_layer_names": layer_names,
        "updated_param_names": encoder_names,
    }


@torch.no_grad()
def _cpca_ts_basis(model, anchor_encoder, tasks, train_loaders, device, k, samples: int = 1024):
    # Task-sensitive core selection: score_i = norm(sigma_i) * norm(g_i),
    # g_i = E_x[(dL_old/dz . u_i)^2] using the frozen Stage-II model.
    feats = []
    n = 0
    for t in tasks:
        for xb, _yb in train_loaders.get(t.spec.name, []):
            with torch.no_grad():
                z = model.encode_with_encoder(anchor_encoder, xb.to(device), task_name=t.spec.name)
            feats.append(z.detach().float().cpu())
            n += int(z.shape[0])
            if n >= samples:
                break
        if n >= samples:
            break
    Z = torch.cat(feats, 0)[:samples]
    Zc = Z - Z.mean(0, keepdim=True)
    _, s, vt = torch.linalg.svd(Zc, full_matrices=False)
    vt = vt.to(device).float()
    s = s.to(device).float()
    g = torch.zeros(vt.shape[0], device=device)
    torch.set_grad_enabled(True)
    for t in tasks:
        for xb, yb in train_loaders.get(t.spec.name, []):
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                z0 = model.encode_with_encoder(anchor_encoder, xb, task_name=t.spec.name)
            with torch.enable_grad():
                z0 = z0.detach().clone().requires_grad_(True)
                head = model.heads[t.spec.name]
                ot = head(z0)
                if t.spec.kind == "cls":
                    ls = F.cross_entropy(ot, yb.long())
                else:
                    ls = F.smooth_l1_loss(ot.squeeze(-1), yb.float())
                dg = torch.autograd.grad(ls, z0)[0]
            g = g + (dg @ vt.t()).pow(2).mean(0)
            break
    g = g / max(1, len(tasks))
    score = (s / (s.max() + 1e-8)) * (g / (g.max() + 1e-8))
    kk = min(int(k), int(vt.shape[0]))
    top = torch.topk(score, kk).indices
    basis = vt[top].T.contiguous()
    return {"basis": basis, "reuse_k": int(kk), "sigma": s, "sensitivity": g, "score": score}


def _old_task_head_basis(model: "MultiTaskModel", old_task_data: List[Any], stage1_encoder=None, include_fm: bool = False,
                         fm_pool: Optional[List[Any]] = None, device=None):
    """Model-only protection basis: row space of the previously-seen task heads (U_task),
    optionally unioned with the Stage-I encoder's dominant feature directions (U_FM).
    Uses only seen-task heads + the frozen model; no unseen-task information."""
    Ws = []
    for td in old_task_data:
        name = td.runtime.spec.name
        head = model.heads[name] if (hasattr(model, "heads") and name in model.heads) else None
        if head is None:
            continue
        lins = [m for m in head.modules() if isinstance(m, nn.Linear)]
        if not lins:
            continue
        Ws.append(lins[-1].weight.detach().float().cpu())
    if not Ws:
        return None
    W = torch.cat(Ws, 0)
    _, _, Vh = torch.linalg.svd(W, full_matrices=False)
    basis = Vh.T.contiguous()  # (d, k)
    if include_fm and stage1_encoder is not None and fm_pool:
        model.eval()
        feats = []
        with torch.no_grad():
            for td in fm_pool:
                name = td.runtime.spec.name
                X = torch.as_tensor(np.asarray(td.split_X["train"]), dtype=torch.float32)
                n = min(len(X), 512)
                if n == 0:
                    continue
                xb = X[:n].to(device)
                z = model.encode_with_encoder(stage1_encoder, xb, task_name=name)
                feats.append(z.detach().float().cpu())
        if feats:
            F = torch.cat(feats); F = F - F.mean(0, keepdim=True)
            _, S, Vh2 = torch.linalg.svd(F, full_matrices=False)
            e = (S ** 2).cumsum(0) / (S ** 2).sum()
            k2 = int((e < 0.95).sum()) + 1
            basis = torch.linalg.qr(torch.cat([basis, Vh2[:k2].T.contiguous()], 1))[0]
    if device is not None:
        basis = basis.to(device=device, dtype=torch.float32)
    return {"basis": basis, "reuse_k": int(basis.shape[1])}


def _build_care_auto_anchor_basis(
    model: MultiTaskModel,
    anchor_encoder: nn.Module,
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    feature_space: str,
    anchor_source: str,
    requested_k: int,
    max_batches: int,
    requested_samples: int = 0,
    public_pool_path: str = "",
    energy_threshold: float = 0.95,
    max_rank_ratio: float = 0.5,
    min_rank: int = 4,
) -> Dict[str, Any]:
    if str(feature_space) != "encoder_pooled":
        raise ValueError(f"Unsupported care_auto feature space '{feature_space}'")
    source_key = str(anchor_source).strip().lower()
    sample_budget = int(requested_samples) if int(requested_samples) > 0 else _default_anchor_sample_budget(tasks, train_loaders, max_batches)

    if source_key == "source_union":
        features_cpu, total_batches = _collect_source_union_anchor_features(
            model=model,
            anchor_encoder=anchor_encoder,
            tasks=tasks,
            train_loaders=train_loaders,
            device=device,
            sample_budget=sample_budget,
        )
    elif source_key == "stage1_public_pool":
        features_cpu, total_batches = _collect_public_pool_anchor_features(
            model=model,
            anchor_encoder=anchor_encoder,
            tasks=tasks,
            train_loaders=train_loaders,
            device=device,
            public_pool_path=public_pool_path,
            sample_budget=sample_budget,
        )
    elif source_key == "mixed_source_public":
        source_budget = max(1, sample_budget // 2)
        public_budget = max(1, sample_budget - source_budget)
        source_features, source_batches = _collect_source_union_anchor_features(
            model=model,
            anchor_encoder=anchor_encoder,
            tasks=tasks,
            train_loaders=train_loaders,
            device=device,
            sample_budget=source_budget,
        )
        public_features, public_batches = _collect_public_pool_anchor_features(
            model=model,
            anchor_encoder=anchor_encoder,
            tasks=tasks,
            train_loaders=train_loaders,
            device=device,
            public_pool_path=public_pool_path,
            sample_budget=public_budget,
        )
        features_cpu = [*source_features, *public_features]
        total_batches = int(source_batches + public_batches)
    else:
        raise NotImplementedError(f"Unsupported care_auto anchor source '{anchor_source}'")
    if not features_cpu:
        raise ValueError("Unable to build care_auto anchor basis from empty source loaders")

    anchor_features = torch.cat(features_cpu, dim=0)
    feature_dim = int(anchor_features.shape[-1])
    anchor_mean = anchor_features.mean(dim=0, keepdim=True)
    anchor_std = anchor_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    centered = anchor_features - anchor_mean
    auto_energy_selected = False
    if centered.shape[0] <= 1:
        k_min = max(1, int(min_rank))
        reuse_k = k_min
        candidate_ks = [reuse_k]
        basis = torch.eye(feature_dim, dtype=torch.float32)[:, :reuse_k]
    else:
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        use_auto = (int(requested_k) <= 0) and (float(energy_threshold) > 0.0)
        if use_auto:
            # auto_energy: find smallest k such that cumulative energy >= threshold
            s_sq = s.pow(2)
            total_energy = float(s_sq.sum().item())
            if total_energy > 0:
                cumsum = torch.cumsum(s_sq, dim=0)
                mask = (cumsum / total_energy) >= float(energy_threshold)
                k_energy = int(mask.nonzero(as_tuple=True)[0][0].item()) + 1 if mask.any() else int(s.shape[0])
            else:
                k_energy = 1
            k_cap = max(1, int(math.floor(feature_dim * float(max_rank_ratio))))
            k_min = max(1, int(min_rank))
            reuse_k = max(k_min, min(k_energy, k_cap))
            candidate_ks = sorted({reuse_k, k_min, k_cap, k_energy, min(32, feature_dim)})
            auto_energy_selected = True
        else:
            reuse_k, candidate_ks = _resolve_care_auto_reuse_k(feature_dim, int(requested_k))
        top_k = min(reuse_k, int(vh.shape[0]))
        basis = vh[:top_k].T.contiguous().float()
        reuse_k = int(top_k)
    return {
        "basis": basis.to(device=device),
        "anchor_mean": anchor_mean.to(device=device),
        "anchor_std": anchor_std.to(device=device),
        "anchor_feature_norm_mean": float(anchor_features.norm(dim=-1).mean().item()),
        "feature_dim": int(feature_dim),
        "reuse_k": int(reuse_k),
        "candidate_ks": [int(v) for v in candidate_ks],
        "num_anchor_samples": int(anchor_features.shape[0]),
        "num_anchor_batches": int(total_batches),
        "auto_energy_selected": bool(auto_energy_selected),
    }


def _care_auto_reuse_terms(
    current_features: torch.Tensor,
    anchor_features: torch.Tensor,
    basis: torch.Tensor,
    loss_type: str,
    anchor_mean: Optional[torch.Tensor] = None,
    anchor_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if loss_type == "full_mse":
        reuse_loss = F.mse_loss(current_features, anchor_features)
        drift_mse = F.mse_loss(current_features, anchor_features)
        drift_cosine = F.cosine_similarity(current_features, anchor_features, dim=-1).mean()
        stats: Dict[str, torch.Tensor] = {
            "feature_norm_before": anchor_features.norm(dim=-1).mean(),
            "feature_norm_after": current_features.norm(dim=-1).mean(),
        }
        stats["feature_norm_ratio"] = stats["feature_norm_after"] / stats["feature_norm_before"].clamp_min(1e-6)
        return reuse_loss, drift_mse, drift_cosine, stats

    proj_current = current_features.matmul(basis)
    proj_anchor = anchor_features.matmul(basis)
    drift_mse = F.mse_loss(proj_current, proj_anchor)
    drift_cosine = F.cosine_similarity(proj_current, proj_anchor, dim=-1).mean()
    stats: Dict[str, torch.Tensor] = {
        "feature_norm_before": anchor_features.norm(dim=-1).mean(),
        "feature_norm_after": current_features.norm(dim=-1).mean(),
    }
    stats["feature_norm_ratio"] = stats["feature_norm_after"] / stats["feature_norm_before"].clamp_min(1e-6)
    if loss_type == "mse":
        reuse_loss = drift_mse
    elif loss_type == "cosine":
        reuse_loss = 1.0 - drift_cosine
    elif loss_type == "norm_mse":
        reuse_loss = F.mse_loss(
            F.normalize(proj_current, dim=-1, eps=1e-6),
            F.normalize(proj_anchor, dim=-1, eps=1e-6),
        )
    elif loss_type == "statnorm_mse":
        ref_mean = anchor_mean if anchor_mean is not None else anchor_features.mean(dim=0, keepdim=True)
        ref_std = anchor_std if anchor_std is not None else anchor_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        whitened_current = (current_features - ref_mean) / ref_std
        whitened_anchor = (anchor_features - ref_mean) / ref_std
        reuse_loss = F.mse_loss(whitened_current.matmul(basis), whitened_anchor.matmul(basis))
        current_mean = current_features.mean(dim=0)
        anchor_batch_mean = anchor_features.mean(dim=0)
        current_std = current_features.std(dim=0, unbiased=False)
        anchor_batch_std = anchor_features.std(dim=0, unbiased=False)
        stats["stat_loss"] = F.mse_loss(current_mean, anchor_batch_mean) + F.mse_loss(current_std, anchor_batch_std)
    elif loss_type == "huber":
        reuse_loss = F.huber_loss(proj_current, proj_anchor, delta=1.0)
    else:
        raise ValueError(f"Unsupported care_auto reuse loss '{loss_type}'")
    return reuse_loss, drift_mse, drift_cosine, stats


def estimate_encoder_importance(
    model: MultiTaskModel,
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    max_batches_per_task: int,
) -> Dict[str, torch.Tensor]:
    enc_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad and name.startswith("encoder.")]
    if not enc_params:
        return {}

    importance = {name: torch.zeros_like(p, device="cpu") for name, p in enc_params}
    model = model.to(device)
    model.eval()
    batches_seen = 0
    for rt in tasks:
        loader = train_loaders[rt.spec.name]
        for batch_idx, (xb, yb) in enumerate(loader):
            if batch_idx >= max_batches_per_task:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            _, out = model(xb, task_name=rt.spec.name)
            loss = task_losses(out, {rt.spec.name: yb}, [rt])[rt.spec.name]
            loss.backward()
            for name, p in enc_params:
                if p.grad is not None:
                    importance[name] += p.grad.detach().pow(2).cpu()
            batches_seen += 1

    if batches_seen <= 0:
        return importance

    flat_means: List[torch.Tensor] = []
    for name in list(importance):
        importance[name] = importance[name] / float(batches_seen)
        flat_means.append(importance[name].mean())
    denom = torch.stack(flat_means).mean().clamp_min(1e-8) if flat_means else torch.tensor(1.0)
    for name in list(importance):
        importance[name] = (importance[name] / denom).clamp_min(0.0)
    model.zero_grad(set_to_none=True)
    return importance


def _importance_score_map(importance: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {
        name: float(tensor.detach().float().mean().item())
        for name, tensor in importance.items()
    }


def _normalize_score_map(score_map: Dict[str, float]) -> Dict[str, float]:
    positive = [float(v) for v in score_map.values() if float(v) > 0.0]
    if not positive:
        return {name: 0.0 for name in score_map}
    denom = max(float(np.mean(positive)), 1e-8)
    return {
        name: max(float(value) / denom, 0.0)
        for name, value in score_map.items()
    }


def _merge_score_maps(score_maps: Sequence[Dict[str, float]]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for score_map in score_maps:
        for name, value in score_map.items():
            merged[name] = merged.get(name, 0.0) + float(value)
    return merged


def _top_score_names(score_map: Dict[str, float], budget: int, exclude: Optional[set[str]] = None) -> set[str]:
    if int(budget) <= 0:
        return set()
    blocked = exclude or set()
    ranked = sorted(
        ((name, float(value)) for name, value in score_map.items() if name not in blocked),
        key=lambda item: item[1],
        reverse=True,
    )
    return {name for name, _ in ranked[: int(budget)]}


def _encoder_param_name_set(model: MultiTaskModel) -> set[str]:
    return {f"encoder.{name}" for name, _ in model.encoder.named_parameters()}


def _care_v4_stage2_candidate_names(
    model: MultiTaskModel,
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    scope: str,
    max_batches: int,
    extra_ratio: float,
) -> set[str]:
    all_names = _encoder_param_name_set(model)
    if scope == "full":
        return all_names

    base_names = _care_v3_candidate_param_names(model, scope)
    remaining_names = max(0, len(all_names) - len(base_names))
    remaining_budget = min(
        remaining_names,
        max(1, int(math.ceil(float(extra_ratio) * max(1, remaining_names)))),
    )
    if remaining_budget <= 0 or not tasks:
        return base_names

    shared_scores = _normalize_score_map(
        _importance_score_map(
            estimate_encoder_importance(
                model=model,
                tasks=tasks,
                train_loaders=train_loaders,
                device=device,
                max_batches_per_task=max_batches,
            )
        )
    )
    per_task_scores = [
        _normalize_score_map(
            _importance_score_map(
                estimate_encoder_importance(
                    model=model,
                    tasks=[rt],
                    train_loaders=train_loaders,
                    device=device,
                    max_batches_per_task=max_batches,
                )
            )
        )
        for rt in tasks
    ]

    selected = set(base_names)
    shared_budget = max(1, remaining_budget // 2) if shared_scores else 0
    per_task_budget = max(1, (remaining_budget - shared_budget) // max(1, len(per_task_scores))) if per_task_scores else 0
    selected |= _top_score_names(shared_scores, shared_budget, exclude=selected)
    for score_map in per_task_scores:
        selected |= _top_score_names(score_map, per_task_budget, exclude=selected)
    missing = len(base_names) + remaining_budget - len(selected)
    if missing > 0:
        selected |= _top_score_names(_merge_score_maps([shared_scores] + per_task_scores), missing, exclude=selected)
    return selected


def _care_v4_stage3_candidate_names(
    model: MultiTaskModel,
    new_task: TaskRuntime,
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    old_task_importance: Dict[str, Dict[str, torch.Tensor]],
    scope: str,
    max_batches: int,
    extra_ratio: float,
) -> Tuple[set[str], Dict[str, torch.Tensor]]:
    all_names = _encoder_param_name_set(model)
    new_importance = estimate_encoder_importance(
        model=model,
        tasks=[new_task],
        train_loaders=train_loaders,
        device=device,
        max_batches_per_task=max_batches,
    )
    if scope == "full":
        return all_names, new_importance

    base_names = _care_v3_candidate_param_names(model, scope)
    remaining_names = max(0, len(all_names) - len(base_names))
    new_scores = _normalize_score_map(_importance_score_map(new_importance))
    old_score_maps = [
        _normalize_score_map(_importance_score_map(importance))
        for importance in old_task_importance.values()
    ]
    remaining_budget = min(
        remaining_names,
        max(1, int(math.ceil(float(extra_ratio) * max(1, remaining_names)))),
    )
    if remaining_budget <= 0:
        return base_names, new_importance

    selected = set(base_names)
    new_budget = max(1, remaining_budget // 3) if new_scores else 0
    old_shared_budget = max(1, remaining_budget // 3) if old_score_maps else 0
    per_old_budget = max(1, (remaining_budget - new_budget - old_shared_budget) // max(1, len(old_score_maps))) if old_score_maps else 0
    selected |= _top_score_names(new_scores, new_budget, exclude=selected)
    selected |= _top_score_names(_merge_score_maps(old_score_maps), old_shared_budget, exclude=selected)
    for score_map in old_score_maps:
        selected |= _top_score_names(score_map, per_old_budget, exclude=selected)
    missing = len(base_names) + remaining_budget - len(selected)
    if missing > 0:
        selected |= _top_score_names(_merge_score_maps([new_scores] + old_score_maps), missing, exclude=selected)
    return selected, new_importance


def _mean_or_zero(losses: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.stack(losses).mean() if losses else torch.zeros((), device=device)


def _care_stage2_family_losses(
    task_loss_pairs: List[Tuple[TaskRuntime, torch.Tensor]],
    care_scales: Dict[str, float],
    care_loss_ema: float,
    reg_floor_lambda: float,
    cls_floor_lambda: float,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    entries: List[Dict[str, Any]] = []
    reg_losses: List[torch.Tensor] = []
    cls_losses: List[torch.Tensor] = []
    for rt, loss_i in task_loss_pairs:
        name = rt.spec.name
        scale = max(care_scales.get(name, float(loss_i.detach().item())), 1e-6)
        norm_loss = loss_i / scale
        entry = {"runtime": rt, "norm_loss": norm_loss}
        entries.append(entry)
        if rt.spec.kind == "reg":
            reg_losses.append(norm_loss)
        else:
            cls_losses.append(norm_loss)
        care_scales[name] = max(care_loss_ema * scale + (1.0 - care_loss_ema) * float(loss_i.detach().item()), 1e-6)

    reg_mean = _mean_or_zero(reg_losses, device)
    cls_mean = _mean_or_zero(cls_losses, device)
    if reg_losses and cls_losses:
        balanced = 0.5 * reg_mean + 0.5 * cls_mean
    else:
        balanced = reg_mean + cls_mean

    reg_floor = torch.topk(torch.stack(reg_losses), k=max(1, int(math.ceil(len(reg_losses) / 2.0)))).values.mean() if reg_losses else torch.zeros((), device=device)
    cls_floor = torch.topk(torch.stack(cls_losses), k=max(1, int(math.ceil(len(cls_losses) / 2.0)))).values.mean() if cls_losses else torch.zeros((), device=device)
    reg_obj = reg_mean + reg_floor_lambda * reg_floor if reg_losses else torch.zeros((), device=device)
    cls_obj = cls_mean + cls_floor_lambda * cls_floor if cls_losses else torch.zeros((), device=device)
    total = balanced + reg_floor_lambda * reg_floor + cls_floor_lambda * cls_floor
    return entries, total, reg_obj, cls_obj


def _project_conflicting(primary: torch.Tensor, reference: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if primary.numel() == 0 or reference.numel() == 0:
        return primary
    dot = torch.dot(primary, reference)
    if float(dot.detach().item()) >= 0.0:
        return primary
    denom = reference.pow(2).sum().clamp_min(1e-12)
    return primary - float(scale) * dot / denom * reference


def _care_v2b_param_bonus(name: str) -> float:
    return 1.0 if ".bn" in name or "first_block_bn" in name or name.endswith("norm.weight") or name.endswith("norm.bias") else 0.65


def _care_v3_param_bonus(name: str) -> float:
    if _is_norm_param_name(name):
        return 1.0
    if name.endswith(".bias"):
        return 0.85
    return 0.65


def _tensor_negative_cosine_fraction(task_tensors: Sequence[torch.Tensor]) -> float:
    if len(task_tensors) < 2:
        return 0.0
    negatives = 0
    pairs = 0
    for i in range(len(task_tensors)):
        gi = task_tensors[i].reshape(-1)
        ni = gi.norm().clamp_min(1e-12)
        if float(ni.detach().item()) <= 1e-12:
            continue
        for j in range(i + 1, len(task_tensors)):
            gj = task_tensors[j].reshape(-1)
            nj = gj.norm().clamp_min(1e-12)
            if float(nj.detach().item()) <= 1e-12:
                continue
            cos = torch.dot(gi, gj) / (ni * nj)
            negatives += int(float(cos.detach().item()) < 0.0)
            pairs += 1
    return float(negatives) / float(max(1, pairs))


def _care_v2b_stage2_encoder_grads(
    enc_params_named: List[Tuple[str, nn.Parameter]],
    reg_grads: List[Optional[torch.Tensor]],
    cls_grads: List[Optional[torch.Tensor]],
    anchor_grads: List[Optional[torch.Tensor]],
) -> Tuple[List[torch.Tensor], float, float]:
    reg_norms: List[torch.Tensor] = []
    cls_norms: List[torch.Tensor] = []
    for g_reg, g_cls in zip(reg_grads, cls_grads):
        reg_norms.append((torch.zeros((), device=enc_params_named[0][1].device) if g_reg is None else g_reg.norm()))
        cls_norms.append((torch.zeros((), device=enc_params_named[0][1].device) if g_cls is None else g_cls.norm()))
    totals = [r + c for r, c in zip(reg_norms, cls_norms)]
    denom = torch.stack([t for t in totals if float(t.detach().item()) > 0.0]).mean().clamp_min(1e-8) if any(float(t.detach().item()) > 0.0 for t in totals) else torch.tensor(1.0, device=enc_params_named[0][1].device)

    final_grads: List[torch.Tensor] = []
    gate_vals: List[float] = []
    reg_ratio_vals: List[float] = []
    for (name, param), g_reg, g_cls, g_anchor, u_reg, u_cls, total in zip(enc_params_named, reg_grads, cls_grads, anchor_grads, reg_norms, cls_norms, totals):
        reg_t = torch.zeros_like(param) if g_reg is None else g_reg
        cls_t = torch.zeros_like(param) if g_cls is None else g_cls
        anchor_t = torch.zeros_like(param) if g_anchor is None else g_anchor
        total_val = float(total.detach().item())
        if total_val <= 0.0:
            gate = 0.0
            reg_ratio = 0.5
        else:
            conflict = 0.0
            if float(u_reg.detach().item()) > 0.0 and float(u_cls.detach().item()) > 0.0:
                cos = torch.dot(reg_t.reshape(-1), cls_t.reshape(-1)) / (u_reg * u_cls).clamp_min(1e-12)
                conflict = max(0.0, -float(cos.detach().item()))
            total_rel = float((total / denom).detach().item())
            gate = min(1.0, _care_v2b_param_bonus(name) * math.sqrt(max(total_rel, 0.0)) / (1.0 + conflict))
            reg_ratio = float((u_reg / (u_reg + u_cls + 1e-12)).detach().item()) if float((u_reg + u_cls).detach().item()) > 0.0 else 0.5
        cls_proj = _project_conflicting(cls_t.reshape(-1), reg_t.reshape(-1), scale=1.0).view_as(param)
        reg_priority = 1.0 + 0.5 * reg_ratio
        final = gate * (reg_ratio * reg_t + (1.0 - reg_ratio) * cls_proj) + reg_priority * anchor_t
        final_grads.append(final)
        gate_vals.append(gate)
        reg_ratio_vals.append(reg_ratio)
    return final_grads, float(np.mean(gate_vals)) if gate_vals else 0.0, float(np.mean(reg_ratio_vals)) if reg_ratio_vals else 0.5


def _care_v2b_stage3_encoder_grads(
    enc_params_named: List[Tuple[str, nn.Parameter]],
    new_grads: List[Optional[torch.Tensor]],
    reg_preserve_grads: List[Optional[torch.Tensor]],
    cls_preserve_grads: List[Optional[torch.Tensor]],
    reg_importance: Dict[str, torch.Tensor],
    cls_importance: Dict[str, torch.Tensor],
) -> Tuple[List[torch.Tensor], float]:
    final_grads: List[torch.Tensor] = []
    reg_priority_vals: List[float] = []
    for (name, param), g_new, g_reg, g_cls in zip(enc_params_named, new_grads, reg_preserve_grads, cls_preserve_grads):
        new_t = torch.zeros_like(param) if g_new is None else g_new
        reg_t = torch.zeros_like(param) if g_reg is None else g_reg
        cls_t = torch.zeros_like(param) if g_cls is None else g_cls
        reg_imp = reg_importance.get(name)
        cls_imp = cls_importance.get(name)
        reg_imp_mean = float(reg_imp.mean().item()) if reg_imp is not None else 0.0
        cls_imp_mean = float(cls_imp.mean().item()) if cls_imp is not None else 0.0
        reg_priority = 0.5 + reg_imp_mean / (reg_imp_mean + cls_imp_mean + 1e-8)
        new_proj = _project_conflicting(new_t.reshape(-1), reg_t.reshape(-1), scale=1.0).view_as(param)
        if float(cls_t.norm().detach().item()) > 0.0:
            new_proj = _project_conflicting(new_proj.reshape(-1), cls_t.reshape(-1), scale=0.35).view_as(param)
        final = new_proj + reg_priority * reg_t + 0.35 * cls_t
        final_grads.append(final)
        reg_priority_vals.append(reg_priority)
    return final_grads, float(np.mean(reg_priority_vals)) if reg_priority_vals else 0.0


def _care_v3_stage2_encoder_grads(
    enc_params_named: List[Tuple[str, nn.Parameter]],
    task_grads: List[List[Optional[torch.Tensor]]],
    anchor_grads: List[Optional[torch.Tensor]],
) -> Tuple[List[torch.Tensor], float, float, float]:
    if not enc_params_named:
        return [], 0.0, 0.0, 0.0

    device = enc_params_named[0][1].device
    total_norms: List[torch.Tensor] = []
    for (name, param), grads_for_param in zip(enc_params_named, zip(*task_grads)):
        del name
        norm_vals = [torch.zeros((), device=device) if g is None else g.norm() for g in grads_for_param]
        total_norms.append(torch.stack(norm_vals).sum() if norm_vals else torch.zeros((), device=device))
    nonzero_totals = [t for t in total_norms if float(t.detach().item()) > 0.0]
    denom = torch.stack(nonzero_totals).mean().clamp_min(1e-8) if nonzero_totals else torch.tensor(1.0, device=device)

    final_grads: List[torch.Tensor] = []
    gate_vals: List[float] = []
    focus_vals: List[float] = []
    conflict_vals: List[float] = []
    for (name, param), grads_for_param, g_anchor, total in zip(enc_params_named, zip(*task_grads), anchor_grads, total_norms):
        task_tensors = [torch.zeros_like(param) if g is None else g for g in grads_for_param]
        anchor_t = torch.zeros_like(param) if g_anchor is None else g_anchor
        total_val = float(total.detach().item())
        if total_val <= 0.0:
            gate = 0.0
            focus = 0.0
            conflict = 0.0
            merged = torch.zeros_like(param)
        else:
            local_stack = torch.stack([t.reshape(-1) for t in task_tensors])
            merged = (pcgrad_merge(local_stack) if local_stack.shape[0] > 1 else local_stack[0]).view_as(param)
            norm_stack = torch.stack([t.norm() for t in task_tensors])
            focus = float((norm_stack.max() / norm_stack.sum().clamp_min(1e-12)).detach().item())
            conflict = _tensor_negative_cosine_fraction(task_tensors)
            total_rel = float((total / denom).detach().item())
            gate = min(
                1.0,
                _care_v3_param_bonus(name) * math.sqrt(max(total_rel, 0.0)) * (0.5 + 0.5 * focus) / (1.0 + conflict),
            )
        final_grads.append(gate * merged + anchor_t)
        gate_vals.append(gate)
        focus_vals.append(focus)
        conflict_vals.append(conflict)
    return (
        final_grads,
        float(np.mean(gate_vals)) if gate_vals else 0.0,
        float(np.mean(focus_vals)) if focus_vals else 0.0,
        float(np.mean(conflict_vals)) if conflict_vals else 0.0,
    )


def _care_v3_stage3_encoder_grads(
    enc_params_named: List[Tuple[str, nn.Parameter]],
    new_grads: List[Optional[torch.Tensor]],
    preserve_grads: List[Optional[torch.Tensor]],
    importance: Dict[str, torch.Tensor],
) -> Tuple[List[torch.Tensor], float, float]:
    if not enc_params_named:
        return [], 0.0, 0.0

    device = enc_params_named[0][1].device
    total_norms: List[torch.Tensor] = []
    importance_means: List[torch.Tensor] = []
    for (name, param), g_new, g_preserve in zip(enc_params_named, new_grads, preserve_grads):
        del param
        new_norm = torch.zeros((), device=device) if g_new is None else g_new.norm()
        preserve_norm = torch.zeros((), device=device) if g_preserve is None else g_preserve.norm()
        total_norms.append(new_norm + preserve_norm)
        imp = importance.get(name)
        importance_means.append(torch.tensor(0.0, device=device) if imp is None else imp.to(device=device, dtype=torch.float32).mean())
    nonzero_totals = [t for t in total_norms if float(t.detach().item()) > 0.0]
    total_denom = torch.stack(nonzero_totals).mean().clamp_min(1e-8) if nonzero_totals else torch.tensor(1.0, device=device)
    nonzero_importance = [t for t in importance_means if float(t.detach().item()) > 0.0]
    importance_denom = torch.stack(nonzero_importance).mean().clamp_min(1e-8) if nonzero_importance else torch.tensor(1.0, device=device)

    final_grads: List[torch.Tensor] = []
    gate_vals: List[float] = []
    plastic_vals: List[float] = []
    for (name, param), g_new, g_preserve, total, imp_mean in zip(enc_params_named, new_grads, preserve_grads, total_norms, importance_means):
        new_t = torch.zeros_like(param) if g_new is None else g_new
        preserve_t = torch.zeros_like(param) if g_preserve is None else g_preserve
        new_norm = new_t.norm()
        preserve_norm = preserve_t.norm()
        total_val = float(total.detach().item())
        if total_val <= 0.0:
            gate = 0.0
            plastic = 0.0
            new_proj = torch.zeros_like(param)
        else:
            plastic = float((new_norm / (new_norm + preserve_norm + 1e-12)).detach().item())
            conflict = 0.0
            if float(new_norm.detach().item()) > 0.0 and float(preserve_norm.detach().item()) > 0.0:
                cos = torch.dot(new_t.reshape(-1), preserve_t.reshape(-1)) / (new_norm * preserve_norm).clamp_min(1e-12)
                conflict = max(0.0, -float(cos.detach().item()))
                new_proj = _project_conflicting(new_t.reshape(-1), preserve_t.reshape(-1), scale=1.0).view_as(param)
            else:
                new_proj = new_t
            total_rel = float((total / total_denom).detach().item())
            imp_rel = float((imp_mean / importance_denom).detach().item()) if float(imp_mean.detach().item()) > 0.0 else 0.0
            gate = min(
                1.0,
                _care_v3_param_bonus(name) * math.sqrt(max(total_rel, 0.0)) * plastic / (1.0 + conflict + 0.35 * imp_rel),
            )
        final_grads.append(gate * new_proj + preserve_t)
        gate_vals.append(gate)
        plastic_vals.append(plastic)
    return final_grads, float(np.mean(gate_vals)) if gate_vals else 0.0, float(np.mean(plastic_vals)) if plastic_vals else 0.0


def _care_v4_stage2_encoder_grads(
    enc_params_named: List[Tuple[str, nn.Parameter]],
    task_grads: List[List[Optional[torch.Tensor]]],
    anchor_grads: List[Optional[torch.Tensor]],
    candidate_names: set[str],
) -> Tuple[List[torch.Tensor], float, float, float, float]:
    final_grads, gate_mean, focus_mean, negcos = _care_v3_stage2_encoder_grads(
        enc_params_named=enc_params_named,
        task_grads=task_grads,
        anchor_grads=anchor_grads,
    )
    masked_grads: List[torch.Tensor] = []
    selected = 0
    for (name, param), grad in zip(enc_params_named, final_grads):
        if name in candidate_names:
            masked_grads.append(grad)
            selected += 1
        else:
            masked_grads.append(torch.zeros_like(param))
    select_ratio = float(selected) / float(max(1, len(enc_params_named)))
    return masked_grads, gate_mean, focus_mean, negcos, select_ratio


def _care_v4_stage3_encoder_grads(
    enc_params_named: List[Tuple[str, nn.Parameter]],
    new_grads: List[Optional[torch.Tensor]],
    shared_preserve_grads: List[Optional[torch.Tensor]],
    old_task_preserve_grads: Dict[str, List[Optional[torch.Tensor]]],
    old_task_importance: Dict[str, Dict[str, torch.Tensor]],
    candidate_names: set[str],
) -> Tuple[List[torch.Tensor], float, float, float]:
    if not enc_params_named:
        return [], 0.0, 0.0, 0.0

    device = enc_params_named[0][1].device
    total_norms: List[torch.Tensor] = []
    importance_means: List[torch.Tensor] = []
    old_task_names = list(old_task_preserve_grads)
    for idx, ((name, _), g_new, g_shared) in enumerate(zip(enc_params_named, new_grads, shared_preserve_grads)):
        new_norm = torch.zeros((), device=device) if g_new is None else g_new.norm()
        shared_norm = torch.zeros((), device=device) if g_shared is None else g_shared.norm()
        old_norm_total = torch.zeros((), device=device)
        for task_name in old_task_names:
            g_old = old_task_preserve_grads[task_name][idx]
            if g_old is not None:
                old_norm_total = old_norm_total + g_old.norm()
        total_norms.append(new_norm + shared_norm + old_norm_total)
        old_imp_vals = []
        for task_name in old_task_names:
            imp = old_task_importance.get(task_name, {}).get(name)
            if imp is not None:
                old_imp_vals.append(imp.to(device=device, dtype=torch.float32).mean())
        importance_means.append(torch.stack(old_imp_vals).mean() if old_imp_vals else torch.tensor(0.0, device=device))

    nonzero_totals = [t for t in total_norms if float(t.detach().item()) > 0.0]
    total_denom = torch.stack(nonzero_totals).mean().clamp_min(1e-8) if nonzero_totals else torch.tensor(1.0, device=device)
    nonzero_importance = [t for t in importance_means if float(t.detach().item()) > 0.0]
    importance_denom = torch.stack(nonzero_importance).mean().clamp_min(1e-8) if nonzero_importance else torch.tensor(1.0, device=device)

    final_grads: List[torch.Tensor] = []
    gate_vals: List[float] = []
    plastic_vals: List[float] = []
    route_vals: List[float] = []
    for idx, ((name, param), g_new, g_shared, total, imp_mean) in enumerate(zip(enc_params_named, new_grads, shared_preserve_grads, total_norms, importance_means)):
        new_t = torch.zeros_like(param) if g_new is None else g_new
        shared_t = torch.zeros_like(param) if g_shared is None else g_shared
        new_norm = new_t.norm()
        old_terms: List[torch.Tensor] = []
        route_scores: List[float] = []
        for task_name in old_task_names:
            g_old = old_task_preserve_grads[task_name][idx]
            old_t = torch.zeros_like(param) if g_old is None else g_old
            old_norm = old_t.norm()
            if float(old_norm.detach().item()) <= 0.0:
                continue
            imp = old_task_importance.get(task_name, {}).get(name)
            imp_val = float(imp.mean().item()) if imp is not None else 0.0
            conflict = 0.0
            if float(new_norm.detach().item()) > 0.0:
                cos = torch.dot(new_t.reshape(-1), old_t.reshape(-1)) / (new_norm * old_norm).clamp_min(1e-12)
                conflict = max(0.0, -float(cos.detach().item()))
            route_scores.append((0.25 + conflict) * (0.5 + imp_val))
            old_terms.append(old_t)
        old_mix = torch.zeros_like(param)
        route_peak = 0.0
        if old_terms:
            route_sum = max(sum(route_scores), 1e-8)
            route_weights = [score / route_sum for score in route_scores]
            route_peak = max(route_weights)
            for weight, term in zip(route_weights, old_terms):
                old_mix = old_mix + float(weight) * term
        preserve_t = shared_t + old_mix
        preserve_norm = preserve_t.norm()
        total_val = float(total.detach().item())
        if name not in candidate_names or total_val <= 0.0:
            gate = 0.0
            plastic = 0.0
            final = torch.zeros_like(param)
        else:
            plastic = float((new_norm / (new_norm + preserve_norm + 1e-12)).detach().item())
            conflict = 0.0
            if float(new_norm.detach().item()) > 0.0 and float(preserve_norm.detach().item()) > 0.0:
                cos = torch.dot(new_t.reshape(-1), preserve_t.reshape(-1)) / (new_norm * preserve_norm).clamp_min(1e-12)
                conflict = max(0.0, -float(cos.detach().item()))
                new_proj = _project_conflicting(new_t.reshape(-1), preserve_t.reshape(-1), scale=1.0).view_as(param)
            else:
                new_proj = new_t
            total_rel = float((total / total_denom).detach().item())
            imp_rel = float((imp_mean / importance_denom).detach().item()) if float(imp_mean.detach().item()) > 0.0 else 0.0
            gate = min(
                1.0,
                _care_v3_param_bonus(name) * math.sqrt(max(total_rel, 0.0)) * (0.4 + 0.6 * plastic) / (1.0 + conflict + 0.25 * imp_rel),
            )
            final = preserve_t + gate * new_proj
        final_grads.append(final)
        gate_vals.append(gate)
        plastic_vals.append(plastic)
        route_vals.append(route_peak)
    return (
        final_grads,
        float(np.mean(gate_vals)) if gate_vals else 0.0,
        float(np.mean(plastic_vals)) if plastic_vals else 0.0,
        float(np.mean(route_vals)) if route_vals else 0.0,
    )


def _care_stage2_entries(
    task_loss_pairs: List[Tuple[TaskRuntime, torch.Tensor]],
    care_scales: Dict[str, float],
    care_loss_ema: float,
    care_floor_lambda: float,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    entries: List[Dict[str, Any]] = []
    reg_indices: List[int] = []
    cls_indices: List[int] = []
    for rt, loss_i in task_loss_pairs:
        name = rt.spec.name
        scale = max(care_scales.get(name, float(loss_i.detach().item())), 1e-6)
        norm_loss = loss_i / scale
        entry = {"runtime": rt, "norm_loss": norm_loss, "weight": 0.0}
        entries.append(entry)
        if rt.spec.kind == "reg":
            reg_indices.append(len(entries) - 1)
        else:
            cls_indices.append(len(entries) - 1)
        care_scales[name] = max(care_loss_ema * scale + (1.0 - care_loss_ema) * float(loss_i.detach().item()), 1e-6)

    if not entries:
        return entries, torch.zeros(())

    if reg_indices and cls_indices:
        for idx in reg_indices:
            entries[idx]["weight"] += 0.5 / float(len(reg_indices))
        for idx in cls_indices:
            entries[idx]["weight"] += 0.5 / float(len(cls_indices))
    else:
        base_weight = 1.0 / float(len(entries))
        for entry in entries:
            entry["weight"] += base_weight

    norm_stack = torch.stack([entry["norm_loss"] for entry in entries])
    floor_k = max(1, int(math.ceil(len(entries) / 2.0)))
    floor_indices = torch.topk(norm_stack, k=floor_k).indices.tolist()
    floor_weight = float(care_floor_lambda) / float(floor_k)
    for idx in floor_indices:
        entries[idx]["weight"] += floor_weight

    total = entries[0]["norm_loss"].new_zeros(())
    for entry in entries:
        total = total + float(entry["weight"]) * entry["norm_loss"]
    return entries, total


def _care_gradient_agreement_mask(task_grads: torch.Tensor, mask_min: float, mask_power: float) -> torch.Tensor:
    if task_grads.ndim != 2:
        raise ValueError("task_grads must have shape [n_tasks, n_params]")
    if task_grads.shape[0] == 0:
        return torch.zeros(task_grads.shape[-1], device=task_grads.device, dtype=task_grads.dtype)
    mean_abs = task_grads.abs().mean(0)
    agree = task_grads.mean(0).abs() / mean_abs.clamp_min(1e-12)
    mask = agree.clamp(0.0, 1.0).pow(max(float(mask_power), 1e-6))
    if float(mask_min) > 0.0:
        mask = mask.clamp_min(float(mask_min))
    return mask


def _negative_cosine_fraction(task_grads: torch.Tensor) -> float:
    if task_grads.ndim != 2 or task_grads.shape[0] < 2:
        return 0.0
    negatives = 0
    pairs = 0
    for i in range(task_grads.shape[0]):
        gi = task_grads[i]
        ni = gi.norm().clamp_min(1e-12)
        for j in range(i + 1, task_grads.shape[0]):
            gj = task_grads[j]
            nj = gj.norm().clamp_min(1e-12)
            cos = torch.dot(gi, gj) / (ni * nj)
            negatives += int(float(cos.detach().item()) < 0.0)
            pairs += 1
    return float(negatives) / float(max(1, pairs))


def set_stage2_trainable(
    model: MultiTaskModel,
    method: str,
    care_stage2_scope: str = "top2",
    care_v5_stage2_scope: str = "top2",
    care_auto_stage2_scope: str = "full_encoder",
) -> None:
    if method == "lora_tune":
        set_lora_encoder_trainable(model, enabled=True)
    elif method in CARE_AUTO_STAGE2_METHODS:
        set_encoder_trainable_scope(model, _care_auto_trainable_scope(care_auto_stage2_scope))
    elif method in CARE_V5_STAGE2_METHODS:
        _set_encoder_trainable_names(model, _care_v5_candidate_param_names(model, care_v5_stage2_scope))
    elif method == "care_direct_v4":
        set_encoder_trainable_scope(model, "full")
    elif method in SCOPE_MATCHED_STAGE2_METHODS:
        _set_encoder_trainable_names(model, _care_v3_candidate_param_names(model, care_stage2_scope))
    elif method == "care_direct_v2b":
        _set_encoder_trainable_names(model, _care_v2b_candidate_param_names(model, care_stage2_scope))
    elif method == "care_direct_v3":
        _set_encoder_trainable_names(model, _care_v3_candidate_param_names(model, care_stage2_scope))
    else:
        set_encoder_trainable_scope(model, _stage2_encoder_scope(method, care_stage2_scope, care_v5_stage2_scope))
    for p in model.heads.parameters():
        p.requires_grad = True


def fit_stage2(
    model: MultiTaskModel,
    method: str,
    tasks: List[TaskRuntime],
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    test_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    epochs: int,
    patience: int,
    head_lr: float,
    encoder_lr_mult: float,
    weight_decay: float,
    log_prefix: str,
    care_stage2_scope: str = "top2",
    care_stage2_anchor_lambda: float = 0.1,
    care_floor_lambda: float = 0.25,
    care_loss_ema: float = 0.95,
    care_grad_mask_min: float = 0.10,
    care_grad_mask_power: float = 1.5,
    care_reg_floor_lambda: float = 0.35,
    care_cls_floor_lambda: float = 0.15,
    care_importance_batches: int = 8,
    care_v4_extra_ratio: float = 0.18,
    task_interfaces: Optional[Dict[str, TaskInterface]] = None,
    care_v5_stage2_scope: str = "top2",
    care_v5_anchor_lambda: float = 0.1,
    care_v5_floor_lambda: float = 0.25,
    care_v5_loss_ema: float = 0.95,
    care_auto_anchor_source: str = "source_union",
    care_auto_reuse_feature_space: str = "encoder_pooled",
    care_auto_reuse_k: int = 0,
    care_auto_reuse_lambda: float = 0.05,
    care_auto_stage2_scope: str = "full_encoder",
    care_auto_reuse_loss_type: str = "mse",
    care_auto_anchor_max_batches: int = 8,
    care_auto_anchor_basis_samples: int = 0,
    care_auto_public_pool_path: str = "",
    care_auto_drift_budget: float = 0.02,
    care_auto_lambda_scale_min: float = 0.5,
    care_auto_lambda_scale_max: float = 5.0,
    care_auto_stat_lambda: float = 0.01,
    care_auto_energy_threshold: float = 0.95,
    care_auto_max_rank_ratio: float = 0.5,
    care_auto_min_rank: int = 4,
    enable_early_stop: bool = True,
    restore_best_checkpoint: bool = True,
    task_loss_reduction: str = "mean",
    fairgrad_alpha: float = 1.0,
    l2sp_lambda: float = 1e-2,
) -> Tuple[MultiTaskModel, Dict[str, Dict[str, float]], Dict[str, Any]]:
    if task_loss_reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported task_loss_reduction: {task_loss_reduction}")
    model = model.to(device)
    set_stage2_trainable(
        model,
        method,
        care_stage2_scope=care_stage2_scope,
        care_v5_stage2_scope=care_v5_stage2_scope,
        care_auto_stage2_scope=care_auto_stage2_scope,
    )
    trainable_summary = _trainable_param_summary(model)
    init_regression_biases(model, tasks)
    stage1_encoder_snapshot = _encoder_param_snapshot(model)
    opt = make_optimizer(model, head_lr, encoder_lr_mult, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    task_logits: Optional[nn.Parameter] = None
    task_opt: Optional[torch.optim.Optimizer] = None
    init_losses: Optional[torch.Tensor] = None
    if method == "gradnorm_joint":
        task_logits = nn.Parameter(torch.zeros(len(tasks), device=device))
        task_opt = torch.optim.Adam([task_logits], lr=2.5e-2)

    anchor_encoder: Optional[nn.Module] = None
    care_scales: Dict[str, float] = {}
    care_auto_basis: Optional[torch.Tensor] = None
    care_auto_anchor_mean: Optional[torch.Tensor] = None
    care_auto_anchor_std: Optional[torch.Tensor] = None
    care_auto_feature_dim = 0
    care_auto_selected_k = 0
    care_auto_candidate_ks: List[int] = []
    care_auto_basis_batches = 0
    care_auto_basis_samples = 0
    care_auto_anchor_feature_norm_mean = float("nan")
    care_auto_active_loss_type = _resolve_care_auto_reuse_loss_type(method, care_auto_reuse_loss_type)
    if method in ALL_CARE_STAGE2_METHODS:
        anchor_encoder = copy.deepcopy(model.encoder).to(device).eval()
        anchor_encoder.requires_grad_(False)
    if method in CARE_AUTO_STAGE2_METHODS:
        care_auto_basis_info = _build_care_auto_anchor_basis(
            model=model,
            anchor_encoder=anchor_encoder,
            tasks=tasks,
            train_loaders=train_loaders,
            device=device,
            feature_space=care_auto_reuse_feature_space,
            anchor_source=care_auto_anchor_source,
            requested_k=int(care_auto_reuse_k),
            max_batches=int(care_auto_anchor_max_batches),
            requested_samples=int(care_auto_anchor_basis_samples),
            public_pool_path=str(care_auto_public_pool_path),
            energy_threshold=float(care_auto_energy_threshold),
            max_rank_ratio=float(care_auto_max_rank_ratio),
            min_rank=int(care_auto_min_rank),
        )
        care_auto_basis = care_auto_basis_info["basis"]
        care_auto_anchor_mean = care_auto_basis_info["anchor_mean"]
        care_auto_anchor_std = care_auto_basis_info["anchor_std"]
        care_auto_feature_dim = int(care_auto_basis_info["feature_dim"])
        care_auto_selected_k = int(care_auto_basis_info["reuse_k"])
        care_auto_candidate_ks = list(care_auto_basis_info["candidate_ks"])
        care_auto_basis_batches = int(care_auto_basis_info["num_anchor_batches"])
        care_auto_basis_samples = int(care_auto_basis_info["num_anchor_samples"])
        care_auto_anchor_feature_norm_mean = float(care_auto_basis_info.get("anchor_feature_norm_mean", float("nan")))
        _auto_tag = "(auto_energy)" if care_auto_basis_info.get("auto_energy_selected") else "(fixed)"
        print(
            f"    {log_prefix}CARE anchor basis: k={care_auto_selected_k}{_auto_tag}"
            f" samples={care_auto_basis_samples} lambda={care_auto_reuse_lambda} loss={care_auto_active_loss_type}",
            flush=True,
        )

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    last_aux = 0.0
    last_fair_weights: Optional[np.ndarray] = None
    last_fair_gtg_trace = float("nan")
    last_anchor = 0.0
    last_mask_mean = 1.0
    last_negcos = 0.0
    last_route_ratio = 0.5
    last_select_ratio = 1.0
    last_reuse_lambda = float(care_auto_reuse_lambda)
    last_anchor_drift_mse = float("nan")
    last_anchor_drift_cosine = float("nan")
    last_feature_norm_before = float("nan")
    last_feature_norm_after = float("nan")
    last_feature_norm_ratio = float("nan")
    last_source_loss = float("nan")
    last_reuse_scale_relative = float("nan")
    last_stat_loss = float("nan")
    reuse_history: List[float] = []
    source_loss_history: List[float] = []
    reuse_scale_history: List[float] = []
    source_loss_ema: Dict[str, torch.Tensor] = {}
    care_v4_candidate_names: Optional[set[str]] = None
    if method == "care_direct_v4":
        care_v4_candidate_names = _care_v4_stage2_candidate_names(
            model=model,
            tasks=tasks,
            train_loaders=train_loaders,
            device=device,
            scope=care_stage2_scope,
            max_batches=care_importance_batches,
            extra_ratio=care_v4_extra_ratio,
        )

    for ep in range(epochs):
        model.train()
        _keep_frozen_encoder_norms_eval(model)
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
                # Memory-efficient PCGrad: sequential per-task backward passes,
                # no retain_graph. Gradients are cloned after each task's backward.
                params = [p for p in model.parameters() if p.requires_grad]
                task_grads: List[torch.Tensor] = []
                task_scalars: List[float] = []
                for rt, xb, yb in batches:
                    opt.zero_grad(set_to_none=True)
                    _, out = model(xb, task_name=rt.spec.name)
                    loss_i = task_losses(out, {rt.spec.name: yb}, [rt])[rt.spec.name]
                    loss_i.backward()
                    grads_i = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                    task_grads.append(flatten_grads(grads_i, params))
                    task_scalars.append(float(loss_i.detach().item()))
                assign_flat_grad(params, pcgrad_merge(torch.stack(task_grads)))
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                train_losses.append(float(np.mean(task_scalars)))
                continue

            if method == "gradnorm_joint":
                assert task_logits is not None and task_opt is not None
                # Memory-efficient GradNorm: sequential per-task passes (no retain_graph,
                # no create_graph). Uses G_i = w_i * ||grad_enc(L_i)|| which equals
                # ||grad_enc(w_i * L_i)|| by linearity, keeping G differentiable
                # w.r.t. task_logits without a retained computation graph.
                enc_params = [p for name, p in model.named_parameters()
                              if p.requires_grad and name.startswith("encoder.")]
                raw_norms: List[torch.Tensor] = []
                raw_losses_gn: List[torch.Tensor] = []
                for rt, xb, yb in batches:
                    opt.zero_grad(set_to_none=True)
                    _, out_i = model(xb, task_name=rt.spec.name)
                    loss_i = task_losses(out_i, {rt.spec.name: yb}, [rt])[rt.spec.name]
                    loss_i.backward()
                    flat_g = flatten_grads([p.grad for p in enc_params], enc_params)
                    raw_norms.append(flat_g.norm(2).detach())
                    raw_losses_gn.append(loss_i.detach())
                raw_norms_tensor = torch.stack(raw_norms)
                losses_tensor = torch.stack(raw_losses_gn)
                if init_losses is None:
                    init_losses = losses_tensor.clamp_min(1e-6)
                weights = len(tasks) * torch.softmax(task_logits, dim=0)
                G = weights * raw_norms_tensor
                loss_ratio = losses_tensor.clamp_min(1e-6) / init_losses
                inv_rate = loss_ratio / loss_ratio.mean()
                target = G.detach().mean() * inv_rate.pow(1.5)
                gradnorm_loss = torch.abs(G - target).sum()
                task_opt.zero_grad(set_to_none=True)
                gradnorm_loss.backward()
                task_opt.step()
                weights_det = (len(tasks) * torch.softmax(task_logits, dim=0)).detach()
                opt.zero_grad(set_to_none=True)
                total_loss_val = 0.0
                for i, (rt, xb, yb) in enumerate(batches):
                    _, out_i = model(xb, task_name=rt.spec.name)
                    loss_i = task_losses(out_i, {rt.spec.name: yb}, [rt])[rt.spec.name]
                    (weights_det[i] * loss_i).backward()
                    total_loss_val += float(loss_i.detach().item())
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                opt.step()
                last_aux = float(gradnorm_loss.detach().item())
                train_losses.append(total_loss_val / max(len(batches), 1))
                continue

            if method == "fairgrad_joint":
                # Memory-efficient FairGrad: sequential per-task backward passes,
                # no retain_graph. Encoder grads are saved per-task; head grads
                # are accumulated across tasks in p.grad.
                enc_params_fg = [p for name, p in model.named_parameters()
                                 if p.requires_grad and name.startswith("encoder.")]
                head_params_fg = [p for name, p in model.named_parameters()
                                  if p.requires_grad and not name.startswith("encoder.")]
                task_enc_grads_fg: List[torch.Tensor] = []
                task_scalars_fg: List[float] = []
                opt.zero_grad(set_to_none=True)
                for rt, xb, yb in batches:
                    _, out_i = model(xb, task_name=rt.spec.name)
                    loss_i = task_losses(out_i, {rt.spec.name: yb}, [rt])[rt.spec.name]
                    loss_i.backward()
                    if enc_params_fg:
                        enc_gi = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                                  for p in enc_params_fg]
                        task_enc_grads_fg.append(flatten_grads(enc_gi, enc_params_fg))
                        for p in enc_params_fg:
                            if p.grad is not None:
                                p.grad.zero_()
                    task_scalars_fg.append(float(loss_i.detach().item()))
                if enc_params_fg and task_enc_grads_fg:
                    grad_stack = torch.stack(task_enc_grads_fg)
                    fair_gram = torch.mm(grad_stack, grad_stack.t())
                    gram_np = fair_gram.detach().cpu().numpy()
                    x_start = np.full(len(task_scalars_fg), 1.0 / max(len(task_scalars_fg), 1), dtype=np.float64)

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
                    assign_flat_grad(enc_params_fg, merged_grad)
                    last_fair_weights = weights_np.copy()
                    last_fair_gtg_trace = float(np.trace(gram_np)) if gram_np.size else float("nan")
                    last_aux = float(np.linalg.norm(weights_np, ord=1))
                else:
                    gram_np = np.zeros((0, 0))
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                opt.step()
                train_losses.append(sum(task_scalars_fg) / max(len(task_scalars_fg), 1))
                continue

            task_loss_pairs: List[Tuple[TaskRuntime, torch.Tensor]] = []
            care_anchor_terms: List[torch.Tensor] = []
            care_stat_terms: List[torch.Tensor] = []
            care_anchor_drift_mses: List[torch.Tensor] = []
            care_anchor_drift_cosines: List[torch.Tensor] = []
            for rt, xb, yb in batches:
                z, out = model(xb, task_name=rt.spec.name)
                task_loss_pairs.append((rt, task_losses(out, {rt.spec.name: yb}, [rt])[rt.spec.name]))
                if method in ALL_CARE_STAGE2_METHODS and anchor_encoder is not None:
                    with torch.no_grad():
                        z_anchor = model.encode_with_encoder(anchor_encoder, xb, task_name=rt.spec.name)
                    if method in CARE_AUTO_STAGE2_METHODS:
                        assert care_auto_basis is not None
                        reuse_term, drift_mse, drift_cosine, reuse_stats = _care_auto_reuse_terms(
                            current_features=z,
                            anchor_features=z_anchor,
                            basis=care_auto_basis,
                            loss_type=care_auto_active_loss_type,
                            anchor_mean=care_auto_anchor_mean,
                            anchor_std=care_auto_anchor_std,
                        )
                        care_anchor_terms.append(reuse_term)
                        care_anchor_drift_mses.append(drift_mse)
                        care_anchor_drift_cosines.append(drift_cosine)
                        if "stat_loss" in reuse_stats:
                            care_stat_terms.append(reuse_stats["stat_loss"])
                            last_stat_loss = float(reuse_stats["stat_loss"].detach().item())
                        last_feature_norm_before = float(reuse_stats["feature_norm_before"].detach().item())
                        last_feature_norm_after = float(reuse_stats["feature_norm_after"].detach().item())
                        last_feature_norm_ratio = float(reuse_stats["feature_norm_ratio"].detach().item())
                    else:
                        care_anchor_terms.append(F.mse_loss(z, z_anchor))
            per_task_losses = [loss for _, loss in task_loss_pairs]
            if not per_task_losses:
                continue

            if method in CARE_AUTO_STAGE2_METHODS:
                if method == "care_v2_tasknorm":
                    normalized_task_losses: List[torch.Tensor] = []
                    for rt, loss_i in task_loss_pairs:
                        ema_prev = source_loss_ema.get(rt.spec.name)
                        ema_next = loss_i.detach() if ema_prev is None else (0.9 * ema_prev + 0.1 * loss_i.detach())
                        source_loss_ema[rt.spec.name] = ema_next
                        normalized_task_losses.append(loss_i / ema_next.clamp_min(1e-6))
                    losses_tensor = torch.stack(normalized_task_losses)
                else:
                    losses_tensor = torch.stack(per_task_losses)
                source_loss = losses_tensor.sum() if task_loss_reduction == "sum" else losses_tensor.mean()
                anchor_loss = torch.stack(care_anchor_terms).mean() if care_anchor_terms else torch.zeros((), device=device)
                stat_loss = torch.stack(care_stat_terms).mean() if care_stat_terms else torch.zeros((), device=device)
                drift_mse = torch.stack(care_anchor_drift_mses).mean() if care_anchor_drift_mses else torch.zeros((), device=device)
                drift_cosine = torch.stack(care_anchor_drift_cosines).mean() if care_anchor_drift_cosines else torch.ones((), device=device)
                lambda_eff = float(care_auto_reuse_lambda)
                if method in {"care_v2", "care_v3"} and float(care_auto_drift_budget) > 0.0:
                    scale = float(anchor_loss.detach().item()) / max(float(care_auto_drift_budget), 1e-8)
                    scale = float(np.clip(scale, float(care_auto_lambda_scale_min), float(care_auto_lambda_scale_max)))
                    lambda_eff *= scale
                loss = source_loss + float(lambda_eff) * anchor_loss + float(care_auto_stat_lambda) * stat_loss
                last_anchor = float(anchor_loss.detach().item())
                last_anchor_drift_mse = float(drift_mse.detach().item())
                last_anchor_drift_cosine = float(drift_cosine.detach().item())
                last_reuse_lambda = float(lambda_eff)
                last_source_loss = float(source_loss.detach().item())
                last_reuse_scale_relative = float((float(lambda_eff) * last_anchor) / max(last_source_loss, 1e-8))
                last_aux = last_anchor
                reuse_history.append(last_anchor)
                source_loss_history.append(last_source_loss)
                reuse_scale_history.append(last_reuse_scale_relative)
            elif method in CARE_STAGE2_METHODS:
                anchor_loss = torch.stack(care_anchor_terms).mean() if care_anchor_terms else torch.zeros((), device=device)
                last_anchor = float(anchor_loss.detach().item())
                last_aux = last_anchor

                final_encoder_grad: Optional[torch.Tensor] = None
                head_grads: Optional[List[Optional[torch.Tensor]]] = None
                encoder_params_named = [(name, p) for name, p in model.named_parameters() if p.requires_grad and name.startswith("encoder.")]
                encoder_params = [p for _, p in encoder_params_named]
                head_params = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("encoder.")]
                if method == "care_direct_v2b" and encoder_params:
                    _, family_total, reg_obj, cls_obj = _care_stage2_family_losses(
                        task_loss_pairs=task_loss_pairs,
                        care_scales=care_scales,
                        care_loss_ema=care_loss_ema,
                        reg_floor_lambda=care_reg_floor_lambda,
                        cls_floor_lambda=care_cls_floor_lambda,
                        device=device,
                    )
                    loss = family_total + care_stage2_anchor_lambda * anchor_loss
                    reg_grads = list(torch.autograd.grad(reg_obj, encoder_params, retain_graph=True, allow_unused=True)) if float(reg_obj.detach().item()) > 0.0 else [None for _ in encoder_params]
                    cls_grads = list(torch.autograd.grad(cls_obj, encoder_params, retain_graph=True, allow_unused=True)) if float(cls_obj.detach().item()) > 0.0 else [None for _ in encoder_params]
                    anchor_grads = list(torch.autograd.grad(care_stage2_anchor_lambda * anchor_loss, encoder_params, retain_graph=True, allow_unused=True))
                    final_grads, last_mask_mean, last_route_ratio = _care_v2b_stage2_encoder_grads(
                        enc_params_named=encoder_params_named,
                        reg_grads=reg_grads,
                        cls_grads=cls_grads,
                        anchor_grads=anchor_grads,
                    )
                    final_encoder_grad = flatten_grads(final_grads, encoder_params)
                    head_grads = list(
                        torch.autograd.grad(
                            loss,
                            head_params,
                            allow_unused=True,
                        )
                    ) if head_params else []
                elif method in CARE_V5_STAGE2_METHODS and encoder_params:
                    care_entries, care_task_loss = _care_stage2_entries(
                        task_loss_pairs=task_loss_pairs,
                        care_scales=care_scales,
                        care_loss_ema=care_v5_loss_ema,
                        care_floor_lambda=care_v5_floor_lambda,
                    )
                    loss = care_task_loss + care_v5_anchor_lambda * anchor_loss
                    task_grads = [
                        list(
                            torch.autograd.grad(
                                float(entry["weight"]) * entry["norm_loss"],
                                encoder_params,
                                retain_graph=True,
                                allow_unused=True,
                            )
                        )
                        for entry in care_entries
                    ]
                    anchor_grads = list(torch.autograd.grad(care_v5_anchor_lambda * anchor_loss, encoder_params, retain_graph=True, allow_unused=True))
                    final_grads, last_mask_mean, last_route_ratio, last_negcos = _care_v3_stage2_encoder_grads(
                        enc_params_named=encoder_params_named,
                        task_grads=task_grads,
                        anchor_grads=anchor_grads,
                    )
                    final_encoder_grad = flatten_grads(final_grads, encoder_params)
                    head_grads = list(
                        torch.autograd.grad(
                            care_task_loss,
                            head_params,
                            allow_unused=True,
                        )
                    ) if head_params else []
                elif method == "care_direct_v3" and encoder_params:
                    care_entries, care_task_loss = _care_stage2_entries(
                        task_loss_pairs=task_loss_pairs,
                        care_scales=care_scales,
                        care_loss_ema=care_loss_ema,
                        care_floor_lambda=care_floor_lambda,
                    )
                    loss = care_task_loss + care_stage2_anchor_lambda * anchor_loss
                    task_grads = [
                        list(
                            torch.autograd.grad(
                                float(entry["weight"]) * entry["norm_loss"],
                                encoder_params,
                                retain_graph=True,
                                allow_unused=True,
                            )
                        )
                        for entry in care_entries
                    ]
                    anchor_grads = list(torch.autograd.grad(care_stage2_anchor_lambda * anchor_loss, encoder_params, retain_graph=True, allow_unused=True))
                    final_grads, last_mask_mean, last_route_ratio, last_negcos = _care_v3_stage2_encoder_grads(
                        enc_params_named=encoder_params_named,
                        task_grads=task_grads,
                        anchor_grads=anchor_grads,
                    )
                    final_encoder_grad = flatten_grads(final_grads, encoder_params)
                    head_grads = list(
                        torch.autograd.grad(
                            loss,
                            head_params,
                            allow_unused=True,
                        )
                    ) if head_params else []
                elif method == "care_direct_v4" and encoder_params:
                    care_entries, care_task_loss = _care_stage2_entries(
                        task_loss_pairs=task_loss_pairs,
                        care_scales=care_scales,
                        care_loss_ema=care_loss_ema,
                        care_floor_lambda=care_floor_lambda,
                    )
                    loss = care_task_loss + care_stage2_anchor_lambda * anchor_loss
                    task_grads = [
                        list(
                            torch.autograd.grad(
                                float(entry["weight"]) * entry["norm_loss"],
                                encoder_params,
                                retain_graph=True,
                                allow_unused=True,
                            )
                        )
                        for entry in care_entries
                    ]
                    anchor_grads = list(torch.autograd.grad(care_stage2_anchor_lambda * anchor_loss, encoder_params, retain_graph=True, allow_unused=True))
                    final_grads, last_mask_mean, last_route_ratio, last_negcos, last_select_ratio = _care_v4_stage2_encoder_grads(
                        enc_params_named=encoder_params_named,
                        task_grads=task_grads,
                        anchor_grads=anchor_grads,
                        candidate_names=care_v4_candidate_names or set(),
                    )
                    final_encoder_grad = flatten_grads(final_grads, encoder_params)
                    head_grads = list(
                        torch.autograd.grad(
                            loss,
                            head_params,
                            allow_unused=True,
                        )
                    ) if head_params else []
                else:
                    care_entries, care_task_loss = _care_stage2_entries(
                        task_loss_pairs=task_loss_pairs,
                        care_scales=care_scales,
                        care_loss_ema=care_loss_ema,
                        care_floor_lambda=care_floor_lambda,
                    )
                    loss = care_task_loss + care_stage2_anchor_lambda * anchor_loss
                if method == "care_direct_v2" and encoder_params and method != "care_direct_v2b" and care_entries:
                    task_grad_stack = torch.stack(
                        [
                            flatten_grads(
                                torch.autograd.grad(
                                    float(entry["weight"]) * entry["norm_loss"],
                                    encoder_params,
                                    retain_graph=True,
                                    allow_unused=True,
                                ),
                                encoder_params,
                            )
                            for entry in care_entries
                        ]
                    )
                    merged_grad = pcgrad_merge(task_grad_stack) if task_grad_stack.shape[0] > 1 else task_grad_stack[0]
                    grad_mask = _care_gradient_agreement_mask(task_grad_stack, care_grad_mask_min, care_grad_mask_power)
                    masked_grad = merged_grad * grad_mask
                    anchor_grad = flatten_grads(
                        torch.autograd.grad(
                            care_stage2_anchor_lambda * anchor_loss,
                            encoder_params,
                            retain_graph=True,
                            allow_unused=True,
                        ),
                        encoder_params,
                    )
                    final_encoder_grad = masked_grad + anchor_grad
                    head_grads = list(
                        torch.autograd.grad(
                            loss,
                            head_params,
                            allow_unused=True,
                        )
                    ) if head_params else []
                    last_mask_mean = float(grad_mask.detach().mean().item())
                    last_negcos = _negative_cosine_fraction(task_grad_stack)
            else:
                losses_tensor = torch.stack(per_task_losses)
                loss = losses_tensor.sum() if task_loss_reduction == "sum" else losses_tensor.mean()
                if method == "l2sp" and float(l2sp_lambda) > 0.0:
                    l2sp_penalty = torch.zeros((), device=device)
                    for name, p in model.named_parameters():
                        if p.requires_grad and name.startswith("encoder.") and name in stage1_encoder_snapshot:
                            l2sp_penalty = l2sp_penalty + (p - stage1_encoder_snapshot[name].to(p.device)).pow(2).sum()
                    loss = loss + float(l2sp_lambda) * l2sp_penalty
            opt.zero_grad(set_to_none=True)
            if method in {"care_direct_v2", "care_direct_v2b", "care_direct_v3", "care_direct_v4", "care_v5a", "care_v5b"} and final_encoder_grad is not None and head_grads is not None:
                for p, g in zip(head_params, head_grads):
                    p.grad = None if g is None else g.detach().clone()
                assign_flat_grad(encoder_params, final_encoder_grad)
            else:
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
            if method == "gradnorm_joint":
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
            elif method in CARE_V5_STAGE2_METHODS:
                extra = f" v5_anchor={last_anchor:.4g} gate={last_mask_mean:.3f} focus={last_route_ratio:.3f} negcos={last_negcos:.3f}"
            elif method in CARE_AUTO_STAGE2_METHODS:
                extra = (
                    f" reuse={last_anchor:.4g} lambda={last_reuse_lambda:.4g}"
                    f" drift_mse={last_anchor_drift_mse:.4g} cos={last_anchor_drift_cosine:.4f}"
                    f" k={care_auto_selected_k}/{care_auto_feature_dim}"
                )
            elif method == "care_direct":
                extra = f" care_anchor={last_anchor:.4g}"
            elif method == "care_direct_v2":
                extra = f" care_anchor={last_anchor:.4g} mask={last_mask_mean:.3f} negcos={last_negcos:.3f}"
            elif method == "care_direct_v2b":
                extra = f" care_anchor={last_anchor:.4g} gate={last_mask_mean:.3f} regmix={last_route_ratio:.3f}"
            elif method == "care_direct_v3":
                extra = f" care_anchor={last_anchor:.4g} gate={last_mask_mean:.3f} focus={last_route_ratio:.3f} negcos={last_negcos:.3f}"
            elif method == "care_direct_v4":
                extra = f" care_anchor={last_anchor:.4g} gate={last_mask_mean:.3f} focus={last_route_ratio:.3f} negcos={last_negcos:.3f} select={last_select_ratio:.3f}"
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
    stage2_encoder_snapshot = _encoder_param_snapshot(model)
    stage2_drift = _encoder_drift_metrics(stage2_encoder_snapshot, stage1_encoder_snapshot)
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
        "reuse_loss_last": float(last_anchor) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "reuse_loss_mean": float(np.mean(reuse_history)) if reuse_history else float("nan"),
        "anchor_feature_drift_mse": float(last_anchor_drift_mse) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "anchor_feature_drift_cosine": float(last_anchor_drift_cosine) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "param_drift_rel": float(stage2_drift.get("rel", float("nan"))),
        "feature_norm_before": float(last_feature_norm_before) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "feature_norm_after": float(last_feature_norm_after) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "feature_norm_ratio": float(last_feature_norm_ratio) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "source_loss_mean": float(np.mean(source_loss_history)) if source_loss_history else float("nan"),
        "reuse_loss_scale_relative_to_source": float(np.mean(reuse_scale_history)) if reuse_scale_history else float("nan"),
        "stat_loss_last": float(last_stat_loss) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "anchor_feature_norm_mean": float(care_auto_anchor_feature_norm_mean) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "trainable_param_count": int(trainable_summary["trainable_param_count"]),
        "encoder_trainable_param_count": int(trainable_summary["encoder_trainable_param_count"]),
        "updated_layer_names": list(trainable_summary["updated_layer_names"]),
        "updated_param_names": list(trainable_summary["updated_param_names"]),
        "reuse_feature_space": str(care_auto_reuse_feature_space) if method in CARE_AUTO_STAGE2_METHODS else "",
        "reuse_k": int(care_auto_selected_k) if method in CARE_AUTO_STAGE2_METHODS else 0,
        "reuse_k_candidates": list(care_auto_candidate_ks) if method in CARE_AUTO_STAGE2_METHODS else [],
        "reuse_lambda": float(care_auto_reuse_lambda) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "reuse_lambda_effective_last": float(last_reuse_lambda) if method in CARE_AUTO_STAGE2_METHODS else float("nan"),
        "anchor_source": str(care_auto_anchor_source) if method in CARE_AUTO_STAGE2_METHODS else "",
        "care_variant": str(method) if method in CARE_AUTO_STAGE2_METHODS else "",
        "train_scope": str(care_auto_stage2_scope) if method in CARE_AUTO_STAGE2_METHODS else "",
        "feature_dim": int(care_auto_feature_dim) if method in CARE_AUTO_STAGE2_METHODS else 0,
        "anchor_basis_batches": int(care_auto_basis_batches) if method in CARE_AUTO_STAGE2_METHODS else 0,
        "anchor_basis_samples": int(care_auto_basis_samples) if method in CARE_AUTO_STAGE2_METHODS else 0,
    }


def _head_dropout(model: MultiTaskModel) -> float:
    head = next(iter(model.heads.values()))
    drop = head.net[1]
    return float(getattr(drop, "p", 0.0))


def _infer_enc_dim(model: MultiTaskModel, sample_x: np.ndarray, task_name: str, device: torch.device) -> int:
    x = torch.as_tensor(sample_x[:2], dtype=torch.float32, device=device)
    with torch.no_grad():
        z = model.encode(x, task_name=task_name)
    return int(z.shape[-1])


def _add_future_task_support(model: MultiTaskModel, task_name: str) -> None:
    if task_name not in model.temporal_task_supports_sec:
        native_sec = float(model.input_len) / float(model.hz) if model.input_len > 0 else 0.0
        model.temporal_task_supports_sec[task_name] = native_sec
    if task_name not in model.temporal_task_support_lens:
        model.temporal_task_support_lens[task_name] = model.input_len


def add_task_head(model: MultiTaskModel, task: TaskRuntime, sample_x: np.ndarray, device: torch.device) -> None:
    if task.spec.name in model.heads:
        return
    _add_future_task_support(model, task.spec.name)
    enc_dim = _infer_enc_dim(model, sample_x, task.spec.name, device)
    head = TaskHead(enc_dim, task.n_out, _head_dropout(model)).to(device)
    model.heads[task.spec.name] = head


def set_stage3_trainable(
    model: MultiTaskModel,
    new_task_name: str,
    stage3_policy: str,
    care_stage3_scope: str = "top1",
    care_v5_stage3_scope: str = "top1",
) -> None:
    if stage3_policy == "lora_ft":
        set_lora_encoder_trainable(model, enabled=True)
    elif stage3_policy in CARE_V5_STAGE3_POLICIES:
        _set_encoder_trainable_names(model, _care_v5_candidate_param_names(model, care_v5_stage3_scope))
    elif stage3_policy == "care_direct_v4":
        set_encoder_trainable_scope(model, "full")
    elif stage3_policy in SCOPE_MATCHED_STAGE3_POLICIES:
        _set_encoder_trainable_names(model, _care_v3_candidate_param_names(model, care_stage3_scope))
    elif stage3_policy == "care_direct_v2b":
        _set_encoder_trainable_names(model, _care_v2b_candidate_param_names(model, care_stage3_scope))
    elif stage3_policy == "care_direct_v3":
        _set_encoder_trainable_names(model, _care_v3_candidate_param_names(model, care_stage3_scope))
    else:
        set_encoder_trainable_scope(model, _stage3_encoder_scope(stage3_policy, care_stage3_scope, care_v5_stage3_scope))
    for name, head in model.heads.items():
        req = name == new_task_name
        for p in head.parameters():
            p.requires_grad = req



# ---------------------------------------------------------------------------
# SGP: gradient projection onto the complement of previous-task gradient bases
# (GPM/SGP family). Replay-free; uses only the current task's data. Stores one
# orthonormal basis per task (not samples). No new encoder parameters.
# ---------------------------------------------------------------------------
_SGP_SCALE = 0.0          # 0.0 = orthogonal projection (GPM); >0 = scaled projection
_SGP_BASIS_SIZE = 32      # max stored directions (memory: size x #encoder params)
_SGP_GRAD_BATCHES = 16    # batches used to build the current task's basis


def _sgp_trainable_encoder_params(model: nn.Module):
    return [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]


def _sgp_flat_grads(params) -> Optional[torch.Tensor]:
    gs = [p.grad.detach().reshape(-1) for _, p in params if p.grad is not None]
    return torch.cat(gs) if gs else None


def _sgp_project_encoder_grads(model: nn.Module, basis: Optional[torch.Tensor], scale: float = 0.0) -> None:
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


def _sgp_append_basis(model: nn.Module, loader, device, task, size: int, n_batches: int) -> None:
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
        params = _sgp_trainable_encoder_params(model)
        g = _sgp_flat_grads(params)
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
    new_task_data: TaskData,
    device: torch.device,
    epochs: int,
    patience: int,
    head_lr: float,
    encoder_lr_mult: float,
    weight_decay: float,
    stage3_policy: str,
    log_prefix: str,
    care_stage3_scope: str = "top1",
    care_stage3_anchor_lambda: float = 0.1,
    care_old_distill_lambda: float = 0.25,
    care_importance_lambda: float = 1e-4,
    care_importance: Optional[Dict[str, torch.Tensor]] = None,
    care_reg_importance: Optional[Dict[str, torch.Tensor]] = None,
    care_cls_importance: Optional[Dict[str, torch.Tensor]] = None,
    care_task_importance: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    care_stage3_reg_distill_lambda: float = 0.35,
    care_stage3_cls_distill_lambda: float = 0.15,
    care_importance_batches: int = 8,
    care_v4_extra_ratio: float = 0.18,
    lwf_old_distill_lambda: float = 0.25,
    task_interfaces: Optional[Dict[str, TaskInterface]] = None,
    care_v5_stage3_scope: str = "top1",
    care_v5_anchor_lambda: float = 0.1,
    care_v5_lwf_lambda: float = 0.25,
    care_v5_importance_lambda: float = 1e-4,
    care_v5_support_anchor_lambda: float = 0.0,
    lpr_buffer_size: int = 256,
    lpr_alpha: float = 1.0,
    lpr_omega0: float = 1.0,
    lpr_refresh_interval: int = 10,
    lpr_lr: float = 1e-3,
    old_task_data: Optional[List[Any]] = None,
    seed: int = 0,
    cpca_core_k: int = 0,
    lwp_lam_old: float = 1.0,
    lwp_lam_dwdp: float = 0.01,
    core_lambda: float = 0.1,
    core_energy: float = 0.95,
    core_mode: str = "refresh",
    core_budget: int = 0,
    ewc_lambda: float = 1000.0,
    cpca_beta: float = 1.0,
) -> Tuple[MultiTaskModel, Dict[str, Dict[str, float]], Dict[str, Any]]:
    model = model.to(device)
    stage2_anchor_model: Optional[MultiTaskModel] = None
    stage2_encoder_ref: Dict[str, torch.Tensor] = {}
    care_importance_dev: Dict[str, torch.Tensor] = {}
    care_reg_importance_dev: Dict[str, torch.Tensor] = {}
    care_cls_importance_dev: Dict[str, torch.Tensor] = {}
    care_task_importance_dev: Dict[str, Dict[str, torch.Tensor]] = {}
    if stage3_policy in DISTILL_STAGE3_POLICIES or stage3_policy in {"lpr_core", "core_task", "sgp_core"}:
        stage2_anchor_model = copy.deepcopy(model).to(device)
        _add_future_task_support(stage2_anchor_model, new_task_data.runtime.spec.name)
        stage2_anchor_model.eval()
        stage2_anchor_model.requires_grad_(False)
    if stage3_policy in CARE_STAGE3_POLICIES:
        stage2_encoder_ref = {
            name: tensor.to(device=device, dtype=torch.float32)
            for name, tensor in _encoder_param_snapshot(stage2_anchor_model).items()
        }
        care_importance_dev = {
            name: tensor.to(device=device, dtype=torch.float32)
            for name, tensor in (care_importance or {}).items()
        }
        care_reg_importance_dev = {
            name: tensor.to(device=device, dtype=torch.float32)
            for name, tensor in (care_reg_importance or {}).items()
        }
        care_cls_importance_dev = {
            name: tensor.to(device=device, dtype=torch.float32)
            for name, tensor in (care_cls_importance or {}).items()
        }
        care_task_importance_dev = {
            task_name: {
                name: tensor.to(device=device, dtype=torch.float32)
                for name, tensor in importance.items()
            }
            for task_name, importance in (care_task_importance or {}).items()
        }
    add_task_head(model, new_task_data.runtime, new_task_data.split_X["train"], device)
    set_stage3_trainable(
        model,
        new_task_data.runtime.spec.name,
        stage3_policy,
        care_stage3_scope=care_stage3_scope,
        care_v5_stage3_scope=care_v5_stage3_scope,
    )
    if stage3_policy == "head_mlp":
        _h = model.heads[new_task_data.runtime.spec.name]
        _net = getattr(_h, "net", None)
        if _net is not None and len(_net) >= 1 and isinstance(_net[0], nn.Linear):
            _din = int(_net[0].in_features); _dout = int(_net[-1].out_features); _hid = max(64, _din // 2)
            _h.net = nn.Sequential(nn.Linear(_din, _hid), nn.GELU(), nn.Linear(_hid, _dout)).to(device)
            for p in _h.net.parameters():
                p.requires_grad = True
    init_regression_biases(model, [new_task_data.runtime])

    train_loaders, val_loaders, test_loaders_new = make_task_loaders_for_device([new_task_data], batch_size=64, device=device)
    if stage3_policy in {"lpr", "lpr_core"}:
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lpr_lr, momentum=0.0, weight_decay=0.0)
    else:
        opt = make_optimizer(model, head_lr, encoder_lr_mult, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    lpr_layers: List[Tuple[str, nn.Module, str]] = []
    lpr_X_list: List[np.ndarray] = []
    lpr_y_list: List[np.ndarray] = []
    lpr_names: List[Any] = []
    lpr_precond: Dict[str, torch.Tensor] = {}
    lpr_step = 0
    if stage3_policy in {"lpr", "lpr_core"}:
        lpr_layers = _lpr_preconditionable_layers(model.encoder)
        lpr_rng = np.random.default_rng(seed)
        lpr_X_list, lpr_y_list, lpr_names = _lpr_build_replay(old_task_data or old_tasks, lpr_buffer_size, lpr_rng)
        lpr_step_rng = np.random.default_rng(int(seed) + 9999)
        lpr_omega = lpr_omega0 / max(1, sum(int(X.shape[0]) for X in lpr_X_list))
        lpr_precond = _lpr_build_preconditioners(model.encoder, lpr_layers, lpr_X_list, lpr_omega, device)

    cpca_basis: Optional[torch.Tensor] = None
    cpca_k = 0
    cpca_basis_info: Dict[str, Any] = {}
    sema_wrapper = None
    sema_rd_lambda = 0.1
    if stage3_policy in {"cpca_core", "cpca_ts", "cpca_ca", "cpca_tsca", "lwf_core", "lwp_core", "lpr_core", "core_dwdp", "core_affinity", "core_affinity_hybrid", "core_task", "lwp_task", "sgp_core"} and stage2_anchor_model is not None:
        _old_td = list(old_task_data) if old_task_data else []
        _old_train = make_task_loaders_for_device(_old_td, batch_size=64, device=device)[0]
        if stage3_policy in {"core_task", "lwp_task"}:
            cpca_basis_info = _old_task_head_basis(model, _old_td, device=device)
        else:
            cpca_basis_info = _build_care_auto_anchor_basis(
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
            energy_threshold=core_energy,
        )
        if stage3_policy in {"cpca_ts", "cpca_tsca"}:
            _k = int(cpca_core_k) if int(cpca_core_k) > 0 else int(cpca_basis_info["reuse_k"])
            cpca_basis_info = _cpca_ts_basis(
                model, stage2_anchor_model.encoder, [td.runtime for td in _old_td], _old_train, device, _k
            )
            print(
                f"    {log_prefix}[cpca_ts] task-sensitive basis k={int(cpca_basis_info['reuse_k'])}/"
                f"{int(cpca_basis_info['sigma'].shape[0])} sigma_max={float(cpca_basis_info['sigma'][0]):.4g} "
                f"g_max={float(cpca_basis_info['sensitivity'].max()):.4g}",
                flush=True,
            )
        cpca_basis = cpca_basis_info["basis"]
        cpca_k = int(cpca_basis_info["reuse_k"])
        if stage3_policy == "lwp_core":
            _prev = getattr(model, "_carried_core", None)
            if _prev is not None:
                _prev = _prev.to(device=device, dtype=torch.float32)
                if core_mode == "fixed":
                    cpca_basis = _prev
                    cpca_k = int(_prev.shape[1])
                elif core_mode in {"fuse_grow", "fuse_budget"}:
                    _cat = torch.cat([_prev, cpca_basis], dim=1)
                    _q = torch.linalg.qr(_cat)[0]
                    if core_mode == "fuse_budget" and int(core_budget) > 0 and _q.shape[1] > int(core_budget):
                        _q = _q[:, :int(core_budget)]
                    cpca_basis = _q.contiguous()
                    cpca_k = int(cpca_basis.shape[1])
            model._carried_core = cpca_basis.detach().clone().cpu()
        print(
            f"    {log_prefix}[{stage3_policy}] core basis k={cpca_k}",
            flush=True,
        )

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
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.parameters():
            if not any(p is hp for hp in _head_params):
                p.requires_grad = False
        _trainable = list(sema_wrapper.parameters()) + [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(_trainable, lr=0.005, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        print(f"    {log_prefix}[sema] adapters={sum(len(l.funcs) for l in sema_wrapper.layers.values())} "
              f"frozen_backbone={sum(1 for p in model.parameters() if not p.requires_grad)}", flush=True)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    new_task = new_task_data.runtime
    old_task_map = {rt.spec.name: rt for rt in old_tasks}
    last_anchor = 0.0
    last_distill = 0.0
    last_importance = 0.0
    last_conflict = 0.0
    last_lambda = 0.0
    last_reg_distill = 0.0
    last_cls_distill = 0.0
    last_reg_priority = 0.0
    last_gate_mean = 0.0
    last_plasticity = 0.0
    last_route_focus = 0.0
    care_v4_candidate_names: Optional[set[str]] = None
    if stage3_policy == "care_direct_v4":
        care_v4_candidate_names, _ = _care_v4_stage3_candidate_names(
            model=model,
            new_task=new_task,
            train_loaders=train_loaders,
            device=device,
            old_task_importance=care_task_importance or {},
            scope=care_stage3_scope,
            max_batches=care_importance_batches,
            extra_ratio=care_v4_extra_ratio,
        )

    if stage3_policy == "sema":
        for p in model.encoder.parameters():
            p.requires_grad = False
    for ep in range(epochs):
        model.train()
        _keep_frozen_encoder_norms_eval(model)
        losses_ep: List[float] = []
        for xb, yb in train_loaders[new_task.spec.name]:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if stage3_policy in {"lpr", "lpr_core"}:
                z, out = model(xb, task_name=new_task.spec.name)
                new_loss = task_losses(out, {new_task.spec.name: yb}, [new_task])[new_task.spec.name]
                replay_loss = _lpr_replay_loss(model, lpr_X_list, lpr_y_list, lpr_names, device, lpr_step_rng, 32)
                if stage3_policy == "lpr_core":
                    with torch.no_grad():
                        z_anchor_lpr, _ = stage2_anchor_model(xb, task_name=new_task.spec.name)
                    core_loss = F.mse_loss(z @ cpca_basis, z_anchor_lpr @ cpca_basis)
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
            anchor_loss = torch.zeros((), device=device)
            support_anchor_loss = torch.zeros((), device=device)
            distill_loss = torch.zeros((), device=device)
            reg_distill_loss = torch.zeros((), device=device)
            cls_distill_loss = torch.zeros((), device=device)
            importance_loss = torch.zeros((), device=device)
            old_task_distill_terms: Dict[str, torch.Tensor] = {}
            if stage3_policy in DISTILL_STAGE3_POLICIES:
                assert stage2_anchor_model is not None
                with torch.no_grad():
                    if stage3_policy == "cpca_core":
                        z_anchor, out_anchor = stage2_anchor_model(xb, task_name=new_task.spec.name)
                    elif stage3_policy in CARE_STAGE3_POLICIES:
                        z_anchor, out_anchor = stage2_anchor_model(xb, task_name=new_task.spec.name)
                    else:
                        z_anchor, out_anchor = stage2_anchor_model(xb, task_name=new_task.spec.name)
                # Anchor loss must be OUTSIDE no_grad so the encoder receives its gradient.
                if stage3_policy in {"cpca_core", "cpca_ts", "cpca_ca", "cpca_tsca", "lwf_core", "lwp_core", "core_task", "lwp_task"}:
                    anchor_loss = F.mse_loss(z @ cpca_basis, z_anchor @ cpca_basis)
                elif stage3_policy in CARE_STAGE3_POLICIES:
                    anchor_loss = F.mse_loss(z, z_anchor)
                distill_terms: List[torch.Tensor] = []
                reg_distill_terms: List[torch.Tensor] = []
                cls_distill_terms: List[torch.Tensor] = []
                for old_name, old_rt in old_task_map.items():
                    if old_name not in out or old_name not in out_anchor:
                        continue
                    iface = (task_interfaces or {}).get(old_name)
                    if stage3_policy == "care_v5b" and iface is not None:
                        term = _task_preserve_distance(iface, out[old_name], out_anchor[old_name])
                        distill_terms.append(term)
                        if iface.output_structure == "class_distribution":
                            cls_distill_terms.append(term)
                        else:
                            reg_distill_terms.append(term)
                    elif old_rt.spec.kind == "cls":
                        term = F.kl_div(
                            F.log_softmax(out[old_name], dim=-1),
                            F.softmax(out_anchor[old_name], dim=-1),
                            reduction="batchmean",
                        )
                        distill_terms.append(term)
                        cls_distill_terms.append(term)
                    else:
                        term = F.mse_loss(out[old_name], out_anchor[old_name])
                        distill_terms.append(term)
                        reg_distill_terms.append(term)
                    old_task_distill_terms[old_name] = term
                distill_loss = torch.stack(distill_terms).mean() if distill_terms else torch.zeros((), device=device)
                reg_distill_loss = torch.stack(reg_distill_terms).mean() if reg_distill_terms else torch.zeros((), device=device)
                cls_distill_loss = torch.stack(cls_distill_terms).mean() if cls_distill_terms else torch.zeros((), device=device)

                if stage3_policy in CARE_STAGE3_POLICIES:
                    importance_terms: List[torch.Tensor] = []
                    for name, p in model.named_parameters():
                        if not p.requires_grad or not name.startswith("encoder."):
                            continue
                        ref = stage2_encoder_ref.get(name)
                        if ref is None:
                            continue
                        diff2 = (p - ref.to(dtype=p.dtype)).pow(2)
                        weight = care_importance_dev.get(name)
                        importance_terms.append((weight.to(dtype=p.dtype) * diff2).mean() if weight is not None else diff2.mean())
                    importance_loss = torch.stack(importance_terms).mean() if importance_terms else torch.zeros((), device=device)
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
                    loss = new_loss + lwf_old_distill_lambda * distill_loss
                elif stage3_policy == "lwf_core":
                    loss = new_loss + lwf_old_distill_lambda * distill_loss + core_lambda * anchor_loss
                elif stage3_policy == "lwp_core":
                    dwdp_loss = _dwdp_loss(z_anchor, z, yb, new_task.spec.kind)
                    loss = new_loss + lwp_lam_old * distill_loss + lwp_lam_dwdp * dwdp_loss + core_lambda * anchor_loss
                    last_anchor = float(dwdp_loss.detach().item())
                elif stage3_policy in CARE_V5_STAGE3_POLICIES:
                    loss = (
                        new_loss
                        + care_v5_anchor_lambda * anchor_loss
                        + care_v5_lwf_lambda * distill_loss
                        + care_v5_importance_lambda * importance_loss
                        + care_v5_support_anchor_lambda * support_anchor_loss
                    )
                elif stage3_policy == "care_direct_v2b":
                    loss = (
                        new_loss
                        + care_stage3_anchor_lambda * anchor_loss
                        + care_stage3_reg_distill_lambda * reg_distill_loss
                        + care_stage3_cls_distill_lambda * cls_distill_loss
                        + care_importance_lambda * importance_loss
                    )
                elif stage3_policy == "care_direct_v4":
                    loss = (
                        new_loss
                        + care_stage3_anchor_lambda * anchor_loss
                        + care_old_distill_lambda * distill_loss
                        + care_importance_lambda * importance_loss
                    )
                elif stage3_policy == "anchor_only":
                    loss = new_loss + care_stage3_anchor_lambda * anchor_loss
                elif stage3_policy in {"cpca_core", "cpca_ts", "core_task"}:
                    loss = new_loss + care_stage3_anchor_lambda * anchor_loss
                elif stage3_policy in {"cpca_ca", "cpca_tsca"}:
                    _params = [p for p in model.parameters() if p.requires_grad]
                    _gn = torch.autograd.grad(new_loss, _params, retain_graph=True, allow_unused=True)
                    _gc = torch.autograd.grad(anchor_loss, _params, retain_graph=True, allow_unused=True)
                    _vn = torch.cat([(t if t is not None else torch.zeros(p.numel(), device=device)).reshape(-1) for t, p in zip(_gn, _params)])
                    _vc = torch.cat([(t if t is not None else torch.zeros(p.numel(), device=device)).reshape(-1) for t, p in zip(_gc, _params)])
                    _c = float((_vn @ _vc) / (_vn.norm() * _vc.norm() + 1e-8))
                    _lam = float(care_stage3_anchor_lambda) * (1.0 + float(cpca_beta) * max(0.0, -_c))
                    loss = new_loss + _lam * anchor_loss
                    last_anchor = float(anchor_loss.detach().item())
                    last_conflict = _c
                    last_lambda = _lam
                elif stage3_policy == "lwp":
                    dwdp_loss = _dwdp_loss(z_anchor, z, yb, new_task.spec.kind)
                    loss = new_loss + lwp_lam_old * distill_loss + lwp_lam_dwdp * dwdp_loss
                    last_anchor = float(dwdp_loss.detach().item())
                elif stage3_policy == "lwp_task":
                    dwdp_loss = _dwdp_loss(z_anchor, z, yb, new_task.spec.kind)
                    loss = new_loss + lwp_lam_old * distill_loss + lwp_lam_dwdp * dwdp_loss + care_stage3_anchor_lambda * anchor_loss
                    last_anchor = float(dwdp_loss.detach().item())
                elif stage3_policy in {"core_dwdp", "core_affinity", "core_affinity_hybrid"}:
                    dwdp_loss = _core_dwdp_loss(
                        z_anchor, z, yb, new_task.spec.kind, cpca_basis,
                        affinity=(stage3_policy in {"core_affinity", "core_affinity_hybrid"}),
                        hybrid=(stage3_policy == "core_affinity_hybrid"),
                    )
                    loss = new_loss + lwp_lam_old * distill_loss + lwp_lam_dwdp * dwdp_loss
                    last_anchor = float(dwdp_loss.detach().item())
                elif stage3_policy == "sema":
                    assert sema_wrapper is not None
                    rd_loss = sema_wrapper.adapter_rd_loss()
                    loss = new_loss + sema_rd_lambda * rd_loss
                    last_anchor = float(rd_loss.detach().item())
                else:
                    loss = (
                        new_loss
                        + care_stage3_anchor_lambda * anchor_loss
                        + care_old_distill_lambda * distill_loss
                        + care_importance_lambda * importance_loss
                    )
                last_anchor = float(anchor_loss.detach().item())
                last_distill = float(distill_loss.detach().item())
                last_importance = float(importance_loss.detach().item())
                last_reg_distill = float(reg_distill_loss.detach().item())
                last_cls_distill = float(cls_distill_loss.detach().item())
            opt.zero_grad(set_to_none=True)
            if stage3_policy in CARE_V5_STAGE3_POLICIES:
                enc_params_named = [(name, p) for name, p in model.named_parameters() if p.requires_grad and name.startswith("encoder.")]
                enc_params = [p for _, p in enc_params_named]
                head_params = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("encoder.")]
                new_grads = list(torch.autograd.grad(new_loss, enc_params, retain_graph=True, allow_unused=True)) if enc_params else []
                preserve_obj = (
                    care_v5_anchor_lambda * anchor_loss
                    + care_v5_lwf_lambda * distill_loss
                    + care_v5_importance_lambda * importance_loss
                    + care_v5_support_anchor_lambda * support_anchor_loss
                )
                preserve_grads = (
                    list(torch.autograd.grad(preserve_obj, enc_params, retain_graph=True, allow_unused=True))
                    if enc_params and preserve_obj.requires_grad
                    else [None] * len(enc_params)
                )
                head_grads = list(torch.autograd.grad(loss, head_params, allow_unused=True)) if head_params else []
                final_grads, last_gate_mean, last_plasticity = _care_v3_stage3_encoder_grads(
                    enc_params_named=enc_params_named,
                    new_grads=new_grads,
                    preserve_grads=preserve_grads,
                    importance=care_importance_dev,
                )
                for p, g in zip(head_params, head_grads):
                    p.grad = None if g is None else g.detach().clone()
                assign_flat_grad(enc_params, flatten_grads(final_grads, enc_params))
            elif stage3_policy == "care_direct_v2b":
                enc_params_named = [(name, p) for name, p in model.named_parameters() if p.requires_grad and name.startswith("encoder.")]
                enc_params = [p for _, p in enc_params_named]
                head_params = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("encoder.")]
                new_grads = list(torch.autograd.grad(new_loss, enc_params, retain_graph=True, allow_unused=True)) if enc_params else []
                reg_preserve_obj = care_stage3_anchor_lambda * anchor_loss + care_stage3_reg_distill_lambda * reg_distill_loss + care_importance_lambda * importance_loss
                cls_preserve_obj = care_stage3_cls_distill_lambda * cls_distill_loss
                reg_preserve_grads = (
                    list(torch.autograd.grad(reg_preserve_obj, enc_params, retain_graph=True, allow_unused=True))
                    if enc_params and reg_preserve_obj.requires_grad
                    else [None] * len(enc_params)
                )
                cls_preserve_grads = (
                    list(torch.autograd.grad(cls_preserve_obj, enc_params, retain_graph=True, allow_unused=True))
                    if enc_params and cls_preserve_obj.requires_grad
                    else [None] * len(enc_params)
                )
                head_grads = list(torch.autograd.grad(loss, head_params, allow_unused=True)) if head_params else []
                final_grads, last_reg_priority = _care_v2b_stage3_encoder_grads(
                    enc_params_named=enc_params_named,
                    new_grads=new_grads,
                    reg_preserve_grads=reg_preserve_grads,
                    cls_preserve_grads=cls_preserve_grads,
                    reg_importance=care_reg_importance_dev,
                    cls_importance=care_cls_importance_dev,
                )
                for p, g in zip(head_params, head_grads):
                    p.grad = None if g is None else g.detach().clone()
                assign_flat_grad(enc_params, flatten_grads(final_grads, enc_params))
            elif stage3_policy == "care_direct_v3":
                enc_params_named = [(name, p) for name, p in model.named_parameters() if p.requires_grad and name.startswith("encoder.")]
                enc_params = [p for _, p in enc_params_named]
                head_params = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("encoder.")]
                new_grads = list(torch.autograd.grad(new_loss, enc_params, retain_graph=True, allow_unused=True)) if enc_params else []
                preserve_obj = care_stage3_anchor_lambda * anchor_loss + care_old_distill_lambda * distill_loss + care_importance_lambda * importance_loss
                preserve_grads = (
                    list(torch.autograd.grad(preserve_obj, enc_params, retain_graph=True, allow_unused=True))
                    if enc_params and preserve_obj.requires_grad
                    else [None] * len(enc_params)
                )
                head_grads = list(torch.autograd.grad(loss, head_params, allow_unused=True)) if head_params else []
                final_grads, last_gate_mean, last_plasticity = _care_v3_stage3_encoder_grads(
                    enc_params_named=enc_params_named,
                    new_grads=new_grads,
                    preserve_grads=preserve_grads,
                    importance=care_importance_dev,
                )
                for p, g in zip(head_params, head_grads):
                    p.grad = None if g is None else g.detach().clone()
                assign_flat_grad(enc_params, flatten_grads(final_grads, enc_params))
            elif stage3_policy == "care_direct_v4":
                enc_params_named = [(name, p) for name, p in model.named_parameters() if p.requires_grad and name.startswith("encoder.")]
                enc_params = [p for _, p in enc_params_named]
                head_params = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("encoder.")]
                new_grads = list(torch.autograd.grad(new_loss, enc_params, retain_graph=True, allow_unused=True)) if enc_params else []
                shared_preserve_obj = care_stage3_anchor_lambda * anchor_loss + care_importance_lambda * importance_loss
                shared_preserve_grads = (
                    list(torch.autograd.grad(shared_preserve_obj, enc_params, retain_graph=True, allow_unused=True))
                    if enc_params and shared_preserve_obj.requires_grad
                    else [None] * len(enc_params)
                )
                old_task_preserve_grads = {
                    old_name: list(
                        torch.autograd.grad(
                            care_old_distill_lambda * term,
                            enc_params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                    )
                    for old_name, term in old_task_distill_terms.items()
                }
                head_grads = list(torch.autograd.grad(loss, head_params, allow_unused=True)) if head_params else []
                final_grads, last_gate_mean, last_plasticity, last_route_focus = _care_v4_stage3_encoder_grads(
                    enc_params_named=enc_params_named,
                    new_grads=new_grads,
                    shared_preserve_grads=shared_preserve_grads,
                    old_task_preserve_grads=old_task_preserve_grads,
                    old_task_importance=care_task_importance_dev,
                    candidate_names=care_v4_candidate_names or set(),
                )
                for p, g in zip(head_params, head_grads):
                    p.grad = None if g is None else g.detach().clone()
                assign_flat_grad(enc_params, flatten_grads(final_grads, enc_params))
            else:
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
            extra = ""
            if stage3_policy == "care_direct":
                extra = f" anchor={last_anchor:.4g} distill={last_distill:.4g} imp={last_importance:.4g}"
            elif stage3_policy in CARE_V5_STAGE3_POLICIES:
                extra = f" v5_anchor={last_anchor:.4g} distill={last_distill:.4g} imp={last_importance:.4g} gate={last_gate_mean:.3f} plastic={last_plasticity:.3f}"
            elif stage3_policy == "care_direct_v2b":
                extra = f" anchor={last_anchor:.4g} regd={last_reg_distill:.4g} clsd={last_cls_distill:.4g} imp={last_importance:.4g} regprio={last_reg_priority:.3f}"
            elif stage3_policy == "care_direct_v3":
                extra = f" anchor={last_anchor:.4g} distill={last_distill:.4g} imp={last_importance:.4g} gate={last_gate_mean:.3f} plastic={last_plasticity:.3f}"
            elif stage3_policy == "care_direct_v4":
                extra = f" anchor={last_anchor:.4g} distill={last_distill:.4g} imp={last_importance:.4g} gate={last_gate_mean:.3f} plastic={last_plasticity:.3f} route={last_route_focus:.3f}"
            elif stage3_policy == "lwf_ft":
                extra = f" distill={last_distill:.4g}"
            elif stage3_policy in {"anchor_only", "cpca_core"}:
                extra = f" anchor={last_anchor:.4g}"
            print(
                f"    {log_prefix}ep{ep:03d} train={np.mean(losses_ep):.4f} "
                f"val_loss={val:.4f} best={best_val:.4f} bad={bad}{extra}",
                flush=True,
            )
        if bad >= patience:
            print(f"    {log_prefix}early-stop @ ep{ep}", flush=True)
            break

    model.load_state_dict(best_state)
    if stage3_policy == "ewc":
        model.eval()
        _F = dict(getattr(model, "_ewc_fisher", {}) or {})
        _nb = 0
        for _i, (_xb, _yb) in enumerate(train_loaders[new_task.spec.name]):
            if _i >= 4:
                break
            _xb = _xb.to(device); _yb = _yb.to(device)
            _, _out = model(_xb, task_name=new_task.spec.name)
            _l = task_losses(_out, {new_task.spec.name: _yb}, [new_task])[new_task.spec.name]
            _gs = torch.autograd.grad(_l, [p for p in model.encoder.parameters()], allow_unused=True)
            for (n, p), gg in zip(model.encoder.named_parameters(), _gs):
                if gg is None:
                    continue
                _F[n] = (_F[n].to(device) if n in _F else 0) + gg.detach() ** 2
            _nb += 1
        model._ewc_fisher = {k: (v / max(1, _nb)).cpu() for k, v in _F.items()}
        model._ewc_params = {n: p.detach().clone().cpu() for n, p in model.encoder.named_parameters()}
        model.train()
    if stage3_policy in {"sgp", "sgp_core"}:
        _sgp_append_basis(model, train_loaders[new_task.spec.name], device, new_task,
                          _SGP_BASIS_SIZE, _SGP_GRAD_BATCHES)
    all_test_loaders = dict(test_loaders_new)
    task_runtimes = list(old_tasks) + [new_task]
    for td in old_tasks:
        pass
    metrics = evaluate(model, all_test_loaders, [new_task], device)
    return model, metrics, {"val_loss_best": float(best_val), "epochs_trained": int(ep + 1)}


def evaluate_all_tasks(model: MultiTaskModel, task_data_all: List[TaskData], device: torch.device, batch_size: int) -> Dict[str, Dict[str, float]]:
    _, _, test_loaders = make_task_loaders_for_device(task_data_all, batch_size, device)
    return evaluate(model, test_loaders, [td.runtime for td in task_data_all], device)


def fit_stage1_single_task_oracle(
    student: str,
    seed: int,
    kd_method: str,
    ckpt_dir: Path,
    task_data: TaskData,
    dataset_name: str,
    temporal_policy: str,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    enc, ck_meta = load_ppg_encoder_init(
        student=student,
        ckpt_dir=ckpt_dir,
        kd_method=kd_method,
        seed=args.stage1_ckpt_seed,
        device=device,
        encoder_init=str(args.encoder_init),
    )
    model = MultiTaskModel(
        enc,
        int(ck_meta["enc_dim"]),
        [task_data.runtime],
        args.dropout,
        dataset_name=dataset_name,
        hz=args.hz,
        input_len=int(task_data.split_X["train"].shape[-1]),
        temporal_policy=temporal_policy,
    )
    model, metrics, _ = fit_stage2(
        model=model,
        method="joint_finetune",
        tasks=[task_data.runtime],
        train_loaders=make_task_loaders_for_device([task_data], args.batch_size, device)[0],
        val_loaders=make_task_loaders_for_device([task_data], args.batch_size, device)[1],
        test_loaders=make_task_loaders_for_device([task_data], args.batch_size, device)[2],
        device=device,
        epochs=args.stage3_epochs,
        patience=args.stage3_patience,
        head_lr=args.stage3_head_lr,
        encoder_lr_mult=args.stage3_encoder_lr_mult,
        weight_decay=args.weight_decay,
        log_prefix=f"[oracle/{student}/s{seed}/{task_data.runtime.spec.name}] ",
    )
    return metrics


def compare_metric(curr: float, ref: float, higher_better: bool) -> float:
    if not np.isfinite(curr) or not np.isfinite(ref) or curr == 0 or ref == 0:
        return float("nan")
    return float(curr / ref) if higher_better else float(ref / curr)


def expected_phase_pairs(initial_tasks: List[str], future_tasks: List[str]) -> set[Tuple[str, str, str]]:
    expected: set[Tuple[str, str, str]] = set()
    for task in initial_tasks:
        expected.add(("stage2", "", task))
    old = list(initial_tasks)
    for fut in future_tasks:
        old_plus_new = old + [fut]
        for task in old_plus_new:
            expected.add(("stage3", fut, task))
        expected.add(("oracle", fut, fut))
        old = old_plus_new
    return expected


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
    got = set(zip(sub["phase"].astype(str), sub["added_task"].fillna("").astype(str), sub["task"].astype(str)))
    return expected_phase_pairs(cfg["initial_tasks"], cfg["future_tasks"]).issubset(got)


def stage2_base_record(pack: str, pack_cfg: Dict[str, Any], student: str, seed: int, method: str, args: argparse.Namespace, ckpt_path: str) -> Dict[str, Any]:
    method_cfg: Dict[str, Any]
    if method in CARE_V5_STAGE2_METHODS or args.stage3_policy in CARE_V5_STAGE3_POLICIES:
        method_cfg = {
            "care_v5_stage2_scope": args.care_v5_stage2_scope,
            "care_v5_stage3_scope": args.care_v5_stage3_scope,
            "care_v5_anchor_lambda": args.care_v5_anchor_lambda,
            "care_v5_lwf_lambda": args.care_v5_lwf_lambda,
            "care_v5_importance_lambda": args.care_v5_importance_lambda,
            "care_v5_support_anchor_lambda": args.care_v5_support_anchor_lambda,
            "care_v5_loss_ema": args.care_v5_loss_ema,
            "care_v5_floor_lambda": args.care_v5_floor_lambda,
            "care_v5_importance_batches": args.care_v5_importance_batches,
        }
    elif method in CARE_STAGE2_METHODS or args.stage3_policy in CARE_STAGE3_POLICIES:
        method_cfg = {
            "care_stage2_scope": args.care_stage2_scope,
            "care_stage3_scope": args.care_stage3_scope,
            "care_stage2_anchor_lambda": args.care_stage2_anchor_lambda,
            "care_floor_lambda": args.care_floor_lambda,
            "care_loss_ema": args.care_loss_ema,
            "care_stage3_anchor_lambda": args.care_stage3_anchor_lambda,
            "care_old_distill_lambda": args.care_old_distill_lambda,
            "care_importance_lambda": args.care_importance_lambda,
            "care_importance_batches": args.care_importance_batches,
            "care_grad_mask_min": args.care_grad_mask_min,
            "care_grad_mask_power": args.care_grad_mask_power,
            "care_reg_floor_lambda": args.care_reg_floor_lambda,
            "care_cls_floor_lambda": args.care_cls_floor_lambda,
            "care_stage3_reg_distill_lambda": args.care_stage3_reg_distill_lambda,
            "care_stage3_cls_distill_lambda": args.care_stage3_cls_distill_lambda,
            "care_v4_extra_ratio": args.care_v4_extra_ratio,
        }
    elif method in SCOPE_MATCHED_STAGE2_METHODS or args.stage3_policy in SCOPE_MATCHED_STAGE3_POLICIES:
        method_cfg = {
            "care_stage2_scope": args.care_stage2_scope,
            "care_stage3_scope": args.care_stage3_scope,
        }
    elif args.stage3_policy == "lwf_ft":
        method_cfg = {
            "lwf_old_distill_lambda": args.lwf_old_distill_lambda,
        }
    elif method == "lora_tune" or args.stage3_policy == "lora_ft":
        method_cfg = {
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        }
    else:
        method_cfg = {}
    return {
        "pack": pack,
        "dataset": pack_cfg["dataset"],
        "student": student,
        "seed": seed,
        "encoder_init": str(args.encoder_init),
        "kd_method": args.kd_method,
        "stage2_method": method,
        "stage3_policy": args.stage3_policy,
        "stage23_baseline": stage23_baseline_label(method, args.stage3_policy, str(args.encoder_init)),
        "stage2_encoder_scope": _stage2_encoder_scope(method, args.care_stage2_scope, args.care_v5_stage2_scope),
        "stage3_encoder_scope": _stage3_encoder_scope(args.stage3_policy, args.care_stage3_scope, args.care_v5_stage3_scope),
        "temporal_policy": pack_cfg["temporal_policy"],
        "input_window_policy": str(pack_cfg.get("input_window_policy", "shared_native")),
        "input_window_sec_map": json.dumps(pack_cfg.get("input_window_sec_map", {}), sort_keys=True),
        "care_config_json": json.dumps(method_cfg, sort_keys=True),
        "initial_tasks": json.dumps(pack_cfg["initial_tasks"]),
        "future_tasks": json.dumps(pack_cfg["future_tasks"]),
        "stage1_ckpt": ckpt_path,
    }


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df.copy()
    if "encoder_init" not in df.columns:
        df["encoder_init"] = "stage1"
    if "stage23_baseline" not in df.columns:
        df["stage23_baseline"] = [
            stage23_baseline_label(str(m), str(p), str(init))
            for m, p, init in zip(
                df["stage2_method"].astype(str),
                df["stage3_policy"].astype(str),
                df["encoder_init"].astype(str),
            )
        ]
    if "input_window_policy" not in df.columns:
        df["input_window_policy"] = "shared_native"
    if "input_window_sec_map" not in df.columns:
        df["input_window_sec_map"] = "{}"
    if "stage2_encoder_scope" not in df.columns:
        df["stage2_encoder_scope"] = np.where(df["stage2_method"].astype(str).eq("frozen_heads"), "none", "full")
    if "stage3_encoder_scope" not in df.columns:
        df["stage3_encoder_scope"] = np.where(df["stage3_policy"].astype(str).eq("new_head_only"), "none", "full")
    if "care_config_json" not in df.columns:
        df["care_config_json"] = "{}"

    summary_rows: List[Dict[str, Any]] = []
    keys = [
        "pack",
        "dataset",
        "student",
        "seed",
        "encoder_init",
        "kd_method",
        "stage2_method",
        "stage3_policy",
        "stage23_baseline",
        "stage2_encoder_scope",
        "stage3_encoder_scope",
        "temporal_policy",
        "input_window_policy",
        "input_window_sec_map",
        "care_config_json",
    ]
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
            rec["encoder_drift_stage1_to_stage2_rel"] = float(stage2["encoder_drift_stage1_to_stage2_rel"].mean()) if "encoder_drift_stage1_to_stage2_rel" in stage2.columns else np.nan
            rec["encoder_drift_stage2_to_stage3_rel"] = float(new_rows["encoder_drift_stage2_to_stage3_rel"].mean()) if "encoder_drift_stage2_to_stage3_rel" in new_rows.columns and not new_rows.empty else np.nan
            rec["encoder_drift_stage1_to_stage3_rel"] = float(new_rows["encoder_drift_stage1_to_stage3_rel"].mean()) if "encoder_drift_stage1_to_stage3_rel" in new_rows.columns and not new_rows.empty else np.nan
            summary_rows.append(rec)

    by_seed = pd.DataFrame(summary_rows)
    if by_seed.empty:
        return by_seed, pd.DataFrame()
    agg = (
        by_seed.groupby(
            [
                "pack",
                "dataset",
                "student",
                "encoder_init",
                "kd_method",
                "stage2_method",
                "stage3_policy",
                "stage23_baseline",
                "stage2_encoder_scope",
                "stage3_encoder_scope",
                "temporal_policy",
                "input_window_policy",
                "input_window_sec_map",
                "care_config_json",
            ],
            dropna=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            old_avg_ret_mean=("old_avg_ret_from_stage2", "mean"),
            old_avg_ret_std=("old_avg_ret_from_stage2", "std"),
            old_floor_ret_mean=("old_floor_ret_from_stage2", "mean"),
            old_floor_ret_std=("old_floor_ret_from_stage2", "std"),
            new_task_rel_mean=("new_task_rel_vs_oracle", "mean"),
            new_task_rel_std=("new_task_rel_vs_oracle", "std"),
            stage2_drift_rel_mean=("encoder_drift_stage1_to_stage2_rel", "mean"),
            stage2_drift_rel_std=("encoder_drift_stage1_to_stage2_rel", "std"),
            stage3_delta_drift_rel_mean=("encoder_drift_stage2_to_stage3_rel", "mean"),
            stage3_delta_drift_rel_std=("encoder_drift_stage2_to_stage3_rel", "std"),
            stage3_total_drift_rel_mean=("encoder_drift_stage1_to_stage3_rel", "mean"),
            stage3_total_drift_rel_std=("encoder_drift_stage1_to_stage3_rel", "std"),
        )
        .reset_index()
    )
    return by_seed, agg


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.stage3_policy == "lora_ft" or "lora_tune" in args.stage2_methods:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        print("[runtime] cuDNN disabled for LoRA stability", flush=True)
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
            existing = pd.read_csv(out_csv)
            if "encoder_init" not in existing.columns:
                existing["encoder_init"] = "stage1"
            print(f"[append:{pack}] loaded {len(existing)} rows", flush=True)
        if args.resume and out_csv.exists():
            by_seed, agg = summarize(existing)
            by_seed.to_csv(by_seed_csv, index=False)
            agg.to_csv(summary_csv, index=False)
            print(agg.to_string(index=False), flush=True)
            continue

        all_rows: List[Dict[str, Any]] = existing.to_dict("records") if not existing.empty else []
        for student in args.students:
            ckpt_dir = DEFAULT_CKPT_DIRS.get(student, ROOT / f"kd_sweep_results/pulseppg_preliminary/{student.lower()}_ckpts")
            for seed in args.seeds:
                initial_task_data, future_task_data = build_pack_task_data(pack, args, seed)
                full_task_data = initial_task_data + future_task_data
                if not existing.empty:
                    done_methods = [
                        m
                        for m in args.stage2_methods
                        if combo_completed(
                            existing[
                                existing["encoder_init"].astype(str).eq(str(args.encoder_init))
                            ],
                            pack,
                            student,
                            seed,
                            m,
                            args.stage3_policy,
                        )
                    ]
                else:
                    done_methods = []
                oracle_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
                for stage2_method in args.stage2_methods:
                    if stage2_method in done_methods:
                        print(f"[skip:{pack}] {student} seed={seed} {stage2_method} [cached]", flush=True)
                        continue
                    if args.stage3_policy == "lora_ft" and stage2_method != "lora_tune":
                        raise ValueError("stage3_policy=lora_ft requires stage2_method=lora_tune")
                    robust_scalar_preserve = stage2_method == "care_v5b" or args.stage3_policy == "care_v5b"
                    initial_task_interfaces = build_task_interfaces(initial_task_data, args.hz, robust_scalar=robust_scalar_preserve)
                    importance_batches = (
                        args.care_v5_importance_batches
                        if stage2_method in CARE_V5_STAGE2_METHODS or args.stage3_policy in CARE_V5_STAGE3_POLICIES
                        else args.care_importance_batches
                    )
                    set_seed(seed)
                    print(
                        f"\n{'='*88}\n"
                        f"[pack={pack}] student={student} seed={seed} stage2={stage2_method}\n"
                        f"dataset={pack_cfg['dataset']} initial={pack_cfg['initial_tasks']} future={pack_cfg['future_tasks']}\n"
                        f"{'='*88}",
                        flush=True,
                    )
                    enc, ck_meta = load_ppg_encoder_init(
                        student=student,
                        ckpt_dir=ckpt_dir,
                        kd_method=args.kd_method,
                        seed=args.stage1_ckpt_seed,
                        device=device,
                        encoder_init=str(args.encoder_init),
                    )
                    input_len = max(int(td.split_X["train"].shape[-1]) for td in initial_task_data)
                    model = MultiTaskModel(
                        enc,
                        int(ck_meta["enc_dim"]),
                        [td.runtime for td in initial_task_data],
                        args.dropout,
                        dataset_name=pack_cfg["dataset"],
                        hz=args.hz,
                        input_len=input_len,
                        temporal_policy=pack_cfg["temporal_policy"],
                    )
                    if stage2_method == "lora_tune":
                        inject_lora_into_encoder(
                            model.encoder,
                            rank=args.lora_rank,
                            alpha=args.lora_alpha,
                            dropout=args.lora_dropout,
                        )
                    stage1_encoder_snapshot = _encoder_param_snapshot(model)
                    train_loaders, val_loaders, test_loaders = make_task_loaders_for_device(initial_task_data, args.batch_size, device)
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
                            log_prefix=f"[{pack}/{student}/s{seed}/{stage2_method}/stage2] ",
                            care_stage2_scope=args.care_stage2_scope,
                            care_stage2_anchor_lambda=args.care_stage2_anchor_lambda,
                            care_floor_lambda=args.care_floor_lambda,
                            care_loss_ema=args.care_loss_ema,
                            care_grad_mask_min=args.care_grad_mask_min,
                            care_grad_mask_power=args.care_grad_mask_power,
                            care_reg_floor_lambda=args.care_reg_floor_lambda,
                            care_cls_floor_lambda=args.care_cls_floor_lambda,
                            care_importance_batches=args.care_importance_batches,
                            care_v4_extra_ratio=args.care_v4_extra_ratio,
                            task_interfaces=initial_task_interfaces,
                            care_v5_stage2_scope=args.care_v5_stage2_scope,
                            care_v5_anchor_lambda=args.care_v5_anchor_lambda,
                            care_v5_floor_lambda=args.care_v5_floor_lambda,
                            care_v5_loss_ema=args.care_v5_loss_ema,
                        )
                        if stage2_cache_path is not None:
                            stage2_cache_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(
                                {"state_dict": model.state_dict(), "stage2_metrics": stage2_metrics, "stage2_meta": stage2_meta},
                                str(stage2_cache_path),
                            )
                            print(f"    [{pack}/{student}/s{seed}/{stage2_method}/stage2] saved Stage-II checkpoint to {stage2_cache_path}", flush=True)
                    base = stage2_base_record(pack, pack_cfg, student, seed, stage2_method, args, ck_meta["ckpt_path"])
                    base["time_s_stage2"] = float(time.time() - t0)
                    base["encoder_params"] = count_params(model.encoder)
                    base["total_params"] = count_params(model)
                    base["trainable_params_stage2"] = count_params(model, trainable_only=True)
                    stage2_encoder_snapshot = _encoder_param_snapshot(model)
                    stage2_drift = _encoder_drift_metrics(stage2_encoder_snapshot, stage1_encoder_snapshot)
                    base["encoder_drift_stage1_to_stage2_l2"] = stage2_drift["l2"]
                    base["encoder_drift_stage1_to_stage2_rmse"] = stage2_drift["rmse"]
                    base["encoder_drift_stage1_to_stage2_rel"] = stage2_drift["rel"]
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
                                "rel_vs_stage2": 1.0,
                                "rel_vs_oracle": np.nan,
                            }
                        )

                    stage2_snapshot = {k: dict(v) for k, v in stage2_metrics.items()}
                    care_importance = (
                        estimate_encoder_importance(
                            model=model,
                            tasks=[td.runtime for td in initial_task_data],
                            train_loaders=train_loaders,
                            device=device,
                            max_batches_per_task=importance_batches,
                        )
                        if stage2_method in CARE_STAGE2_METHODS
                        else None
                    )
                    care_reg_importance = (
                        estimate_encoder_importance(
                            model=model,
                            tasks=[td.runtime for td in initial_task_data if td.runtime.spec.kind == "reg"],
                            train_loaders=train_loaders,
                            device=device,
                            max_batches_per_task=importance_batches,
                        )
                        if stage2_method == "care_direct_v2b" and any(td.runtime.spec.kind == "reg" for td in initial_task_data)
                        else None
                    )
                    care_cls_importance = (
                        estimate_encoder_importance(
                            model=model,
                            tasks=[td.runtime for td in initial_task_data if td.runtime.spec.kind == "cls"],
                            train_loaders=train_loaders,
                            device=device,
                            max_batches_per_task=importance_batches,
                        )
                        if stage2_method == "care_direct_v2b" and any(td.runtime.spec.kind == "cls" for td in initial_task_data)
                        else None
                    )
                    care_task_importance = (
                        {
                            td.runtime.spec.name: estimate_encoder_importance(
                                model=model,
                                tasks=[td.runtime],
                                train_loaders=train_loaders,
                                device=device,
                                max_batches_per_task=importance_batches,
                            )
                            for td in initial_task_data
                        }
                        if stage2_method == "care_direct_v4"
                        else None
                    )
                    current_old = list(initial_task_data)
                    current_model = copy.deepcopy(model).cpu()
                    for added_task_data in future_task_data:
                        new_name = added_task_data.runtime.spec.name
                        stage3_task_data = current_old + [added_task_data]
                        stage3_task_interfaces = build_task_interfaces(stage3_task_data, args.hz, robust_scalar=robust_scalar_preserve)
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
                                    kd_method=args.kd_method,
                                    ckpt_dir=ckpt_dir,
                                    task_data=added_task_data,
                                    dataset_name=pack_cfg["dataset"],
                                    temporal_policy=pack_cfg["temporal_policy"],
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
                                "rel_vs_stage2": np.nan,
                                "rel_vs_oracle": 1.0,
                            }
                        )

                        ext_model = copy.deepcopy(current_model).to(device)
                        stage2_prev_snapshot = _encoder_param_snapshot(ext_model)
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
                            log_prefix=f"[{pack}/{student}/s{seed}/{stage2_method}/add:{new_name}] ",
                            care_stage3_scope=args.care_stage3_scope,
                            care_stage3_anchor_lambda=args.care_stage3_anchor_lambda,
                            care_old_distill_lambda=args.care_old_distill_lambda,
                            care_importance_lambda=args.care_importance_lambda,
                            care_importance=care_importance,
                            care_reg_importance=care_reg_importance,
                            care_cls_importance=care_cls_importance,
                            care_task_importance=care_task_importance,
                            care_stage3_reg_distill_lambda=args.care_stage3_reg_distill_lambda,
                            care_stage3_cls_distill_lambda=args.care_stage3_cls_distill_lambda,
                            care_importance_batches=args.care_importance_batches,
                            care_v4_extra_ratio=args.care_v4_extra_ratio,
                            lwf_old_distill_lambda=args.lwf_old_distill_lambda,
                            task_interfaces=stage3_task_interfaces,
                            care_v5_stage3_scope=args.care_v5_stage3_scope,
                            care_v5_anchor_lambda=args.care_v5_anchor_lambda,
                            care_v5_lwf_lambda=args.care_v5_lwf_lambda,
                            care_v5_importance_lambda=args.care_v5_importance_lambda,
                            care_v5_support_anchor_lambda=args.care_v5_support_anchor_lambda,
                            lpr_buffer_size=args.lpr_buffer_size,
                            lpr_alpha=args.lpr_alpha,
                            lpr_omega0=args.lpr_omega0,
                            lpr_refresh_interval=args.lpr_refresh_interval,
                            lpr_lr=args.lpr_lr,
                            old_task_data=current_old,
                            seed=seed,
                            cpca_core_k=args.cpca_core_k,
                            lwp_lam_old=args.lwp_lam_old,
                            lwp_lam_dwdp=args.lwp_lam_dwdp,
                            core_lambda=args.core_lambda,
                            core_energy=args.core_energy,
                            core_mode=args.core_mode,
                            core_budget=args.core_budget,
                            ewc_lambda=args.ewc_lambda,
                            cpca_beta=args.cpca_beta,
                        )
                        current_model = copy.deepcopy(ext_model).cpu()
                        stage3_encoder_snapshot = _encoder_param_snapshot(ext_model)
                        stage3_delta = _encoder_drift_metrics(stage3_encoder_snapshot, stage2_prev_snapshot)
                        stage3_total = _encoder_drift_metrics(stage3_encoder_snapshot, stage1_encoder_snapshot)
                        eval_task_data = stage3_task_data
                        stage3_metrics = evaluate_all_tasks(ext_model, eval_task_data, device, args.batch_size)
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
                                    "rel_vs_stage2": rel_stage2,
                                    "rel_vs_oracle": rel_oracle,
                                    "time_s_stage3": float(time.time() - t1),
                                    "trainable_params_stage3": count_params(ext_model, trainable_only=True),
                                    "encoder_drift_stage2_to_stage3_l2": stage3_delta["l2"],
                                    "encoder_drift_stage2_to_stage3_rmse": stage3_delta["rmse"],
                                    "encoder_drift_stage2_to_stage3_rel": stage3_delta["rel"],
                                    "encoder_drift_stage1_to_stage3_l2": stage3_total["l2"],
                                    "encoder_drift_stage1_to_stage3_rmse": stage3_total["rmse"],
                                    "encoder_drift_stage1_to_stage3_rel": stage3_total["rel"],
                                }
                            )
                        stage2_snapshot.update({k: dict(v) for k, v in stage3_metrics.items()})
                        current_old = current_old + [added_task_data]

                    df = pd.DataFrame(all_rows)
                    df.to_csv(out_csv, index=False)
                    by_seed, agg = summarize(df)
                    by_seed.to_csv(by_seed_csv, index=False)
                    agg.to_csv(summary_csv, index=False)
                    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

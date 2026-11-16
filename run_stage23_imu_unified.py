#!/usr/bin/env python3
"""Unified Stage-2 -> Stage-3 evaluation for compressed IMU students.

The IMU first pass mirrors the PPG/ECG unified protocol:

1. start from a Stage-1 KD-compressed NormWear student,
2. fit a multi-task Stage-2 model on a same-dataset task pack,
3. add one held-out future task with Stage 3,
4. measure both new-task learnability and old-task forgetting.

The first IMU loop should cover more than repeated phone activity
classification. The current runner therefore supports:

- `mhealth_core`
- `opportunity_interaction_core`
- `wisdm_core`
- `motionsense_core`
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from imu_stage1_data import PAMAP2_CORE12_IDS, PAMAP2_CORE12_NAMES, IMUBundle, build_eval_bundles
from normwear_kd_experiment import NormWearStudentEncoder
from run_stage23_ppg_unified import (
    compare_metric,
    evaluate_all_tasks,
    fit_new_task_extension,
    fit_stage2,
    summarize,
)
from run_stage2_multitask_ppg import (
    MultiTaskModel,
    TaskData,
    TaskRuntime,
    TaskSpec,
    count_params,
    make_task_loaders,
    resolve_device,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "kd_sweep_results/stage23_imu_unified"
DEFAULT_CACHE_DIR = ROOT / "kd_sweep_results/imu_preliminary/cache"
DEFAULT_CKPT_DIRS = {
    "imu_tiny": ROOT / "kd_sweep_results/imu_preliminary/exp_normwear_imu_stage1_imu_tiny",
    "imu2m": ROOT / "kd_sweep_results/imu_preliminary/exp_normwear_imu_stage1_pilot",
    "imu3m": ROOT / "kd_sweep_results/imu_preliminary/exp_normwear_imu_stage1_imu3m",
}
STAGE2_METHODS = ("frozen_heads", "joint_finetune", "gradnorm_joint", "pcgrad_joint", "scope_matched_naive", "care_v1")
STAGE3_POLICIES = ("naive_ft", "new_head_only", "scope_matched_naive", "lwf_ft", "anchor_only", "lpr", "cpca_core", "lwp", "cpca_ts", "cpca_ca", "cpca_tsca", "sema", "lwf_core", "lwp_core", "lpr_core", "core_dwdp", "core_affinity", "core_affinity_hybrid", "core_task", "lwp_task", "head_mlp", "ewc", "sgp", "sgp_core")

WISDM_GROUP6_ORDER = [
    "locomotion",
    "stairs",
    "static",
    "eat_drink",
    "sports",
    "upper_body",
]
WISDM_18_TO_GROUP6 = {
    "walking": "locomotion",
    "jogging": "locomotion",
    "stairs": "stairs",
    "sitting": "static",
    "standing": "static",
    "soup": "eat_drink",
    "chips": "eat_drink",
    "pasta": "eat_drink",
    "drinking": "eat_drink",
    "sandwich": "eat_drink",
    "kicking": "sports",
    "catch": "sports",
    "dribbling": "sports",
    "typing": "upper_body",
    "teeth": "upper_body",
    "writing": "upper_body",
    "clapping": "upper_body",
    "folding": "upper_body",
}
WISDM_DYNAMIC_POS = {
    "walking",
    "jogging",
    "stairs",
    "kicking",
    "catch",
    "dribbling",
}
WISDM_UPPER_BODY_POS = {
    "typing",
    "teeth",
    "writing",
    "clapping",
    "folding",
}

MOTIONSENSE_GROUP3_ORDER = ["locomotion", "jogging", "static"]
MOTIONSENSE_6_TO_GROUP3 = {
    "downstairs": "locomotion",
    "upstairs": "locomotion",
    "walking": "locomotion",
    "jogging": "jogging",
    "sitting": "static",
    "standing": "static",
}
MOTIONSENSE_DYNAMIC_POS = {"downstairs", "upstairs", "walking", "jogging"}
MOTIONSENSE_STAIRS_POS = {"downstairs", "upstairs"}

MHEALTH_SEMANTIC3_ORDER = ["static", "locomotion", "exercise"]
MHEALTH_12_TO_SEMANTIC3 = {
    "standing_still": "static",
    "sitting_relaxing": "static",
    "lying_down": "static",
    "walking": "locomotion",
    "climbing_stairs": "locomotion",
    "jogging": "locomotion",
    "running": "locomotion",
    "waist_bends_forward": "exercise",
    "frontal_arm_elevation": "exercise",
    "knees_bending": "exercise",
    "cycling": "exercise",
    "jump_front_back": "exercise",
}
MHEALTH_AMBULATION_POS = {"walking", "climbing_stairs", "jogging", "running"}
MHEALTH_VIGOROUS_POS = {"cycling", "jogging", "running", "jump_front_back"}

OPPORTUNITY_OBJECT4_ORDER = ["appliance", "drawer", "door", "misc"]
OPPORTUNITY_GESTURE_TO_OBJECT4 = {
    "close_dishwasher": "appliance",
    "close_fridge": "appliance",
    "open_dishwasher": "appliance",
    "open_fridge": "appliance",
    "close_drawer_1": "drawer",
    "close_drawer_2": "drawer",
    "close_drawer_3": "drawer",
    "open_drawer_1": "drawer",
    "open_drawer_2": "drawer",
    "open_drawer_3": "drawer",
    "close_door_1": "door",
    "close_door_2": "door",
    "open_door_1": "door",
    "open_door_2": "door",
    "toggle_switch": "misc",
    "drink_from_cup": "misc",
    "clean_table": "misc",
}
OPPORTUNITY_OPEN_CLOSE_POS = {
    "close_dishwasher",
    "close_drawer_3",
    "close_drawer_2",
    "close_door_1",
    "close_door_2",
    "close_drawer_1",
    "close_fridge",
    "open_dishwasher",
    "open_drawer_3",
    "open_drawer_2",
    "open_door_1",
    "open_door_2",
    "open_drawer_1",
    "open_fridge",
}
OPPORTUNITY_DRINK_CLEAN_POS = {"drink_from_cup", "clean_table"}

PAMAP2_LABEL_NAMES = {
    f"activity_{int(activity_id)}": str(PAMAP2_CORE12_NAMES[int(activity_id)])
    for activity_id in PAMAP2_CORE12_IDS
}
PAMAP2_POSTURE3_ORDER = ["lying", "sitting", "standing"]
PAMAP2_12_TO_POSTURE3 = {
    label_key: {
        "lying": "lying",
        "sitting": "sitting",
        "standing": "standing",
        "walking": "standing",
        "running": "standing",
        "cycling": "sitting",
        "nordic_walking": "standing",
        "ascending_stairs": "standing",
        "descending_stairs": "standing",
        "vacuum_cleaning": "standing",
        "ironing": "standing",
        "rope_jumping": "standing",
    }[activity_name]
    for label_key, activity_name in PAMAP2_LABEL_NAMES.items()
}
PAMAP2_STAIRS_POS = {
    label_key for label_key, activity_name in PAMAP2_LABEL_NAMES.items() if activity_name in {"ascending_stairs", "descending_stairs"}
}
PAMAP2_HOUSEWORK_POS = {
    label_key for label_key, activity_name in PAMAP2_LABEL_NAMES.items() if activity_name in {"vacuum_cleaning", "ironing"}
}
PAMAP2_COARSE4_ORDER = ["static", "ambulation", "stairs", "housework"]
PAMAP2_12_TO_COARSE4 = {
    label_key: {
        "lying": "static",
        "sitting": "static",
        "standing": "static",
        "walking": "ambulation",
        "running": "ambulation",
        "cycling": "ambulation",
        "nordic_walking": "ambulation",
        "ascending_stairs": "stairs",
        "descending_stairs": "stairs",
        "vacuum_cleaning": "housework",
        "ironing": "housework",
        "rope_jumping": "ambulation",
    }[activity_name]
    for label_key, activity_name in PAMAP2_LABEL_NAMES.items()
}
PAMAP2_LOCOMOTION_POS = {
    label_key
    for label_key, activity_name in PAMAP2_LABEL_NAMES.items()
    if activity_name in {
        "walking",
        "running",
        "cycling",
        "nordic_walking",
        "ascending_stairs",
        "descending_stairs",
        "rope_jumping",
    }
}

PACKS: Dict[str, Dict[str, Any]] = {
    "mhealth_core": {
        "dataset": "mhealth",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "activity12": 6.0,
            "semantic3": 3.0,
            "ambulation_bin": 4.0,
            "vigorous_bin": 3.0,
        },
        "bundle_task_id": "mhealth_activity12",
        "bundle_family": "mhealth",
        "initial_tasks": ["activity12", "semantic3", "ambulation_bin"],
        "future_tasks": ["vigorous_bin"],
        "description": "Task-specific IMU raw windows: 6s fine activity, 3s coarse semantics/vigorous, and 4s ambulation detection.",
    },
    "opportunity_interaction_core": {
        "dataset": "opportunity",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "gesture17": 3.0,
            "object4": 2.5,
            "open_close_bin": 2.0,
            "drink_clean_bin": 4.0,
        },
        "bundle_task_id": "opportunity_gesture17",
        "bundle_family": "opportunity_gesture",
        "initial_tasks": ["gesture17", "object4", "open_close_bin"],
        "future_tasks": ["drink_clean_bin"],
        "description": "Task-specific IMU raw windows: 3s gesture, 2.5s object family, 2s open/close, and 4s drink/clean interactions.",
    },
    "opportunity_to_mhealth_hr": {
        "dataset": "opportunity_to_mhealth",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "opportunity_locomotion4": 3.0,
            "opportunity_gesture17": 3.0,
            "mhealth_hr_reg": 5.0,
        },
        "split_per_task": True,
        "task_defs": {
            "opportunity_locomotion4": {
                "bundle_task_id": "opportunity_locomotion4",
                "bundle_family": "opportunity_locomotion",
                "label_task": "locomotion4",
            },
            "opportunity_gesture17": {
                "bundle_task_id": "opportunity_gesture17",
                "bundle_family": "opportunity_gesture",
                "label_task": "gesture17",
            },
            "mhealth_hr_reg": {
                "bundle_task_id": "mhealth_hr",
                "bundle_family": "mhealth",
                "label_task": "hr_regression",
            },
        },
        "initial_tasks": ["opportunity_locomotion4", "opportunity_gesture17"],
        "future_tasks": ["mhealth_hr_reg"],
        "description": "Cross-modal Stage-III IMU setting (paper): adapt on the two Opportunity tasks (locomotion4 + gesture17), then add MHEALTH IMU-only HR regression.",
    },
    "opportunity_to_mhealth_two": {
        "dataset": "opportunity_to_mhealth",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "opportunity_locomotion4": 3.0,
            "opportunity_gesture17": 3.0,
            "mhealth_hr_reg": 5.0,
            "mhealth_activity12": 6.0,
        },
        "split_per_task": True,
        "task_defs": {
            "opportunity_locomotion4": {
                "bundle_task_id": "opportunity_locomotion4",
                "bundle_family": "opportunity_locomotion",
                "label_task": "locomotion4",
            },
            "opportunity_gesture17": {
                "bundle_task_id": "opportunity_gesture17",
                "bundle_family": "opportunity_gesture",
                "label_task": "gesture17",
            },
            "mhealth_hr_reg": {
                "bundle_task_id": "mhealth_hr",
                "bundle_family": "mhealth",
                "label_task": "hr_regression",
            },
            "mhealth_activity12": {
                "bundle_task_id": "mhealth_activity12",
                "bundle_family": "mhealth",
                "label_task": "activity12",
            },
        },
        "initial_tasks": ["opportunity_locomotion4", "opportunity_gesture17"],
        "future_tasks": ["mhealth_hr_reg", "mhealth_activity12"],
        "description": "Two-step Stage-III IMU: adapt on Opportunity locomotion4 + gesture17, add MHEALTH IMU-only HR regression (step 1), then add MHEALTH 12-class activity (step 2).",
    },
    "opportunity_to_mhealth_two_seq": {
        "dataset": "opportunity_to_mhealth",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "opportunity_locomotion4": 3.0,
            "opportunity_gesture17": 3.0,
            "mhealth_hr_reg": 5.0,
            "mhealth_activity12": 6.0,
        },
        "split_per_task": True,
        "task_defs": {
            "opportunity_locomotion4": {
                "bundle_task_id": "opportunity_locomotion4",
                "bundle_family": "opportunity_locomotion",
                "label_task": "locomotion4",
            },
            "opportunity_gesture17": {
                "bundle_task_id": "opportunity_gesture17",
                "bundle_family": "opportunity_gesture",
                "label_task": "gesture17",
            },
            "mhealth_hr_reg": {
                "bundle_task_id": "mhealth_hr",
                "bundle_family": "mhealth",
                "label_task": "hr_regression",
            },
            "mhealth_activity12": {
                "bundle_task_id": "mhealth_activity12",
                "bundle_family": "mhealth",
                "label_task": "activity12",
            },
        },
        "initial_tasks": ["opportunity_locomotion4"],
        "future_tasks": ["opportunity_gesture17", "mhealth_hr_reg", "mhealth_activity12"],
        "description": "Pipeline A (pure CL) IMU: one initial task, then sequentially add opportunity_gesture17, mhealth_hr_reg, mhealth_activity12.",
    },
    "wisdm_core": {
        "dataset": "wisdm",
        "bundle_task_id": "wisdm_phone_activity18",
        "bundle_family": "wisdm",
        "initial_tasks": ["activity18", "activity6", "dynamic_bin"],
        "future_tasks": ["upper_body_bin"],
        "description": "Temporal-support supplement: fine-grained, grouped, and movement-aware WISDM pack; then add an upper-body specialization task.",
    },
    "motionsense_core": {
        "dataset": "motionsense",
        "bundle_task_id": "motionsense_activity6",
        "bundle_family": "motionsense",
        "initial_tasks": ["activity6", "activity3", "dynamic_bin"],
        "future_tasks": ["stairs_bin"],
        "description": "Temporal-support supplement: clean motion pack with grouped and movement tasks; then add stairs specialization.",
    },
    "pamap2_hand_to_ankle_coarse4": {
        "dataset": "pamap2_hand_to_ankle",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "hand_posture3": 5.0,
            "hand_stairs_bin": 5.0,
            "hand_housework_bin": 5.0,
            "ankle_coarse4": 5.0,
        },
        "split_bundle_task_id": "pamap2_hand_activity12",
        "split_window_sec": 5.0,
        "task_defs": {
            "hand_posture3": {
                "bundle_task_id": "pamap2_hand_activity12",
                "bundle_family": "pamap2",
                "label_task": "posture3",
            },
            "hand_stairs_bin": {
                "bundle_task_id": "pamap2_hand_activity12",
                "bundle_family": "pamap2",
                "label_task": "stairs_bin",
            },
            "hand_housework_bin": {
                "bundle_task_id": "pamap2_hand_activity12",
                "bundle_family": "pamap2",
                "label_task": "housework_bin",
            },
            "ankle_coarse4": {
                "bundle_task_id": "pamap2_ankle_activity12",
                "bundle_family": "pamap2",
                "label_task": "coarse4",
            },
        },
        "initial_tasks": ["hand_posture3", "hand_stairs_bin", "hand_housework_bin"],
        "future_tasks": ["ankle_coarse4"],
        "description": "Primary shifted IMU candidate: learn source hand posture, stairs, and housework labels, then probe ankle coarse4 under a strong placement shift.",
    },
    "pamap2_chest_to_ankle_coarse4": {
        "dataset": "pamap2_chest_to_ankle",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "chest_posture3": 5.0,
            "chest_stairs_bin": 5.0,
            "chest_housework_bin": 5.0,
            "ankle_coarse4": 5.0,
        },
        "split_bundle_task_id": "pamap2_chest_activity12",
        "split_window_sec": 5.0,
        "task_defs": {
            "chest_posture3": {
                "bundle_task_id": "pamap2_chest_activity12",
                "bundle_family": "pamap2",
                "label_task": "posture3",
            },
            "chest_stairs_bin": {
                "bundle_task_id": "pamap2_chest_activity12",
                "bundle_family": "pamap2",
                "label_task": "stairs_bin",
            },
            "chest_housework_bin": {
                "bundle_task_id": "pamap2_chest_activity12",
                "bundle_family": "pamap2",
                "label_task": "housework_bin",
            },
            "ankle_coarse4": {
                "bundle_task_id": "pamap2_ankle_activity12",
                "bundle_family": "pamap2",
                "label_task": "coarse4",
            },
        },
        "initial_tasks": ["chest_posture3", "chest_stairs_bin", "chest_housework_bin"],
        "future_tasks": ["ankle_coarse4"],
        "description": "Secondary shifted IMU candidate: same coarse transfer protocol with chest as the source placement and ankle as the probe placement.",
    },
    "pamap2_hand_activity_to_ankle_hr_probe": {
        "dataset": "pamap2_hand_activity_to_ankle_hr",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "hand_coarse4": 5.0,
            "hand_locomotion_bin": 5.0,
            "ankle_hr_regression": 5.0,
        },
        "split_bundle_task_id": "pamap2_hand_activity12",
        "split_window_sec": 5.0,
        "task_defs": {
            "hand_coarse4": {
                "bundle_task_id": "pamap2_hand_activity12",
                "bundle_family": "pamap2",
                "label_task": "coarse4",
            },
            "hand_locomotion_bin": {
                "bundle_task_id": "pamap2_hand_activity12",
                "bundle_family": "pamap2",
                "label_task": "locomotion_bin",
            },
            "ankle_hr_regression": {
                "bundle_task_id": "pamap2_ankle_hr",
                "bundle_family": "pamap2",
                "label_task": "hr_regression",
            },
        },
        "initial_tasks": ["hand_coarse4", "hand_locomotion_bin"],
        "future_tasks": ["ankle_hr_regression"],
        "description": "Priority-A PAMAP2 stress candidate: source hand activity-family classification, then probe ankle IMU-only heart-rate regression.",
    },
    "pamap2_chest_activity_to_ankle_hr_probe": {
        "dataset": "pamap2_chest_activity_to_ankle_hr",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "chest_coarse4": 5.0,
            "chest_locomotion_bin": 5.0,
            "ankle_hr_regression": 5.0,
        },
        "split_bundle_task_id": "pamap2_chest_activity12",
        "split_window_sec": 5.0,
        "task_defs": {
            "chest_coarse4": {
                "bundle_task_id": "pamap2_chest_activity12",
                "bundle_family": "pamap2",
                "label_task": "coarse4",
            },
            "chest_locomotion_bin": {
                "bundle_task_id": "pamap2_chest_activity12",
                "bundle_family": "pamap2",
                "label_task": "locomotion_bin",
            },
            "ankle_hr_regression": {
                "bundle_task_id": "pamap2_ankle_hr",
                "bundle_family": "pamap2",
                "label_task": "hr_regression",
            },
        },
        "initial_tasks": ["chest_coarse4", "chest_locomotion_bin"],
        "future_tasks": ["ankle_hr_regression"],
        "description": "Priority-A PAMAP2 stress candidate: source chest activity-family classification, then probe ankle IMU-only heart-rate regression.",
    },
    "pamap2_hand_hr_to_ankle_activity_probe": {
        "dataset": "pamap2_hand_hr_to_ankle_activity",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "hand_hr_regression": 5.0,
            "ankle_commonk_fine": 5.0,
        },
        "split_bundle_task_id": "pamap2_hand_hr",
        "split_window_sec": 5.0,
        "task_defs": {
            "hand_hr_regression": {
                "bundle_task_id": "pamap2_hand_hr",
                "bundle_family": "pamap2",
                "label_task": "hr_regression",
            },
            "ankle_commonk_fine": {
                "bundle_task_id": "pamap2_ankle_activity12",
                "bundle_family": "pamap2",
                "label_task": "commonk_fine",
            },
        },
        "initial_tasks": ["hand_hr_regression"],
        "future_tasks": ["ankle_commonk_fine"],
        "description": "Priority-B PAMAP2 stress candidate: source hand IMU-only heart-rate regression, then probe ankle fine activity classification.",
    },
    "pamap2_chest_hr_to_ankle_activity_probe": {
        "dataset": "pamap2_chest_hr_to_ankle_activity",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "chest_hr_regression": 5.0,
            "ankle_commonk_fine": 5.0,
        },
        "split_bundle_task_id": "pamap2_chest_hr",
        "split_window_sec": 5.0,
        "task_defs": {
            "chest_hr_regression": {
                "bundle_task_id": "pamap2_chest_hr",
                "bundle_family": "pamap2",
                "label_task": "hr_regression",
            },
            "ankle_commonk_fine": {
                "bundle_task_id": "pamap2_ankle_activity12",
                "bundle_family": "pamap2",
                "label_task": "commonk_fine",
            },
        },
        "initial_tasks": ["chest_hr_regression"],
        "future_tasks": ["ankle_commonk_fine"],
        "description": "Priority-B PAMAP2 stress candidate: source chest IMU-only heart-rate regression, then probe ankle fine activity classification.",
    },
    "pamap2_hand_to_ankle_commonk_fine": {
        "dataset": "pamap2_hand_to_ankle_commonk_fine",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "hand_coarse4": 5.0,
            "hand_locomotion_bin": 5.0,
            "ankle_commonk_fine": 5.0,
        },
        "split_bundle_task_id": "pamap2_hand_activity12",
        "split_window_sec": 5.0,
        "task_defs": {
            "hand_coarse4": {
                "bundle_task_id": "pamap2_hand_activity12",
                "bundle_family": "pamap2",
                "label_task": "coarse4",
            },
            "hand_locomotion_bin": {
                "bundle_task_id": "pamap2_hand_activity12",
                "bundle_family": "pamap2",
                "label_task": "locomotion_bin",
            },
            "ankle_commonk_fine": {
                "bundle_task_id": "pamap2_ankle_activity12",
                "bundle_family": "pamap2",
                "label_task": "commonk_fine",
            },
        },
        "initial_tasks": ["hand_coarse4", "hand_locomotion_bin"],
        "future_tasks": ["ankle_commonk_fine"],
        "description": "Priority-C PAMAP2 placement plus task-family shift: source hand coarse/binary activity labels, then probe ankle fine activity classification.",
    },
    "pamap2_chest_to_ankle_commonk_fine": {
        "dataset": "pamap2_chest_to_ankle_commonk_fine",
        "temporal_policy": "shared_window",
        "input_window_policy": "task_specific_input",
        "input_window_sec_map": {
            "chest_coarse4": 5.0,
            "chest_locomotion_bin": 5.0,
            "ankle_commonk_fine": 5.0,
        },
        "split_bundle_task_id": "pamap2_chest_activity12",
        "split_window_sec": 5.0,
        "task_defs": {
            "chest_coarse4": {
                "bundle_task_id": "pamap2_chest_activity12",
                "bundle_family": "pamap2",
                "label_task": "coarse4",
            },
            "chest_locomotion_bin": {
                "bundle_task_id": "pamap2_chest_activity12",
                "bundle_family": "pamap2",
                "label_task": "locomotion_bin",
            },
            "ankle_commonk_fine": {
                "bundle_task_id": "pamap2_ankle_activity12",
                "bundle_family": "pamap2",
                "label_task": "commonk_fine",
            },
        },
        "initial_tasks": ["chest_coarse4", "chest_locomotion_bin"],
        "future_tasks": ["ankle_commonk_fine"],
        "description": "Priority-C PAMAP2 placement plus task-family shift: source chest coarse/binary activity labels, then probe ankle fine activity classification.",
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--packs",
        nargs="+",
        default=["mhealth_core", "opportunity_interaction_core"],
        choices=sorted(PACKS),
    )
    ap.add_argument("--students", nargs="+", default=["imu2m"], choices=sorted(DEFAULT_CKPT_DIRS))
    ap.add_argument("--kd-method", default="ns_kd_lam0.1")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument(
        "--stage2-methods",
        nargs="+",
        default=["frozen_heads", "joint_finetune", "gradnorm_joint", "pcgrad_joint"],
        choices=STAGE2_METHODS,
    )
    ap.add_argument("--stage3-policy", default="naive_ft", choices=STAGE3_POLICIES)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--stage2-epochs", type=int, default=30)
    ap.add_argument("--stage2-patience", type=int, default=6)
    ap.add_argument("--stage2-head-lr", type=float, default=1e-3)
    ap.add_argument("--stage2-encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--stage3-epochs", type=int, default=20)
    ap.add_argument("--stage3-patience", type=int, default=5)
    ap.add_argument("--stage3-head-lr", type=float, default=1e-3)
    ap.add_argument("--stage3-encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--care-stage2-scope", choices=["none", "full", "top1", "top2"], default="top2")
    ap.add_argument("--care-stage3-scope", choices=["none", "full", "top1", "top2"], default="top1")
    ap.add_argument("--lwf-old-distill-lambda", type=float, default=0.25)
    ap.add_argument("--care-stage3-anchor-lambda", type=float, default=0.1)
    ap.add_argument("--core-lambda", type=float, default=0.1)
    ap.add_argument("--core-mode", type=str, default="refresh")
    ap.add_argument("--core-budget", type=int, default=0)
    ap.add_argument("--ewc-lambda", type=float, default=1000.0)
    ap.add_argument("--cpca-core-k", type=int, default=0)
    ap.add_argument("--stage1-ckpt-seed", type=int, default=None)
    ap.add_argument("--stage2-cache-dir", type=Path, default=None)
    ap.add_argument("--oracle-cache-dir", type=Path, default=None)
    ap.add_argument("--lpr-buffer-size", type=int, default=256)
    ap.add_argument("--lpr-alpha", type=float, default=1.0)
    ap.add_argument("--lpr-omega0", type=float, default=1.0)
    ap.add_argument("--lpr-refresh-interval", type=int, default=10)
    ap.add_argument("--lpr-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--win-sec", type=float, default=10.0)
    ap.add_argument("--stride-sec", type=float, default=10.0)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def disable_dropout(model: nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            mod.p = 0.0


def configure_attention_backends(*, math_only: bool = False) -> None:
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return
    if hasattr(cuda_backends, "enable_flash_sdp"):
        cuda_backends.enable_flash_sdp(not math_only)
    if hasattr(cuda_backends, "enable_mem_efficient_sdp"):
        cuda_backends.enable_mem_efficient_sdp(not math_only)
    if hasattr(cuda_backends, "enable_math_sdp"):
        cuda_backends.enable_math_sdp(True)


def split_subjects(subjects: Iterable[int], seed: int) -> Tuple[set[int], set[int], set[int]]:
    unique = np.asarray(sorted(set(int(x) for x in subjects)), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique)
    n = len(unique)
    n_train = max(1, int(round(0.6 * n)))
    n_val = max(1, int(round(0.2 * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train = set(unique[:n_train].tolist())
    val = set(unique[n_train : n_train + n_val].tolist())
    test = set(unique[n_train + n_val :].tolist())
    return train, val, test


def maybe_limit_split(
    X: np.ndarray,
    ys: Dict[str, np.ndarray],
    limit: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if limit <= 0 or len(X) <= limit:
        return X, ys
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(X), int(limit), replace=False))
    return np.asarray(X[idx]), {k: np.asarray(v[idx]) for k, v in ys.items()}


def _map_by_name(bundle: IMUBundle, mapping: Dict[str, str], order: List[str]) -> np.ndarray:
    idx_to_name = list(bundle.label_names)
    group_to_idx = {name: i for i, name in enumerate(order)}
    y_group = np.empty_like(bundle.y)
    for old_idx, name in enumerate(idx_to_name):
        y_group[bundle.y == old_idx] = group_to_idx[mapping[name]]
    return y_group.astype(np.int64)


def _binary_by_name(bundle: IMUBundle, positives: set[str]) -> np.ndarray:
    idx_to_name = list(bundle.label_names)
    pos_idx = {i for i, name in enumerate(idx_to_name) if name in positives}
    return np.asarray([1 if int(y) in pos_idx else 0 for y in bundle.y], dtype=np.int64)


def bundle_task_arrays(bundle_name: str, bundle: IMUBundle) -> Tuple[Dict[str, np.ndarray], Dict[str, TaskSpec]]:
    if bundle_name == "wisdm":
        ys = {
            "activity18": np.asarray(bundle.y, dtype=np.int64),
            "activity6": _map_by_name(bundle, WISDM_18_TO_GROUP6, WISDM_GROUP6_ORDER),
            "dynamic_bin": _binary_by_name(bundle, WISDM_DYNAMIC_POS),
            "upper_body_bin": _binary_by_name(bundle, WISDM_UPPER_BODY_POS),
        }
        specs = {
            "activity18": TaskSpec("activity18", "cls", "", "f1_macro"),
            "activity6": TaskSpec("activity6", "cls", "", "f1_macro"),
            "dynamic_bin": TaskSpec("dynamic_bin", "cls", "", "f1_macro"),
            "upper_body_bin": TaskSpec("upper_body_bin", "cls", "", "f1_macro"),
        }
        return ys, specs

    if bundle_name == "motionsense":
        ys = {
            "activity6": np.asarray(bundle.y, dtype=np.int64),
            "activity3": _map_by_name(bundle, MOTIONSENSE_6_TO_GROUP3, MOTIONSENSE_GROUP3_ORDER),
            "dynamic_bin": _binary_by_name(bundle, MOTIONSENSE_DYNAMIC_POS),
            "stairs_bin": _binary_by_name(bundle, MOTIONSENSE_STAIRS_POS),
        }
        specs = {
            "activity6": TaskSpec("activity6", "cls", "", "f1_macro"),
            "activity3": TaskSpec("activity3", "cls", "", "f1_macro"),
            "dynamic_bin": TaskSpec("dynamic_bin", "cls", "", "f1_macro"),
            "stairs_bin": TaskSpec("stairs_bin", "cls", "", "f1_macro"),
        }
        return ys, specs

    if bundle_name == "mhealth":
        if np.issubdtype(np.asarray(bundle.y).dtype, np.floating):
            ys = {
                "hr_regression": np.asarray(bundle.y, dtype=np.float32),
            }
            specs = {
                "hr_regression": TaskSpec("hr_regression", "reg", "", "mae"),
            }
            return ys, specs
        ys = {
            "activity12": np.asarray(bundle.y, dtype=np.int64),
            "semantic3": _map_by_name(bundle, MHEALTH_12_TO_SEMANTIC3, MHEALTH_SEMANTIC3_ORDER),
            "ambulation_bin": _binary_by_name(bundle, MHEALTH_AMBULATION_POS),
            "vigorous_bin": _binary_by_name(bundle, MHEALTH_VIGOROUS_POS),
        }
        specs = {
            "activity12": TaskSpec("activity12", "cls", "", "f1_macro"),
            "semantic3": TaskSpec("semantic3", "cls", "", "f1_macro"),
            "ambulation_bin": TaskSpec("ambulation_bin", "cls", "", "f1_macro"),
            "vigorous_bin": TaskSpec("vigorous_bin", "cls", "", "f1_macro"),
        }
        return ys, specs

    if bundle_name == "opportunity_locomotion":
        ys = {
            "locomotion4": np.asarray(bundle.y, dtype=np.int64),
        }
        specs = {
            "locomotion4": TaskSpec("locomotion4", "cls", "", "f1_macro"),
        }
        return ys, specs

    if bundle_name == "opportunity_gesture":
        ys = {
            "gesture17": np.asarray(bundle.y, dtype=np.int64),
            "object4": _map_by_name(bundle, OPPORTUNITY_GESTURE_TO_OBJECT4, OPPORTUNITY_OBJECT4_ORDER),
            "open_close_bin": _binary_by_name(bundle, OPPORTUNITY_OPEN_CLOSE_POS),
            "drink_clean_bin": _binary_by_name(bundle, OPPORTUNITY_DRINK_CLEAN_POS),
        }
        specs = {
            "gesture17": TaskSpec("gesture17", "cls", "", "f1_macro"),
            "object4": TaskSpec("object4", "cls", "", "f1_macro"),
            "open_close_bin": TaskSpec("open_close_bin", "cls", "", "f1_macro"),
            "drink_clean_bin": TaskSpec("drink_clean_bin", "cls", "", "f1_macro"),
        }
        return ys, specs

    if bundle_name == "pamap2":
        if np.issubdtype(np.asarray(bundle.y).dtype, np.floating):
            ys = {
                "hr_regression": np.asarray(bundle.y, dtype=np.float32),
            }
            specs = {
                "hr_regression": TaskSpec("hr_regression", "reg", "", "mae"),
            }
            return ys, specs
        ys = {
            "activity12": np.asarray(bundle.y, dtype=np.int64),
            "commonk_fine": np.asarray(bundle.y, dtype=np.int64),
            "posture3": _map_by_name(bundle, PAMAP2_12_TO_POSTURE3, PAMAP2_POSTURE3_ORDER),
            "stairs_bin": _binary_by_name(bundle, PAMAP2_STAIRS_POS),
            "housework_bin": _binary_by_name(bundle, PAMAP2_HOUSEWORK_POS),
            "coarse4": _map_by_name(bundle, PAMAP2_12_TO_COARSE4, PAMAP2_COARSE4_ORDER),
            "locomotion_bin": _binary_by_name(bundle, PAMAP2_LOCOMOTION_POS),
        }
        specs = {
            "activity12": TaskSpec("activity12", "cls", "", "f1_macro"),
            "commonk_fine": TaskSpec("commonk_fine", "cls", "", "f1_macro"),
            "posture3": TaskSpec("posture3", "cls", "", "f1_macro"),
            "stairs_bin": TaskSpec("stairs_bin", "cls", "", "f1_macro"),
            "housework_bin": TaskSpec("housework_bin", "cls", "", "f1_macro"),
            "coarse4": TaskSpec("coarse4", "cls", "", "f1_macro"),
            "locomotion_bin": TaskSpec("locomotion_bin", "cls", "", "f1_macro"),
        }
        return ys, specs

    raise KeyError(f"Unsupported IMU bundle {bundle_name}")


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
    task_data: TaskData,
    device: torch.device,
    batch_size: int,
) -> float:
    ref = copy.deepcopy(encoder_ref).to(device).eval()
    cur = copy.deepcopy(encoder_cur).to(device).eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(task_data.split_X["test"], dtype=np.float32))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    cos_vals: List[torch.Tensor] = []
    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        z_ref = ref(xb).reshape(xb.shape[0], -1)
        z_cur = cur(xb).reshape(xb.shape[0], -1)
        cos_vals.append(torch.sum(F.normalize(z_ref, dim=1) * F.normalize(z_cur, dim=1), dim=1).detach().cpu())
    if not cos_vals:
        return float("nan")
    return float(1.0 - torch.cat(cos_vals).mean().item())


def build_bundle_by_window(
    cache_dir: Path,
    bundle_task_id: str,
    win_sec: float,
    stride_sec: float,
) -> IMUBundle:
    bundles = build_eval_bundles(
        cache_dir=cache_dir,
        target_len=500,
        win_sec=float(win_sec),
        stride_sec=float(stride_sec),
        task_ids=[bundle_task_id],
    )
    return bundles[bundle_task_id]


def clone_task_spec(spec: TaskSpec, name: str) -> TaskSpec:
    return TaskSpec(
        name=name,
        kind=spec.kind,
        y_template=spec.y_template,
        metric=spec.metric,
        label_transform=spec.label_transform,
    )


def resolve_pack_task_defs(cfg: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    task_defs_raw = cfg.get("task_defs")
    if not task_defs_raw:
        bundle_task_id = str(cfg["bundle_task_id"])
        bundle_family = str(cfg["bundle_family"])
        return {
            str(task_name): {
                "bundle_task_id": bundle_task_id,
                "bundle_family": bundle_family,
                "label_task": str(task_name),
            }
            for task_name in list(cfg["initial_tasks"]) + list(cfg["future_tasks"])
        }

    task_defs: Dict[str, Dict[str, str]] = {}
    for task_name, raw in dict(task_defs_raw).items():
        task_defs[str(task_name)] = {
            "bundle_task_id": str(raw["bundle_task_id"]),
            "bundle_family": str(raw["bundle_family"]),
            "label_task": str(raw.get("label_task", task_name)),
        }
    return task_defs


def pack_bundle_descriptor(cfg: Dict[str, Any]) -> str:
    if "bundle_task_id" in cfg:
        return str(cfg["bundle_task_id"])

    seen: List[str] = []
    for task_name in list(cfg["initial_tasks"]) + list(cfg["future_tasks"]):
        bundle_task_id = str(cfg["task_defs"][task_name]["bundle_task_id"])
        if bundle_task_id not in seen:
            seen.append(bundle_task_id)
    return " -> ".join(seen)


def build_pack_task_data(pack_name: str, args: argparse.Namespace, seed: int) -> Tuple[List[TaskData], List[TaskData], Dict[str, Any]]:
    cfg = PACKS[pack_name]
    task_names = list(cfg["initial_tasks"]) + list(cfg["future_tasks"])
    task_defs = resolve_pack_task_defs(cfg)
    input_window_policy = str(cfg.get("input_window_policy", "shared_native"))
    sec_map = {str(k): float(v) for k, v in (cfg.get("input_window_sec_map") or {}).items()}
    default_win_sec = float(args.win_sec)
    task_window_sec = {
        task_name: float(sec_map.get(task_name, default_win_sec))
        for task_name in task_names
    }
    bundle_keys = sorted({
        (str(task_defs[task_name]["bundle_task_id"]), float(task_window_sec[task_name]))
        for task_name in task_names
    })
    bundles_by_key = {
        (bundle_task_id, float(win_sec)): build_bundle_by_window(
            cache_dir=args.cache_dir,
            bundle_task_id=bundle_task_id,
            win_sec=float(win_sec),
            stride_sec=float(win_sec) if input_window_policy == "task_specific_input" else float(args.stride_sec),
        )
        for bundle_task_id, win_sec in bundle_keys
    }

    split_bundle_task_id = str(cfg.get("split_bundle_task_id", task_defs[task_names[0]]["bundle_task_id"]))
    split_window_sec = float(cfg.get("split_window_sec", task_window_sec[task_names[0]]))
    ref_bundle = bundles_by_key[(split_bundle_task_id, split_window_sec)]
    train_subj, val_subj, test_subj = split_subjects(ref_bundle.groups, seed)
    split_per_task = bool(cfg.get("split_per_task", False))

    task_data_map: Dict[str, TaskData] = {}
    task_positive_rates: Dict[str, float] = {}
    for task_name in task_names:
        task_def = task_defs[task_name]
        bundle = bundles_by_key[(task_def["bundle_task_id"], float(task_window_sec[task_name]))]
        ys_all, specs = bundle_task_arrays(task_def["bundle_family"], bundle)
        label_task = str(task_def["label_task"])
        spec = clone_task_spec(specs[label_task], task_name)
        y_all = ys_all[label_task]
        if split_per_task:
            t_subj, v_subj, te_subj = split_subjects(bundle.groups, int(seed))
        else:
            t_subj, v_subj, te_subj = train_subj, val_subj, test_subj
        split_X: Dict[str, np.ndarray] = {}
        split_y: Dict[str, np.ndarray] = {}
        for split, subjects in (("train", t_subj), ("val", v_subj), ("test", te_subj)):
            mask = np.isin(bundle.groups, list(subjects))
            Xi = np.asarray(bundle.X[mask], dtype=np.float32)
            yi = np.asarray(y_all[mask])
            limit = {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test}[split]
            if int(limit) > 0 and len(Xi) > int(limit):
                Xi, ys_limited = maybe_limit_split(
                    Xi,
                    {task_name: yi},
                    int(limit),
                    int(seed) + {"train": 0, "val": 1, "test": 2}[split],
                )
                yi = np.asarray(ys_limited[task_name])
            split_X[split] = Xi
            split_y[split] = yi
        if spec.kind == "cls":
            runtime = TaskRuntime(spec, n_out=int(np.max(y_all)) + 1)
        else:
            y_train = split_y["train"].astype(np.float32)
            y_mean = float(y_train.mean())
            y_std = float(y_train.std())
            if not np.isfinite(y_std) or y_std < 1e-6:
                y_std = 1.0
            runtime = TaskRuntime(spec, n_out=1, y_mean=y_mean, y_std=y_std)
        task_data_map[task_name] = TaskData(
            runtime=runtime,
            split_X=split_X,
            split_y=split_y,
        )
        if spec.kind == "cls" and runtime.n_out == 2:
            task_positive_rates[task_name] = float(np.mean(y_all.astype(np.float32)))

    meta = {
        "dataset": str(cfg["dataset"]),
        "bundle_task_id": pack_bundle_descriptor(cfg),
        "bundle_task_ids": sorted({bundle_task_id for bundle_task_id, _ in bundle_keys}),
        "split_bundle_task_id": split_bundle_task_id,
        "n_samples": {
            f"{bundle_task_id}@{win_sec:g}s": int(bundle.n_samples)
            for (bundle_task_id, win_sec), bundle in bundles_by_key.items()
        },
        "n_subjects": int(np.unique(ref_bundle.groups).size),
        "channels": list(ref_bundle.channels),
        "target_len": int(ref_bundle.target_len),
        "input_window_policy": input_window_policy,
        "input_window_sec_map": task_window_sec,
        "task_positive_rates": task_positive_rates,
        "task_sources": task_defs,
    }
    initial = [task_data_map[name] for name in cfg["initial_tasks"]]
    future = [task_data_map[name] for name in cfg["future_tasks"]]
    return initial, future, meta


def load_stage1_imu_encoder(student: str, ckpt_dir: Path, method: str, seed: int, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt_path = ckpt_dir / "checkpoints" / f"{method}_seed{int(seed)}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing IMU Stage-1 checkpoint: {ckpt_path}")
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = dict(obj["student_cfg"])
    enc = NormWearStudentEncoder(
        device=device,
        sampling_rate=50,
        patch_pool="mean",
        channel_pool="mean",
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        decoder_embed_dim=cfg["decoder_embed_dim"],
        decoder_depth=cfg["decoder_depth"],
        mlp_ratio=cfg["mlp_ratio"],
        fuse_freq=cfg["fuse_freq"],
        nvar=cfg["nvar"],
    ).to(device)
    enc.load_state_dict(obj["encoder_state_dict"], strict=True)
    disable_dropout(enc)
    with torch.inference_mode():
        enc_dim = int(enc(torch.zeros(2, int(cfg["nvar"]), 500, device=device)).shape[-1])
    meta = {k: v for k, v in obj.items() if k not in {"state_dict", "encoder_state_dict"}}
    meta["ckpt_path"] = str(ckpt_path)
    meta["enc_dim"] = int(enc_dim)
    return enc, meta


def fit_stage1_single_task_oracle(
    student: str,
    seed: int,
    kd_method: str,
    ckpt_dir: Path,
    task_data: TaskData,
    dataset_name: str,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    ckpt_seed = int(args.stage1_ckpt_seed) if getattr(args, "stage1_ckpt_seed", None) is not None else int(seed)
    enc, ck_meta = load_stage1_imu_encoder(student, ckpt_dir, kd_method, ckpt_seed, device)
    model = MultiTaskModel(
        enc,
        int(ck_meta["enc_dim"]),
        [task_data.runtime],
        args.dropout,
        dataset_name=dataset_name,
        hz=50,
        input_len=int(task_data.split_X["train"].shape[-1]),
        temporal_policy="shared_window",
    )
    train_loaders, val_loaders, test_loaders = make_task_loaders([task_data], args.batch_size)
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


def expected_phase_pairs(
    initial_tasks: List[str],
    future_tasks: List[str],
    *,
    include_oracle: bool = True,
) -> set[Tuple[str, str, str]]:
    expected: set[Tuple[str, str, str]] = set()
    for task in initial_tasks:
        expected.add(("stage2", "", task))
    old = list(initial_tasks)
    for fut in future_tasks:
        old_plus_new = old + [fut]
        for task in old_plus_new:
            expected.add(("stage3", fut, task))
        if include_oracle:
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
    return expected_phase_pairs(cfg["initial_tasks"], cfg["future_tasks"], include_oracle=False).issubset(got)


def stage2_base_record(
    pack: str,
    pack_cfg: Dict[str, Any],
    student: str,
    seed: int,
    method: str,
    args: argparse.Namespace,
    ckpt_path: str,
) -> Dict[str, Any]:
    return {
        "pack": pack,
        "dataset": pack_cfg["dataset"],
        "bundle_task_id": pack_bundle_descriptor(pack_cfg),
        "student": student,
        "seed": seed,
        "kd_method": args.kd_method,
        "stage2_method": method,
        "stage3_policy": args.stage3_policy,
        "temporal_policy": str(pack_cfg.get("temporal_policy", "shared_window")),
        "input_window_policy": str(pack_cfg.get("input_window_policy", "shared_native")),
        "input_window_sec_map": json.dumps(pack_cfg.get("input_window_sec_map", {}), sort_keys=True),
        "initial_tasks": json.dumps(pack_cfg["initial_tasks"]),
        "future_tasks": json.dumps(pack_cfg["future_tasks"]),
        "stage1_ckpt": ckpt_path,
    }


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # GradNorm needs second-order gradients through attention, which the
    # efficient SDP kernels do not provide on this stack.
    configure_attention_backends(math_only=("gradnorm_joint" in args.stage2_methods))
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
        if args.resume and out_csv.exists():
            by_seed, agg = summarize(existing)
            by_seed.to_csv(by_seed_csv, index=False)
            agg.to_csv(summary_csv, index=False)
            print(agg.to_string(index=False), flush=True)
            continue

        all_rows: List[Dict[str, Any]] = existing.to_dict("records") if not existing.empty else []
        for student in args.students:
            ckpt_dir = DEFAULT_CKPT_DIRS[student]
            for seed in args.seeds:
                initial_task_data, future_task_data, pack_meta = build_pack_task_data(pack, args, seed)
                (out_dir / f"pack_meta_seed{int(seed)}.json").write_text(json.dumps(pack_meta, indent=2))
                if not existing.empty:
                    done_methods = [m for m in args.stage2_methods if combo_completed(existing, pack, student, seed, m, args.stage3_policy)]
                else:
                    done_methods = []

                oracle_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
                for stage2_method in args.stage2_methods:
                    if stage2_method in done_methods:
                        print(f"[skip:{pack}] {student} seed={seed} {stage2_method} [cached]", flush=True)
                        continue

                    bundle_desc = pack_bundle_descriptor(pack_cfg)
                    set_seed(int(seed))
                    print(
                        f"\n{'='*88}\n"
                        f"[pack={pack}] student={student} seed={seed} stage2={stage2_method}\n"
                        f"dataset={pack_cfg['dataset']} bundle_task_id={bundle_desc} initial={pack_cfg['initial_tasks']} future={pack_cfg['future_tasks']}\n"
                        f"{'='*88}",
                        flush=True,
                    )

                    ckpt_seed = int(args.stage1_ckpt_seed) if args.stage1_ckpt_seed is not None else int(seed)
                    enc, ck_meta = load_stage1_imu_encoder(student, ckpt_dir, args.kd_method, ckpt_seed, device)
                    enc_ref = copy.deepcopy(enc).cpu()
                    input_len = max(int(td.split_X["train"].shape[-1]) for td in initial_task_data)
                    model = MultiTaskModel(
                        enc,
                        int(ck_meta["enc_dim"]),
                        [td.runtime for td in initial_task_data],
                        args.dropout,
                        dataset_name=pack_cfg["dataset"],
                        hz=50,
                        input_len=input_len,
                        temporal_policy=str(pack_cfg.get("temporal_policy", "shared_window")),
                    )
                    train_loaders, val_loaders, test_loaders = make_task_loaders(initial_task_data, args.batch_size)
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
                    base = stage2_base_record(pack, pack_cfg, student, seed, stage2_method, args, ck_meta["ckpt_path"])
                    base["time_s_stage2"] = float(time.time() - t0)
                    base["encoder_params"] = count_params(model.encoder)
                    base["total_params"] = count_params(model)
                    base["trainable_params_stage2"] = count_params(model, trainable_only=True)
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
                                    kd_method=args.kd_method,
                                    ckpt_dir=ckpt_dir,
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
                            care_stage3_scope=args.care_stage3_scope,
                            lwf_old_distill_lambda=args.lwf_old_distill_lambda,
                            care_stage3_anchor_lambda=args.care_stage3_anchor_lambda,
                            cpca_core_k=args.cpca_core_k,
                            lpr_buffer_size=args.lpr_buffer_size,
                            core_lambda=args.core_lambda,
                            core_mode=args.core_mode,
                            core_budget=args.core_budget,
                            ewc_lambda=args.ewc_lambda,
                            lpr_alpha=args.lpr_alpha,
                            lpr_omega0=args.lpr_omega0,
                            lpr_refresh_interval=args.lpr_refresh_interval,
                            lpr_lr=args.lpr_lr,
                            old_task_data=current_old,
                            seed=int(seed),
                            log_prefix=f"[{pack}/{student}/s{seed}/{stage2_method}/add:{new_name}] ",
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

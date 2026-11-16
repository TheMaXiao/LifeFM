# Data layout

Datasets are **not** included. The code reads prepared arrays under `pulseppg/data/datasets/` (relative to the
release root). Expected files for the PPG pipeline:

```
pulseppg/data/datasets/
├── dalia/
│   ├── train_X_ppg_50Hz.npy      # (N, 1, 400)   KD pool source
│   ├── val_X_ppg_50Hz.npy
│   └── test_X_ppg_50Hz.npy
├── sdb/
│   ├── train_X_ppg_50Hz.npy      # (N, 1, 1500?)  initial tasks (binary / multiclass)
│   ├── val_X_ppg_50Hz.npy
│   └── test_X_ppg_50Hz.npy       # + label arrays used by the task builders
├── wesad/{binary,multiclass}/X_ppg_50Hz.npy
├── wfppg/{wrist,finger}/X_ppg_50Hz.npy
├── ppgbp/, ppgarrhythmia/, mimicperform/, earset/, bidmc/, vitaldb/
└── pulsedb/                      # if present; otherwise the cached teacher pool is required
```

ECG and IMU data live in the sibling `ecg/` and `imu_stage1/` trees of the original project; this release expects the
same relative layout (`ecg/data/...`, and the IMU arrays referenced by `imu_stage1_data.py`).

## Stage-I KD pools

| Modality | Sources | Samples |
|---|---|---|
| PPG | PPG-DaLiA + PulseDB | 5K + 5K = 10K |
| ECG | PTB-XL | 10K |
| IMU | MotionSense | 10K |

The KD pool is unlabeled. Per the paper, **subjects used for KD are disjoint from the downstream test splits**;
enforce this when building the pool.

## Stage-II / Stage-III tasks used in the paper

| Modality | Initial tasks | New₁ | New₂ |
|---|---|---|---|
| PPG | SDB binary + SDB multiclass | BIDMC HR | WF-PPG finger HR |
| ECG | Chapman rhythm4 + abnormality | PTB-XL STTC | PTB-XL CD |
| IMU | Opportunity locomotion4 + gesture17 | MHEALTH HR | MHEALTH activity-12 |

Task packs are defined in the runners (e.g. `sdb_to_bidmc_finger`, `chapman2_to_ptbxl_morph`,
`opportunity_to_mhealth_two`). Splits are subject-level and seeded; the same split must be reused across methods.

## Preprocessing

PPG is resampled to **50 Hz** and windowed per task (e.g. 30 s for SDB, 6 s for BIDMC HR, 10 s for WF-PPG finger);
ECG to its native lead layout; IMU to per-task windows. The prepared `.npy` arrays are what this code consumes —
rebuilding them from raw datasets is out of scope for this release.

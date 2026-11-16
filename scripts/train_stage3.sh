#!/usr/bin/env bash
# Stage III — core-preserved continual evolution (LifeFM `lwp_core`; λ_evo_core = 0.1, core refresh).
# Two successive new tasks per modality, no historical data. Starts from the LifeFM Stage-II model.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

# ── PPG: SDB → BIDMC HR → WF-PPG finger ───────────────────────────────────────────────────
$PY -u run_stage23_ppg_unified.py \
  --packs sdb_to_bidmc_finger --students BF32_NB8 \
  --stage2-methods care_v1 --stage3-policy lwp_core \
  --core-energy 0.95 --core-lambda 0.1 --core-mode refresh \
  --care-stage2-scope top2 --care-stage3-scope top1 \
  --seeds 0 1 2 --stage2-epochs 40 --stage3-epochs 30 \
  --out-root results/stage3_ppg

# ── ECG: Chapman → PTB-XL STTC → CD ───────────────────────────────────────────────────────
$PY -u run_stage23_ecg_unified.py \
  --packs chapman2_to_ptbxl_morph --students BF6_NB7 \
  --stage1-method ns_kd --stage1-ckpt-seed 0 \
  --stage2-methods care_v1 --stage3-policy lwp_core \
  --core-lambda 0.1 --core-mode refresh \
  --care-stage2-scope top2 --care-stage3-scope full \
  --seeds 0 1 2 --stage2-epochs 30 --stage3-epochs 20 \
  --out-root results/stage3_ecg

# ── IMU: Opportunity → MHEALTH HR → activity-12 ───────────────────────────────────────────
$PY -u run_stage23_imu_unified.py \
  --packs opportunity_to_mhealth_two --students imu2m \
  --kd-method ns_kd_lam0.1 --stage1-ckpt-seed 0 \
  --stage2-methods care_v1 --stage3-policy lwp_core \
  --core-lambda 0.1 --core-mode refresh \
  --care-stage2-scope top2 --care-stage3-scope full \
  --seeds 0 1 2 --batch-size 64 \
  --stage2-epochs 30 --stage3-epochs 20 \
  --out-root results/stage3_imu

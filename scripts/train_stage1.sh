#!/usr/bin/env bash
# Stage I — LifeFM compression (core alignment + non-core suppression).
# Requires the teacher FM checkpoint (not included in this release).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

# ── PPG (Pulse-PPG teacher) ──────────────────────────────────────────────────────────────
$PY -u run_ns_kd.py --teacher pulseppg --student BF32_NB8 \
  --methods ns_kd --lambdas 0.1 --k-values 50 \
  --n-seeds 3 --epochs 30 --n-data 10000 --max-samples 30000 --gpu-probe

# ── ECG (ECGFounder teacher) ─────────────────────────────────────────────────────────────
$PY -u ecg/run_ecgfounder_kd.py --student BF6_NB7 --methods ns_kd \
  --lam 0.05 --pool-n 10000 --epochs 30 --seeds 0 \
  --out-dir kd_sweep_results/ecg_preliminary

# ── IMU (NormWear teacher) ───────────────────────────────────────────────────────────────
# The IMU Stage-I encoder is trained with the NormWear KD pipeline in the sibling `imu_stage1`
# project (model definitions: normwear_kd_experiment.py). The released Stage-I IMU checkpoint is
# provided in checkpoints/stage1/imu_imu2m/, so Stage II/III can run without redoing Stage I.
echo "IMU Stage-I: use the imu_stage1 pipeline (released checkpoint provided)."

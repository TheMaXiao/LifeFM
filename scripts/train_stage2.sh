#!/usr/bin/env bash
# Stage II — core-anchored multi-task adaptation (LifeFM `lifefm_adapt`, λ_core = 0.05, 95% energy core).
# Requires the Stage-I encoder (released) and the downstream data under pulseppg/data/datasets/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

# ── PPG (initial tasks: SDB binary + multiclass) ──────────────────────────────────────────
$PY -u run_stage2_mtl_future_probe.py \
  --student BF32_NB8 --stage1_method lifefm_kd \
  --methods lifefm_adapt --cases all --lambda-core 0.05 \
  --batch_size 64 --stage2_epochs 40 --stage2_patience 8 \
  --stage2_head_lr 1e-3 --stage2_encoder_lr_mult 0.1 \
  --seeds 0 1 2 --output_dir results/stage2_ppg

# ── ECG (initial tasks: Chapman rhythm4 + abnormality) ────────────────────────────────────
$PY -u run_stage2_mtl_future_probe_ecg.py \
  --student BF6_NB7 --stage1_method lifefm_kd \
  --methods lifefm_adapt --cases all --lambda-core 0.05 \
  --batch_size 64 --stage2_epochs 30 --stage2_patience 8 \
  --stage2_head_lr 1e-3 --stage2_encoder_lr_mult 0.1 \
  --seeds 0 1 2 --output_dir results/stage2_ecg

# ── IMU (initial tasks: Opportunity locomotion4 + gesture17) ──────────────────────────────
$PY -u run_stage2_mtl_future_probe_imu.py \
  --student imu2m --stage1_method lifefm_kd \
  --methods lifefm_adapt --cases all --lambda-core 0.05 \
  --batch_size 64 --stage2_epochs 30 --stage2_patience 8 \
  --stage2_head_lr 1e-3 --stage2_encoder_lr_mult 0.1 \
  --seeds 0 1 2 --output_dir results/stage2_imu

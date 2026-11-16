#!/usr/bin/env bash
# Populate the checkpoint locations expected by the Stage-II/III runners.
# Released Stage-I checkpoints live in checkpoints/stage1/; the runners look for them under
# kd_sweep_results/... (the original working layout). This script creates that layout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PPG=kd_sweep_results/pulseppg_preliminary
ECG=kd_sweep_results/ecg_preliminary/exp_ecgfounder_BF6_NB7_10K_nskd_tune_lam005_e40/exp_ecgfounder_BF6_NB7/checkpoints
IMU=kd_sweep_results/imu_preliminary/exp_normwear_imu_stage1_pilot/checkpoints
mkdir -p "$PPG/bf20_ckpts" "$PPG/bf32_ckpts" "$PPG/bf32_nb8_ckpts" "$PPG/bf48_ckpts" "$ECG" "$IMU"

cp -n checkpoints/stage1/ppg_bf20/lifefm_kd_seed0.pt "$PPG/bf20_ckpts/"
cp -n checkpoints/stage1/ppg_bf32/lifefm_kd_seed0.pt "$PPG/bf32_ckpts/"
cp -n checkpoints/stage1/ppg_bf32/lifefm_kd_seed0.pt "$PPG/bf32_nb8_ckpts/"
cp -n checkpoints/stage1/ppg_bf48/lifefm_kd_seed0.pt "$PPG/bf48_ckpts/"
cp -n checkpoints/stage1/ecg_bf6/lifefm_kd_seed0.pt "$ECG/"
cp -n checkpoints/stage1/imu_imu2m/lifefm_kd_lam0.1_seed0.pt "$IMU/"

echo "Stage-I checkpoints installed under kd_sweep_results/."

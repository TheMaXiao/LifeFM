# LifeFM

Reference implementation of **LifeFM**, a three-stage lifecycle for compressing, specializing, and continually
evolving wearable foundation models (IMWUT'27).

LifeFM treats a large wearable foundation model (FM) as a teacher and a small on-device encoder as the student,
and applies the same *core-knowledge* principle at every stage:

| Stage | Goal | Method (paper) | Code |
|---|---|---|---|
| **I. Compression** | Distill the FM into a wearable-scale encoder | capacity-matched budget `B_KD = (1+ρ)·B_S`, top-`B_KD` teacher-SVD directions, core-alignment MSE + non-core suppression (`λ_sup = 0.1`) | `run_ns_kd.py` (`--methods ns_kd`) |
| **II. Initial specialization** | Adapt the compressed encoder to an initial task set without losing generalizability | 95%-energy core basis of the compressed encoder + core-anchor loss (`λ_core = 0.05`) | `run_stage2_multitask_ppg.py` / `run_stage2_mtl_future_probe_ecg.py` / `run_stage2_mtl_future_probe_imu.py` (`care_v1`) |
| **III. Continual evolution** | Integrate new tasks without historical data | core-preserved update + core refresh (`λ_evo_core = 0.1`), added on top of an underlying continual-learning objective | `run_stage23_ppg_unified.py` / `..._ecg_...` / `..._imu_...` (`--stage3-policy lwp_core`) |

The paper is included for reference: `LifeFM_IMWUT27.pdf`, `LifeFM_IMWUT27_Appendix.pdf`.

---

## 1. What is in this release

```
.
├── run_ns_kd.py                       # Stage-I (LifeFM KD: core alignment + non-core suppression)
├── run_k_probe_pulseppg.py            # PPG teacher/student/data utilities used by Stage-I
├── downstream_eval.py                 # frozen-encoder linear-probe evaluation (retention)
├── run_stage2_mtl_future_probe.py     # Stage-II (PPG), includes `care_v1`
├── run_stage2_mtl_future_probe_ecg.py # Stage-II (ECG)
├── run_stage2_mtl_future_probe_imu.py # Stage-II (IMU)
├── run_stage2_multitask_ppg.py        # multi-task trainer / dataset specs used by Stage-II
├── run_specialization_sweep_ppg.py    # Stage-II core-basis / probe helpers
├── run_specialization_sweep_ecg.py
├── run_stage2_student_subspace.py     # gradient utilities used by Stage-III
├── run_stage23_ppg_unified.py         # Stage-III (PPG): two-step core-preserved evolution
├── run_stage23_ecg_unified.py         # Stage-III (ECG)
├── run_stage23_imu_unified.py         # Stage-III (IMU)
├── normwear_kd_experiment.py          # IMU student/teacher definitions (Stage-I)
├── imu_stage1_data.py                 # IMU data loading
├── sdb.py, wesad.py, wfppg.py, sportppg.py   # dataset readers
├── ecg/                               # ECGFounder encoder/teacher definitions
├── NormWear/                          # NormWear model definitions (IMU)
├── pulseppg/                          # ResNet1D nets, utils, experiment configs, data processing
├── checkpoints/stage1/                # released Stage-I encoder checkpoints (see §6)
├── scripts/                           # example commands (setup, Stage-I/II/III)
├── docs/METHOD.md                     # paper §4–§6 → code map + hyperparameters
├── docs/DATA.md                       # expected dataset layout and KD sources
└── requirements.txt
```

**Not included:** datasets, teacher checkpoints, baseline implementations, and experiment outputs. This release
contains only the LifeFM method.

---

## 2. Installation

```bash
python >= 3.10
pip install -r requirements.txt
```

Tested with PyTorch 2.x on NVIDIA GPUs (CUDA 12). CPU-only execution is possible for evaluation but not recommended
for training.

---

## 3. Data

Datasets are **not** shipped. Put them under `pulseppg/data/datasets/` using the layout in
[`docs/DATA.md`](docs/DATA.md). In short:

- Downstream tasks: PPG (SDB, BIDMC, WF-PPG, PPG-BP, WESAD, PPG-DaLiA, PPG-Arrhythmia, MIMIC-Perform, EarSet,
  VitalDB), ECG (Chapman, PTB-XL, CINC), IMU (Opportunity, MHEALTH, MotionSense/PAMAP2).
- Stage-I KD data (unlabeled, 10K samples): PPG-DaLiA + PulseDB (5K each) for PPG; PTB-XL for ECG; MotionSense
  for IMU. Following the paper, KD subjects must be disjoint from downstream test subjects.

TODO: dataset preprocessing scripts (the released code consumes the prepared `.npy` arrays).

---

## 4. Quick start (released checkpoints)

The Stage-I encoder checkpoints are included, so you can skip Stage I and go straight to adaptation:

```bash
# 1) put the released checkpoints where the runners expect them
bash scripts/setup_checkpoints.sh

# 2) Stage-II: core-anchored multi-task adaptation (PPG example)
bash scripts/train_stage2.sh

# 3) Stage-III: two-step core-preserved continual evolution (PPG example)
bash scripts/train_stage3.sh
```

---

## 5. Training

### Stage I — compression (`run_ns_kd.py`)

```bash
python -u run_ns_kd.py \
  --teacher pulseppg --student BF32_NB8 \
  --methods ns_kd --lambdas 0.1 \
  --k-values 50 \
  --n-seeds 3 --epochs 30 --n-data 10000 --max-samples 30000 --gpu-probe
```

- `--teacher` ∈ `{pulseppg, normwear, papagei}` selects the FM.
- `--methods ns_kd` is LifeFM's KD objective (core alignment + non-core suppression).
- `--k-values` sets the distillation budget `B_KD` (number of top teacher-SVD directions).
- Optimizer AdamW, lr `1e-4`, batch 64, up to 30 epochs (best-train-loss checkpoint).
- **The teacher checkpoint is not included** — obtain it from the original FM release and place it as expected by
  `run_k_probe_pulseppg.load_pulseppg_teacher` (or the corresponding ECG/IMU loader).

### Stage II — initial specialization (`care_v1`)

```bash
python -u run_stage2_mtl_future_probe.py \
  --student BF32_NB8 --stage1_method ns_kd \
  --methods care_v1 --cases all --care_reuse_lambda 0.05 \
  --batch_size 64 --stage2_epochs 40 --stage2_patience 8 \
  --stage2_head_lr 1e-3 --stage2_encoder_lr_mult 0.1 \
  --seeds 0 1 2 --output_dir results/stage2_ppg
```

- Core basis: SVD of the compressed encoder's embeddings on the initial-task training inputs, 95% cumulative energy.
- Core-anchor loss weight: `--care_reuse_lambda 0.05` (paper `λ_core`).
- Optimizer AdamW, encoder lr `1e-4`, heads lr `1e-3`, batch 64, early stopping on validation loss.

### Stage III — continual evolution (`lwp_core`)

```bash
python -u run_stage23_ppg_unified.py \
  --packs sdb_to_bidmc_finger --students BF32_NB8 \
  --stage2-methods care_v1 --stage3-policy lwp_core \
  --core-energy 0.95 --core-lambda 0.1 --core-mode refresh \
  --care-stage2-scope top2 --care-stage3-scope top1 \
  --seeds 0 1 2 --stage2-epochs 40 --stage3-epochs 30 \
  --out-root results/stage3_ppg
```

- `--stage2-methods care_v1` starts from the LifeFM Stage-II model.
- `--stage3-policy lwp_core` = LwP + evolution core anchor (LifeFM Stage-III).
- `--core-mode refresh` recomputes the core basis from the updated encoder after each task (paper §6.2).
- `--core-lambda 0.1` is the evolution anchor weight `λ_evo_core`.

---

## 6. Released Stage-I checkpoints

| File | Modality / student | Params | Recorded KD budget `k` | `λ_sup` |
|---|---|---:|---:|---:|
| `checkpoints/stage1/ppg_bf20/ns_kd_seed{0,1,2}.pt` | PPG / BF20_NB8 (width 40) | 0.19 M | 40 | 0.1 |
| `checkpoints/stage1/ppg_bf32/ns_kd_seed0.pt` | PPG / BF32_NB8 (width 64) | 0.46 M | 50 | 0.1 |
| `checkpoints/stage1/ppg_bf48/ns_kd_seed{0,1,2}.pt` | PPG / BF48_NB8 (width 96) | 1.02 M | 40 | 0.1 |
| `checkpoints/stage1/ecg_bf6/ns_kd_seed0.pt` | ECG / BF6_NB7 | 0.46 M | 117 | 0.05 |
| `checkpoints/stage1/imu_imu2m/ns_kd_lam0.1_seed0.pt` | IMU / imu2m | 2.20 M | 4 | — |

These are the exact artifacts used for the paper's Stage-I results; each file stores its own metadata
(`student`, `method`, `seed`, `k`, `lam`, `student_cfg`, `state_dict`).

---

## 7. Evaluation and metrics

`downstream_eval.py` runs frozen-encoder linear probes on the downstream tasks. **Retention** is the ratio between
the evaluated model and the reference model: `value / reference` for higher-is-better metrics and
`reference / value` for MAE. Values above 1.0 mean the compressed/adapted model beats its reference.

- Stage I: per-seed average and minimum retention across tasks, then mean ± std across seeds.
- Stage II: per-group average raw score → retention vs the group reference (inverse ratio for MAE).
- Stage III: initial tasks normalized against the Stage-II model; new tasks against their single-task oracles;
  `ASP = (2·Initial + New₁ + New₂) / 4`.

---

## 8. Notes and limitations

- **Teacher checkpoints are not included.** Stage-I training requires the original FM checkpoints (Pulse-PPG,
  ECGFounder, NormWear). The released Stage-I students let you run Stages II–III without them.
- **Datasets are not included** and must be placed under `pulseppg/data/datasets/` (see `docs/DATA.md`).
- **Baselines are not included.** The runners expose baseline policy names internally, but only the LifeFM paths
  (`ns_kd`, `care_v1`, `lwp_core`) are part of this release.
- The paper's budget rule is `B_KD = (1+ρ)·B_S` with `ρ = 0.1`; set it explicitly via `--k-values` (or
  `--adaptive-k` for an energy-based alternative). The released checkpoints list their recorded `k`.
- Runtime paths are relative to the repository root; run the commands from the release root.

---

## 9. Citation

```bibtex
@article{lifefm2027,
  title   = {LifeFM: Enabling On-Device Deployment and Lifecycle Adaptation of Wearable Foundation Models},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)},
  year    = {2027}
}
```

## 10. License

To be added by the authors before public release.

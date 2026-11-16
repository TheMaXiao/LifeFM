# Method ↔ code map

Maps the paper's method sections (`LifeFM_IMWUT27.pdf`, §4–§6) to this release.

## Stage I — Compressing a wearable FM (paper §4)

**§4.1 Distill at the right level.**
`B_KD = (1 + ρ)·B_S`, with `B_S` = width of the final representation-producing encoder block and `ρ = 0.1`.
Implemented as the distillation budget `k` (`--k-values`). `B_S` per student:
BF20 → 40, BF32 → 64, BF48 → 96 (`2 × base_filters`; see `run_k_probe_pulseppg.make_student_builder`).

**§4.2 Distill the right knowledge.**
Teacher embeddings on the KD pool are centered and decomposed by SVD; the top-`B_KD` directions are the **core**.
Implementation: `run_ns_kd.compute_svd_artifacts` (centered PCA basis) and `make_training_targets`-equivalent target
construction (`Z @ V_kᵀ`).

**§4.3 KD objective (Eq. 2).**
`L_KD = L_core + λ_sup · L_sup`:
- `L_core = MSE(z_s Vᵀ, z_t Vᵀ)` on the top-`k` subspace,
- `L_sup = ‖(I − V Vᵀ) z_s‖²` on the non-core subspace,
- `λ_sup = 0.1`.

Implementation: `run_ns_kd.train_ns_kd` (weights `--lambdas`). Optimizer AdamW, lr `1e-4`, batch 64, 30 epochs,
best-train-loss checkpoint.

## Stage II — Initial task specialization (paper §5)

**§5.1 Identifying core knowledge.**
Collect the compressed encoder's embeddings on the initial-task training inputs (**labels not used**), SVD, and
keep the smallest leading set whose cumulative energy reaches **95%**.

Implementation: `run_stage2_multitask_ppg._build_care_auto_anchor_basis` /
`_care_auto_reuse_terms` (`--alpha-anchor`, `--care-auto-*`), and the same machinery in the ECG/IMU Stage-II
scripts. Stage-II policy name: **`care_v1`**.

**§5.2 Core-anchored multi-task adaptation (Eq. 3).**
`L_adapt = L_task + λ_core · L_anchor`, where the anchor is the MSE between the adapting and frozen reference
encoder embeddings projected on the core basis. `λ_core = 0.05` (`--alpha-anchor 0.05`).
Optimizer AdamW, encoder lr `1e-4`, heads lr `1e-3`, batch 64, early stopping.

## Stage III — Core-preserved continual evolution (paper §6)

**§6.1 Core-preserved task integration (Eq. 4–5).**
Before discarding the initial data, save the core basis `U_evo` (95% energy) of the Stage-II encoder. For each new
task, attach a new head, freeze a reference copy of the latest model, and add
`L_evo_anchor = E[‖U_evoᵀ E_θ(x) − U_evoᵀ E_ref(x)‖²]` on the **new task's** data only.
`L_evolution = L_CL + λ_evo_core · L_evo_anchor`, with `λ_evo_core = 0.1`.

**§6.2 Core refresh.**
After each update, recompute `U_evo` from the updated encoder using only the current task's inputs, then discard
the data. Both the basis and the reference model are refreshed between updates and fixed within an update.

Implementation: `run_stage23_*_unified.py` with **`--stage3-policy lwp_core`**, `--core-mode refresh`,
`--core-energy 0.95`, `--core-lambda 0.1`, `--stage2-methods care_v1`.
Underlying continual-learning objective: **LwP** (`--stage3-policy lwp_core` builds on LwP; the paper reports
output-distillation weight 1.0 and distance-preservation weight 0.01).

## Hyperparameters used for the released checkpoints

| Modality | Student | `k` (B_KD) | `λ_sup` |
|---|---|---:|---:|
| PPG | BF20_NB8 | 40 | 0.1 |
| PPG | BF32_NB8 | 50 | 0.1 |
| PPG | BF48_NB8 | 40 | 0.1 |
| ECG | BF6_NB7 | 117 | 0.05 |
| IMU | imu2m | 4 | — |

## Key symbols

| Paper | Code |
|---|---|
| `B_S` (final block width) | `2 × base_filters` (`native_width`) |
| `B_KD` | `--k-values` / `--k` |
| `V`, top-`k` SVD directions | SVD basis from `compute_svd_artifacts`, `Vt[:k]` |
| `λ_sup` | `--lambdas` (Stage I) |
| `U_evo`, 95% energy core | `--core-energy 0.95` (`care_v1`, `lwp_core`) |
| `λ_core` (Stage II) | `--alpha-anchor` |
| `λ_evo_core` (Stage III) | `--core-lambda` |
| `core refresh` | `--core-mode refresh` |

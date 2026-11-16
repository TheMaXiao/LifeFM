from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, TensorDataset
from tqdm import tqdm

from NormWear.main_model import NormWearModel
from NormWear.modules.normwear import NormWear, cwt_wrap
from pulseppg.nets.ResNet1D.ResNet1D_Net import Net as ResNet1DNet
from pulseppg.utils.kfold import make_split
from sdb import SDBNpyDataset
from sportppg import SportPPGConfig, SportPPGWindowDataset
from wesad import make_wesad_dataset
from wfppg import make_wfppg_dataset


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ExperimentConfig:
    repo_root: Path = Path(".")
    device: str = "cuda"
    teacher_checkpoint: Path = Path("NormWear/checkpoints/normwear_pretrain_ckpt.pth")
    teacher_sampling_rate: int = 50
    teacher_patch_pool: str = "mean"
    teacher_channel_pool: str = "mean"

    public_n: int = 100_000
    public_seed: int = 20260314
    kd_init_seed: int = 20260314
    kd_batch_size: int = 64
    kd_num_workers: int = 4
    kd_epochs: int = 30
    kd_lr: float = 1e-4
    kd_weight_decay: float = 0.0
    responsekd_normalize: bool = True
    rkd_w_dist: float = 1.0
    rkd_w_angle: float = 2.0

    student_base_filters: int = 40
    student_n_block: int = 14
    student_kernel_size: int = 11
    student_stride: int = 2
    student_groups: int = 1
    student_downsample_gap: int = 2
    student_increasefilter_gap: int = 4
    student_finalpool: str = "max"
    student_use_bn: bool = True
    student_use_do: bool = True
    student_arch: str = "resnet1d"

    normwear_student_embed_dim: int = 256
    normwear_student_depth: int = 4
    normwear_student_num_heads: int = 4
    normwear_student_decoder_embed_dim: int = 192
    normwear_student_decoder_depth: int = 2
    normwear_student_mlp_ratio: float = 4.0
    normwear_student_fuse_freq: int = 2
    normwear_student_nvar: int = 1

    sport_data_root: Path = Path("/home/xiaoma/datasets/ppg/PPG-Sport-F5B0/dataset/")
    sport_gt_root: Path = Path("/home/xiaoma/datasets/ppg/PPG-Sport-F5B0/cleaned_gt/")
    sport_subjects: tuple[str, ...] = tuple(f"S{sid:02d}" for sid in range(1, 31))
    sport_acts: tuple[str, ...] = (
        "stationary",
        "walking",
        "running",
        "badminton",
        "tabletennis",
        "basketball",
    )
    sport_target_fs: int = 50

    wesad_base_dir: Path = Path("pulseppg/data/datasets/wesad")
    wesad_hz: int = 50

    wfppg_hz: int = 50

    sdb_public_root: Path = Path("pulseppg/data/datasets/sdb")
    sdb_public_fs: int = 50
    pulsedb_public_ppg_path: Path = Path("/home/xiaoma/datasets/ppg/PulseDB/subset_100k_h5_ppg.npy")
    pulsedb_public_subj_path: Path = Path("/home/xiaoma/datasets/ppg/PulseDB/subset_100k_h5_subj.npy")
    pulsedb_public_orig_fs: int = 125
    public_target_fs: int = 50
    public_win_sec: float = 10.0
    public_normalize: bool = True

    eval_batch_size: int = 256
    eval_num_workers: int = 4
    eval_n_splits: int = 5
    eval_seed: int = 42
    eval_use_raw_student_encoder: bool = True

    cache_dir: Path = Path("kd_cache/normwear")
    results_dir: Path = Path("kd_results")
    amp: bool = True
    public_val_n: int = 10_000

    public_refine_enabled: bool = False
    public_refine_candidate_n: int | None = None
    public_refine_seed: int | None = None
    public_refine_batch_size: int = 128
    public_refine_num_workers: int | None = None
    public_refine_kmeans_k: int = 2048
    public_refine_kmeans_batch: int = 4096
    public_refine_core_ratio: float = 0.7
    public_refine_dist_hi_q: float = 0.9
    public_refine_inc_hi_q: float = 0.9
    public_refine_core_inc_q: float = 0.5
    public_refine_hard_inc_lo_q: float = 0.6
    public_refine_hard_inc_hi_q: float = 0.9
    public_refine_hard_dist_lo_q: float = 0.5
    public_refine_hard_dist_hi_q: float = 0.9

    hptkd_num_prototypes: int = 64
    hptkd_proto_momentum: float = 0.99
    hptkd_alpha_max: float = 0.5
    hptkd_tau_reliability: float = 1.0
    hptkd_tau_learnability: float = 0.2
    hptkd_lambda_steer: float = 1.0
    hptkd_lambda_weak: float = 0.2
    hptkd_align_mode: str = "projection"
    hptkd_patience: int = 5
    hptkd_min_delta: float = 1e-4
    hptkd_val_lambda_point: float = 1.0
    hptkd_val_lambda_cos: float = 1.0
    hptkd_val_lambda_rel: float = 0.2
    hptkd_experiment_tag: str = ""

    v2_num_prototypes: int = 64
    v2_proto_momentum: float = 0.99
    v2_relation_temp: float = 0.07
    v2_relation_ema_momentum: float = 0.95
    v2_lambda_relation: float = 1.0
    v2_relation_align_mode: str = "projection"
    v2_use_occupancy_weighting: bool = False
    v2_occupancy_ema_momentum: float = 0.95
    v2_occupancy_weight_alpha: float = 0.5
    v2_occupancy_weight_min: float = 0.5
    v2_occupancy_weight_max: float = 2.0
    v2_use_embedding_steering: bool = False
    v2_steer_beta_max: float = 0.15
    v2_steer_centrality_tau: float = 0.6
    v2_use_tempered_resampling: bool = False
    v2_sample_ema_momentum: float = 0.9
    v2_resample_warmup_epochs: int = 1
    v2_resample_temperature: float = 1.5
    v2_resample_mix: float = 0.7
    v2_resample_proto_alpha: float = 0.75
    v2_resample_diff_alpha: float = 0.5
    v2_resample_coverage_alpha: float = 0.5
    v2_resample_reliability_alpha: float = 0.0
    v2_resample_novelty_alpha: float = 0.5
    v2_resample_weight_min: float = 0.25
    v2_resample_weight_max: float = 4.0
    v2_best_epoch_min: int = 1
    v2_patience: int = 5
    v2_min_delta: float = 1e-4
    v2_val_lambda_relation: float = 1.0
    v2_experiment_tag: str = "v2"

    def resolve(self) -> "ExperimentConfig":
        cfg = copy.deepcopy(self)
        cfg.repo_root = cfg.repo_root.resolve()
        cfg.teacher_checkpoint = _resolve_path(cfg.repo_root, cfg.teacher_checkpoint)
        cfg.wesad_base_dir = _resolve_path(cfg.repo_root, cfg.wesad_base_dir)
        cfg.sdb_public_root = _resolve_path(cfg.repo_root, cfg.sdb_public_root)
        cfg.cache_dir = _resolve_path(cfg.repo_root, cfg.cache_dir)
        cfg.results_dir = _resolve_path(cfg.repo_root, cfg.results_dir)
        return cfg


def _resolve_path(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def seed_all(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def get_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def disable_dropout_inplace(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.p = 0.0


class NormWearTeacherEncoder(nn.Module):
    def __init__(
        self,
        weight_path: str | Path,
        *,
        device: str | torch.device,
        sampling_rate: int = 50,
        patch_pool: str = "mean",
        channel_pool: str = "mean",
        optimized_cwt: bool = True,
    ):
        super().__init__()
        self.runtime_device = get_device(device)
        self.sampling_rate = int(sampling_rate)
        self.patch_pool = str(patch_pool)
        self.channel_pool = str(channel_pool)
        self.encoder = NormWearModel(
            weight_path=str(weight_path),
            optimized_cwt=bool(optimized_cwt),
        ).to(self.runtime_device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x_cpu = x.detach().to("cpu", dtype=torch.float32)
        out = self.encoder.get_embedding(
            x_cpu,
            sampling_rate=self.sampling_rate,
            device=self.runtime_device,
        )
        if self.patch_pool == "mean":
            out = out.mean(dim=2)
        elif self.patch_pool == "cls":
            out = out[:, :, 0, :]
        else:
            raise ValueError(f"Unsupported patch_pool={self.patch_pool}")

        if self.channel_pool == "mean":
            out = out.mean(dim=1)
        elif self.channel_pool == "flatten":
            out = out.flatten(1)
        else:
            raise ValueError(f"Unsupported channel_pool={self.channel_pool}")
        return out.float()


class NormWearStudentEncoder(nn.Module):
    def __init__(
        self,
        *,
        device: str | torch.device,
        sampling_rate: int,
        patch_pool: str,
        channel_pool: str,
        embed_dim: int,
        depth: int,
        num_heads: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        mlp_ratio: float,
        fuse_freq: int,
        nvar: int,
    ):
        super().__init__()
        self.runtime_device = get_device(device)
        self.sampling_rate = int(sampling_rate)
        self.patch_pool = str(patch_pool)
        self.channel_pool = str(channel_pool)
        self.backbone = NormWear(
            img_size=(387, 65),
            patch_size=(9, 5),
            in_chans=3,
            target_len=388,
            nvar=int(nvar),
            embed_dim=int(embed_dim),
            decoder_embed_dim=int(decoder_embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            decoder_depth=int(decoder_depth),
            mlp_ratio=float(mlp_ratio),
            fuse_freq=int(fuse_freq),
            is_pretrain=True,
            mask_scheme="random",
            mask_prob=0.8,
            use_cwt=True,
            no_fusion=False,
            mean_fuse=False,
        ).to(self.runtime_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = x.detach().to(self.runtime_device, dtype=torch.float32)
        batch_size, nvar, length = x.shape
        cwt_res = cwt_wrap(x.view(batch_size * nvar, length), 0.1, 64)
        _, n_, new_length, n_scale = cwt_res.shape
        spec = cwt_res.view(batch_size, nvar, n_, new_length, n_scale).float()
        out = self._signal_embedding_with_grad(spec.to(self.runtime_device))
        if self.patch_pool == "mean":
            out = out.mean(dim=2)
        elif self.patch_pool == "cls":
            out = out[:, :, 0, :]
        else:
            raise ValueError(f"Unsupported patch_pool={self.patch_pool}")

        if self.channel_pool == "mean":
            out = out.mean(dim=1)
        elif self.channel_pool == "flatten":
            out = out.flatten(1)
        else:
            raise ValueError(f"Unsupported channel_pool={self.channel_pool}")
        return out.float()

    def _signal_embedding_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, nvar, channels, length, n_freq = x.shape
        patches = self.backbone.patch_embed(x.reshape(batch_size * nvar, channels, length, n_freq))
        pos_embed = self.backbone.pos_adjust((length, n_freq), device=self.runtime_device)
        patches = patches + pos_embed[:, 1:, :]

        cls_token = self.backbone.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(patches.shape[0], -1, -1)
        patches = torch.cat((cls_tokens, patches), dim=1)

        for block in self.backbone.encoder_blocks:
            block.nvar = nvar
            patches = block(patches)

        patches = self.backbone.norm(patches)
        _, n_patch, embed_dim = patches.shape
        patches = patches.reshape(batch_size, nvar, n_patch, embed_dim)
        if self.backbone.comb_freq:
            patches = self.backbone.combine_freq_patch(patches)
        return patches


class StudentOutputAdapter(nn.Module):
    def __init__(self, encoder: nn.Module, output_dim: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.output_head: nn.Module | None = None
        self.output_dim = None if output_dim is None else int(output_dim)

    def attach_output_head(self, sample_x: torch.Tensor, device: torch.device, output_dim: int) -> None:
        raw_dim = infer_model_embed_dim(self.encoder, sample_x, device)
        self.output_dim = int(output_dim)
        if raw_dim == int(output_dim):
            self.output_head = None
            return
        # Guard against callers wrapping head attachment in inference_mode().
        # Trainable parameters created under inference_mode become inference
        # tensors and later fail when the optimizer performs in-place updates.
        with torch.inference_mode(False):
            self.output_head = nn.Linear(raw_dim, int(output_dim), bias=False).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if z.ndim > 2:
            z = z.flatten(1)
        if self.output_head is not None:
            z = self.output_head(z)
        return z.float()


class PulseDBPublicDataset(Dataset):
    def __init__(
        self,
        x_path: str | Path,
        subj_path: str | Path,
        *,
        normalize: bool,
        target_len: int,
    ):
        self.X = np.load(str(x_path), mmap_mode="r")
        self.subject = np.load(str(subj_path), mmap_mode="r")
        if len(self.X) != len(self.subject):
            raise ValueError("PulseDB public X and subject arrays have different lengths")
        self.normalize = bool(normalize)
        self.target_len = int(target_len)

    def __len__(self) -> int:
        return int(len(self.X))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32)).view(1, -1)
        if x.shape[-1] != self.target_len:
            x = F.interpolate(x.unsqueeze(0), size=self.target_len, mode="linear", align_corners=False).squeeze(0)
        if self.normalize:
            x = zscore_per_sample(x)
        subject = torch.tensor(int(self.subject[idx]), dtype=torch.long)
        return x, subject


class SDBPublicDataset(Dataset):
    def __init__(self, root_dir: str | Path, *, fs: int, normalize: bool):
        base_datasets = [
            SDBNpyDataset(str(root_dir), split, fs=fs, mmap=True)
            for split in ("train", "val", "test")
        ]
        self.dataset = ConcatDataset(base_datasets)
        self.subject_arrays = [
            np.load(Path(root_dir) / f"{split}_subject_{fs}Hz.npy", mmap_mode="r")
            for split in ("train", "val", "test")
        ]
        self.split_offsets = np.cumsum([0] + [len(arr) for arr in self.subject_arrays])
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return int(len(self.dataset))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.dataset[idx]
        if self.normalize:
            x = zscore_per_sample(x)
        split_idx = int(np.searchsorted(self.split_offsets[1:], idx, side="right"))
        local_idx = idx - int(self.split_offsets[split_idx])
        subject = torch.tensor(int(self.subject_arrays[split_idx][local_idx]), dtype=torch.long)
        return x.float(), subject


def zscore_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (std + float(eps))


def build_public_dataset(source: str, cfg: ExperimentConfig) -> Dataset:
    target_len = int(round(float(cfg.public_target_fs) * float(cfg.public_win_sec)))
    if source == "pulsedb":
        return PulseDBPublicDataset(
            cfg.pulsedb_public_ppg_path,
            cfg.pulsedb_public_subj_path,
            normalize=cfg.public_normalize,
            target_len=target_len,
        )
    if source == "sdb":
        return SDBPublicDataset(
            cfg.sdb_public_root,
            fs=cfg.sdb_public_fs,
            normalize=cfg.public_normalize,
        )
    raise ValueError(f"Unsupported public source: {source}")


def build_public_subset_loader(
    dataset: Dataset,
    *,
    n: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> tuple[DataLoader, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    total = len(dataset)
    n = min(int(n), int(total))
    idx = rng.choice(total, size=n, replace=False).astype(np.int64)
    subset = Subset(dataset, idx.tolist())
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return loader, idx


def cache_public_teacher_embeddings(
    teacher: NormWearTeacherEncoder,
    public_dataset: Dataset,
    *,
    source: str,
    cfg: ExperimentConfig,
) -> tuple[TensorDataset, dict[str, Any]]:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.cache_dir / f"normwear_public_cache_{source}_n{int(cfg.public_n)}_seed{int(cfg.public_seed)}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        ds = TensorDataset(payload["x"], payload["t"])
        return ds, payload["meta"]

    device = teacher.runtime_device
    loader, sampled_idx = build_public_subset_loader(
        public_dataset,
        n=cfg.public_n,
        seed=cfg.public_seed,
        batch_size=cfg.kd_batch_size,
        num_workers=cfg.kd_num_workers,
        device=device,
        shuffle=False,
    )

    x_batches: list[torch.Tensor] = []
    t_batches: list[torch.Tensor] = []
    teacher.eval()
    with torch.inference_mode():
        for x, _ in tqdm(loader, desc=f"Cache teacher {source}", leave=False):
            x = x.float()
            t = teacher(x).detach().cpu()
            x_batches.append(x.cpu())
            t_batches.append(t)

    x_all = torch.cat(x_batches, dim=0)
    t_all = torch.cat(t_batches, dim=0)
    meta = {
        "source": str(source),
        "public_n": int(len(sampled_idx)),
        "sampled_idx": sampled_idx.tolist(),
        "x_shape": tuple(x_all.shape),
        "t_shape": tuple(t_all.shape),
    }
    torch.save({"x": x_all, "t": t_all, "meta": meta}, cache_path)
    return TensorDataset(x_all, t_all), meta


def cache_mixed_public_teacher_embeddings(
    teacher: NormWearTeacherEncoder,
    *,
    source_counts: dict[str, int],
    mix_name: str,
    cfg: ExperimentConfig,
) -> tuple[TensorDataset, dict[str, Any]]:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    total_n = int(sum(int(count) for count in source_counts.values()))
    cache_path = cfg.cache_dir / f"normwear_public_cache_{mix_name}_n{total_n}_seed{int(cfg.public_seed)}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        ds = TensorDataset(payload["x"], payload["t"])
        return ds, payload["meta"]

    x_parts: list[torch.Tensor] = []
    t_parts: list[torch.Tensor] = []
    component_meta: dict[str, Any] = {}

    for source, count in source_counts.items():
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg.public_n = int(count)
        public_dataset = build_public_dataset(str(source), sub_cfg)
        cached_dataset, meta = cache_public_teacher_embeddings(
            teacher,
            public_dataset,
            source=str(source),
            cfg=sub_cfg,
        )
        x_parts.append(cached_dataset.tensors[0].cpu())
        t_parts.append(cached_dataset.tensors[1].cpu())
        component_meta[str(source)] = meta

    x_all = torch.cat(x_parts, dim=0)
    t_all = torch.cat(t_parts, dim=0)

    rng = np.random.default_rng(int(cfg.public_seed))
    perm = torch.from_numpy(rng.permutation(len(x_all)).astype(np.int64))
    x_all = x_all[perm]
    t_all = t_all[perm]

    meta = {
        "source": str(mix_name),
        "public_n": int(total_n),
        "source_counts": {str(source): int(count) for source, count in source_counts.items()},
        "x_shape": tuple(x_all.shape),
        "t_shape": tuple(t_all.shape),
        "component_meta": component_meta,
    }
    torch.save({"x": x_all, "t": t_all, "meta": meta}, cache_path)
    return TensorDataset(x_all, t_all), meta


def build_tiny_student(
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    output_dim: int | None = None,
) -> nn.Module:
    if cfg.student_arch == "normwear_tiny":
        encoder = NormWearStudentEncoder(
            device=device,
            sampling_rate=cfg.teacher_sampling_rate,
            patch_pool=cfg.teacher_patch_pool,
            channel_pool=cfg.teacher_channel_pool,
            embed_dim=cfg.normwear_student_embed_dim,
            depth=cfg.normwear_student_depth,
            num_heads=cfg.normwear_student_num_heads,
            decoder_embed_dim=cfg.normwear_student_decoder_embed_dim,
            decoder_depth=cfg.normwear_student_decoder_depth,
            mlp_ratio=cfg.normwear_student_mlp_ratio,
            fuse_freq=cfg.normwear_student_fuse_freq,
            nvar=cfg.normwear_student_nvar,
        ).to(device)
        return StudentOutputAdapter(encoder, output_dim=output_dim).to(device)

    encoder = ResNet1DNet(
        in_channels=1,
        base_filters=int(cfg.student_base_filters),
        kernel_size=int(cfg.student_kernel_size),
        stride=int(cfg.student_stride),
        groups=int(cfg.student_groups),
        n_block=int(cfg.student_n_block),
        finalpool=str(cfg.student_finalpool),
        downsample_gap=int(cfg.student_downsample_gap),
        increasefilter_gap=int(cfg.student_increasefilter_gap),
        use_bn=bool(cfg.student_use_bn),
        use_do=bool(cfg.student_use_do),
        verbose=False,
    )
    return StudentOutputAdapter(encoder, output_dim=output_dim).to(device)


def infer_model_embed_dim(model: nn.Module, x: torch.Tensor, device: torch.device) -> int:
    with torch.inference_mode():
        z = model(x.to(device))
    if z.ndim > 2:
        z = z.flatten(1)
    return int(z.shape[-1])


def infer_embed_dim(model: nn.Module, loader: DataLoader, device: torch.device) -> int:
    batch = next(iter(loader))
    return infer_model_embed_dim(model, batch[0], device)


def make_cached_distill_loader(dataset: Dataset, cfg: ExperimentConfig, device: torch.device, *, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg.kd_batch_size),
        shuffle=bool(shuffle),
        num_workers=int(cfg.kd_num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def resolve_cached_dataset_tensors(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dataset, TensorDataset):
        if len(dataset.tensors) < 2:
            raise ValueError("Expected cached TensorDataset to contain at least two tensors")
        return dataset.tensors[0], dataset.tensors[1]
    if isinstance(dataset, Subset):
        x_all, t_all = resolve_cached_dataset_tensors(dataset.dataset)
        subset_idx = torch.as_tensor(dataset.indices, dtype=torch.long)
        return x_all.index_select(0, subset_idx), t_all.index_select(0, subset_idx)
    raise TypeError(f"Unsupported cached dataset type: {type(dataset)!r}")


def iter_cached_epoch_batches(
    x_all: torch.Tensor,
    t_all: torch.Tensor,
    sample_idx: torch.Tensor,
    *,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    total = int(sample_idx.numel())
    for start in range(0, total, int(batch_size)):
        batch_idx = sample_idx[start : start + int(batch_size)]
        yield batch_idx, x_all.index_select(0, batch_idx), t_all.index_select(0, batch_idx)


def split_cached_dataset(
    dataset: Dataset,
    *,
    val_n: int,
    seed: int,
) -> tuple[Dataset, Dataset | None, dict[str, Any]]:
    total_n = int(len(dataset))
    val_n = max(0, min(int(val_n), total_n - 1))
    if val_n == 0:
        return dataset, None, {"total_n": total_n, "train_n": total_n, "val_n": 0}

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(total_n).astype(np.int64)
    val_idx = perm[:val_n].tolist()
    train_idx = perm[val_n:].tolist()
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    return train_ds, val_ds, {"total_n": total_n, "train_n": len(train_idx), "val_n": len(val_idx)}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _as_2d_embedding(z: torch.Tensor) -> torch.Tensor:
    if z.ndim == 2:
        return z
    if z.ndim == 3:
        return z.mean(dim=-1)
    return z.flatten(1)


def _augment_ppg_for_refine(
    x: torch.Tensor,
    *,
    noise_scale: float = 0.02,
    drop_p: float = 0.05,
) -> torch.Tensor:
    x2 = x.detach().clone().float()
    batch_n = int(x2.shape[0])
    amp = 0.8 + 0.4 * torch.rand((batch_n, 1, 1), device=x2.device, dtype=x2.dtype)
    x2 = x2 * amp

    sig = noise_scale * x2.std(dim=-1, keepdim=True).clamp_min(1e-6)
    x2 = x2 + torch.randn_like(x2) * sig

    if drop_p > 0.0:
        keep_p = float(1.0 - float(drop_p))
        mask = (torch.rand_like(x2) < keep_p).to(x2.dtype) / max(1e-6, keep_p)
        x2 = x2 * mask
    return x2


def _allocate_subset_counts(counts: np.ndarray, target: int) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    target = int(target)
    total = int(counts.sum())
    if total <= 0 or target <= 0:
        return np.zeros_like(counts, dtype=np.int64)

    raw = (counts.astype(np.float64) / float(total)) * float(target)
    base = np.floor(raw).astype(np.int64)
    base = np.minimum(base, counts)
    remain = int(target - base.sum())

    if remain > 0:
        frac = raw - base.astype(np.float64)
        for cluster_idx in np.argsort(-frac):
            if remain <= 0:
                break
            if base[cluster_idx] < counts[cluster_idx]:
                base[cluster_idx] += 1
                remain -= 1
    elif remain < 0:
        frac = raw - base.astype(np.float64)
        for cluster_idx in np.argsort(frac):
            if remain >= 0:
                break
            if base[cluster_idx] > 0:
                base[cluster_idx] -= 1
                remain += 1

    base = np.minimum(base, counts)
    remain = int(target - base.sum())
    if remain > 0:
        for cluster_idx in np.argsort(-counts):
            if remain <= 0:
                break
            if base[cluster_idx] < counts[cluster_idx]:
                base[cluster_idx] += 1
                remain -= 1
    elif remain < 0:
        for cluster_idx in np.argsort(counts):
            if remain >= 0:
                break
            if base[cluster_idx] > 0:
                base[cluster_idx] -= 1
                remain += 1
    return base


def refine_cached_public_dataset(
    teacher: NormWearTeacherEncoder,
    cached_dataset: Dataset,
    *,
    source: str,
    cfg: ExperimentConfig,
) -> tuple[TensorDataset, dict[str, Any]]:
    x_all, t_all = resolve_cached_dataset_tensors(cached_dataset)
    candidate_n = int(len(x_all))
    final_n = int(cfg.public_n)
    if candidate_n < final_n:
        raise ValueError(f"Refine candidate pool too small for {source}: candidate_n={candidate_n} final_n={final_n}")
    if candidate_n == final_n:
        meta = {
            "subset_mode": "refined",
            "source": str(source),
            "candidate_n": int(candidate_n),
            "final_n": int(final_n),
            "note": "candidate_n equals final_n; returned candidate cache unchanged",
        }
        return TensorDataset(x_all.clone(), t_all.clone()), meta

    refine_seed = int(cfg.public_refine_seed if cfg.public_refine_seed is not None else cfg.public_seed)
    core_ratio = _clip01(getattr(cfg, "public_refine_core_ratio", 0.7))
    if not (0.0 < core_ratio < 1.0):
        raise ValueError("public_refine_core_ratio must be in (0, 1)")

    refine_tag = (
        f"{str(source)}_final{final_n}_cand{candidate_n}_seed{refine_seed}"
        f"_k{int(getattr(cfg, 'public_refine_kmeans_k', 2048))}"
    )
    cache_path = cfg.cache_dir / f"normwear_public_refine_{refine_tag}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        refined_ds = TensorDataset(payload["x"], payload["t"])
        return refined_ds, payload["meta"]

    batch_size = int(getattr(cfg, "public_refine_batch_size", cfg.kd_batch_size) or cfg.kd_batch_size)
    num_workers = getattr(cfg, "public_refine_num_workers", None)
    num_workers = int(cfg.kd_num_workers if num_workers is None else num_workers)

    z_base = _as_2d_embedding(t_all.float()).cpu()
    x_dataset = TensorDataset(x_all)
    x_loader = DataLoader(
        x_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(teacher.runtime_device.type == "cuda"),
        drop_last=False,
    )

    inc_chunks: list[torch.Tensor] = []
    teacher.eval()
    with torch.inference_mode():
        offset = 0
        for (x_batch,) in tqdm(x_loader, desc=f"Refine public {source}", leave=False):
            z_batch = z_base[offset : offset + int(x_batch.shape[0])]
            x_aug = _augment_ppg_for_refine(x_batch)
            z_aug = _as_2d_embedding(teacher(x_aug)).float().cpu()
            cos = F.cosine_similarity(z_batch, z_aug, dim=1, eps=1e-8)
            inc_chunks.append((1.0 - cos).detach().cpu())
            offset += int(x_batch.shape[0])

    inc = torch.cat(inc_chunks, dim=0).numpy().astype(np.float32)
    z_np = z_base.numpy().astype(np.float32)
    good = np.isfinite(z_np).all(axis=1) & np.isfinite(inc)
    if not bool(good.all()):
        good_idx = torch.from_numpy(np.where(good)[0].astype(np.int64)).long()
        z_np = z_np[good]
        inc = inc[good]
        x_all = x_all.index_select(0, good_idx)
        t_all = t_all.index_select(0, good_idx)
        candidate_n = int(len(x_all))
        if candidate_n < final_n:
            raise RuntimeError(f"Too many non-finite rows removed for {source}: candidate_n={candidate_n} final_n={final_n}")

    z_np = z_np / (np.linalg.norm(z_np, axis=1, keepdims=True) + 1e-12)
    cluster_k = int(min(int(getattr(cfg, "public_refine_kmeans_k", 2048)), int(z_np.shape[0])))
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_k,
        random_state=refine_seed,
        batch_size=int(min(int(getattr(cfg, "public_refine_kmeans_batch", 4096)), int(z_np.shape[0]))),
        n_init=10,
        reassignment_ratio=0.01,
    )
    kmeans.fit(z_np)
    labels = kmeans.predict(z_np).astype(np.int64)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    dist = np.linalg.norm(z_np - centers[labels], axis=1).astype(np.float32)

    dist_hi = float(np.quantile(dist, float(getattr(cfg, "public_refine_dist_hi_q", 0.9))))
    inc_hi = float(np.quantile(inc, float(getattr(cfg, "public_refine_inc_hi_q", 0.9))))
    inlier = (dist <= dist_hi) & (inc <= inc_hi)
    if int(inlier.sum()) < final_n:
        inlier = np.ones_like(inlier, dtype=bool)

    core_inc_max = float(np.quantile(inc[inlier], float(getattr(cfg, "public_refine_core_inc_q", 0.5))))
    hard_inc_lo = float(np.quantile(inc[inlier], float(getattr(cfg, "public_refine_hard_inc_lo_q", 0.6))))
    hard_inc_hi = float(np.quantile(inc[inlier], float(getattr(cfg, "public_refine_hard_inc_hi_q", 0.9))))
    hard_dist_lo = float(np.quantile(dist[inlier], float(getattr(cfg, "public_refine_hard_dist_lo_q", 0.5))))
    hard_dist_hi = float(np.quantile(dist[inlier], float(getattr(cfg, "public_refine_hard_dist_hi_q", 0.9))))

    core_pool = np.where(inlier & (inc <= core_inc_max))[0]
    hard_pool = np.where(
        inlier
        & (inc >= hard_inc_lo)
        & (inc <= hard_inc_hi)
        & (dist >= hard_dist_lo)
        & (dist <= hard_dist_hi)
    )[0]
    if int(hard_pool.size) < int((1.0 - core_ratio) * float(final_n)):
        hard_pool = np.where(inlier)[0]

    core_target = int(round(float(final_n) * core_ratio))
    hard_target = int(final_n - core_target)
    counts_core = np.bincount(labels[core_pool], minlength=cluster_k).astype(np.int64)
    counts_hard = np.bincount(labels[hard_pool], minlength=cluster_k).astype(np.int64)
    alloc_core = _allocate_subset_counts(counts_core, core_target)
    alloc_hard = _allocate_subset_counts(counts_hard, hard_target)

    selected: list[int] = []
    selected_set: set[int] = set()
    for cluster_idx in range(cluster_k):
        take_n = int(alloc_core[cluster_idx])
        if take_n <= 0:
            continue
        cluster_pool = core_pool[labels[core_pool] == cluster_idx]
        if cluster_pool.size == 0:
            continue
        if take_n >= cluster_pool.size:
            chosen = cluster_pool
        else:
            chosen = cluster_pool[np.argpartition(dist[cluster_pool], take_n - 1)[:take_n]]
        for sample_idx in chosen.tolist():
            sample_idx = int(sample_idx)
            if sample_idx not in selected_set:
                selected_set.add(sample_idx)
                selected.append(sample_idx)

    for cluster_idx in range(cluster_k):
        take_n = int(alloc_hard[cluster_idx])
        if take_n <= 0:
            continue
        cluster_pool = hard_pool[labels[hard_pool] == cluster_idx]
        if cluster_pool.size == 0:
            continue
        if take_n >= cluster_pool.size:
            chosen = cluster_pool
        else:
            part = np.argpartition(dist[cluster_pool], cluster_pool.size - take_n)
            chosen = cluster_pool[part[-take_n:]]
        for sample_idx in chosen.tolist():
            sample_idx = int(sample_idx)
            if sample_idx not in selected_set:
                selected_set.add(sample_idx)
                selected.append(sample_idx)

    rng = np.random.default_rng(refine_seed)
    if len(selected) < final_n:
        selected_arr = np.asarray(selected, dtype=np.int64)
        remaining_pool = np.where(inlier)[0]
        remaining_pool = remaining_pool[~np.isin(remaining_pool, selected_arr)]
        need = int(final_n - len(selected))
        if remaining_pool.size < need:
            remaining_pool = np.arange(candidate_n, dtype=np.int64)
            remaining_pool = remaining_pool[~np.isin(remaining_pool, selected_arr)]
        extra = rng.choice(remaining_pool, size=need, replace=False).astype(np.int64)
        selected.extend(extra.tolist())

    selected_idx = np.asarray(selected[:final_n], dtype=np.int64)
    rng.shuffle(selected_idx)
    selected_idx_t = torch.from_numpy(selected_idx).long()
    refined_x = x_all.index_select(0, selected_idx_t)
    refined_t = t_all.index_select(0, selected_idx_t)
    meta = {
        "subset_mode": "refined",
        "source": str(source),
        "candidate_n": int(candidate_n),
        "final_n": int(final_n),
        "refine_seed": int(refine_seed),
        "kmeans_k": int(cluster_k),
        "core_ratio": float(core_ratio),
        "dist_hi": float(dist_hi),
        "inc_hi": float(inc_hi),
        "core_inc_max": float(core_inc_max),
        "hard_inc_lo": float(hard_inc_lo),
        "hard_inc_hi": float(hard_inc_hi),
        "hard_dist_lo": float(hard_dist_lo),
        "hard_dist_hi": float(hard_dist_hi),
        "selected_unique_n": int(np.unique(selected_idx).size),
        "selected_idx": selected_idx.tolist(),
    }
    torch.save({"x": refined_x, "t": refined_t, "meta": meta}, cache_path)
    return TensorDataset(refined_x, refined_t), meta


def select_best_history_row(
    history_df: pd.DataFrame,
    *,
    min_epoch: int = 1,
    score_col: str = "val_score",
) -> pd.Series:
    if history_df.empty:
        raise ValueError("Cannot select a best row from an empty history DataFrame")

    min_epoch = max(1, int(min_epoch))
    candidate_df = history_df.copy()
    if "checkpoint_eligible" in candidate_df.columns:
        eligible_df = candidate_df[candidate_df["checkpoint_eligible"].astype(bool)].copy()
        if not eligible_df.empty:
            candidate_df = eligible_df
        elif "epoch" in candidate_df.columns:
            min_epoch_df = candidate_df[candidate_df["epoch"] >= min_epoch].copy()
            if not min_epoch_df.empty:
                candidate_df = min_epoch_df
    elif "epoch" in candidate_df.columns and min_epoch > 1:
        min_epoch_df = candidate_df[candidate_df["epoch"] >= min_epoch].copy()
        if not min_epoch_df.empty:
            candidate_df = min_epoch_df

    if score_col in candidate_df.columns:
        return candidate_df.sort_values([score_col, "epoch"], ascending=[True, True]).iloc[0]
    return candidate_df.sort_values(["epoch"], ascending=[True]).iloc[-1]


def align_student_output(
    student: nn.Module,
    sample_x: torch.Tensor,
    *,
    device: torch.device,
    target_dim: int,
    align_mode: str,
) -> tuple[nn.Module | None, list[nn.Parameter]]:
    align_mode = str(align_mode)
    if align_mode == "same_dim":
        if isinstance(student, StudentOutputAdapter):
            student.attach_output_head(sample_x, device, target_dim)
        else:
            student_dim = infer_model_embed_dim(student, sample_x, device)
            if student_dim != int(target_dim):
                raise ValueError("same_dim mode requires the student to emit teacher-sized embeddings")
        return None, list(student.parameters())

    if align_mode == "projection":
        student_dim = infer_model_embed_dim(student, sample_x, device)
        projector = nn.Linear(student_dim, int(target_dim), bias=False).to(device)
        return projector, list(student.parameters()) + list(projector.parameters())

    raise ValueError(f"Unsupported align_mode={align_mode}")


class HPTKDPrototypeMemory:
    def __init__(self, num_prototypes: int, embed_dim: int, *, momentum: float, device: torch.device):
        self.num_prototypes = int(num_prototypes)
        self.embed_dim = int(embed_dim)
        self.momentum = float(momentum)
        self.device = device
        self.prototypes = torch.zeros(self.num_prototypes, self.embed_dim, device=device, dtype=torch.float32)
        self.initialized = False

    def initialize(self, teacher_z: torch.Tensor) -> None:
        teacher_z = F.normalize(teacher_z.detach().float(), dim=1)
        n = int(teacher_z.shape[0])
        if n == 0:
            raise ValueError("Cannot initialize prototype memory from an empty batch")
        if n >= self.num_prototypes:
            idx = torch.randperm(n, device=teacher_z.device)[: self.num_prototypes]
            proto = teacher_z[idx]
        else:
            repeat = int(np.ceil(self.num_prototypes / max(1, n)))
            proto = teacher_z.repeat(repeat, 1)[: self.num_prototypes]
        self.prototypes.copy_(F.normalize(proto, dim=1))
        self.initialized = True

    def lookup(self, teacher_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher_z = F.normalize(teacher_z.detach().float(), dim=1)
        if not self.initialized:
            self.initialize(teacher_z)
        sim = teacher_z @ self.prototypes.T
        assign = sim.argmax(dim=1)
        proto = self.prototypes[assign]
        dist_sq = (teacher_z - proto).pow(2).sum(dim=1)
        return proto, assign, dist_sq

    @torch.no_grad()
    def update(self, teacher_z: torch.Tensor, assign: torch.Tensor) -> None:
        teacher_z = F.normalize(teacher_z.detach().float(), dim=1)
        for proto_idx in assign.unique(sorted=False):
            mask = assign == proto_idx
            if not bool(mask.any()):
                continue
            batch_mean = F.normalize(teacher_z[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            old = self.prototypes[int(proto_idx)]
            new = self.momentum * old + (1.0 - self.momentum) * batch_mean
            self.prototypes[int(proto_idx)] = F.normalize(new.unsqueeze(0), dim=1).squeeze(0)


class PrototypeRelationEMAMemory:
    def __init__(self, num_prototypes: int, *, momentum: float, device: torch.device):
        self.num_prototypes = int(num_prototypes)
        self.momentum = float(momentum)
        self.device = device
        self.targets = torch.full(
            (self.num_prototypes, self.num_prototypes),
            1.0 / float(max(1, self.num_prototypes)),
            device=device,
            dtype=torch.float32,
        )
        self.initialized = torch.zeros(self.num_prototypes, device=device, dtype=torch.bool)

    @torch.no_grad()
    def get_targets(self, assign: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        out = self.targets[assign].clone()
        missing = ~self.initialized[assign]
        if bool(missing.any()):
            out[missing] = teacher_probs.detach()[missing]
        return out

    @torch.no_grad()
    def update(self, assign: torch.Tensor, teacher_probs: torch.Tensor) -> None:
        teacher_probs = teacher_probs.detach().float()
        for proto_idx in assign.unique(sorted=False):
            mask = assign == proto_idx
            if not bool(mask.any()):
                continue
            batch_mean = teacher_probs[mask].mean(dim=0)
            batch_mean = batch_mean / batch_mean.sum().clamp_min(1e-6)
            proto_id = int(proto_idx)
            if bool(self.initialized[proto_id]):
                updated = self.momentum * self.targets[proto_id] + (1.0 - self.momentum) * batch_mean
                self.targets[proto_id] = updated / updated.sum().clamp_min(1e-6)
            else:
                self.targets[proto_id] = batch_mean
                self.initialized[proto_id] = True


class PrototypeOccupancyEMAMemory:
    def __init__(self, num_prototypes: int, *, momentum: float, device: torch.device):
        self.num_prototypes = int(num_prototypes)
        self.momentum = float(momentum)
        self.device = device
        self.counts = torch.ones(self.num_prototypes, device=device, dtype=torch.float32)

    @torch.no_grad()
    def get_weights(
        self,
        assign: torch.Tensor,
        *,
        alpha: float,
        weight_min: float,
        weight_max: float,
    ) -> torch.Tensor:
        proto_counts = self.counts[assign].clamp_min(1e-6)
        mean_count = self.counts.mean().clamp_min(1e-6)
        weights = torch.pow(mean_count / proto_counts, float(alpha))
        weights = weights.clamp(float(weight_min), float(weight_max))
        return weights / weights.mean().clamp_min(1e-6)

    @torch.no_grad()
    def update(self, assign: torch.Tensor) -> None:
        batch_counts = torch.bincount(assign.detach(), minlength=self.num_prototypes).to(self.counts.dtype)
        self.counts.mul_(self.momentum).add_(batch_counts * (1.0 - self.momentum))


class SampleRelationHistoryMemory:
    def __init__(self, num_samples: int, *, momentum: float, device: torch.device):
        self.num_samples = int(num_samples)
        self.momentum = float(momentum)
        self.device = device
        self.relation_gap = torch.ones(self.num_samples, device=device, dtype=torch.float32)
        self.proto_centrality = torch.zeros(self.num_samples, device=device, dtype=torch.float32)
        self.relation_reliability = torch.ones(self.num_samples, device=device, dtype=torch.float32)
        self.seen_count = torch.zeros(self.num_samples, device=device, dtype=torch.float32)
        self.initialized = torch.zeros(self.num_samples, device=device, dtype=torch.bool)

    @torch.no_grad()
    def update(
        self,
        sample_idx: torch.Tensor,
        relation_gap: torch.Tensor,
        proto_sim: torch.Tensor,
        relation_reliability: torch.Tensor,
    ) -> None:
        idx = sample_idx.to(self.device, dtype=torch.long)
        gap = relation_gap.detach().to(self.device).float()
        centrality = proto_sim.detach().to(self.device).float().clamp(-1.0, 1.0)
        reliability = relation_reliability.detach().to(self.device).float().clamp(0.0, 1.0)
        for sid in idx.unique(sorted=False):
            sid_int = int(sid)
            mask = idx == sid
            gap_mean = gap[mask].mean()
            centrality_mean = centrality[mask].mean()
            reliability_mean = reliability[mask].mean()
            if bool(self.initialized[sid_int]):
                self.relation_gap[sid_int] = self.momentum * self.relation_gap[sid_int] + (1.0 - self.momentum) * gap_mean
                self.proto_centrality[sid_int] = self.momentum * self.proto_centrality[sid_int] + (1.0 - self.momentum) * centrality_mean
                self.relation_reliability[sid_int] = self.momentum * self.relation_reliability[sid_int] + (1.0 - self.momentum) * reliability_mean
            else:
                self.relation_gap[sid_int] = gap_mean
                self.proto_centrality[sid_int] = centrality_mean
                self.relation_reliability[sid_int] = reliability_mean
                self.initialized[sid_int] = True
            self.seen_count[sid_int] += float(mask.sum())

    @torch.no_grad()
    def build_weights(
        self,
        assign: torch.Tensor,
        proto_sim: torch.Tensor,
        relation_reliability: torch.Tensor,
        occupancy_memory: PrototypeOccupancyEMAMemory,
        *,
        cfg: ExperimentConfig,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assign = assign.to(self.device, dtype=torch.long)
        proto_sim = proto_sim.to(self.device).float().clamp(0.0, 1.0)
        relation_reliability = relation_reliability.to(self.device).float().clamp(0.0, 1.0)

        proto_counts = occupancy_memory.counts[assign].clamp_min(1e-6)
        mean_count = occupancy_memory.counts.mean().clamp_min(1e-6)
        proto_boost = torch.pow((mean_count / proto_counts).clamp_min(1e-6), float(getattr(cfg, "v2_resample_proto_alpha", 0.75)))

        gap_hist = self.relation_gap.clamp_min(1e-6)
        gap_scale = gap_hist.mean().clamp_min(1e-6)
        difficulty_boost = torch.pow((gap_hist / gap_scale).clamp(0.25, 4.0), float(getattr(cfg, "v2_resample_diff_alpha", 0.5)))

        seen_hist = self.seen_count
        seen_scale = seen_hist[seen_hist > 0].mean() if bool((seen_hist > 0).any()) else torch.tensor(1.0, device=self.device)
        coverage_boost = torch.pow(((seen_scale + 1.0) / (seen_hist + 1.0)).clamp(0.25, 4.0), float(getattr(cfg, "v2_resample_coverage_alpha", 0.5)))

        reliability_hist = torch.where(self.initialized, self.relation_reliability.clamp(0.0, 1.0), relation_reliability)
        reliability_scale = reliability_hist.mean().clamp_min(1e-6)
        reliability_boost = torch.pow(
            (reliability_hist / reliability_scale).clamp(0.25, 4.0),
            float(getattr(cfg, "v2_resample_reliability_alpha", 0.0)),
        )

        centrality_hist = torch.where(self.initialized, self.proto_centrality.clamp(0.0, 1.0), proto_sim)
        novelty = (1.0 - 0.5 * (centrality_hist + proto_sim)).clamp(0.05, 1.0)
        novelty_scale = novelty.mean().clamp_min(1e-6)
        novelty_boost = torch.pow((novelty / novelty_scale).clamp(0.25, 4.0), float(getattr(cfg, "v2_resample_novelty_alpha", 0.5)))

        weights = proto_boost * difficulty_boost * coverage_boost * reliability_boost * novelty_boost
        weights = weights.clamp(
            float(getattr(cfg, "v2_resample_weight_min", 0.25)),
            float(getattr(cfg, "v2_resample_weight_max", 4.0)),
        )
        weights = weights / weights.mean().clamp_min(1e-6)
        stats = {
            "mean_sampling_weight_std": float(weights.std(unbiased=False).detach().cpu()),
            "mean_sampling_weight_max": float(weights.max().detach().cpu()),
            "mean_sampling_proto_boost": float(proto_boost.mean().detach().cpu()),
            "mean_sampling_diff_boost": float(difficulty_boost.mean().detach().cpu()),
            "mean_sampling_coverage_boost": float(coverage_boost.mean().detach().cpu()),
            "mean_sampling_reliability_boost": float(reliability_boost.mean().detach().cpu()),
            "mean_sampling_novelty_boost": float(novelty_boost.mean().detach().cpu()),
        }
        return weights, stats


@torch.no_grad()
def compute_dataset_prototype_assignments(
    teacher_z_all: torch.Tensor,
    proto_memory: HPTKDPrototypeMemory,
    *,
    device: torch.device,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    assign_chunks: list[torch.Tensor] = []
    sim_chunks: list[torch.Tensor] = []
    total = int(teacher_z_all.shape[0])
    for start in range(0, total, int(chunk_size)):
        teacher_chunk = teacher_z_all[start : start + int(chunk_size)].to(device, non_blocking=True)
        t_norm = F.normalize(teacher_chunk.float(), dim=1)
        proto, assign, _ = proto_memory.lookup(t_norm)
        proto_sim = (t_norm * proto).sum(dim=1).clamp(-1.0, 1.0)
        assign_chunks.append(assign.detach().cpu())
        sim_chunks.append(proto_sim.detach().cpu())
    return torch.cat(assign_chunks, dim=0), torch.cat(sim_chunks, dim=0)


def sample_tempered_epoch_indices(
    weights: torch.Tensor,
    *,
    num_samples: int,
    temperature: float,
    mix_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights_cpu = weights.detach().float().cpu().clamp_min(1e-6)
    probs = torch.pow(weights_cpu, 1.0 / max(float(temperature), 1e-6))
    probs = probs / probs.sum().clamp_min(1e-6)
    mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
    uniform = torch.full_like(probs, 1.0 / float(max(1, probs.numel())))
    probs = mix_ratio * probs + (1.0 - mix_ratio) * uniform
    probs = probs / probs.sum().clamp_min(1e-6)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    epoch_idx = torch.multinomial(probs, num_samples=int(num_samples), replacement=True, generator=generator)
    unique_n = int(epoch_idx.unique().numel())
    stats = {
        "epoch_unique_samples": float(unique_n),
        "epoch_unique_ratio": float(unique_n) / float(max(1, num_samples)),
        "epoch_reuse_ratio": 1.0 - float(unique_n) / float(max(1, num_samples)),
        "epoch_sampling_prob_max": float(probs.max().item()),
        "epoch_sampling_prob_min": float(probs.min().item()),
    }
    return epoch_idx, stats


def _normalized_entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return float("nan")
    probs = counts / total
    probs = probs[probs > 0.0]
    if probs.size <= 1:
        return 0.0
    denom = np.log(float(len(counts)))
    if denom <= 0.0:
        return 0.0
    return float(-(probs * np.log(probs)).sum() / denom)


def _gini_from_counts(counts: np.ndarray) -> float:
    counts = np.sort(np.asarray(counts, dtype=np.float64))
    total = float(counts.sum())
    if total <= 0.0:
        return float("nan")
    n = counts.size
    if n == 0:
        return float("nan")
    weights = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(weights, counts)) / (n * total) - (n + 1) / n)


def _summarize_epoch_prototype_stats(
    batch_counts: list[np.ndarray],
    *,
    num_prototypes: int,
    batch_size: int,
) -> dict[str, float]:
    if not batch_counts:
        return {
            "epoch_active_prototypes": 0.0,
            "epoch_top1_share": float("nan"),
            "epoch_top5_share": float("nan"),
            "epoch_occupancy_entropy": float("nan"),
            "epoch_occupancy_gini": float("nan"),
            "epoch_count_cv": float("nan"),
            "epoch_num_rare_prototypes": 0.0,
            "epoch_rare_cluster_cutoff": float("nan"),
            "mean_batch_dominant_share": float("nan"),
            "p95_batch_dominant_share": float("nan"),
            "mean_batch_active_clusters": float("nan"),
            "mean_batch_active_ratio": float("nan"),
            "mean_batch_entropy": float("nan"),
            "mean_rare_cluster_hit_ratio": float("nan"),
            "worst_rare_batch_missing_prob": float("nan"),
        }

    stacked = np.stack(batch_counts, axis=0).astype(np.float64)
    counts = stacked.sum(axis=0)
    active = counts[counts > 0.0]

    dominant_share = stacked.max(axis=1) / np.clip(stacked.sum(axis=1), a_min=1.0, a_max=None)
    active_clusters = (stacked > 0.0).sum(axis=1)
    batch_entropy = np.array([_normalized_entropy_from_counts(row) for row in stacked], dtype=np.float64)

    rare_cutoff = float(np.quantile(active, 0.25)) if active.size else float("nan")
    rare_ids = np.where((counts > 0.0) & (counts <= rare_cutoff))[0] if active.size else np.array([], dtype=np.int64)
    if rare_ids.size:
        rare_hit_ratio = (stacked[:, rare_ids] > 0.0).sum(axis=1) / float(rare_ids.size)
        rare_shares = counts[rare_ids] / np.clip(counts.sum(), a_min=1.0, a_max=None)
        rare_missing_prob = np.power(1.0 - rare_shares, float(batch_size))
        mean_rare_cluster_hit_ratio = float(np.mean(rare_hit_ratio))
        worst_rare_batch_missing_prob = float(np.max(rare_missing_prob))
    else:
        mean_rare_cluster_hit_ratio = float("nan")
        worst_rare_batch_missing_prob = float("nan")

    return {
        "epoch_active_prototypes": float((counts > 0.0).sum()),
        "epoch_top1_share": float(np.max(counts) / np.clip(counts.sum(), a_min=1.0, a_max=None)),
        "epoch_top5_share": float(np.sort(counts)[::-1][:5].sum() / np.clip(counts.sum(), a_min=1.0, a_max=None)),
        "epoch_occupancy_entropy": _normalized_entropy_from_counts(counts),
        "epoch_occupancy_gini": _gini_from_counts(counts),
        "epoch_count_cv": float(active.std() / max(active.mean(), 1e-6)) if active.size else float("nan"),
        "epoch_num_rare_prototypes": float(rare_ids.size),
        "epoch_rare_cluster_cutoff": rare_cutoff,
        "mean_batch_dominant_share": float(np.mean(dominant_share)),
        "p95_batch_dominant_share": float(np.quantile(dominant_share, 0.95)),
        "mean_batch_active_clusters": float(np.mean(active_clusters)),
        "mean_batch_active_ratio": float(np.mean(active_clusters) / float(max(1, num_prototypes))),
        "mean_batch_entropy": float(np.nanmean(batch_entropy)),
        "mean_rare_cluster_hit_ratio": mean_rare_cluster_hit_ratio,
        "worst_rare_batch_missing_prob": worst_rare_batch_missing_prob,
    }


def _hptkd_targets(
    student_z: torch.Tensor,
    teacher_z: torch.Tensor,
    memory: HPTKDPrototypeMemory,
    *,
    cfg: ExperimentConfig,
) -> dict[str, torch.Tensor]:
    s_norm = F.normalize(student_z.float(), dim=1)
    t_norm = F.normalize(teacher_z.float(), dim=1)
    proto, assign, dist_sq = memory.lookup(t_norm)
    dist_norm = dist_sq / (dist_sq.mean().detach() + 1e-6)
    reliability = torch.exp(-dist_norm / float(cfg.hptkd_tau_reliability)).clamp(0.0, 1.0)

    gap = (1.0 - (s_norm.detach() * t_norm).sum(dim=1)).clamp_min(0.0)
    learnability = torch.exp(-gap / float(cfg.hptkd_tau_learnability)).clamp(0.0, 1.0)
    alpha = float(cfg.hptkd_alpha_max) * reliability * (1.0 - learnability)
    steered = F.normalize((1.0 - alpha.unsqueeze(1)) * t_norm + alpha.unsqueeze(1) * proto, dim=1)

    weights_easy = reliability * learnability
    weights_hard = reliability * (1.0 - learnability)
    weights_risk = 1.0 - reliability
    return {
        "student": s_norm,
        "teacher": t_norm,
        "prototype": proto,
        "assign": assign,
        "reliability": reliability,
        "learnability": learnability,
        "alpha": alpha,
        "steered": steered,
        "w_easy": weights_easy,
        "w_hard": weights_hard,
        "w_risk": weights_risk,
    }


def evaluate_hptkd_validation(
    student: nn.Module,
    val_loader: DataLoader,
    memory: HPTKDPrototypeMemory,
    *,
    cfg: ExperimentConfig,
    device: torch.device,
    projector: nn.Module | None = None,
) -> dict[str, float]:
    student.eval()
    if projector is not None:
        projector.eval()

    point_losses: list[float] = []
    cos_losses: list[float] = []
    rel_losses: list[float] = []
    method_losses: list[float] = []

    with torch.inference_mode():
        for x_cpu, t_cpu in val_loader:
            x = x_cpu.to(device, non_blocking=True)
            t = t_cpu.to(device, non_blocking=True)
            s = student(x).float()
            if projector is not None:
                s = projector(s).float()

            target_stats = _hptkd_targets(s, t, memory, cfg=cfg)
            s_norm = target_stats["student"]
            t_norm = target_stats["teacher"]
            steered = target_stats["steered"]
            w_easy = target_stats["w_easy"]
            w_hard = target_stats["w_hard"]
            w_risk = target_stats["w_risk"]

            direct = (s_norm - t_norm).pow(2).mean(dim=1)
            steer = (s_norm - steered).pow(2).mean(dim=1)
            weak = 1.0 - (s_norm * t_norm).sum(dim=1)
            method_loss = (
                w_easy * direct
                + float(cfg.hptkd_lambda_steer) * w_hard * steer
                + float(cfg.hptkd_lambda_weak) * w_risk * weak
            ).mean()

            point = F.mse_loss(s_norm, t_norm)
            cos = (1.0 - (s_norm * t_norm).sum(dim=1)).mean()
            rel = rkd_distance_loss(s_norm, t_norm)

            point_losses.append(float(point.detach().cpu()))
            cos_losses.append(float(cos.detach().cpu()))
            rel_losses.append(float(rel.detach().cpu()))
            method_losses.append(float(method_loss.detach().cpu()))

    point_mean = float(np.mean(point_losses))
    cos_mean = float(np.mean(cos_losses))
    rel_mean = float(np.mean(rel_losses))
    method_mean = float(np.mean(method_losses))
    score = (
        float(cfg.hptkd_val_lambda_point) * point_mean
        + float(cfg.hptkd_val_lambda_cos) * cos_mean
        + float(cfg.hptkd_val_lambda_rel) * rel_mean
    )
    return {
        "val_score": score,
        "val_method_loss": method_mean,
        "val_point_loss": point_mean,
        "val_cos_loss": cos_mean,
        "val_rel_loss": rel_mean,
    }


def _rkd_ema_relation_targets(
    student_z: torch.Tensor,
    teacher_z: torch.Tensor,
    proto_memory: HPTKDPrototypeMemory,
    relation_memory: PrototypeRelationEMAMemory,
    occupancy_memory: PrototypeOccupancyEMAMemory | None = None,
    *,
    cfg: ExperimentConfig,
) -> dict[str, torch.Tensor]:
    s_norm = F.normalize(student_z.float(), dim=1)
    t_norm = F.normalize(teacher_z.float(), dim=1)
    proto, assign, _ = proto_memory.lookup(t_norm)
    proto_bank = F.normalize(proto_memory.prototypes.detach().float(), dim=1)
    proto_sim = (t_norm * proto).sum(dim=1).clamp(-1.0, 1.0)

    temperature = max(float(cfg.v2_relation_temp), 1e-6)
    teacher_logits = (t_norm @ proto_bank.T) / temperature
    student_logits = (s_norm @ proto_bank.T) / temperature

    teacher_probs = F.softmax(teacher_logits, dim=1)
    student_log_probs = F.log_softmax(student_logits, dim=1)
    student_probs = student_log_probs.exp()
    ema_targets = relation_memory.get_targets(assign, teacher_probs)
    target_probs = ema_targets
    steer_beta = torch.zeros_like(assign, dtype=torch.float32)
    dominance_gate = torch.zeros_like(steer_beta)
    centrality_gate = torch.zeros_like(steer_beta)
    steer_shift = torch.zeros_like(steer_beta)

    if bool(getattr(cfg, "v2_use_embedding_steering", False)) and occupancy_memory is not None:
        assigned_proto = proto_bank[assign]
        proto_sim = (t_norm * assigned_proto).sum(dim=1).clamp(-1.0, 1.0)
        proto_counts = occupancy_memory.counts[assign].clamp_min(1e-6)
        mean_count = occupancy_memory.counts.mean().clamp_min(1e-6)
        dominance_gate = (1.0 - mean_count / proto_counts).clamp(0.0, 1.0)
        tau = float(getattr(cfg, "v2_steer_centrality_tau", 0.6))
        centrality_gate = ((proto_sim - tau) / max(1e-6, 1.0 - tau)).clamp(0.0, 1.0)
        steer_beta = float(getattr(cfg, "v2_steer_beta_max", 0.15)) * dominance_gate * centrality_gate

        ema_bary = F.normalize(ema_targets @ proto_bank, dim=1)
        dominant_component = proto_sim.unsqueeze(1) * assigned_proto
        steered_teacher = F.normalize(
            t_norm - steer_beta.unsqueeze(1) * dominant_component + steer_beta.unsqueeze(1) * ema_bary,
            dim=1,
        )
        steered_teacher_logits = (steered_teacher @ proto_bank.T) / temperature
        steered_teacher_probs = F.softmax(steered_teacher_logits, dim=1)
        target_probs = (1.0 - steer_beta.unsqueeze(1)) * ema_targets + steer_beta.unsqueeze(1) * steered_teacher_probs
        target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        steer_shift = (steered_teacher - t_norm).norm(dim=1)

    relation_gap = 0.5 * (student_probs.detach() - target_probs).abs().sum(dim=1)
    relation_reliability = teacher_probs.max(dim=1).values.clamp(0.0, 1.0)

    return {
        "teacher": t_norm,
        "student": s_norm,
        "assign": assign,
        "teacher_probs": teacher_probs,
        "student_probs": student_probs,
        "student_log_probs": student_log_probs,
        "ema_targets": ema_targets,
        "target_probs": target_probs,
        "relation_gap": relation_gap,
        "relation_reliability": relation_reliability,
        "proto_sim": proto_sim,
        "steer_beta": steer_beta,
        "dominance_gate": dominance_gate,
        "centrality_gate": centrality_gate,
        "steer_shift": steer_shift,
    }


def evaluate_hptkd_v2_validation(
    student: nn.Module,
    val_loader: DataLoader,
    proto_memory: HPTKDPrototypeMemory,
    relation_memory: PrototypeRelationEMAMemory,
    occupancy_memory: PrototypeOccupancyEMAMemory | None = None,
    *,
    cfg: ExperimentConfig,
    device: torch.device,
    projector: nn.Module | None = None,
) -> dict[str, float]:
    student.eval()
    if projector is not None:
        projector.eval()

    relation_losses: list[float] = []
    rel_gap_vals: list[float] = []
    distance_losses: list[float] = []
    angle_losses: list[float] = []

    with torch.inference_mode():
        for x_cpu, t_cpu in val_loader:
            x = x_cpu.to(device, non_blocking=True)
            t = t_cpu.to(device, non_blocking=True)
            s_raw = student(x).float()
            s_rel = s_raw
            if projector is not None:
                s_rel = projector(s_raw).float()

            stats = _rkd_ema_relation_targets(s_rel, t, proto_memory, relation_memory, occupancy_memory, cfg=cfg)
            relation = F.kl_div(stats["student_log_probs"], stats["target_probs"], reduction="none").sum(dim=1).mean()
            dist = rkd_distance_loss(s_raw, t)
            angle = rkd_angle_loss(s_raw, t)

            relation_losses.append(float(relation.detach().cpu()))
            rel_gap_vals.append(float(stats["relation_gap"].mean().detach().cpu()))
            distance_losses.append(float(dist.detach().cpu()))
            angle_losses.append(float(angle.detach().cpu()))

    relation_mean = float(np.mean(relation_losses))
    dist_mean = float(np.mean(distance_losses))
    angle_mean = float(np.mean(angle_losses))
    score = (
        float(cfg.rkd_w_dist) * dist_mean
        + float(cfg.rkd_w_angle) * angle_mean
        + float(cfg.v2_val_lambda_relation) * relation_mean
    )
    return {
        "val_score": score,
        "val_relation_loss": relation_mean,
        "val_mean_relation_gap": float(np.mean(rel_gap_vals)),
        "val_rkd_dist_loss": dist_mean,
        "val_rkd_angle_loss": angle_mean,
    }


def distill_hptkd_v2_cached(
    student: nn.Module,
    train_loader: DataLoader,
    *,
    cfg: ExperimentConfig,
    device: torch.device,
    val_loader: DataLoader | None = None,
    align_mode: str | None = None,
) -> tuple[nn.Module, pd.DataFrame]:
    student = student.to(device)
    disable_dropout_inplace(student)
    align_mode = str(align_mode or cfg.v2_relation_align_mode)

    train_x_all, train_t_all = resolve_cached_dataset_tensors(train_loader.dataset)
    sample_x = train_x_all[: min(int(cfg.kd_batch_size), int(train_x_all.shape[0]))]
    target_dim = int(train_t_all.shape[-1])
    projector, projector_params = align_student_output(
        student,
        sample_x,
        device=device,
        target_dim=target_dim,
        align_mode=align_mode,
    )
    proto_memory = HPTKDPrototypeMemory(
        cfg.v2_num_prototypes,
        target_dim,
        momentum=cfg.v2_proto_momentum,
        device=device,
    )
    relation_memory = PrototypeRelationEMAMemory(
        cfg.v2_num_prototypes,
        momentum=cfg.v2_relation_ema_momentum,
        device=device,
    )
    occupancy_memory = PrototypeOccupancyEMAMemory(
        cfg.v2_num_prototypes,
        momentum=float(getattr(cfg, "v2_occupancy_ema_momentum", 0.95)),
        device=device,
    )
    sample_memory = SampleRelationHistoryMemory(
        int(train_x_all.shape[0]),
        momentum=float(getattr(cfg, "v2_sample_ema_momentum", 0.9)),
        device=device,
    )

    student.train()
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector_params),
        lr=float(cfg.kd_lr),
        weight_decay=float(cfg.kd_weight_decay),
    )
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_epoch_min = max(1, int(getattr(cfg, "v2_best_epoch_min", 1)))
    best_state = copy.deepcopy(student.state_dict())
    best_score = float("inf")
    best_epoch = 0
    fallback_state = copy.deepcopy(best_state)
    fallback_score = float("inf")
    fallback_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    batch_size = int(cfg.kd_batch_size)
    default_weight_stats = {
        "mean_sampling_weight_std": 0.0,
        "mean_sampling_weight_max": 1.0,
        "mean_sampling_proto_boost": 1.0,
        "mean_sampling_diff_boost": 1.0,
        "mean_sampling_coverage_boost": 1.0,
        "mean_sampling_reliability_boost": 1.0,
        "mean_sampling_novelty_boost": 1.0,
    }

    for epoch in range(int(cfg.kd_epochs)):
        student.train()
        if projector is not None:
            projector.train()

        if bool(getattr(cfg, "v2_use_tempered_resampling", False)) and epoch >= int(getattr(cfg, "v2_resample_warmup_epochs", 1)):
            assign_all, proto_sim_all = compute_dataset_prototype_assignments(
                train_t_all,
                proto_memory,
                device=device,
                chunk_size=max(1024, int(cfg.kd_batch_size) * 16),
            )
            proto_bank = F.normalize(proto_memory.prototypes.detach().float(), dim=1)
            teacher_logits_all = train_t_all.to(device, non_blocking=True).float()
            teacher_logits_all = F.normalize(teacher_logits_all, dim=1) @ proto_bank.T
            teacher_probs_all = F.softmax(teacher_logits_all / max(float(getattr(cfg, "v2_relation_temp", 0.07)), 1e-6), dim=1)
            relation_reliability_all = teacher_probs_all.max(dim=1).values.detach()
            sampling_weights, sampling_weight_stats = sample_memory.build_weights(
                assign_all.to(device, non_blocking=True),
                proto_sim_all.to(device, non_blocking=True),
                relation_reliability_all,
                occupancy_memory,
                cfg=cfg,
            )
            epoch_indices, epoch_sampling_stats = sample_tempered_epoch_indices(
                sampling_weights,
                num_samples=int(train_x_all.shape[0]),
                temperature=float(getattr(cfg, "v2_resample_temperature", 1.5)),
                mix_ratio=float(getattr(cfg, "v2_resample_mix", 0.7)),
                seed=int(cfg.kd_init_seed + epoch),
            )
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(cfg.kd_init_seed + epoch))
            epoch_indices = torch.randperm(int(train_x_all.shape[0]), generator=generator)
            sampling_weight_stats = dict(default_weight_stats)
            epoch_sampling_stats = {
                "epoch_unique_samples": float(train_x_all.shape[0]),
                "epoch_unique_ratio": 1.0,
                "epoch_reuse_ratio": 0.0,
                "epoch_sampling_prob_max": 1.0 / float(max(1, train_x_all.shape[0])),
                "epoch_sampling_prob_min": 1.0 / float(max(1, train_x_all.shape[0])),
            }

        epoch_total: list[float] = []
        epoch_relation: list[float] = []
        epoch_rel_gap: list[float] = []
        epoch_dist: list[float] = []
        epoch_angle: list[float] = []
        epoch_batch_counts: list[np.ndarray] = []
        epoch_weight_std: list[float] = []
        epoch_weight_max: list[float] = []
        epoch_steer_beta: list[float] = []
        epoch_steer_shift: list[float] = []
        epoch_dominance_gate: list[float] = []
        epoch_centrality_gate: list[float] = []

        for sample_idx_cpu, x_cpu, t_cpu in iter_cached_epoch_batches(train_x_all, train_t_all, epoch_indices, batch_size=batch_size):
            x = x_cpu.to(device, non_blocking=True)
            t = t_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                s_raw = student(x).float()
                s_rel = s_raw
                if projector is not None:
                    s_rel = projector(s_raw).float()

                stats = _rkd_ema_relation_targets(s_rel, t, proto_memory, relation_memory, occupancy_memory, cfg=cfg)
                loss_d = rkd_distance_loss(s_raw, t)
                loss_a = rkd_angle_loss(s_raw, t)
                relation = F.kl_div(stats["student_log_probs"], stats["target_probs"], reduction="none").sum(dim=1)
                if bool(getattr(cfg, "v2_use_occupancy_weighting", False)):
                    relation_weights = occupancy_memory.get_weights(
                        stats["assign"],
                        alpha=float(getattr(cfg, "v2_occupancy_weight_alpha", 0.5)),
                        weight_min=float(getattr(cfg, "v2_occupancy_weight_min", 0.5)),
                        weight_max=float(getattr(cfg, "v2_occupancy_weight_max", 2.0)),
                    )
                else:
                    relation_weights = torch.ones_like(relation)
                weighted_relation = (relation * relation_weights).mean()
                loss = float(cfg.rkd_w_dist) * loss_d + float(cfg.rkd_w_angle) * loss_a + float(cfg.v2_lambda_relation) * weighted_relation

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            proto_memory.update(stats["teacher"], stats["assign"])
            relation_memory.update(stats["assign"], stats["teacher_probs"])
            occupancy_memory.update(stats["assign"])
            sample_memory.update(sample_idx_cpu, stats["relation_gap"], stats["proto_sim"], stats["relation_reliability"])

            epoch_total.append(float(loss.detach().cpu()))
            epoch_relation.append(float(weighted_relation.detach().cpu()))
            epoch_rel_gap.append(float(stats["relation_gap"].mean().detach().cpu()))
            epoch_dist.append(float(loss_d.detach().cpu()))
            epoch_angle.append(float(loss_a.detach().cpu()))
            epoch_weight_std.append(float(relation_weights.std(unbiased=False).detach().cpu()))
            epoch_weight_max.append(float(relation_weights.max().detach().cpu()))
            epoch_steer_beta.append(float(stats["steer_beta"].mean().detach().cpu()))
            epoch_steer_shift.append(float(stats["steer_shift"].mean().detach().cpu()))
            epoch_dominance_gate.append(float(stats["dominance_gate"].mean().detach().cpu()))
            epoch_centrality_gate.append(float(stats["centrality_gate"].mean().detach().cpu()))
            epoch_batch_counts.append(
                np.bincount(
                    stats["assign"].detach().cpu().numpy(),
                    minlength=int(cfg.v2_num_prototypes),
                )
            )

        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(epoch_total)),
            "train_rkd_dist_loss": float(np.mean(epoch_dist)),
            "train_rkd_angle_loss": float(np.mean(epoch_angle)),
            "train_relation_loss": float(np.mean(epoch_relation)),
            "mean_relation_gap": float(np.mean(epoch_rel_gap)),
            "mean_relation_weight_std": float(np.mean(epoch_weight_std)),
            "mean_relation_weight_max": float(np.mean(epoch_weight_max)),
            "mean_steer_beta": float(np.mean(epoch_steer_beta)),
            "mean_steer_shift": float(np.mean(epoch_steer_shift)),
            "mean_dominance_gate": float(np.mean(epoch_dominance_gate)),
            "mean_centrality_gate": float(np.mean(epoch_centrality_gate)),
            "checkpoint_selection_start_epoch": int(best_epoch_min),
            "checkpoint_eligible": bool((epoch + 1) >= best_epoch_min),
        }
        row.update(sampling_weight_stats)
        row.update(epoch_sampling_stats)
        row.update(
            _summarize_epoch_prototype_stats(
                epoch_batch_counts,
                num_prototypes=int(cfg.v2_num_prototypes),
                batch_size=batch_size,
            )
        )

        if val_loader is not None:
            val_metrics = evaluate_hptkd_v2_validation(
                student,
                val_loader,
                proto_memory,
                relation_memory,
                occupancy_memory,
                cfg=cfg,
                device=device,
                projector=projector,
            )
            row.update(val_metrics)
            current_score = float(val_metrics["val_score"])
            if current_score < (fallback_score - float(cfg.v2_min_delta)) or fallback_epoch == 0:
                fallback_score = current_score
                fallback_state = copy.deepcopy(student.state_dict())
                fallback_epoch = int(epoch + 1)
            improved = False
            if bool(row["checkpoint_eligible"]):
                improved = current_score < (best_score - float(cfg.v2_min_delta))
                if improved:
                    best_score = current_score
                    best_state = copy.deepcopy(student.state_dict())
                    best_epoch = int(epoch + 1)
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            else:
                stale_epochs = 0
        else:
            current_score = float(row["train_loss"])
            if current_score < (fallback_score - float(cfg.v2_min_delta)) or fallback_epoch == 0:
                fallback_score = current_score
                fallback_state = copy.deepcopy(student.state_dict())
                fallback_epoch = int(epoch + 1)
            improved = False
            if bool(row["checkpoint_eligible"]):
                improved = current_score < (best_score - float(cfg.v2_min_delta))
                if improved:
                    best_score = current_score
                    best_state = copy.deepcopy(student.state_dict())
                    best_epoch = int(epoch + 1)

        history.append(row)
        msg = (
            f"[HPTKD-V2][{epoch + 1}/{cfg.kd_epochs}] train={row['train_loss']:.6f} "
            f"rkd_d={row['train_rkd_dist_loss']:.6f} rkd_a={row['train_rkd_angle_loss']:.6f} "
            f"relation={row['train_relation_loss']:.6f} relation_gap={row['mean_relation_gap']:.4f} "
            f"top1={row['epoch_top1_share']:.4f} batch_dom={row['mean_batch_dominant_share']:.4f}"
        )
        if "val_score" in row:
            msg += f" val={row['val_score']:.6f}"
        msg += f" eligible={int(bool(row['checkpoint_eligible']))}"
        print(msg)

        if val_loader is not None and bool(row["checkpoint_eligible"]) and stale_epochs >= int(cfg.v2_patience):
            print(f"[HPTKD-V2] Early stop at epoch {epoch + 1} (patience={cfg.v2_patience})")
            break

    if best_epoch == 0 and fallback_epoch > 0:
        print(
            f"[HPTKD-V2] No eligible checkpoint found for epoch >= {best_epoch_min}; "
            f"falling back to epoch {fallback_epoch}."
        )
        best_state = fallback_state
        best_score = fallback_score
        best_epoch = fallback_epoch

    student.load_state_dict(best_state)
    student.eval()
    return student, pd.DataFrame(history)


def distill_hptkd_cached(
    student: nn.Module,
    train_loader: DataLoader,
    *,
    cfg: ExperimentConfig,
    device: torch.device,
    val_loader: DataLoader | None = None,
    align_mode: str | None = None,
) -> tuple[nn.Module, pd.DataFrame]:
    student = student.to(device)
    disable_dropout_inplace(student)
    align_mode = str(align_mode or cfg.hptkd_align_mode)

    sample_x, sample_t = next(iter(train_loader))
    target_dim = int(sample_t.shape[-1])
    projector, trainable_params = align_student_output(
        student,
        sample_x,
        device=device,
        target_dim=target_dim,
        align_mode=align_mode,
    )
    memory = HPTKDPrototypeMemory(
        cfg.hptkd_num_prototypes,
        target_dim,
        momentum=cfg.hptkd_proto_momentum,
        device=device,
    )

    student.train()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.kd_lr),
        weight_decay=float(cfg.kd_weight_decay),
    )
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state = copy.deepcopy(student.state_dict())
    best_score = float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(int(cfg.kd_epochs)):
        student.train()
        if projector is not None:
            projector.train()

        epoch_total: list[float] = []
        epoch_direct: list[float] = []
        epoch_steer: list[float] = []
        epoch_weak: list[float] = []
        epoch_alpha: list[float] = []
        epoch_rel: list[float] = []
        epoch_learn: list[float] = []

        for x_cpu, t_cpu in train_loader:
            x = x_cpu.to(device, non_blocking=True)
            t = t_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                s = student(x).float()
                if projector is not None:
                    s = projector(s).float()

                target_stats = _hptkd_targets(s, t, memory, cfg=cfg)
                s_norm = target_stats["student"]
                t_norm = target_stats["teacher"]
                steered = target_stats["steered"]
                w_easy = target_stats["w_easy"]
                w_hard = target_stats["w_hard"]
                w_risk = target_stats["w_risk"]

                direct = (s_norm - t_norm).pow(2).mean(dim=1)
                steer = (s_norm - steered).pow(2).mean(dim=1)
                weak = 1.0 - (s_norm * t_norm).sum(dim=1)

                loss = (
                    w_easy * direct
                    + float(cfg.hptkd_lambda_steer) * w_hard * steer
                    + float(cfg.hptkd_lambda_weak) * w_risk * weak
                ).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            memory.update(t_norm, target_stats["assign"])

            epoch_total.append(float(loss.detach().cpu()))
            epoch_direct.append(float(direct.mean().detach().cpu()))
            epoch_steer.append(float(steer.mean().detach().cpu()))
            epoch_weak.append(float(weak.mean().detach().cpu()))
            epoch_alpha.append(float(target_stats["alpha"].mean().detach().cpu()))
            epoch_rel.append(float(target_stats["reliability"].mean().detach().cpu()))
            epoch_learn.append(float(target_stats["learnability"].mean().detach().cpu()))

        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(epoch_total)),
            "train_direct_loss": float(np.mean(epoch_direct)),
            "train_steer_loss": float(np.mean(epoch_steer)),
            "train_weak_loss": float(np.mean(epoch_weak)),
            "mean_alpha": float(np.mean(epoch_alpha)),
            "mean_reliability": float(np.mean(epoch_rel)),
            "mean_learnability": float(np.mean(epoch_learn)),
        }

        if val_loader is not None:
            val_metrics = evaluate_hptkd_validation(
                student,
                val_loader,
                memory,
                cfg=cfg,
                device=device,
                projector=projector,
            )
            row.update(val_metrics)
            improved = val_metrics["val_score"] < (best_score - float(cfg.hptkd_min_delta))
            if improved:
                best_score = float(val_metrics["val_score"])
                best_state = copy.deepcopy(student.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
        else:
            improved = row["train_loss"] < (best_score - float(cfg.hptkd_min_delta))
            if improved:
                best_score = float(row["train_loss"])
                best_state = copy.deepcopy(student.state_dict())

        history.append(row)
        msg = (
            f"[HPTKD][{epoch + 1}/{cfg.kd_epochs}] "
            f"train={row['train_loss']:.6f} alpha={row['mean_alpha']:.4f} "
            f"reliability={row['mean_reliability']:.4f} learnability={row['mean_learnability']:.4f}"
        )
        if "val_score" in row:
            msg += f" val={row['val_score']:.6f}"
        print(msg)

        if val_loader is not None and stale_epochs >= int(cfg.hptkd_patience):
            print(f"[HPTKD] Early stop at epoch {epoch + 1} (patience={cfg.hptkd_patience})")
            break

    student.load_state_dict(best_state)
    student.eval()
    return student, pd.DataFrame(history)


def distill_responsekd_cached(
    student: nn.Module,
    cached_loader: DataLoader,
    *,
    cfg: ExperimentConfig,
    device: torch.device,
    align_mode: str = "projection",
) -> nn.Module:
    student = student.to(device)
    disable_dropout_inplace(student)
    align_mode = str(align_mode)

    target_dim = int(cached_loader.dataset.tensors[1].shape[-1])
    sample_x = next(iter(cached_loader))[0]
    if align_mode == "same_dim":
        if isinstance(student, StudentOutputAdapter):
            student.attach_output_head(sample_x, device, target_dim)
        else:
            student_dim = infer_model_embed_dim(student, sample_x, device)
            if student_dim != target_dim:
                raise ValueError("same_dim mode requires the student to emit teacher-sized embeddings")
        projector = None
        trainable_params = list(student.parameters())
    elif align_mode == "projection":
        student_dim = infer_embed_dim(student, cached_loader, device)
        projector = nn.Linear(student_dim, target_dim, bias=False).to(device)
        trainable_params = list(student.parameters()) + list(projector.parameters())
    else:
        raise ValueError(f"Unsupported ResponseKD align_mode={align_mode}")

    student.train()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.kd_lr),
        weight_decay=float(cfg.kd_weight_decay),
    )
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(int(cfg.kd_epochs)):
        losses: list[float] = []
        for x_cpu, t_cpu in cached_loader:
            x = x_cpu.to(device, non_blocking=True)
            t = t_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                s = student(x).float()
                if projector is not None:
                    s = projector(s)
                if cfg.responsekd_normalize:
                    s = F.normalize(s, dim=1)
                    t = F.normalize(t, dim=1)
                loss = F.mse_loss(s, t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        print(f"[ResponseKD][{epoch + 1}/{cfg.kd_epochs}] loss={np.mean(losses):.6f}")

    student.eval()
    return student


def rkd_distance_loss(student_z: torch.Tensor, teacher_z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    with torch.no_grad():
        teacher_dist = torch.pdist(teacher_z, p=2)
        teacher_dist = teacher_dist / (teacher_dist.mean() + float(eps))
    student_dist = torch.pdist(student_z, p=2)
    student_dist = student_dist / (student_dist.mean() + float(eps))
    return F.smooth_l1_loss(student_dist, teacher_dist)


def rkd_angle_loss(student_z: torch.Tensor, teacher_z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    with torch.no_grad():
        t_diff = teacher_z.unsqueeze(0) - teacher_z.unsqueeze(1)
        t_norm = F.normalize(t_diff, p=2, dim=2, eps=float(eps))
        t_angle = torch.bmm(t_norm, t_norm.transpose(1, 2)).reshape(-1)
    s_diff = student_z.unsqueeze(0) - student_z.unsqueeze(1)
    s_norm = F.normalize(s_diff, p=2, dim=2, eps=float(eps))
    s_angle = torch.bmm(s_norm, s_norm.transpose(1, 2)).reshape(-1)
    return F.smooth_l1_loss(s_angle, t_angle)


def distill_rkd_cached(
    student: nn.Module,
    cached_loader: DataLoader,
    *,
    cfg: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    student = student.to(device)
    disable_dropout_inplace(student)
    student.train()

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(cfg.kd_lr),
        weight_decay=float(cfg.kd_weight_decay),
    )
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(int(cfg.kd_epochs)):
        losses: list[float] = []
        for x_cpu, t_cpu in cached_loader:
            x = x_cpu.to(device, non_blocking=True)
            t = t_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                s = student(x).float()
                loss_d = rkd_distance_loss(s, t)
                loss_a = rkd_angle_loss(s, t)
                loss = float(cfg.rkd_w_dist) * loss_d + float(cfg.rkd_w_angle) * loss_a
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        print(f"[RKD][{epoch + 1}/{cfg.kd_epochs}] loss={np.mean(losses):.6f}")

    student.eval()
    return student


def extract_embeddings_from_loader(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    cfg: ExperimentConfig,
    device: torch.device,
    desc: str,
    regression: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    z_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            x = batch[0]
            y = batch[1]
            if isinstance(model, NormWearTeacherEncoder):
                z = model(x)
            elif bool(cfg.eval_use_raw_student_encoder) and isinstance(model, StudentOutputAdapter):
                z = model.encoder(x.to(device, non_blocking=True))
            else:
                z = model(x.to(device, non_blocking=True))
            if z.ndim > 2:
                z = z.flatten(1)
            z_list.append(z.detach().cpu().numpy())
            if regression:
                y_list.append(y.detach().cpu().float().numpy())
            else:
                y_list.append(y.detach().cpu().numpy().astype(np.int64))

    return np.concatenate(z_list, axis=0), np.concatenate(y_list, axis=0)


def eval_kfold_clf(Z: np.ndarray, y: np.ndarray, groups: list[Any], *, n_splits: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_idx in range(int(n_splits)):
        split = make_split(
            split_mode="kfold_subject",
            groups=groups,
            n_splits=int(n_splits),
            fold_idx=int(fold_idx),
            seed=int(seed),
        )
        tr_idx = split["train_idx"]
        te_idx = split["test_idx"]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(Z[tr_idx])
        X_te = scaler.transform(Z[te_idx])
        clf = LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(X_tr, y[tr_idx])
        pred = clf.predict(X_te)
        rows.append(
            {
                "fold": int(fold_idx),
                "acc": float(accuracy_score(y[te_idx], pred)),
                "f1_macro": float(f1_score(y[te_idx], pred, average="macro")),
                "recall_macro": float(recall_score(y[te_idx], pred, average="macro")),
            }
        )
    return pd.DataFrame(rows)


def eval_kfold_ridge(Z: np.ndarray, y: np.ndarray, groups: list[Any], *, n_splits: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    alphas = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    for fold_idx in range(int(n_splits)):
        split = make_split(
            split_mode="kfold_subject",
            groups=groups,
            n_splits=int(n_splits),
            fold_idx=int(fold_idx),
            seed=int(seed),
        )
        tr_idx = split["train_idx"]
        te_idx = split["test_idx"]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(Z[tr_idx])
        X_te = scaler.transform(Z[te_idx])
        print(f"[Eval][Ridge][fold {fold_idx + 1}/{n_splits}] train={len(tr_idx)} test={len(te_idx)}")
        grid = RidgeCV(alphas=alphas, cv=4, scoring="neg_mean_squared_error")
        grid.fit(X_tr, y[tr_idx])
        pred = grid.predict(X_te)
        rows.append(
            {
                "fold": int(fold_idx),
                "mae": float(mean_absolute_error(y[te_idx], pred)),
                "rmse": float(np.sqrt(np.mean((y[te_idx] - pred) ** 2))),
                "r2": float(r2_score(y[te_idx], pred)),
                "best_alpha": float(grid.alpha_),
            }
        )
    return pd.DataFrame(rows)


def build_downstream_resources(cfg: ExperimentConfig, device: torch.device) -> dict[str, Any]:
    sport_cfg = SportPPGConfig(
        data_root=cfg.sport_data_root,
        gt_root=cfg.sport_gt_root,
        subjects=list(cfg.sport_subjects),
        acts=list(cfg.sport_acts),
        hand="L",
        use_acc=False,
        orig_fs=64,
        target_fs=cfg.sport_target_fs,
        win_sec=10.0,
        hop_sec=2.0,
    )
    ds_sport = SportPPGWindowDataset(sport_cfg, task="activity", mmap=True)
    dl_sport = DataLoader(
        ds_sport,
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.eval_num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    groups_sport = [ds_sport.records[ri].subject for (ri, _) in ds_sport.master_index]

    wesad_dl: dict[str, DataLoader] = {}
    wesad_groups: dict[str, list[str]] = {}
    for task in ("binary", "multiclass"):
        ds = make_wesad_dataset(
            base_dir=str(cfg.wesad_base_dir),
            task=task,
            split="full",
            hz=cfg.wesad_hz,
            mmap=True,
            return_meta=True,
        )
        wesad_groups[task] = ds.subject.tolist()
        wesad_dl[task] = DataLoader(
            ds,
            batch_size=int(cfg.eval_batch_size),
            shuffle=False,
            num_workers=int(cfg.eval_num_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    wfppg_dl: dict[str, DataLoader] = {}
    wfppg_groups: dict[str, list[str]] = {}
    for source in ("wrist", "finger"):
        ds = make_wfppg_dataset(ppg_source=source, split="full", hz=cfg.wfppg_hz, mmap=True, return_meta=False)
        wfppg_groups[source] = ds.subject.tolist()
        wfppg_dl[source] = DataLoader(
            ds,
            batch_size=int(cfg.eval_batch_size),
            shuffle=False,
            num_workers=int(cfg.eval_num_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    return {
        "dl_sport": dl_sport,
        "groups_sport": groups_sport,
        "wesad_dl": wesad_dl,
        "wesad_groups": wesad_groups,
        "wfppg_dl": wfppg_dl,
        "wfppg_groups": wfppg_groups,
    }


def teacher_baseline_cache_path(cfg: ExperimentConfig) -> Path:
    name = (
        f"normwear_teacher_baseline_"
        f"patch-{cfg.teacher_patch_pool}_channel-{cfg.teacher_channel_pool}_"
        f"splits-{int(cfg.eval_n_splits)}.csv"
    )
    return cfg.results_dir / name


def eval_encoder_linearprobe_3datasets(
    model: nn.Module,
    *,
    cfg: ExperimentConfig,
    device: torch.device,
    resources: dict[str, Any],
    mode: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    print(f"[Eval] Start SportPPG for mode={mode}")

    Z, y = extract_embeddings_from_loader(
        model,
        resources["dl_sport"],
        cfg=cfg,
        device=device,
        desc=f"SportPPG {mode}",
        regression=False,
    )
    df = eval_kfold_clf(Z, y, resources["groups_sport"], n_splits=cfg.eval_n_splits, seed=cfg.eval_seed)
    df["dataset"] = "SportPPG"
    df["task"] = "activity"
    df["mode"] = mode
    frames.append(df)
    print(f"[Eval] Finished SportPPG for mode={mode}")

    for task in ("binary", "multiclass"):
        print(f"[Eval] Start WESAD-{task} for mode={mode}")
        Z, y = extract_embeddings_from_loader(
            model,
            resources["wesad_dl"][task],
            cfg=cfg,
            device=device,
            desc=f"WESAD-{task} {mode}",
            regression=False,
        )
        df = eval_kfold_clf(Z, y, resources["wesad_groups"][task], n_splits=cfg.eval_n_splits, seed=cfg.eval_seed)
        df["dataset"] = "WESAD"
        df["task"] = task
        df["mode"] = mode
        frames.append(df)
        print(f"[Eval] Finished WESAD-{task} for mode={mode}")

    for source in ("wrist", "finger"):
        print(f"[Eval] Start WFPPG-{source} for mode={mode}")
        Z, y = extract_embeddings_from_loader(
            model,
            resources["wfppg_dl"][source],
            cfg=cfg,
            device=device,
            desc=f"WFPPG-{source} {mode}",
            regression=True,
        )
        df = eval_kfold_ridge(Z, y, resources["wfppg_groups"][source], n_splits=cfg.eval_n_splits, seed=cfg.eval_seed)
        df["dataset"] = "WFPPG"
        df["task"] = f"hr_{source}"
        df["mode"] = mode
        frames.append(df)
        print(f"[Eval] Finished WFPPG-{source} for mode={mode}")

    return pd.concat(frames, ignore_index=True)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in ("acc", "f1_macro", "recall_macro", "mae", "rmse", "r2") if c in results.columns]
    return results.groupby(["public_source", "method", "dataset", "task"], as_index=False)[metric_cols].agg(["mean", "std"]).reset_index()


def summarize_architecture_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in ("acc", "f1_macro", "recall_macro", "mae", "rmse", "r2") if c in results.columns]
    return (
        results.groupby(["student_arch", "public_source", "method", "dataset", "task"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )


def run_architecture_comparison(
    config: ExperimentConfig | None = None,
    *,
    student_archs: tuple[str, ...] = ("resnet1d", "normwear_tiny"),
    public_sources: tuple[str, ...] = ("pulsedb", "sdb"),
    methods: tuple[str, ...] = ("responsekd", "rkd"),
) -> dict[str, Any]:
    cfg = (config or ExperimentConfig()).resolve()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    seed_all(cfg.kd_init_seed)
    device = get_device(cfg.device)
    teacher = NormWearTeacherEncoder(
        cfg.teacher_checkpoint,
        device=device,
        sampling_rate=cfg.teacher_sampling_rate,
        patch_pool=cfg.teacher_patch_pool,
        channel_pool=cfg.teacher_channel_pool,
    )
    resources = build_downstream_resources(cfg, device)

    public_cache: dict[str, tuple[TensorDataset, dict[str, Any]]] = {}
    for public_source in public_sources:
        public_dataset = build_public_dataset(public_source, cfg)
        public_cache[public_source] = cache_public_teacher_embeddings(teacher, public_dataset, source=public_source, cfg=cfg)

    all_frames: list[pd.DataFrame] = []
    student_stats: list[dict[str, Any]] = []
    teacher_params = count_parameters(teacher)

    for student_arch in student_archs:
        arch_cfg = copy.deepcopy(cfg)
        arch_cfg.student_arch = str(student_arch)
        seed_all(arch_cfg.kd_init_seed)
        proto = build_tiny_student(arch_cfg, device)
        student_params = count_parameters(proto)
        student_stats.append(
            {
                "student_arch": str(student_arch),
                "student_params": int(student_params),
                "teacher_params": int(teacher_params),
                "param_ratio": float(student_params) / float(max(1, teacher_params)),
            }
        )
        del proto

        for public_source in public_sources:
            cached_dataset, _ = public_cache[public_source]
            cached_loader = make_cached_distill_loader(cached_dataset, arch_cfg, device)
            for method in methods:
                seed_all(arch_cfg.kd_init_seed)
                student = build_tiny_student(arch_cfg, device)
                if method == "responsekd":
                    student = distill_responsekd_cached(student, cached_loader, cfg=arch_cfg, device=device)
                else:
                    student = distill_rkd_cached(student, cached_loader, cfg=arch_cfg, device=device)
                df = eval_encoder_linearprobe_3datasets(
                    student,
                    cfg=arch_cfg,
                    device=device,
                    resources=resources,
                    mode=f"{student_arch}_{public_source}_{method}",
                )
                df["student_arch"] = str(student_arch)
                df["public_source"] = str(public_source)
                df["method"] = str(method)
                df["teacher_params"] = int(teacher_params)
                df["student_params"] = int(student_params)
                df["student_param_ratio"] = float(student_params) / float(max(1, teacher_params))
                all_frames.append(df)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    all_results = pd.concat(all_frames, ignore_index=True)
    summary = summarize_architecture_results(all_results)
    csv_path = cfg.results_dir / f"normwear_kd_arch_compare_n{int(cfg.public_n)}_ep{int(cfg.kd_epochs)}.csv"
    summary_path = cfg.results_dir / f"normwear_kd_arch_compare_n{int(cfg.public_n)}_ep{int(cfg.kd_epochs)}_meanstd.csv"
    all_results.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    return {
        "config": cfg,
        "results": all_results,
        "summary": summary,
        "student_stats": pd.DataFrame(student_stats),
        "results_csv": csv_path,
        "summary_csv": summary_path,
    }


def run_normwear_kd_comparison(config: ExperimentConfig | None = None) -> dict[str, Any]:
    cfg = (config or ExperimentConfig()).resolve()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    seed_all(cfg.kd_init_seed)
    device = get_device(cfg.device)

    teacher = NormWearTeacherEncoder(
        cfg.teacher_checkpoint,
        device=device,
        sampling_rate=cfg.teacher_sampling_rate,
        patch_pool=cfg.teacher_patch_pool,
        channel_pool=cfg.teacher_channel_pool,
    )
    teacher_params = count_parameters(teacher)
    student_proto = build_tiny_student(cfg, device)
    student_params = count_parameters(student_proto)
    del student_proto

    resources = build_downstream_resources(cfg, device)

    all_frames: list[pd.DataFrame] = []
    teacher_df = eval_encoder_linearprobe_3datasets(
        teacher,
        cfg=cfg,
        device=device,
        resources=resources,
        mode="normwear_teacher",
    )
    teacher_df["public_source"] = "teacher"
    teacher_df["method"] = "teacher"
    teacher_df["teacher_params"] = int(teacher_params)
    teacher_df["student_params"] = np.nan
    teacher_df["student_param_ratio"] = np.nan
    all_frames.append(teacher_df)

    cached_meta: dict[str, Any] = {}
    for public_source in ("pulsedb", "sdb"):
        public_dataset = build_public_dataset(public_source, cfg)
        cached_dataset, meta = cache_public_teacher_embeddings(teacher, public_dataset, source=public_source, cfg=cfg)
        cached_meta[public_source] = meta
        cached_loader = make_cached_distill_loader(cached_dataset, cfg, device)

        for method in ("responsekd", "rkd"):
            seed_all(cfg.kd_init_seed)
            student = build_tiny_student(cfg, device)
            if method == "responsekd":
                student = distill_responsekd_cached(student, cached_loader, cfg=cfg, device=device)
            else:
                student = distill_rkd_cached(student, cached_loader, cfg=cfg, device=device)

            result_df = eval_encoder_linearprobe_3datasets(
                student,
                cfg=cfg,
                device=device,
                resources=resources,
                mode=f"normwear_{public_source}_{method}",
            )
            result_df["public_source"] = str(public_source)
            result_df["method"] = str(method)
            result_df["teacher_params"] = int(teacher_params)
            result_df["student_params"] = int(student_params)
            result_df["student_param_ratio"] = float(student_params) / float(max(1, teacher_params))
            all_frames.append(result_df)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    all_results = pd.concat(all_frames, ignore_index=True)
    summary = summarize_results(all_results[all_results["public_source"] != "teacher"].copy())

    csv_path = cfg.results_dir / f"normwear_kd_compare_n{int(cfg.public_n)}_ep{int(cfg.kd_epochs)}.csv"
    summary_path = cfg.results_dir / f"normwear_kd_compare_n{int(cfg.public_n)}_ep{int(cfg.kd_epochs)}_meanstd.csv"
    all_results.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)

    return {
        "config": cfg,
        "teacher": teacher,
        "results": all_results,
        "summary": summary,
        "cached_meta": cached_meta,
        "results_csv": csv_path,
        "summary_csv": summary_path,
        "teacher_params": int(teacher_params),
        "student_params": int(student_params),
    }


def run_normwear_mixed_public_kd(
    config: ExperimentConfig | None = None,
    *,
    source_counts: dict[str, int] | None = None,
    methods: tuple[str, ...] = ("responsekd", "rkd"),
    mix_name: str | None = None,
) -> dict[str, Any]:
    cfg = (config or ExperimentConfig()).resolve()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    source_counts = source_counts or {"pulsedb": 50_000, "sdb": 50_000}
    source_counts = {str(source): int(count) for source, count in source_counts.items()}
    total_n = int(sum(source_counts.values()))
    if total_n <= 0:
        raise ValueError("source_counts must sum to a positive total")

    run_cfg = copy.deepcopy(cfg)
    run_cfg.public_n = int(total_n)
    mix_name = mix_name or "mix_" + "_".join(f"{source}{count}" for source, count in source_counts.items())

    seed_all(run_cfg.kd_init_seed)
    device = get_device(run_cfg.device)

    teacher = NormWearTeacherEncoder(
        run_cfg.teacher_checkpoint,
        device=device,
        sampling_rate=run_cfg.teacher_sampling_rate,
        patch_pool=run_cfg.teacher_patch_pool,
        channel_pool=run_cfg.teacher_channel_pool,
    )
    teacher_params = count_parameters(teacher)
    student_proto = build_tiny_student(run_cfg, device)
    student_params = count_parameters(student_proto)
    del student_proto

    resources = build_downstream_resources(run_cfg, device)

    all_frames: list[pd.DataFrame] = []
    teacher_df = eval_encoder_linearprobe_3datasets(
        teacher,
        cfg=run_cfg,
        device=device,
        resources=resources,
        mode="normwear_teacher",
    )
    teacher_df["public_source"] = "teacher"
    teacher_df["method"] = "teacher"
    teacher_df["teacher_params"] = int(teacher_params)
    teacher_df["student_params"] = np.nan
    teacher_df["student_param_ratio"] = np.nan
    all_frames.append(teacher_df)

    cached_dataset, meta = cache_mixed_public_teacher_embeddings(
        teacher,
        source_counts=source_counts,
        mix_name=mix_name,
        cfg=run_cfg,
    )
    cached_loader = make_cached_distill_loader(cached_dataset, run_cfg, device)

    for method in methods:
        seed_all(run_cfg.kd_init_seed)
        student = build_tiny_student(run_cfg, device)
        if method == "responsekd":
            student = distill_responsekd_cached(student, cached_loader, cfg=run_cfg, device=device)
        elif method == "rkd":
            student = distill_rkd_cached(student, cached_loader, cfg=run_cfg, device=device)
        else:
            raise ValueError(f"Unsupported KD method: {method}")

        result_df = eval_encoder_linearprobe_3datasets(
            student,
            cfg=run_cfg,
            device=device,
            resources=resources,
            mode=f"normwear_{mix_name}_{method}",
        )
        result_df["public_source"] = str(mix_name)
        result_df["method"] = str(method)
        result_df["teacher_params"] = int(teacher_params)
        result_df["student_params"] = int(student_params)
        result_df["student_param_ratio"] = float(student_params) / float(max(1, teacher_params))
        all_frames.append(result_df)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_results = pd.concat(all_frames, ignore_index=True)
    summary = summarize_results(all_results[all_results["public_source"] != "teacher"].copy())

    csv_path = run_cfg.results_dir / f"normwear_kd_compare_{mix_name}_n{int(run_cfg.public_n)}_ep{int(run_cfg.kd_epochs)}.csv"
    summary_path = run_cfg.results_dir / f"normwear_kd_compare_{mix_name}_n{int(run_cfg.public_n)}_ep{int(run_cfg.kd_epochs)}_meanstd.csv"
    all_results.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)

    return {
        "config": run_cfg,
        "teacher": teacher,
        "results": all_results,
        "summary": summary,
        "cached_meta": meta,
        "results_csv": csv_path,
        "summary_csv": summary_path,
        "teacher_params": int(teacher_params),
        "student_params": int(student_params),
    }


def load_public_cached_dataset(
    teacher: NormWearTeacherEncoder,
    public_setup: str | dict[str, int],
    *,
    name: str,
    cfg: ExperimentConfig,
) -> tuple[Dataset, dict[str, Any]]:
    if isinstance(public_setup, str):
        public_dataset = build_public_dataset(public_setup, cfg)
        return cache_public_teacher_embeddings(teacher, public_dataset, source=public_setup, cfg=cfg)
    if isinstance(public_setup, dict):
        return cache_mixed_public_teacher_embeddings(
            teacher,
            source_counts=public_setup,
            mix_name=name,
            cfg=cfg,
        )
    raise TypeError("public_setup must be a source name or a source-count dictionary")


def run_normwear_hptkd_experiment(
    config: ExperimentConfig | None = None,
    *,
    public_setups: dict[str, str | dict[str, int]] | None = None,
) -> dict[str, Any]:
    cfg = (config or ExperimentConfig()).resolve()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    public_setups = public_setups or {
        "pulsedb": "pulsedb",
        "sdb": "sdb",
        "combo_pulsedb50k_sdb50k": {"pulsedb": 50_000, "sdb": 50_000},
    }

    seed_all(cfg.kd_init_seed)
    device = get_device(cfg.device)
    student_arch = str(getattr(cfg, "student_arch", "resnet1d"))
    teacher = NormWearTeacherEncoder(
        cfg.teacher_checkpoint,
        device=device,
        sampling_rate=cfg.teacher_sampling_rate,
        patch_pool=cfg.teacher_patch_pool,
        channel_pool=cfg.teacher_channel_pool,
    )
    teacher_params = count_parameters(teacher)
    student_proto = build_tiny_student(cfg, device)
    student_params = count_parameters(student_proto)
    del student_proto

    print("[HPTKD] Configuration summary")
    print(
        {
            "device": str(device),
            "public_n": int(cfg.public_n),
            "public_val_n": int(cfg.public_val_n),
            "kd_epochs": int(cfg.kd_epochs),
            "student_arch": student_arch,
            "hptkd_num_prototypes": int(cfg.hptkd_num_prototypes),
            "hptkd_patience": int(cfg.hptkd_patience),
            "public_setups": list(public_setups.keys()),
        }
    )
    print(
        f"[HPTKD] teacher_params={teacher_params} student_params={student_params} "
        f"student_ratio={float(student_params) / float(max(1, teacher_params)):.4%}"
    )

    print("[HPTKD] Building downstream evaluation resources...")
    resources = build_downstream_resources(cfg, device)

    all_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    split_meta: dict[str, Any] = {}
    cached_meta: dict[str, Any] = {}

    teacher_cache = teacher_baseline_cache_path(cfg)
    if teacher_cache.exists():
        print(f"[HPTKD] Reusing cached frozen teacher baseline from {teacher_cache}")
        teacher_df = pd.read_csv(teacher_cache)
    else:
        print("[HPTKD] Evaluating frozen teacher baseline on downstream datasets...")
        teacher_df = eval_encoder_linearprobe_3datasets(
            teacher,
            cfg=cfg,
            device=device,
            resources=resources,
            mode="normwear_teacher",
        )
        teacher_df.to_csv(teacher_cache, index=False)
        print(f"[HPTKD] Cached frozen teacher baseline to {teacher_cache}")
    teacher_df["public_source"] = "teacher"
    teacher_df["method"] = "teacher"
    teacher_df["teacher_params"] = int(teacher_params)
    teacher_df["student_params"] = np.nan
    teacher_df["student_param_ratio"] = np.nan
    teacher_df["student_arch"] = student_arch
    all_frames.append(teacher_df)
    print("[HPTKD] Teacher baseline evaluation finished.")

    total_setups = len(public_setups)
    for setup_idx, (public_name, public_setup) in enumerate(public_setups.items(), start=1):
        print(f"[HPTKD] [{setup_idx}/{total_setups}] Preparing public source: {public_name}")
        cached_dataset, meta = load_public_cached_dataset(teacher, public_setup, name=public_name, cfg=cfg)
        cached_meta[str(public_name)] = meta
        x_shape = tuple(cached_dataset.tensors[0].shape)
        t_shape = tuple(cached_dataset.tensors[1].shape)
        print(f"[HPTKD] [{public_name}] cached_x_shape={x_shape} cached_t_shape={t_shape}")
        train_ds, val_ds, split_info = split_cached_dataset(
            cached_dataset,
            val_n=cfg.public_val_n,
            seed=cfg.public_seed,
        )
        split_meta[str(public_name)] = split_info
        train_n = len(train_ds)
        val_n = 0 if val_ds is None else len(val_ds)
        print(f"[HPTKD] [{public_name}] train_n={train_n} val_n={val_n} split={split_info}")
        train_loader = make_cached_distill_loader(train_ds, cfg, device, shuffle=True)
        val_loader = None if val_ds is None else make_cached_distill_loader(val_ds, cfg, device, shuffle=False)

        seed_all(cfg.kd_init_seed)
        student = build_tiny_student(cfg, device)
        print(f"[HPTKD] [{public_name}] Start training...")
        student, train_history = distill_hptkd_cached(
            student,
            train_loader,
            cfg=cfg,
            device=device,
            val_loader=val_loader,
        )
        if not train_history.empty:
            best_row = train_history.sort_values(["val_score", "epoch"], ascending=[True, True]).iloc[0] if "val_score" in train_history.columns else train_history.iloc[-1]
            best_epoch = int(best_row["epoch"])
            best_val = float(best_row["val_score"]) if "val_score" in best_row.index else float("nan")
            print(f"[HPTKD] [{public_name}] Training finished. best_epoch={best_epoch} best_val={best_val:.6f}")
        train_history["public_source"] = str(public_name)
        train_history["method"] = "hptkd"
        train_history["student_arch"] = student_arch
        history_frames.append(train_history)

        print(f"[HPTKD] [{public_name}] Running downstream evaluation...")
        result_df = eval_encoder_linearprobe_3datasets(
            student,
            cfg=cfg,
            device=device,
            resources=resources,
            mode=f"normwear_{public_name}_hptkd",
        )
        result_df["public_source"] = str(public_name)
        result_df["method"] = "hptkd"
        result_df["teacher_params"] = int(teacher_params)
        result_df["student_params"] = int(student_params)
        result_df["student_param_ratio"] = float(student_params) / float(max(1, teacher_params))
        result_df["student_arch"] = student_arch
        all_frames.append(result_df)
        print(f"[HPTKD] [{public_name}] Downstream evaluation finished.")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[HPTKD] Aggregating results and writing CSV files...")
    all_results = pd.concat(all_frames, ignore_index=True)
    summary = summarize_results(all_results[all_results["public_source"] != "teacher"].copy())
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()

    tag = str(getattr(cfg, "hptkd_experiment_tag", "") or "").strip()
    suffix = f"{student_arch}_n{int(cfg.public_n)}_val{int(cfg.public_val_n)}_ep{int(cfg.kd_epochs)}"
    if tag:
        suffix = f"{tag}_{suffix}"
    csv_path = cfg.results_dir / f"normwear_hptkd_compare_{suffix}.csv"
    summary_path = cfg.results_dir / f"normwear_hptkd_compare_{suffix}_meanstd.csv"
    history_path = cfg.results_dir / f"normwear_hptkd_history_{suffix}.csv"
    all_results.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    if not history_df.empty:
        history_df.to_csv(history_path, index=False)

    print(f"[HPTKD] Results saved to {csv_path}")
    print(f"[HPTKD] Summary saved to {summary_path}")
    if not history_df.empty:
        print(f"[HPTKD] History saved to {history_path}")

    return {
        "config": cfg,
        "teacher": teacher,
        "results": all_results,
        "summary": summary,
        "history": history_df,
        "cached_meta": cached_meta,
        "split_meta": split_meta,
        "results_csv": csv_path,
        "summary_csv": summary_path,
        "history_csv": history_path,
        "teacher_params": int(teacher_params),
        "student_params": int(student_params),
    }


def run_normwear_hptkd_v2_experiment(
    config: ExperimentConfig | None = None,
    *,
    public_setups: dict[str, str | dict[str, int]] | None = None,
) -> dict[str, Any]:
    cfg = (config or ExperimentConfig()).resolve()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    public_setups = public_setups or {
        "pulsedb": "pulsedb",
        "sdb": "sdb",
    }

    seed_all(cfg.kd_init_seed)
    device = get_device(cfg.device)
    student_arch = str(getattr(cfg, "student_arch", "resnet1d"))
    teacher = NormWearTeacherEncoder(
        cfg.teacher_checkpoint,
        device=device,
        sampling_rate=cfg.teacher_sampling_rate,
        patch_pool=cfg.teacher_patch_pool,
        channel_pool=cfg.teacher_channel_pool,
    )
    teacher_params = count_parameters(teacher)
    student_proto = build_tiny_student(cfg, device)
    student_params = count_parameters(student_proto)
    del student_proto

    print("[HPTKD-V2] Configuration summary")
    refine_candidate_n = getattr(cfg, "public_refine_candidate_n", None)
    refine_seed = getattr(cfg, "public_refine_seed", None)
    print(
        {
            "device": str(device),
            "public_n": int(cfg.public_n),
            "public_val_n": int(cfg.public_val_n),
            "public_refine_enabled": bool(getattr(cfg, "public_refine_enabled", False)),
            "public_refine_candidate_n": None if refine_candidate_n is None else int(refine_candidate_n),
            "public_refine_seed": None if refine_seed is None else int(refine_seed),
            "kd_epochs": int(cfg.kd_epochs),
            "student_arch": student_arch,
            "v2_num_prototypes": int(cfg.v2_num_prototypes),
            "v2_relation_temp": float(cfg.v2_relation_temp),
            "v2_relation_ema_momentum": float(cfg.v2_relation_ema_momentum),
            "v2_lambda_relation": float(cfg.v2_lambda_relation),
            "v2_relation_align_mode": str(cfg.v2_relation_align_mode),
            "v2_use_occupancy_weighting": bool(getattr(cfg, "v2_use_occupancy_weighting", False)),
            "v2_occupancy_weight_alpha": float(getattr(cfg, "v2_occupancy_weight_alpha", 0.5)),
            "v2_occupancy_weight_min": float(getattr(cfg, "v2_occupancy_weight_min", 0.5)),
            "v2_occupancy_weight_max": float(getattr(cfg, "v2_occupancy_weight_max", 2.0)),
            "v2_use_embedding_steering": bool(getattr(cfg, "v2_use_embedding_steering", False)),
            "v2_steer_beta_max": float(getattr(cfg, "v2_steer_beta_max", 0.15)),
            "v2_steer_centrality_tau": float(getattr(cfg, "v2_steer_centrality_tau", 0.6)),
            "v2_use_tempered_resampling": bool(getattr(cfg, "v2_use_tempered_resampling", False)),
            "v2_sample_ema_momentum": float(getattr(cfg, "v2_sample_ema_momentum", 0.9)),
            "v2_resample_warmup_epochs": int(getattr(cfg, "v2_resample_warmup_epochs", 1)),
            "v2_resample_temperature": float(getattr(cfg, "v2_resample_temperature", 1.5)),
            "v2_resample_mix": float(getattr(cfg, "v2_resample_mix", 0.7)),
            "v2_resample_proto_alpha": float(getattr(cfg, "v2_resample_proto_alpha", 0.75)),
            "v2_resample_diff_alpha": float(getattr(cfg, "v2_resample_diff_alpha", 0.5)),
            "v2_resample_coverage_alpha": float(getattr(cfg, "v2_resample_coverage_alpha", 0.5)),
            "v2_resample_reliability_alpha": float(getattr(cfg, "v2_resample_reliability_alpha", 0.0)),
            "v2_resample_novelty_alpha": float(getattr(cfg, "v2_resample_novelty_alpha", 0.5)),
            "v2_best_epoch_min": int(getattr(cfg, "v2_best_epoch_min", 1)),
            "public_setups": list(public_setups.keys()),
        }
    )
    print(
        f"[HPTKD-V2] teacher_params={teacher_params} student_params={student_params} "
        f"student_ratio={float(student_params) / float(max(1, teacher_params)):.4%}"
    )

    print("[HPTKD-V2] Building downstream evaluation resources...")
    resources = build_downstream_resources(cfg, device)

    all_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    split_meta: dict[str, Any] = {}
    cached_meta: dict[str, Any] = {}

    teacher_cache = teacher_baseline_cache_path(cfg)
    if teacher_cache.exists():
        print(f"[HPTKD-V2] Reusing cached frozen teacher baseline from {teacher_cache}")
        teacher_df = pd.read_csv(teacher_cache)
    else:
        print("[HPTKD-V2] Evaluating frozen teacher baseline on downstream datasets...")
        teacher_df = eval_encoder_linearprobe_3datasets(
            teacher,
            cfg=cfg,
            device=device,
            resources=resources,
            mode="normwear_teacher",
        )
        teacher_df.to_csv(teacher_cache, index=False)
        print(f"[HPTKD-V2] Cached frozen teacher baseline to {teacher_cache}")
    teacher_df["public_source"] = "teacher"
    teacher_df["method"] = "teacher"
    teacher_df["teacher_params"] = int(teacher_params)
    teacher_df["student_params"] = np.nan
    teacher_df["student_param_ratio"] = np.nan
    teacher_df["student_arch"] = student_arch
    all_frames.append(teacher_df)

    tag = str(getattr(cfg, "v2_experiment_tag", "") or "v2").strip()
    method_name = "hptkd_v2" if tag == "v2" else f"hptkd_{tag}"
    total_setups = len(public_setups)
    for setup_idx, (public_name, public_setup) in enumerate(public_setups.items(), start=1):
        print(f"[HPTKD-V2] [{setup_idx}/{total_setups}] Preparing public source: {public_name}")
        public_subset = "random"
        if bool(getattr(cfg, "public_refine_enabled", False)):
            if not isinstance(public_setup, str):
                raise ValueError("public_refine_enabled currently supports only single-source public setups")
            candidate_cfg = copy.deepcopy(cfg)
            refine_candidate_n = getattr(cfg, "public_refine_candidate_n", None)
            candidate_cfg.public_n = int(max(int(cfg.public_n), int(refine_candidate_n or cfg.public_n)))
            candidate_dataset, candidate_meta = load_public_cached_dataset(
                teacher,
                public_setup,
                name=public_name,
                cfg=candidate_cfg,
            )
            cached_dataset, refine_meta = refine_cached_public_dataset(
                teacher,
                candidate_dataset,
                source=str(public_name),
                cfg=cfg,
            )
            meta = {
                "subset_mode": "refined",
                "candidate_cache": candidate_meta,
                "refine": refine_meta,
            }
            public_subset = "refined"
        else:
            cached_dataset, meta = load_public_cached_dataset(teacher, public_setup, name=public_name, cfg=cfg)
        cached_meta[str(public_name)] = meta
        x_shape = tuple(cached_dataset.tensors[0].shape)
        t_shape = tuple(cached_dataset.tensors[1].shape)
        print(
            f"[HPTKD-V2] [{public_name}] subset={public_subset} "
            f"cached_x_shape={x_shape} cached_t_shape={t_shape}"
        )
        train_ds, val_ds, split_info = split_cached_dataset(
            cached_dataset,
            val_n=cfg.public_val_n,
            seed=cfg.public_seed,
        )
        split_info["public_subset"] = public_subset
        split_meta[str(public_name)] = split_info
        train_n = len(train_ds)
        val_n = 0 if val_ds is None else len(val_ds)
        print(f"[HPTKD-V2] [{public_name}] subset={public_subset} train_n={train_n} val_n={val_n} split={split_info}")
        train_loader = make_cached_distill_loader(train_ds, cfg, device, shuffle=True)
        val_loader = None if val_ds is None else make_cached_distill_loader(val_ds, cfg, device, shuffle=False)

        seed_all(cfg.kd_init_seed)
        student = build_tiny_student(cfg, device)
        print(f"[HPTKD-V2] [{public_name}] Start training...")
        student, train_history = distill_hptkd_v2_cached(
            student,
            train_loader,
            cfg=cfg,
            device=device,
            val_loader=val_loader,
        )
        if not train_history.empty:
            best_row = select_best_history_row(
                train_history,
                min_epoch=int(getattr(cfg, "v2_best_epoch_min", 1)),
            )
            best_epoch = int(best_row["epoch"])
            best_val = float(best_row["val_score"]) if "val_score" in best_row.index else float("nan")
            print(f"[HPTKD-V2] [{public_name}] Training finished. best_epoch={best_epoch} best_val={best_val:.6f}")
        train_history["public_source"] = str(public_name)
        train_history["public_subset"] = str(public_subset)
        train_history["method"] = method_name
        train_history["student_arch"] = student_arch
        history_frames.append(train_history)

        print(f"[HPTKD-V2] [{public_name}] Running downstream evaluation...")
        result_df = eval_encoder_linearprobe_3datasets(
            student,
            cfg=cfg,
            device=device,
            resources=resources,
            mode=f"normwear_{public_name}_{method_name}",
        )
        result_df["public_source"] = str(public_name)
        result_df["public_subset"] = str(public_subset)
        result_df["method"] = method_name
        result_df["teacher_params"] = int(teacher_params)
        result_df["student_params"] = int(student_params)
        result_df["student_param_ratio"] = float(student_params) / float(max(1, teacher_params))
        result_df["student_arch"] = student_arch
        all_frames.append(result_df)
        print(f"[HPTKD-V2] [{public_name}] Downstream evaluation finished.")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[HPTKD-V2] Aggregating results and writing CSV files...")
    all_results = pd.concat(all_frames, ignore_index=True)
    summary = summarize_results(all_results[all_results["public_source"] != "teacher"].copy())
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()

    suffix = f"{student_arch}_n{int(cfg.public_n)}_val{int(cfg.public_val_n)}_ep{int(cfg.kd_epochs)}"
    if tag:
        suffix = f"{tag}_{suffix}"
    csv_path = cfg.results_dir / f"normwear_hptkd_v2_compare_{suffix}.csv"
    summary_path = cfg.results_dir / f"normwear_hptkd_v2_compare_{suffix}_meanstd.csv"
    history_path = cfg.results_dir / f"normwear_hptkd_v2_history_{suffix}.csv"
    all_results.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    if not history_df.empty:
        history_df.to_csv(history_path, index=False)

    print(f"[HPTKD-V2] Results saved to {csv_path}")
    print(f"[HPTKD-V2] Summary saved to {summary_path}")
    if not history_df.empty:
        print(f"[HPTKD-V2] History saved to {history_path}")

    return {
        "config": cfg,
        "teacher": teacher,
        "results": all_results,
        "summary": summary,
        "history": history_df,
        "cached_meta": cached_meta,
        "split_meta": split_meta,
        "results_csv": csv_path,
        "summary_csv": summary_path,
        "history_csv": history_path,
        "teacher_params": int(teacher_params),
        "student_params": int(student_params),
    }


def smoke_test(config: ExperimentConfig | None = None) -> dict[str, Any]:
    cfg = (config or ExperimentConfig()).resolve()
    device = get_device(cfg.device)
    teacher = NormWearTeacherEncoder(
        cfg.teacher_checkpoint,
        device=device,
        sampling_rate=cfg.teacher_sampling_rate,
        patch_pool=cfg.teacher_patch_pool,
        channel_pool=cfg.teacher_channel_pool,
    )
    public_dataset = build_public_dataset("pulsedb", cfg)
    loader, idx = build_public_subset_loader(
        public_dataset,
        n=min(32, int(cfg.public_n)),
        seed=cfg.public_seed,
        batch_size=8,
        num_workers=0,
        device=device,
        shuffle=False,
    )
    x, _ = next(iter(loader))
    with torch.inference_mode():
        teacher_z = teacher(x)
    student = build_tiny_student(cfg, device)
    with torch.inference_mode():
        student_z = student(x.to(device))
    return {
        "sampled_idx": idx,
        "x_shape": tuple(x.shape),
        "teacher_shape": tuple(teacher_z.shape),
        "student_shape": tuple(student_z.shape),
        "teacher_params": count_parameters(teacher),
        "student_params": count_parameters(student),
    }


def summarize_alignment_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in ("acc", "f1_macro", "recall_macro", "mae", "rmse", "r2") if c in results.columns]
    return (
        results.groupby(["student_arch", "align_mode", "public_source", "method", "dataset", "task"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )


def run_responsekd_alignment_comparison(
    config: ExperimentConfig | None = None,
    *,
    public_source: str = "sdb",
    student_archs: tuple[str, ...] = ("resnet1d", "normwear_tiny"),
    align_modes: tuple[str, ...] = ("projection", "same_dim"),
) -> dict[str, Any]:
    cfg = (config or ExperimentConfig()).resolve()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    seed_all(cfg.kd_init_seed)
    device = get_device(cfg.device)
    teacher = NormWearTeacherEncoder(
        cfg.teacher_checkpoint,
        device=device,
        sampling_rate=cfg.teacher_sampling_rate,
        patch_pool=cfg.teacher_patch_pool,
        channel_pool=cfg.teacher_channel_pool,
    )
    teacher_params = count_parameters(teacher)
    resources = build_downstream_resources(cfg, device)

    public_dataset = build_public_dataset(public_source, cfg)
    cached_dataset, meta = cache_public_teacher_embeddings(teacher, public_dataset, source=public_source, cfg=cfg)
    cached_loader = make_cached_distill_loader(cached_dataset, cfg, device)
    target_dim = int(cached_dataset.tensors[1].shape[-1])

    all_frames: list[pd.DataFrame] = []
    student_stats: list[dict[str, Any]] = []

    for student_arch in student_archs:
        for align_mode in align_modes:
            run_cfg = copy.deepcopy(cfg)
            run_cfg.student_arch = str(student_arch)
            seed_all(run_cfg.kd_init_seed)
            student = build_tiny_student(run_cfg, device)
            student = distill_responsekd_cached(
                student,
                cached_loader,
                cfg=run_cfg,
                device=device,
                align_mode=align_mode,
            )
            student_params = count_parameters(student)
            output_dim = infer_embed_dim(student, cached_loader, device)
            student_stats.append(
                {
                    "student_arch": str(student_arch),
                    "align_mode": str(align_mode),
                    "student_params": int(student_params),
                    "teacher_params": int(teacher_params),
                    "param_ratio": float(student_params) / float(max(1, teacher_params)),
                    "student_output_dim": int(output_dim),
                    "teacher_output_dim": int(target_dim),
                }
            )
            df = eval_encoder_linearprobe_3datasets(
                student,
                cfg=run_cfg,
                device=device,
                resources=resources,
                mode=f"{student_arch}_{public_source}_responsekd_{align_mode}",
            )
            df["student_arch"] = str(student_arch)
            df["align_mode"] = str(align_mode)
            df["public_source"] = str(public_source)
            df["method"] = "responsekd"
            df["teacher_params"] = int(teacher_params)
            df["student_params"] = int(student_params)
            df["student_param_ratio"] = float(student_params) / float(max(1, teacher_params))
            df["student_output_dim"] = int(output_dim)
            df["teacher_output_dim"] = int(target_dim)
            all_frames.append(df)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    all_results = pd.concat(all_frames, ignore_index=True)
    summary = summarize_alignment_results(all_results)
    csv_path = cfg.results_dir / f"normwear_responsekd_align_compare_{public_source}_n{int(cfg.public_n)}_ep{int(cfg.kd_epochs)}.csv"
    summary_path = cfg.results_dir / f"normwear_responsekd_align_compare_{public_source}_n{int(cfg.public_n)}_ep{int(cfg.kd_epochs)}_meanstd.csv"
    all_results.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    return {
        "config": cfg,
        "cached_meta": meta,
        "results": all_results,
        "summary": summary,
        "student_stats": pd.DataFrame(student_stats),
        "results_csv": csv_path,
        "summary_csv": summary_path,
    }

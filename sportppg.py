import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset

from pulseppg.utils.kfold import make_split, make_loaders_from_split

# -----------------------------
# Config
# -----------------------------
@dataclass
class SportPPGConfig:
    data_root: Path
    gt_root: Path

    subjects: List[str]
    acts: List[str]

    hand: str = "L"              # "L" or "R"
    use_acc: bool = True         # input channels: PPG only or PPG+ACC

    orig_fs: int = 64            # original sampling rate in stored npy
    target_fs: int = 125         # requested output sampling rate

    win_sec: float = 10.0
    hop_sec: float = 2.0         # you can change (e.g., 5.0 for less overlap)

    # how the provided GT HR was generated in the "official" code
    gt_win_samples: int = 512    # 8s @ 64Hz
    gt_hop_samples: int = 128    # 2s @ 64Hz

    dtype_np: np.dtype = np.float32


# -----------------------------
# Helpers
# -----------------------------
def ensure_shape(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "bvp":
        return x.squeeze() if x.ndim == 2 else x
    if kind == "acc":
        # want shape (3, T)
        return x.T if x.shape[0] != 3 else x
    if kind == "gt":
        return x.squeeze()
    raise ValueError(f"Unknown kind: {kind}")


def resample_linear_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linear resample 1D array to target_len (no scipy dependency)."""
    n = x.shape[0]
    if n == target_len:
        return x
    if n < 2:
        return np.zeros((target_len,), dtype=x.dtype)
    xp = np.linspace(0.0, 1.0, n, endpoint=True)
    fp = x
    xq = np.linspace(0.0, 1.0, target_len, endpoint=True)
    return np.interp(xq, xp, fp).astype(x.dtype, copy=False)


def resample_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    """Resample (T,) or (C,T) -> same shape with T=target_len."""
    if x.ndim == 1:
        return resample_linear_1d(x, target_len)
    elif x.ndim == 2:
        # (C, T)
        return np.stack([resample_linear_1d(x[c], target_len) for c in range(x.shape[0])], axis=0)
    else:
        raise ValueError(f"Unsupported ndim={x.ndim}")


# -----------------------------
# Record indexer
# -----------------------------
class SportPPGRecord:
    def __init__(self, subject: str, hand: str, act: str, bvp_path: Path, acc_path: Path, gt_path: Path):
        self.subject = subject
        self.hand = hand
        self.act = act
        self.bvp_path = bvp_path
        self.acc_path = acc_path
        self.gt_path = gt_path


class SportPPGWindowDataset(Dataset):
    """
    Returns:
      task="activity": x, y_act (long), meta_dict
      task="hr":       x, y_hr (float), meta_dict
    """
    def __init__(self, cfg: SportPPGConfig, task: str = "activity", mmap: bool = True):
        assert task in ["activity", "hr"]
        self.cfg = cfg
        self.task = task
        self.mmap = mmap

        self.act2id = {a: i for i, a in enumerate(cfg.acts)}

        # window sizes in samples (orig + target)
        self.win_orig = int(round(cfg.win_sec * cfg.orig_fs))
        self.hop_orig = int(round(cfg.hop_sec * cfg.orig_fs))
        self.win_target = int(round(cfg.win_sec * cfg.target_fs))

        # build all records (subject, hand, act)
        self.records: List[SportPPGRecord] = []
        for s in cfg.subjects:
            for a in cfg.acts:
                bvp = cfg.data_root / f"{s}{cfg.hand}_{a}_ppg.npy"
                acc = cfg.data_root / f"{s}{cfg.hand}_{a}_imu.npy"
                gt  = cfg.gt_root   / f"{s}_{a}_hr.npy"
                self.records.append(SportPPGRecord(s, cfg.hand, a, bvp, acc, gt))

        # build window index: list of (rec_idx, win_idx)
        self.master_index: List[Tuple[int, int]] = []
        self._len_cache: Dict[Path, int] = {}

        for ri, rec in enumerate(self.records):
            bvp_sig = self._load_npy(rec.bvp_path, kind="bvp")
            acc_sig = self._load_npy(rec.acc_path, kind="acc")
            T = min(len(bvp_sig), acc_sig.shape[1])
            nwin = 0 if T < self.win_orig else (T - self.win_orig) // self.hop_orig + 1
            for w in range(nwin):
                self.master_index.append((ri, w))

        # lightweight cache for memmap arrays
        self._cache: Dict[Path, np.ndarray] = {}

    def __len__(self):
        return len(self.master_index)

    def _load_raw(self, path: Path) -> np.ndarray:
        if self.mmap:
            return np.load(str(path), mmap_mode="r")
        return np.load(str(path))

    def _load_npy(self, path: Path, kind: str) -> np.ndarray:
        # no persistent caching here (used during build as well)
        x = self._load_raw(path).astype(self.cfg.dtype_np, copy=False)
        return ensure_shape(x, kind)

    def _get_cached(self, path: Path, kind: str) -> np.ndarray:
        arr = self._cache.get(path, None)
        if arr is None:
            arr = self._load_npy(path, kind=kind)
            self._cache[path] = arr
        return arr

    def _hr_label_for_window(self, gt: np.ndarray, w: int) -> float:
        """
        Interpolate GT HR (provided at official-window centers) to this window center time.
        """
        cfg = self.cfg

        # GT time axis (centers) based on official params
        n_gt = len(gt)
        t_gt = (np.arange(n_gt) * cfg.gt_hop_samples + (cfg.gt_win_samples / 2.0)) / cfg.orig_fs

        # this window center time (in seconds, on original timeline)
        t_new = (w * self.hop_orig + (self.win_orig / 2.0)) / cfg.orig_fs

        # interpolate
        y = float(np.interp(t_new, t_gt, gt))
        return y

    def __getitem__(self, idx: int):
        ri, w = self.master_index[idx]
        rec = self.records[ri]

        # load signals
        bvp = self._get_cached(rec.bvp_path, kind="bvp")  # (T,)
        acc = self._get_cached(rec.acc_path, kind="acc")  # (3,T)

        # slice on original sampling grid
        s0 = w * self.hop_orig
        e0 = s0 + self.win_orig

        bvp_seg = bvp[s0:e0]
        acc_seg = acc[:, s0:e0]

        # resample to target_fs (e.g., 1250 samples for 10s@125Hz)
        bvp_rs = resample_linear(bvp_seg, self.win_target)              # (Tt,)
        if self.cfg.use_acc:
            acc_rs = resample_linear(acc_seg, self.win_target)          # (3,Tt)
            x = np.concatenate([bvp_rs[None, :], acc_rs], axis=0)       # (4,Tt)
        else:
            x = bvp_rs[None, :]                                         # (1,Tt)

        x = torch.from_numpy(x.astype(np.float32, copy=False))

        meta = {"subject": rec.subject, "hand": rec.hand, "act": rec.act}

        if self.task == "activity":
            y = torch.tensor(self.act2id[rec.act], dtype=torch.long)
            return x, y, meta

        # HR regression task
        gt = self._get_cached(rec.gt_path, kind="gt")  # (N_gt,)
        y_hr = self._hr_label_for_window(gt, w)
        y = torch.tensor(y_hr, dtype=torch.float32)
        return x, y, meta


# -----------------------------
# Splitting + loaders
# -----------------------------
# NOTE:
# Splitting + DataLoader construction is now shared in pulseppg.utils.kfold.
# For SportPPG we only need to provide per-window subject groups:
#   groups = [ds.records[ri].subject for (ri, _) in ds.master_index]



DATA_ROOT = Path("/home/xiaoma/datasets/ppg/PPG-Sport-F5B0/dataset/")
GT_ROOT   = Path("/home/xiaoma/datasets/ppg/PPG-Sport-F5B0/cleaned_gt/")

SUBJECTS = [f"S{sid:02d}" for sid in range(1, 31)]
ACTS = ["stationary", "walking", "running", "badminton", "tabletennis", "basketball"]

cfg = SportPPGConfig(
    data_root=DATA_ROOT,
    gt_root=GT_ROOT,
    subjects=SUBJECTS,
    acts=ACTS,
    hand="L",          # change to "R" if needed
    use_acc=False,
    orig_fs=64,
    target_fs=125,
    win_sec=10.0,
    hop_sec=2.0,       # change if you want
)


if __name__ == "__main__":
    # Build datasets once; only loaders/splits change per fold
    ds_act = SportPPGWindowDataset(cfg, task="activity", mmap=True)
    ds_hr = SportPPGWindowDataset(cfg, task="hr", mmap=True)

    N_SPLITS = 3
    SEED = 175832

    print("Activity dataset size:", len(ds_act))
    print("HR dataset size:", len(ds_hr))
    print("Running subject-wise K-fold:", N_SPLITS, "folds")

    # Per-window subject group for group-based splits
    groups = [ds_act.records[ri].subject for (ri, _) in ds_act.master_index]

    for fold_idx in range(N_SPLITS):
        split = make_split(
            split_mode="kfold_subject",
            groups=groups,
            n_splits=N_SPLITS,
            fold_idx=fold_idx,
            seed=SEED,
        )

        dl_tr_act, dl_te_act = make_loaders_from_split(ds_act, split, batch_size=64, num_workers=4)
        dl_tr_hr, dl_te_hr = make_loaders_from_split(ds_hr, split, batch_size=64, num_workers=4)

        # --------------------- sanity checks ---------------------
        tr_sub = set(split.get("train_groups", []))
        te_sub = set(split.get("test_groups", []))
        if len(tr_sub) > 0 and len(te_sub) > 0:
            assert tr_sub.isdisjoint(te_sub), "train/test subjects overlap"

        tr_idx = split["train_idx"]
        te_idx = split["test_idx"]
        assert len(tr_idx) + len(te_idx) == len(ds_act), "train+test != full dataset"
        assert set(tr_idx).isdisjoint(set(te_idx)), "train/test indices overlap"

        # Check one test batch's subject meta matches declared test_subjects
        xb, yb, meta = next(iter(dl_te_act))
        if isinstance(meta, dict) and "subject" in meta and len(te_sub) > 0:
            batch_subjects = set(meta["subject"])
            assert batch_subjects.issubset(te_sub), (
                f"Test loader contains subjects outside test_subjects. "
                f"extra={sorted(batch_subjects - te_sub)}"
            )

        # ---------------------------------------------------------
        print(
            f"Fold {fold_idx}/{N_SPLITS-1} | "
            f"#train_sub={len(tr_sub)} #test_sub={len(te_sub)} | "
            f"#train_win={len(tr_idx)} #test_win={len(te_idx)} | "
            f"act_batches(tr/te)={len(dl_tr_act)}/{len(dl_te_act)} | "
            f"hr_batches(tr/te)={len(dl_tr_hr)}/{len(dl_te_hr)} | "
            f"x={tuple(xb.shape)}"
        )
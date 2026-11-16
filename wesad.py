from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _default_wesad_base_dir() -> str:
    # repo_root/pulseppg/data/datasets/wesad
    return os.path.join(os.path.dirname(__file__), "pulseppg", "data", "datasets", "wesad")


def _try_load_subject_ids(root_dir: str, split: str, hz: int, mmap: bool) -> Optional[np.ndarray]:
    """Best-effort: load per-sample subject ids if they exist.

    Notes:
      - Newer preprocessing can export: {split}_subject_{hz}Hz.npy (shape [N])
      - Legacy preprocessing exports: {split}_names_subseq_{hz}Hz.npy (NOT per-sample)
    """
    mode = "r" if mmap else None
    # Support both legacy split-based exports and the new full-dataset export.
    # New preprocessing exports (no split): subject_{hz}Hz.npy
    candidates = [
        f"subject_{hz}Hz.npy",
        "subject.npy",
        f"{split}_subject_{hz}Hz.npy",
        f"{split}_subject.npy",
        f"{split}_subjects_{hz}Hz.npy",
        f"{split}_subjects.npy",
        f"{split}_subj_{hz}Hz.npy",
        f"{split}_subj.npy",
    ]
    for fn in candidates:
        p = os.path.join(root_dir, fn)
        if os.path.exists(p):
            s = np.load(p, mmap_mode=mode)
            s = np.asarray(s)
            # normalize dtype (bytes -> str)
            if s.dtype.kind in ("S", "U", "O"):
                s = s.astype(str)
            return s
    return None


class WESADNpyDataset(Dataset):
    """Loads processed WESAD stress dataset from numpy files.

    Loads:
            Legacy split format:
                {split}_X_ppg_{hz}Hz.npy    -> (N, 1, L)
                {split}_y_stress_{hz}Hz.npy -> (N,) int (0-based)

            New full-dataset format (no split):
                X_ppg_{hz}Hz.npy            -> (N, 1, L)
                y_stress_{hz}Hz.npy         -> (N,) int (0-based)

    Returns:
      - (x, y) by default
      - (x, y, meta) if return_meta=True, where meta contains:
          {"split", "hz", "sec", "subject" (optional)}

    Optional per-sample subject ids (enables subject-wise splits):
      {split}_subject_{hz}Hz.npy  -> (N,)

            New full-dataset subject ids:
                subject_{hz}Hz.npy         -> (N,)

    NOTE: Legacy file {split}_names_subseq_{hz}Hz.npy is NOT per-sample.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        hz: int = 50,
        mmap: bool = True,
        return_meta: bool = False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.hz = hz
        self.return_meta = return_meta

        mode = "r" if mmap else None
        if split in ("all", "full", "total"):
            x_name = f"X_ppg_{hz}Hz.npy"
            y_name = f"y_stress_{hz}Hz.npy"
        else:
            x_name = f"{split}_X_ppg_{hz}Hz.npy"
            y_name = f"{split}_y_stress_{hz}Hz.npy"

        self.X = np.load(os.path.join(root_dir, x_name), mmap_mode=mode)
        self.y = np.load(os.path.join(root_dir, y_name), mmap_mode=mode)

        assert self.X.ndim == 3 and self.X.shape[1] == 1, f"Expected (N,1,L), got {self.X.shape}"
        assert len(self.X) == len(self.y), f"len(X)={len(self.X)} != len(y)={len(self.y)}"

        self.L = int(self.X.shape[-1])
        self.sec_per_sample = float(self.L) / float(self.hz)

        self.subject = _try_load_subject_ids(root_dir=root_dir, split=split, hz=hz, mmap=mmap)
        if self.subject is not None and len(self.subject) != len(self.X):
            raise ValueError(
                f"Subject id array length mismatch: len(subject)={len(self.subject)} vs len(X)={len(self.X)}"
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_np = np.asarray(self.X[idx])
        if not x_np.flags.writeable:
            x_np = x_np.copy()
        x = torch.from_numpy(x_np).float()  # (1, L)
        y = int(self.y[idx])
        if not self.return_meta:
            return x, y

        meta: Dict[str, Any] = {
            "split": self.split,
            "hz": self.hz,
            "sec": self.sec_per_sample,
        }
        if self.subject is not None:
            meta["subject"] = self.subject[idx]
        return x, y, meta

def make_wesad_dataset(
    base_dir: str | None = None,
    task: str = "binary",
    split: str = "train",
    hz: int = 50,
    mmap: bool = True,
    return_meta: bool = False,
) -> WESADNpyDataset:
    """Convenience wrapper to build a processed WESAD dataset."""
    assert task in ("binary", "multiclass")
    assert split in ("train", "val", "test", "all", "full", "total")
    if base_dir is None:
        base_dir = _default_wesad_base_dir()
    root = os.path.join(str(base_dir), task)
    return WESADNpyDataset(root, split, hz=hz, mmap=mmap, return_meta=return_meta)


def make_wesad_loaders(
    base_dir: str | None = None,
    task: str = "binary",
    hz: int = 50,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
    mmap: bool = True,
    return_meta: bool = False,
    drop_last: bool = False,
    shuffle_train: bool = True,
):
    """
    task: "binary" or "multiclass"
    Expects files in:
      base_dir/{task}/{split}_X_ppg_{hz}Hz.npy
      base_dir/{task}/{split}_y_stress_{hz}Hz.npy
    """
    assert task in ("binary", "multiclass")
    if base_dir is None:
        base_dir = _default_wesad_base_dir()
    root = os.path.join(str(base_dir), task)

    ds_tr = WESADNpyDataset(root, "train", hz=hz, mmap=mmap, return_meta=return_meta)
    ds_va = WESADNpyDataset(root, "val",   hz=hz, mmap=mmap, return_meta=return_meta)
    ds_te = WESADNpyDataset(root, "test",  hz=hz, mmap=mmap, return_meta=return_meta)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dl_tr, dl_va, dl_te



import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, random_split

class WESADNpyFullDataset(Dataset):
    """
    Loads ALL 60s samples by concatenating:
      {split}_X_ppg_{HZ}Hz.npy for split in splits
      {split}_y_stress_{HZ}Hz.npy
    and ignores the original split boundary.
    """
    def __init__(self, root_dir: str, splits=("train", "val", "test"),
                 hz: int = 50, mmap: bool = False):
        self.root_dir = root_dir
        self.hz = hz

        xs, ys = [], []
        mode = "r" if mmap else None

        for split in splits:
            x_path = os.path.join(root_dir, f"{split}_X_ppg_{hz}Hz.npy")
            y_path = os.path.join(root_dir, f"{split}_y_stress_{hz}Hz.npy")

            X = np.load(x_path, mmap_mode=mode)
            y = np.load(y_path, mmap_mode=mode)

            xs.append(np.asarray(X))
            ys.append(np.asarray(y))

        self.X = np.concatenate(xs, axis=0)  # (N,1,L60)
        self.y = np.concatenate(ys, axis=0)  # (N,)

        assert self.X.ndim == 3 and self.X.shape[1] == 1, f"Expected (N,1,L), got {self.X.shape}"
        assert len(self.X) == len(self.y), f"len(X)={len(self.X)} != len(y)={len(self.y)}"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # returns 60s sample
        x = torch.from_numpy(np.asarray(self.X[idx])).float()  # (1, L60)
        y = int(self.y[idx])
        return x, y


import bisect

class WESADSubwindowDataset(Dataset):
    """
    Wrap a 60s dataset and expose 10s windows with overlap.

    Each parent sample has a single label (stress class),
    and all derived 10s windows inherit that label.
    """
    def __init__(
        self,
        base_ds: WESADNpyFullDataset,
        win_sec: int = 10,
        overlap_sec: int = 2,
        hz: int = 50,
        return_parent_idx: bool = False,
    ):
        self.base = base_ds
        self.hz = hz
        self.win_sec = win_sec
        self.overlap_sec = overlap_sec
        self.return_parent_idx = return_parent_idx

        self.win_len = int(win_sec * hz)
        self.stride = int((win_sec - overlap_sec) * hz)
        assert self.stride > 0, "overlap_sec must be < win_sec"

        # Precompute number of windows per sample (robust to variable lengths)
        self.counts = []
        for i in range(len(self.base)):
            L = self.base.X[i].shape[-1]
            if L < self.win_len:
                self.counts.append(0)
            else:
                nwin = (L - self.win_len) // self.stride + 1
                self.counts.append(int(nwin))

        # Build prefix sum for O(logN) index mapping
        self.cum = np.cumsum(self.counts).tolist()
        self.total = self.cum[-1] if len(self.cum) > 0 else 0

    def __len__(self):
        return self.total

    def _locate(self, idx: int):
        # find parent sample index
        si = bisect.bisect_right(self.cum, idx)
        prev = self.cum[si - 1] if si > 0 else 0
        wi = idx - prev
        return si, wi

    def __getitem__(self, idx):
        si, wi = self._locate(idx)

        x60 = np.asarray(self.base.X[si])  # (1, L)
        y = int(self.base.y[si])

        start = wi * self.stride
        end = start + self.win_len
        x10 = x60[:, start:end]  # (1, win_len)

        x10 = torch.from_numpy(x10).float()

        if self.return_parent_idx:
            return x10, y, si
        return x10, y



def make_wesad_random_10s_loaders(
    base_dir: str,
    task: str = "binary",
    hz: int = 50,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    win_sec: int = 10,
    overlap_sec: int = 2,
    mmap: bool = False,
):
    """
    Build train/val/test loaders with RANDOM window-wise splits
    from existing 60s npy files.

    Steps:
      1) concat 60s train/val/test
      2) wrap as 10s subwindows
      3) random_split on 10s windows
    """
    assert task in ("binary", "multiclass")
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1

    root = os.path.join(base_dir, task)

    # 1) full 60s dataset
    full60 = WESADNpyFullDataset(
        root_dir=root,
        splits=("train", "val", "test"),
        hz=hz,
        mmap=mmap
    )

    # 2) 10s window dataset
    full10 = WESADSubwindowDataset(
        base_ds=full60,
        win_sec=win_sec,
        overlap_sec=overlap_sec,
        hz=hz
    )

    # 3) split lengths
    n = len(full10)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val

    g = torch.Generator().manual_seed(seed)
    ds_tr, ds_va, ds_te = random_split(full10, [n_train, n_val, n_test], generator=g)

    # 4) loaders
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)

    return dl_tr, dl_va, dl_te


import os
import bisect
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader, Subset


# ============================================================
# 0) Utilities: find subject-id files robustly
# ============================================================

def _find_subject_file(root: str, split: str, hz: int) -> str:
    """
    Tries common naming conventions for subject-id arrays.
    Expected shape: (N,)
    """
    cand = [
        f"{split}_subject.npy",
        f"{split}_subjects.npy",
        f"{split}_subj.npy",
        f"{split}_subject_{hz}Hz.npy",
        f"{split}_subj_{hz}Hz.npy",
    ]
    for fn in cand:
        p = os.path.join(root, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot find subject-id npy for split='{split}' in {root}. "
        f"Please export e.g. '{split}_subject.npy' (shape [N])."
    )


# ============================================================
# 1) Full 60s dataset WITH subject ids
# ============================================================

class WESADNpyFullDatasetWithSubject(Dataset):
    """
    Loads ALL 60s samples by concatenating splits, and keeps subject id per sample.

    Required files under root_dir:
      {split}_X_ppg_{hz}Hz.npy            -> (N, 1, L60)
      {split}_y_stress_{hz}Hz.npy         -> (N,)
      {split}_subject(.npy / variants)    -> (N,)  (int or str)

    Returns: (x60, y, subject)
    """
    def __init__(
        self,
        root_dir: str,
        splits: Tuple[str, ...] = ("train", "val", "test"),
        hz: int = 50,
        mmap: bool = False,
        sort_by_subject: bool = True,
    ):
        self.root_dir = root_dir
        self.hz = hz

        mode = "r" if mmap else None
        xs, ys, subs = [], [], []

        for split in splits:
            x_path = os.path.join(root_dir, f"{split}_X_ppg_{hz}Hz.npy")
            y_path = os.path.join(root_dir, f"{split}_y_stress_{hz}Hz.npy")
            s_path = _find_subject_file(root_dir, split, hz)

            X = np.load(x_path, mmap_mode=mode)
            y = np.load(y_path, mmap_mode=mode)
            s = np.load(s_path, mmap_mode=mode)

            xs.append(np.asarray(X))
            ys.append(np.asarray(y))
            subs.append(np.asarray(s))

        self.X = np.concatenate(xs, axis=0)  # (N,1,L)
        self.y = np.concatenate(ys, axis=0)  # (N,)
        self.subject = np.concatenate(subs, axis=0)  # (N,)

        assert self.X.ndim == 3 and self.X.shape[1] == 1, f"Expected (N,1,L), got {self.X.shape}"
        assert len(self.X) == len(self.y) == len(self.subject)

        # Normalize subject dtype: convert bytes/str -> str; ints keep as int
        if self.subject.dtype.kind in ("S", "U", "O"):
            self.subject = self.subject.astype(str)

        if sort_by_subject:
            # Stable sort by subject to make per-subject "chronological" splits reproducible.
            # (If you have a real timestamp/sequence array, you can extend this to sort by that too.)
            order = np.argsort(self.subject, kind="stable")
            self.X = self.X[order]
            self.y = self.y[order]
            self.subject = self.subject[order]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.X[idx])).float()  # (1, L60)
        y = int(self.y[idx])
        s = self.subject[idx]
        return x, y, s


# ============================================================
# 2) 60s -> 10s windows dataset (inherits label + subject)
# ============================================================

class WESADSubwindowDataset(Dataset):
    """
    Wrap a 60s dataset and expose 10s windows with overlap.
    Each 10s window inherits the (stress) label of its parent 60s segment.
    Returns: (x10, y, subject, parent_idx) by default.
    """
    def __init__(
        self,
        base_ds: WESADNpyFullDatasetWithSubject,
        win_sec: int = 10,
        overlap_sec: int = 2,
        hz: int = 50,
        return_parent_idx: bool = True,
    ):
        self.base = base_ds
        self.hz = hz
        self.win_sec = win_sec
        self.overlap_sec = overlap_sec
        self.return_parent_idx = return_parent_idx

        self.win_len = int(win_sec * hz)
        self.stride = int((win_sec - overlap_sec) * hz)
        assert self.stride > 0, "overlap_sec must be < win_sec"

        # windows-per-parent
        self.counts = []
        for i in range(len(self.base)):
            L = self.base.X[i].shape[-1]
            if L < self.win_len:
                self.counts.append(0)
            else:
                nwin = (L - self.win_len) // self.stride + 1
                self.counts.append(int(nwin))

        self.cum = np.cumsum(self.counts).tolist()
        self.total = self.cum[-1] if len(self.cum) > 0 else 0

    def __len__(self):
        return self.total

    def _locate(self, idx: int):
        si = bisect.bisect_right(self.cum, idx)
        prev = self.cum[si - 1] if si > 0 else 0
        wi = idx - prev
        return si, wi

    def __getitem__(self, idx):
        si, wi = self._locate(idx)

        x60 = np.asarray(self.base.X[si])  # (1, L)
        y = int(self.base.y[si])
        s = self.base.subject[si]

        start = wi * self.stride
        end = start + self.win_len
        x10 = x60[:, start:end]  # (1, win_len)
        x10 = torch.from_numpy(x10).float()

        if self.return_parent_idx:
            return x10, y, s, si
        return x10, y, s


# ============================================================
# 3) Build per-subject window index lists efficiently
# ============================================================

def _build_subject_window_segments(full10: WESADSubwindowDataset):
    """
    Returns:
      subj_to_segments: subject -> list of (start_idx, end_idx) ranges in full10 index space
      subj_to_label_segments: subject -> label -> list of (start,end)
      subjects: sorted unique subject list
      K: number of classes inferred from labels in base
    """
    subj_to_segments = defaultdict(list)
    subj_to_label_segments = defaultdict(lambda: defaultdict(list))

    # scan parents and their window ranges
    prev = 0
    ys = []

    for si, nwin in enumerate(full10.counts):
        if nwin <= 0:
            continue
        s = full10.base.subject[si]
        y = int(full10.base.y[si])
        ys.append(y)

        start = prev
        end = prev + nwin
        subj_to_segments[s].append((start, end))
        subj_to_label_segments[s][y].append((start, end))

        prev = end

    subjects = sorted(subj_to_segments.keys())
    K = int(np.max(ys) + 1) if len(ys) > 0 else 0
    return subj_to_segments, subj_to_label_segments, subjects, K


def _materialize_segments(segments: List[Tuple[int, int]]) -> np.ndarray:
    if len(segments) == 0:
        return np.array([], dtype=int)
    parts = [np.arange(a, b, dtype=int) for (a, b) in segments]
    return np.concatenate(parts, axis=0)


# ============================================================
# 4) Support/query split on per-subject windows (like DALIA)
# ============================================================

def split_support_query_window_ids(
    subj_window_ids: np.ndarray,
    support_minutes: float,
    hop_sec: float,
    win_sec: float,
    strategy: str = "chronological",       # chronological | diverse_round_robin | random
    seed: int = 123,
    gap_windows: Optional[int] = None,
    y_per_window: Optional[np.ndarray] = None,  # only needed for diverse_round_robin
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
      subj_window_ids: all window indices belonging to a subject, in a stable order
    """
    n = int(len(subj_window_ids))
    if n <= 1:
        return subj_window_ids, np.array([], dtype=int)

    if gap_windows is None:
        # safe non-overlap gap in window units
        gap_windows = int(np.ceil(win_sec / hop_sec))

    budget = int(np.floor((support_minutes * 60.0) / hop_sec))
    budget = max(1, min(budget, n - 1))

    rng = np.random.default_rng(seed)

    if strategy == "chronological":
        supp = subj_window_ids[:budget]
        q_start = min(budget + gap_windows, n)
        if q_start >= n:
            q_start = budget
            if q_start >= n:
                q_start = n - 1
        qry = subj_window_ids[q_start:]
        if len(qry) == 0:
            qry = subj_window_ids[-1:]
            supp = subj_window_ids[:-1]
        return supp, qry

    if strategy == "random":
        perm = rng.permutation(n)
        supp_idx = np.sort(perm[:budget])
        supp = subj_window_ids[supp_idx]
        mask = np.ones(n, dtype=bool)
        mask[supp_idx] = False
        qry = subj_window_ids[mask]
        if len(qry) == 0:
            qry = subj_window_ids[-1:]
            supp = subj_window_ids[:-1]
        return supp, qry

    if strategy == "diverse_round_robin":
        if y_per_window is None:
            raise ValueError("diverse_round_robin requires y_per_window for this subject.")
        # group window ids by class
        by_c = defaultdict(list)
        for wid, yy in zip(subj_window_ids.tolist(), y_per_window.tolist()):
            by_c[int(yy)].append(wid)
        # shuffle within class
        for c in by_c:
            rng.shuffle(by_c[c])
        classes = sorted(by_c.keys())
        if len(classes) == 0:
            return split_support_query_window_ids(
                subj_window_ids, support_minutes, hop_sec, win_sec,
                strategy="chronological", seed=seed, gap_windows=gap_windows
            )

        supp = []
        ptr = {c: 0 for c in classes}
        while len(supp) < budget:
            progressed = False
            for c in classes:
                if ptr[c] < len(by_c[c]) and len(supp) < budget:
                    supp.append(by_c[c][ptr[c]])
                    ptr[c] += 1
                    progressed = True
            if not progressed:
                break

        supp = np.array(sorted(set(supp)), dtype=int)

        # exclude neighborhood around support to avoid overlap-leak
        subj_pos = {wid: i for i, wid in enumerate(subj_window_ids.tolist())}
        excluded = np.zeros(n, dtype=bool)
        for wid in supp.tolist():
            i = subj_pos.get(wid, None)
            if i is None:
                continue
            lo = max(0, i - gap_windows)
            hi = min(n, i + gap_windows + 1)
            excluded[lo:hi] = True

        qry = subj_window_ids[~excluded]
        if len(qry) == 0:
            qry = subj_window_ids[-1:]
            supp = subj_window_ids[:-1]
        return supp, qry

    raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================
# 5) LOSO fold builder for WESAD
# ============================================================

def make_wesad_loso_folds(
    base_dir: str,
    task: str = "binary",          # "binary" or "multiclass"
    hz: int = 50,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,

    # windows
    win_sec: int = 10,
    overlap_sec: int = 2,

    # support/query
    support_minutes: float = 1.0,
    strategy: str = "chronological",     # chronological | diverse_round_robin | random
    seed: int = 123,
    gap_windows: Optional[int] = None,

    # val subject selection
    val_mode: str = "cyclic",      # cyclic or random
):
    """
    Returns a DALIA-like folds list. Each fold contains:
      fold["stress"]["dl_tr"], ["dl_va"], ["dl_supp"], ["dl_qry"], ["K"], ["class_w"], ["n_support"], ["n_query"]
    """
    assert task in ("binary", "multiclass")
    root = os.path.join(base_dir, task)

    # load full 60s dataset with subject ids
    full60 = WESADNpyFullDatasetWithSubject(
        root_dir=root,
        splits=("train", "val", "test"),
        hz=hz,
        mmap=False,
        sort_by_subject=True,
    )

    # windowize
    full10 = WESADSubwindowDataset(
        base_ds=full60,
        win_sec=win_sec,
        overlap_sec=overlap_sec,
        hz=hz,
        return_parent_idx=True
    )

    subj_to_segments, subj_to_label_segments, subjects, K = _build_subject_window_segments(full10)
    assert len(subjects) >= 3, "Need at least 3 subjects for LOSO+val."

    hop_sec = float(win_sec - overlap_sec)  # since stride = (win-overlap)

    folds = []
    rng = np.random.default_rng(seed)

    for fold_id, test_subj in enumerate(subjects):
        # pick val subject
        if val_mode == "cyclic":
            val_subj = subjects[(fold_id + 1) % len(subjects)]
            if val_subj == test_subj:
                val_subj = subjects[(fold_id + 2) % len(subjects)]
        elif val_mode == "random":
            others = [s for s in subjects if s != test_subj]
            val_subj = rng.choice(others).item() if hasattr(rng.choice(others), "item") else rng.choice(others)
        else:
            raise ValueError("val_mode must be 'cyclic' or 'random'")

        train_subjects = [s for s in subjects if s not in {test_subj, val_subj}]

        # ---- population train/val window ids
        tr_ids = _materialize_segments([seg for s in train_subjects for seg in subj_to_segments[s]])
        va_ids = _materialize_segments(subj_to_segments[val_subj])

        # ---- test subject support/query window ids
        test_all_ids = _materialize_segments(subj_to_segments[test_subj])

        # y_per_window only needed for diverse_round_robin
        y_per_window = None
        if strategy == "diverse_round_robin":
            # build y list cheaply from label-segments (no dataset indexing)
            y_list = []
            wid_list = []
            for y, segs in subj_to_label_segments[test_subj].items():
                for (a, b) in segs:
                    wid_list.append(np.arange(a, b, dtype=int))
                    y_list.append(np.full((b - a,), int(y), dtype=int))
            if len(wid_list) > 0:
                wid_list = np.concatenate(wid_list, axis=0)
                y_list = np.concatenate(y_list, axis=0)
                # align to test_all_ids order
                # (since wid_list is built by iterating labels, it may be permuted)
                pos = {wid: i for i, wid in enumerate(wid_list.tolist())}
                y_per_window = np.array([y_list[pos[wid]] for wid in test_all_ids.tolist()], dtype=int)
            else:
                y_per_window = None

        supp_ids, qry_ids = split_support_query_window_ids(
            subj_window_ids=test_all_ids,
            support_minutes=support_minutes,
            hop_sec=hop_sec,
            win_sec=float(win_sec),
            strategy=strategy,
            seed=seed + fold_id,
            gap_windows=gap_windows,
            y_per_window=y_per_window,
        )

        # ---- build loaders via Subset(full10, ids)
        dl_tr = DataLoader(Subset(full10, tr_ids.tolist()), batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        dl_va = DataLoader(Subset(full10, va_ids.tolist()), batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        dl_supp = DataLoader(Subset(full10, supp_ids.tolist()), batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        dl_qry = DataLoader(Subset(full10, qry_ids.tolist()), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

        # ---- compute class weights from population train (window counts)
        # (cheap: count windows per label by scanning train parents)
        counts = np.zeros((K,), dtype=float)
        for si, nwin in enumerate(full10.counts):
            if nwin <= 0:
                continue
            s = full10.base.subject[si]
            if s in set(train_subjects):
                y = int(full10.base.y[si])
                if 0 <= y < K:
                    counts[y] += float(nwin)
        if counts.sum() > 0:
            freq = counts / counts.sum()
            w = 1.0 / np.clip(freq, 1e-8, None)
            w = w / w.mean()
            class_w = torch.tensor(w, dtype=torch.float32)
        else:
            class_w = None

        folds.append({
            "fold_id": fold_id,
            "test_subject": test_subj,
            "val_subject": val_subj,
            "train_subjects": train_subjects,
            "stress": {
                "dl_tr": dl_tr,
                "dl_va": dl_va,
                "dl_supp": dl_supp,
                "dl_qry": dl_qry,
                "K": K,
                "class_w": class_w,
                "n_support": int(len(supp_ids)),
                "n_query": int(len(qry_ids)),
            }
        })

    return folds


# ============================================================
# 6) Quick dataset inspection (no split)
# ============================================================
if __name__ == "__main__":
    # Default to this repo's processed WESAD path:
    #   pulseppg/data/datasets/wesad/{binary|multiclass}/{train|val|test}_X_ppg_{hz}Hz.npy
    base_dir = os.environ.get(
        "WESAD_BASE_DIR",
        os.path.join(os.path.dirname(__file__), "pulseppg", "data", "datasets", "wesad"),
    )
    task = os.environ.get("WESAD_TASK", "binary")
    hz = int(os.environ.get("WESAD_HZ", "50"))

    root = os.path.join(base_dir, task)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Cannot find WESAD directory: {root}\n"
            f"Set WESAD_BASE_DIR (currently '{base_dir}') or run: python pulseppg/data/process/WESAD.py"
        )

    # Prefer new full-dataset export if present.
    x_all = os.path.join(root, f"X_ppg_{hz}Hz.npy")
    if os.path.exists(x_all):
        X = np.load(x_all, mmap_mode="r")
        L = int(X.shape[-1])
        print(f"[WESAD] task={task} hz={hz}Hz (full)")
        print(f"  N_total={int(X.shape[0])}")
        print(f"  sample: (1, L={L}) -> {L / float(hz):.2f} sec/sample")
    else:
        # Fallback: legacy split-based exports.
        splits = ("train", "val", "test")
        counts = {}
        lengths = {}

        for split in splits:
            x_path = os.path.join(root, f"{split}_X_ppg_{hz}Hz.npy")
            if not os.path.exists(x_path):
                raise FileNotFoundError(f"Missing file: {x_path}")
            X = np.load(x_path, mmap_mode="r")
            counts[split] = int(X.shape[0])
            lengths[split] = int(X.shape[-1])

        n_total = sum(counts.values())
        same_L = len(set(lengths.values())) == 1
        L = next(iter(lengths.values())) if same_L else None

        print(f"[WESAD] task={task} hz={hz}Hz (legacy splits)")
        print(f"  N_total={n_total} (train={counts['train']}, val={counts['val']}, test={counts['test']})")
        if L is not None:
            print(f"  sample: (1, L={L}) -> {L / float(hz):.2f} sec/sample")
        else:
            print(f"  sample length L per split: {lengths}")


# ============================================================
# WESAD LOSO Personalization Functions for PulsePPG Experiments
# ============================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional


class WESADPersonalizationDataset(Dataset):
    """Dataset for WESAD personalization with subject info."""
    
    def __init__(self, records: List[Dict[str, Any]], return_subject: bool = False):
        self.records = records
        self.return_subject = return_subject
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        x = torch.from_numpy(np.asarray(record["X"])).float()  # (1, L)
        y = int(record["y"])
        
        if self.return_subject:
            return x, y, record["subject"]
        return x, y


def build_records_wesad(
    root_dir: str, 
    task: str = "binary", 
    hz: int = 125, 
    splits: tuple = ("train", "val", "test")
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build WESAD records with proper subject mapping.
    
    Args:
        root_dir: Path to WESAD dataset directory
        task: "binary" or "multiclass" 
        hz: Sampling rate (50 or 125)
        splits: Which data splits to include
        
    Returns:
        records: List of record dicts with X, y, subject, metadata
        subjects: List of unique subject IDs
    """
    records = []
    all_subjects = set()
    
    task_dir = os.path.join(root_dir, task)
    
    for split in splits:
        # Load arrays
        X = np.load(os.path.join(task_dir, f"{split}_X_ppg_{hz}Hz.npy"))
        y = np.load(os.path.join(task_dir, f"{split}_y_stress_{hz}Hz.npy"))  
        names = np.load(os.path.join(task_dir, f"{split}_names_subseq_{hz}Hz.npy"))
        
        print(f"  {split}: X{X.shape}, y{y.shape}, names{names.shape}")
        
        # Map each X sample to subject using names array
        # Each name entry corresponds to ~12 windows on average
        windows_per_name = len(X) // len(names)
        remainder = len(X) % len(names)
        
        current_idx = 0
        for name_idx, subject in enumerate(names):
            # Distribute remainder windows to first few names
            n_windows = windows_per_name + (1 if name_idx < remainder else 0)
            
            for window_idx in range(n_windows):
                if current_idx >= len(X):
                    break
                    
                all_subjects.add(subject)
                records.append({
                    "subject": subject,
                    "X": X[current_idx],  # Shape: (1, L)
                    "y": int(y[current_idx]),
                    "split": split,
                    "task": task,
                    "sample_idx": current_idx,
                    "name_idx": name_idx,
                    "window_in_segment": window_idx
                })
                current_idx += 1
    
    return records, sorted(all_subjects)


def make_loso_personalization_folds_wesad(
    root_dir: str,
    task: str = "binary",
    hz: int = 125,
    support_fraction: float = 0.3,
    batch_size: int = 64,
    strategy: str = "chronological", 
    seed: int = 123,
    splits: tuple = ("train", "val", "test"),
    min_samples_per_subject: int = 10,
    # diversity controls
    pop_balance: str = "none",           # none | balanced
    ensure_diverse_support: bool = True,
    ensure_diverse_query: bool = True,
    min_support_per_class: int = 1,
    min_query_per_class: int = 1,
) -> List[Dict[str, Any]]:
    """
    Create LOSO personalization folds for WESAD dataset.
    
    Args:
        root_dir: Path to WESAD dataset
        task: "binary" or "multiclass"
        hz: Sampling frequency 
        support_fraction: Fraction of test subject's data for support
        batch_size: DataLoader batch size
        strategy: "chronological" or "random" for support/query split
        seed: Random seed
        splits: Data splits to include
        min_samples_per_subject: Minimum samples per subject
        
    Returns:
        folds: List of fold dictionaries
    """
    def _balance_records_by_class(recs: List[Dict[str, Any]], n_classes: int, rng: np.random.Generator):
        by_class = {c: [] for c in range(n_classes)}
        for r in recs:
            y = int(r["y"])
            if 0 <= y < n_classes:
                by_class[y].append(r)

        counts = [len(by_class[c]) for c in range(n_classes)]
        non_zero = [c for c in range(n_classes) if counts[c] > 0]
        if not non_zero:
            return recs

        min_n = min(counts[c] for c in non_zero)
        if min_n <= 0:
            return recs

        balanced = []
        for c in non_zero:
            cls_recs = by_class[c]
            if len(cls_recs) > min_n:
                idx = rng.choice(len(cls_recs), size=min_n, replace=False)
                balanced.extend([cls_recs[i] for i in idx])
            else:
                balanced.extend(cls_recs)

        rng.shuffle(balanced)
        return balanced

    def _split_support_query_stratified(
        recs: List[Dict[str, Any]],
        n_support_target: int,
        n_classes: int,
        rng: np.random.Generator,
        min_supp_per_class: int,
        min_qry_per_class: int,
    ):
        by_class_idx = {c: [] for c in range(n_classes)}
        for i, r in enumerate(recs):
            y = int(r["y"])
            if 0 <= y < n_classes:
                by_class_idx[y].append(i)

        classes = [c for c in range(n_classes) if len(by_class_idx[c]) > 0]
        for c in classes:
            rng.shuffle(by_class_idx[c])

        support_idx = []

        # If support budget is too small, pick as many classes as possible
        if n_support_target < len(classes):
            for c in classes[:n_support_target]:
                support_idx.append(by_class_idx[c].pop())
        else:
            # Ensure minimum per-class support (when possible)
            for c in classes:
                take = min(min_supp_per_class, len(by_class_idx[c]))
                for _ in range(take):
                    support_idx.append(by_class_idx[c].pop())

        # Fill remaining support in round-robin
        while len(support_idx) < n_support_target:
            progressed = False
            for c in classes:
                if len(support_idx) >= n_support_target:
                    break
                if by_class_idx[c]:
                    support_idx.append(by_class_idx[c].pop())
                    progressed = True
            if not progressed:
                break

        query_idx = []
        for c in classes:
            query_idx.extend(by_class_idx[c])

        # Ensure query has at least min_qry_per_class (when possible)
        if min_qry_per_class > 0:
            qry_classes = set(int(recs[i]["y"]) for i in query_idx)
            for c in classes:
                if c not in qry_classes:
                    # Move one from support if it doesn't violate min support
                    cand = [i for i in support_idx if int(recs[i]["y"]) == c]
                    if len(cand) > min_supp_per_class:
                        move_i = cand[0]
                        support_idx.remove(move_i)
                        query_idx.append(move_i)
                        qry_classes.add(c)

        support_recs = [recs[i] for i in support_idx]
        query_recs = [recs[i] for i in query_idx]
        return support_recs, query_recs

    print(f"Building WESAD {task} records from {root_dir}...")  
    records, all_subjects = build_records_wesad(root_dir, task, hz, splits)
    
    # Filter subjects by minimum sample count
    subject_counts = defaultdict(int)
    for r in records:
        subject_counts[r["subject"]] += 1
    
    valid_subjects = [s for s in all_subjects if subject_counts[s] >= min_samples_per_subject]
    print(f"Found {len(all_subjects)} total subjects, {len(valid_subjects)} with >= {min_samples_per_subject} samples")
    
    if len(valid_subjects) < 3:
        raise ValueError(f"Need at least 3 subjects for LOSO, got {len(valid_subjects)}")
    
    # Get number of classes for this task
    all_labels = [r["y"] for r in records]
    num_classes = len(set(all_labels))
    print(f"Task '{task}': {num_classes} classes, label range: {min(all_labels)}-{max(all_labels)}")
    
    folds = []
    nS = len(valid_subjects)
    
    np.random.seed(seed)
    
    for fold_id, test_subj in enumerate(valid_subjects):
        print(f"\nFold {fold_id}: Testing on subject {test_subj}")
        
        # LOSO validation subject (next in cycle)
        val_subj = valid_subjects[(fold_id + 1) % nS]
        if val_subj == test_subj:
            val_subj = valid_subjects[(fold_id + 2) % nS]
        
        train_subjects = [s for s in valid_subjects if s not in {test_subj, val_subj}]
        
        # Split records by subject groups
        test_records = [r for r in records if r["subject"] == test_subj]
        train_records = [r for r in records if r["subject"] in train_subjects]
        val_records = [r for r in records if r["subject"] == val_subj]

        rng = np.random.default_rng(seed + fold_id)

        # Optional population balancing to improve label diversity
        if pop_balance == "balanced":
            train_records = _balance_records_by_class(train_records, num_classes, rng)
            val_records = _balance_records_by_class(val_records, num_classes, rng)
        
        print(f"  Train: {len(train_subjects)} subjects ({len(train_records)} samples)")
        print(f"  Val: 1 subject {val_subj} ({len(val_records)} samples)")
        print(f"  Test: 1 subject {test_subj} ({len(test_records)} samples)")
        
        # Split test subject's data into support and query
        n_support = max(1, int(support_fraction * len(test_records)))

        if strategy == "chronological" and not ensure_diverse_support:
            # Chronological split (no diversity enforcement)
            support_records = test_records[:n_support]
            query_records = test_records[n_support:]
        else:
            # Stratified/random split with diversity enforcement
            if strategy == "chronological":
                # Preserve order as much as possible by not shuffling before stratification
                recs_for_split = list(test_records)
            else:
                recs_for_split = list(test_records)
                rng.shuffle(recs_for_split)

            if ensure_diverse_support or ensure_diverse_query:
                support_records, query_records = _split_support_query_stratified(
                    recs_for_split,
                    n_support_target=n_support,
                    n_classes=num_classes,
                    rng=rng,
                    min_supp_per_class=min_support_per_class,
                    min_qry_per_class=min_query_per_class if ensure_diverse_query else 0,
                )
            else:
                support_records = recs_for_split[:n_support]
                query_records = recs_for_split[n_support:]
        
        print(f"  Support: {len(support_records)} samples, Query: {len(query_records)} samples")
        
        # Check class distribution in support/query sets
        support_labels = [r["y"] for r in support_records]
        support_label_counts = Counter(support_labels)
        query_labels = [r["y"] for r in query_records]
        query_label_counts = Counter(query_labels)
        print(f"  Support label distribution: {dict(support_label_counts)}")
        print(f"  Query label distribution:   {dict(query_label_counts)}")
        
        # Create datasets
        ds_tr = WESADPersonalizationDataset(train_records) if train_records else None
        ds_va = WESADPersonalizationDataset(val_records) if val_records else None
        ds_supp = WESADPersonalizationDataset(support_records) if support_records else None
        ds_qry = WESADPersonalizationDataset(query_records) if query_records else None
        
        # Create dataloaders
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False) if ds_tr else None
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False) if ds_va else None  
        dl_supp = DataLoader(ds_supp, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False) if ds_supp else None
        dl_qry = DataLoader(ds_qry, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False) if ds_qry else None
        
        # Compute class weights for imbalanced data
        train_labels = [r["y"] for r in train_records] if train_records else []
        if len(train_labels) > 0:
            label_counts = Counter(train_labels)
            total = sum(label_counts.values())
            class_weights = torch.tensor([
                total / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)
            ], dtype=torch.float32)
        else:
            class_weights = torch.ones(num_classes, dtype=torch.float32)
        
        # Build fold dictionary  
        fold = {
            "fold_id": fold_id,
            "test_subject": test_subj,
            "train_subjects": train_subjects,
            "val_subject": val_subj,
            "task": task,
            "task_type": "classification",
            "num_classes": num_classes,
            "n_support": len(support_records),
            "n_query": len(query_records),
            "dl_tr": dl_tr,
            "dl_va": dl_va, 
            "dl_supp": dl_supp,
            "dl_qry": dl_qry,
            "class_weights": class_weights,
            "support_label_counts": support_label_counts,
            "query_label_counts": query_label_counts,
            "train_label_counts": Counter(train_labels),
            "val_label_counts": Counter([r["y"] for r in val_records]) if val_records else Counter(),
        }
        
        folds.append(fold)
    
    print(f"\n✓ Created {len(folds)} LOSO folds for WESAD {task} task")
    return folds

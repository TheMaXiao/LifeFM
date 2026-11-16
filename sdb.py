import os
import re
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
from torch_ecg._preprocessors import Normalize
from pulseppg.data.process.utils import resample_batch_signal, preprocess_one_ppg_signal

class SDBNpyDataset(Dataset):
    """
    Loads SDB PPG windows and binary sleep disturbance labels.

    Expects in root_dir:
      train_X_ppg_{fs}Hz.npy, train_y_sdb.npy
      val_X_ppg_{fs}Hz.npy,   val_y_sdb.npy
      test_X_ppg_{fs}Hz.npy,  test_y_sdb.npy

    X shape: (N, 1, L)
    y: 0/1 binary labels (or ints that can be treated as binary)
    """
    def __init__(self, root_dir: str, split: str, fs: int, mmap: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.fs = fs

        mode = "r" if mmap else None

        x_path = os.path.join(root_dir, f"{split}_X_ppg_{fs}Hz.npy")
        y_path = os.path.join(root_dir, f"{split}_y_sdb.npy")

        self.X = np.load(x_path, mmap_mode=mode)
        self.y = np.load(y_path, mmap_mode=mode)

        # Expect (N,1,L)
        if self.X.ndim != 3 or self.X.shape[1] != 1:
            raise ValueError(f"Expected X shape (N,1,L), got {self.X.shape}")

        # Make labels int64
        self.y = self.y.astype(np.int64)
        assert len(self.X) == len(self.y), f"len(X)={len(self.X)} != len(y)={len(self.y)}"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.X[idx])).float()  # (1,L)
        y = int(self.y[idx])                                  # scalar int
        return x, y


def make_sdb_loaders(root_dir: str,
                     fs: int = 125,
                     batch_size: int = 64,
                     num_workers: int = 0,
                     pin_memory: bool = True):
    """
    Build train/val/test DataLoaders for SDB sleep disturbance classification.
    """
    ds_tr = SDBNpyDataset(root_dir, "train", fs=fs, mmap=True)
    ds_va = SDBNpyDataset(root_dir, "val",   fs=fs, mmap=True)
    ds_te = SDBNpyDataset(root_dir, "test",  fs=fs, mmap=True)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)
    return dl_tr, dl_va, dl_te

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split

def make_sdb_random_loaders(root_dir: str,
                            fs: int = 125,
                            batch_size: int = 64,
                            num_workers: int = 0,
                            pin_memory: bool = True,
                            train_frac: float = 0.7,
                            val_frac: float = 0.15,
                            seed: int = 42,
                            use_splits=("train", "val", "test"),
                            mmap: bool = True):
    """
    Build train/val/test DataLoaders for SDB with RANDOM sample-wise splits.
    Ignores original subject-wise split boundaries:
      1) loads datasets from `use_splits`
      2) concatenates them
      3) random_split into new train/val/test subsets.
    """
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and (train_frac + val_frac) < 1

    # 1) Build base datasets and concatenate
    base_datasets = [
        SDBNpyDataset(root_dir, split, fs=fs, mmap=mmap)
        for split in use_splits
    ]
    full_ds = ConcatDataset(base_datasets)

    # 2) Compute sizes
    n_total = len(full_ds)
    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)
    n_test  = n_total - n_train - n_val

    # 3) Random split with fixed seed
    g = torch.Generator().manual_seed(seed)
    ds_tr, ds_va, ds_te = random_split(full_ds, [n_train, n_val, n_test], generator=g)

    # 4) DataLoaders
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)

    return dl_tr, dl_va, dl_te


class SDBPersonalizationDataset(Dataset):
    """
    Dataset for SDB personalization records.
    Each record is a dict with keys: x, y, subject.
    Returns (x, y, subject).
    """
    def __init__(self, records: List[Dict[str, Any]]):
        super().__init__()
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        x = torch.from_numpy(np.asarray(r["x"])).float()
        y = int(r["y"])
        s = r.get("subject", None)
        return x, y, s


def _list_subject_ids(root_dir: str) -> List[int]:
    ids = []
    for fn in os.listdir(root_dir):
        if not fn.startswith("subject") or not fn.endswith(".csv"):
            continue
        m = re.search(r"subject(\d+)", fn)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


SDB_TRAIN_SUBJECTS = [
    1, 2, 4, 7, 9, 11, 13, 18, 19, 20, 26, 27, 30, 31, 32, 33, 35, 40, 41,
    43, 44, 46, 47, 50, 51, 53, 55, 58, 59, 61, 62, 63, 64, 65, 67, 69, 70,
    71, 72, 73, 76, 78, 79, 82, 86, 87, 90, 91, 92, 95, 98, 104, 105, 109,
    110, 112, 114, 117, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129,
    130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 145, 147,
    150, 153, 156, 157, 160,
]
SDB_VAL_SUBJECTS = [
    5, 8, 10, 14, 15, 16, 22, 28, 38, 54, 57, 66, 68, 74, 80, 81, 85, 88,
    89, 94, 96, 102, 103, 118, 125, 143, 146, 148, 154,
]
SDB_TEST_SUBJECTS = [
    3, 12, 23, 24, 25, 36, 37, 42, 45, 48, 49, 52, 56, 60, 77, 83, 84, 97,
    99, 101, 106, 111, 113, 115, 116, 144, 149, 151, 155, 158,
]


def _process_subject_ppg(
    root_dir: str,
    subject_id: int,
    hz: int,
    window_sec: int,
    fs_source: float,
    norm: Normalize,
) -> np.ndarray:
    subject_path = os.path.join(root_dir, f"subject{subject_id}.csv")
    if not os.path.exists(subject_path):
        return np.zeros((0, 1, int(hz * window_sec)), dtype=np.float32)

    df = pd.read_csv(subject_path, usecols=["pleth"])
    signal = df["pleth"].to_numpy(dtype=np.float32)

    signal, _ = norm.apply(signal, fs=fs_source)
    signal, _, _, _ = preprocess_one_ppg_signal(waveform=signal, frequency=fs_source)
    signal = resample_batch_signal(signal, fs_original=fs_source, fs_target=hz, axis=0)

    chunk_size = int(hz * window_sec)
    num_chunks = len(signal) // chunk_size
    if num_chunks <= 0:
        return np.zeros((0, 1, chunk_size), dtype=np.float32)

    chunks = np.array_split(signal[:num_chunks * chunk_size], num_chunks)
    chunks = np.asarray(chunks, dtype=np.float32)[:, None, :]
    return chunks


def _subject_cache_path(root_dir: str, split: str, hz: int) -> str:
    return os.path.join(root_dir, f"{split}_subject_{hz}Hz.npy")


def _load_xy_split(root_dir: str, split: str, hz: int) -> Tuple[np.ndarray, np.ndarray]:
    x_path = os.path.join(root_dir, f"{split}_X_ppg_{hz}Hz.npy")
    y_path = os.path.join(root_dir, f"{split}_y_sdb_{hz}Hz.npy")
    if not os.path.exists(y_path):
        y_path = os.path.join(root_dir, f"{split}_y_sdb.npy")
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    return X, y


def _build_subject_ids_from_raw(
    root_dir: str,
    hz: int,
    window_sec: int,
    fs_source: float,
    subject_ids: List[int],
    save_path: Optional[str] = None,
) -> np.ndarray:
    norm = Normalize(method="z-score")
    subjects_sorted = sorted(subject_ids)
    subject_arr: List[int] = []

    for sid in subjects_sorted:
        chunks = _process_subject_ppg(
            root_dir=root_dir,
            subject_id=sid,
            hz=hz,
            window_sec=window_sec,
            fs_source=fs_source,
            norm=norm,
        )
        if len(chunks) == 0:
            continue
        subject_arr.extend([sid] * len(chunks))

    subject_arr = np.asarray(subject_arr, dtype=np.int64)
    if save_path is not None:
        np.save(save_path, subject_arr)
    return subject_arr


def _load_or_build_subject_ids(
    root_dir: str,
    split: str,
    hz: int,
    window_sec: int,
    fs_source: float,
    subject_ids: List[int],
    use_cache: bool = True,
) -> np.ndarray:
    cache_path = _subject_cache_path(root_dir, split, hz)
    if use_cache and os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")
    return _build_subject_ids_from_raw(
        root_dir=root_dir,
        hz=hz,
        window_sec=window_sec,
        fs_source=fs_source,
        subject_ids=subject_ids,
        save_path=cache_path if use_cache else None,
    )


def build_records_sdb(
    root_dir: str,
    hz: int = 125,
    window_sec: int = 10,
    fs_source: float = 62.5,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    use_cache: bool = True,
    quiet: bool = False,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Build per-window records for SDB from npy X/y files.
    Returns (records, subject_ids_used).
    """
    split_subjects = {
        "train": SDB_TRAIN_SUBJECTS,
        "val": SDB_VAL_SUBJECTS,
        "test": SDB_TEST_SUBJECTS,
    }

    records: List[Dict[str, Any]] = []
    used_subjects: List[int] = []

    for split in splits:
        X, y = _load_xy_split(root_dir, split, hz)
        subject_ids = split_subjects.get(split, [])
        subj_arr = _load_or_build_subject_ids(
            root_dir=root_dir,
            split=split,
            hz=hz,
            window_sec=window_sec,
            fs_source=fs_source,
            subject_ids=subject_ids,
            use_cache=use_cache,
        )

        if len(subj_arr) != len(X):
            raise ValueError(
                f"Subject id array length mismatch for {split}: {len(subj_arr)} != {len(X)}"
            )

        for i in range(len(X)):
            sid = int(subj_arr[i])
            records.append({"x": X[i], "y": int(y[i]), "subject": sid})
            used_subjects.append(sid)

        if not quiet:
            print(f"{split}: {len(X)} samples, subjects={len(set(subj_arr))}")

    return records, sorted(set(used_subjects))


def _balance_records_by_class(records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_class[int(r["y"])].append(r)

    if not by_class:
        return records

    min_n = min(len(v) for v in by_class.values())
    balanced = []
    for cls, rs in by_class.items():
        idx = rng.permutation(len(rs))[:min_n]
        balanced.extend([rs[i] for i in idx])
    rng.shuffle(balanced)
    return balanced


def _split_support_query_stratified(
    records: List[Dict[str, Any]],
    support_fraction: float,
    num_classes: int,
    seed: int,
    ensure_diverse_support: bool,
    ensure_diverse_query: bool,
    min_support_per_class: int,
    min_query_per_class: int,
    balance_support_query: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_class[int(r["y"])].append(i)

    for cls in by_class:
        rng.shuffle(by_class[cls])

    query_idx: List[int] = []
    support_idx: List[int] = []

    if balance_support_query and num_classes > 1:
        class_sizes = [len(by_class.get(cls, [])) for cls in range(num_classes)]
        min_count = min(class_sizes) if class_sizes else 0
        required = min_support_per_class + min_query_per_class
        if min_count >= required:
            per_class_support = max(min_support_per_class, int(round(min_count * support_fraction)))
            per_class_support = min(per_class_support, min_count - min_query_per_class)
            for cls in range(num_classes):
                idxs = by_class.get(cls, [])
                support_idx.extend(idxs[:per_class_support])
                query_idx.extend(idxs[per_class_support:per_class_support + (min_count - per_class_support)])
            support = [records[i] for i in support_idx]
            query = [records[i] for i in query_idx]
            return support, query

    if ensure_diverse_query:
        for cls in range(num_classes):
            idxs = by_class.get(cls, [])
            take = min(min_query_per_class, len(idxs))
            query_idx.extend(idxs[:take])
            by_class[cls] = idxs[take:]

    if ensure_diverse_support:
        for cls in range(num_classes):
            idxs = by_class.get(cls, [])
            take = min(min_support_per_class, len(idxs))
            support_idx.extend(idxs[:take])
            by_class[cls] = idxs[take:]

    remaining = [i for cls in by_class for i in by_class[cls]]
    rng.shuffle(remaining)

    n_support_target = max(1, int(round(len(records) * support_fraction)))
    n_support_target = max(n_support_target, len(support_idx))
    n_support_target = min(n_support_target, len(records) - max(1, len(query_idx)))

    needed = n_support_target - len(support_idx)
    if needed > 0:
        support_idx.extend(remaining[:needed])
        remaining = remaining[needed:]

    query_idx.extend(remaining)

    if len(query_idx) == 0:
        query_idx.append(support_idx.pop())

    support = [records[i] for i in support_idx]
    query = [records[i] for i in query_idx]
    return support, query


def _compute_class_weights(records: List[Dict[str, Any]], num_classes: int) -> torch.Tensor:
    counts = Counter(int(r["y"]) for r in records)
    total = max(1, len(records))
    weights = []
    for c in range(num_classes):
        cnt = max(1, counts.get(c, 0))
        weights.append(total / (num_classes * cnt))
    return torch.tensor(weights, dtype=torch.float32)


def _subject_label_map(records: List[Dict[str, Any]]) -> Dict[int, int]:
    by_subject: Dict[int, Counter] = defaultdict(Counter)
    for r in records:
        sid = int(r["subject"])
        by_subject[sid].update([int(r["y"])])
    label_map = {}
    for sid, cnt in by_subject.items():
        label_map[sid] = cnt.most_common(1)[0][0]
    return label_map


def make_balanced_subject_folds_sdb(
    root_dir: str,
    hz: int = 125,
    window_sec: int = 10,
    fs_source: float = 62.5,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 123,
    num_folds: int = 5,
    subjects_per_class_support: int = 1,
    subjects_per_class_query: int = 1,
    val_fraction: float = 0.1,
    pop_balance: str = "balanced",
    balance_support_query: bool = True,
    records: Optional[List[Dict[str, Any]]] = None,
    subjects: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Build class-balanced folds where support/query are sampled from multiple subjects.
    This is not strict personalization; it balances labels across subjects.
    """
    if records is None or subjects is None:
        records, subjects = build_records_sdb(
            root_dir=root_dir,
            hz=hz,
            window_sec=window_sec,
            fs_source=fs_source,
            quiet=True,
        )

    num_classes = max(1, len(set(int(r["y"]) for r in records)))
    label_map = _subject_label_map(records)

    by_subject: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_subject[int(r["subject"])].append(r)

    subjects = sorted(set(subjects))
    by_class_subjects: Dict[int, List[int]] = defaultdict(list)
    for sid in subjects:
        if sid not in label_map:
            continue
        by_class_subjects[label_map[sid]].append(sid)

    rng = np.random.default_rng(seed)
    folds: List[Dict[str, Any]] = []

    for i in range(num_folds):
        support_subjects: List[int] = []
        query_subjects: List[int] = []

        for cls in range(num_classes):
            cand = list(by_class_subjects.get(cls, []))
            rng.shuffle(cand)
            need = subjects_per_class_support + subjects_per_class_query
            if len(cand) < need:
                raise RuntimeError(
                    f"Not enough subjects for class {cls}: have {len(cand)}, need {need}"
                )
            query_subjects.extend(cand[:subjects_per_class_query])
            support_subjects.extend(cand[subjects_per_class_query:need])

        remaining_subjects = [
            s for s in subjects
            if s not in set(support_subjects) and s not in set(query_subjects)
        ]

        # Build val subjects stratified by class from remaining subjects
        val_subjects: List[int] = []
        by_class_remaining: Dict[int, List[int]] = defaultdict(list)
        for s in remaining_subjects:
            by_class_remaining[label_map[s]].append(s)
        for cls in range(num_classes):
            cand = list(by_class_remaining.get(cls, []))
            rng.shuffle(cand)
            n_val = int(round(len(cand) * val_fraction))
            if n_val > 0:
                val_subjects.extend(cand[:n_val])

        train_subjects = [s for s in remaining_subjects if s not in set(val_subjects)]

        tr_records = [r for s in train_subjects for r in by_subject[s]]
        va_records = [r for s in val_subjects for r in by_subject[s]]
        supp_records = [r for s in support_subjects for r in by_subject[s]]
        qry_records = [r for s in query_subjects for r in by_subject[s]]

        if pop_balance == "balanced":
            tr_records = _balance_records_by_class(tr_records, seed=seed + i)

        if balance_support_query:
            supp_records = _balance_records_by_class(supp_records, seed=seed + i + 11)
            qry_records = _balance_records_by_class(qry_records, seed=seed + i + 37)

        ds_tr = SDBPersonalizationDataset(tr_records)
        ds_va = SDBPersonalizationDataset(va_records)
        ds_supp = SDBPersonalizationDataset(supp_records)
        ds_qry = SDBPersonalizationDataset(qry_records)

        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        dl_supp = DataLoader(ds_supp, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        dl_qry = DataLoader(ds_qry, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        fold = {
            "task": "multiclass",
            "num_classes": num_classes,
            "test_subject": query_subjects,
            "val_subject": val_subjects,
            "train_subjects": train_subjects,
            "dl_tr": dl_tr,
            "dl_va": dl_va,
            "dl_supp": dl_supp,
            "dl_qry": dl_qry,
            "n_support": len(supp_records),
            "n_query": len(qry_records),
            "class_weights": _compute_class_weights(tr_records, num_classes),
            "support_label_counts": Counter(int(r["y"]) for r in supp_records),
            "query_label_counts": Counter(int(r["y"]) for r in qry_records),
            "train_label_counts": Counter(int(r["y"]) for r in tr_records),
            "val_label_counts": Counter(int(r["y"]) for r in va_records),
        }
        folds.append(fold)

    return folds


def make_loso_personalization_folds_sdb(
    root_dir: str,
    task: str = "multiclass",
    hz: int = 125,
    window_sec: int = 10,
    fs_source: float = 62.5,
    support_fraction: float = 0.3,
    strategy: str = "chronological",
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 123,
    min_samples_per_subject: int = 5,
    val_subject_strategy: str = "next",
    pop_balance: str = "none",
    ensure_diverse_support: bool = True,
    ensure_diverse_query: bool = True,
    min_support_per_class: int = 1,
    min_query_per_class: int = 1,
    balance_support_query: bool = False,
    require_all_classes: bool = True,
    records: Optional[List[Dict[str, Any]]] = None,
    subjects: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Build LOSO personalization folds for SDB.

    pop_balance: "none" or "balanced" (class-balanced population train set)
    val_subject_strategy: "next" or "random"
    """
    if records is None or subjects is None:
        records, subjects = build_records_sdb(
            root_dir=root_dir,
            hz=hz,
            window_sec=window_sec,
            fs_source=fs_source,
            quiet=True,
        )

    if task == "binary":
        num_classes = 2
    else:
        num_classes = max(1, len(set(int(r["y"]) for r in records)))

    rng = np.random.default_rng(seed)
    subjects = sorted(set(subjects))

    by_subject: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_subject[int(r["subject"])].append(r)

    eligible_subjects = [s for s in subjects if len(by_subject.get(s, [])) >= min_samples_per_subject]
    if require_all_classes:
        filtered = []
        for s in eligible_subjects:
            counts = Counter(int(r["y"]) for r in by_subject.get(s, []))
            if len(counts) < num_classes:
                continue
            if balance_support_query:
                min_count = min(counts.values())
                if min_count < (min_support_per_class + min_query_per_class):
                    continue
            filtered.append(s)
        eligible_subjects = filtered
    folds: List[Dict[str, Any]] = []

    for i, test_subject in enumerate(eligible_subjects):
        remaining = [s for s in eligible_subjects if s != test_subject]
        if len(remaining) < 2:
            continue

        if val_subject_strategy == "random":
            val_subject = rng.choice(remaining)
        else:
            val_subject = remaining[i % len(remaining)]

        train_subjects = [s for s in remaining if s != val_subject]

        tr_records = [r for s in train_subjects for r in by_subject[s]]
        va_records = list(by_subject[val_subject])
        te_records = list(by_subject[test_subject])

        if balance_support_query:
            te_counts = Counter(int(r["y"]) for r in te_records)
            min_count = min(te_counts.values()) if te_counts else 0
            required = min_support_per_class + min_query_per_class
            if len(te_counts) < num_classes or min_count < required:
                import warnings
                warnings.warn(
                    f"Subject {test_subject}: cannot balance support/query (counts={dict(te_counts)})"
                )

        if pop_balance == "balanced":
            tr_records = _balance_records_by_class(tr_records, seed=seed + i)

        if ensure_diverse_support or ensure_diverse_query:
            supp_records, qry_records = _split_support_query_stratified(
                te_records,
                support_fraction=support_fraction,
                num_classes=num_classes,
                seed=seed + i,
                ensure_diverse_support=ensure_diverse_support,
                ensure_diverse_query=ensure_diverse_query,
                min_support_per_class=min_support_per_class,
                min_query_per_class=min_query_per_class,
                balance_support_query=balance_support_query,
            )
        else:
            split_idx = max(1, int(round(len(te_records) * support_fraction)))
            if strategy == "random":
                order = rng.permutation(len(te_records)).tolist()
                te_records = [te_records[i] for i in order]
            supp_records = te_records[:split_idx]
            qry_records = te_records[split_idx:] if split_idx < len(te_records) else te_records[-1:]

        ds_tr = SDBPersonalizationDataset(tr_records)
        ds_va = SDBPersonalizationDataset(va_records)
        ds_supp = SDBPersonalizationDataset(supp_records)
        ds_qry = SDBPersonalizationDataset(qry_records)

        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        dl_supp = DataLoader(ds_supp, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        dl_qry = DataLoader(ds_qry, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        fold = {
            "task": task,
            "num_classes": num_classes,
            "test_subject": test_subject,
            "val_subject": val_subject,
            "train_subjects": train_subjects,
            "dl_tr": dl_tr,
            "dl_va": dl_va,
            "dl_supp": dl_supp,
            "dl_qry": dl_qry,
            "n_support": len(supp_records),
            "n_query": len(qry_records),
            "class_weights": _compute_class_weights(tr_records, num_classes),
            "support_label_counts": Counter(int(r["y"]) for r in supp_records),
            "query_label_counts": Counter(int(r["y"]) for r in qry_records),
            "train_label_counts": Counter(int(r["y"]) for r in tr_records),
            "val_label_counts": Counter(int(r["y"]) for r in va_records),
        }
        folds.append(fold)

    return folds

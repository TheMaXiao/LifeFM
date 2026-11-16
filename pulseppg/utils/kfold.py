"""Utility functions for deterministic K-fold splitting.

This module intentionally avoids sklearn dependency and focuses on:
- splitting unique *groups* (e.g., subject IDs) into K folds
- producing sample indices given a per-sample group label list

Typical use case:
- You have N samples and each sample belongs to a subject.
- You want K-fold CV where all samples from the same subject stay together.

Example:
    groups = ["S01", "S01", "S02", "S03", ...]  # length N
    split = make_group_kfold_split(groups, n_splits=3, fold_idx=0, seed=123)
    train_idx, test_idx = split["train_idx"], split["test_idx"]
"""

from __future__ import annotations

from typing import Any, Hashable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def make_random_split(
    n_samples: int,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> Dict[str, Any]:
    """Sample-wise random split (train/test) over indices [0..n_samples-1]."""
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_samples)
    rng.shuffle(all_idx)
    n_tr = int(round(train_ratio * n_samples))
    return {
        "train_idx": all_idx[:n_tr].tolist(),
        "test_idx": all_idx[n_tr:].tolist(),
    }


def make_group_holdout_split(
    groups: Sequence[Hashable],
    train_ratio: float = 0.8,
    seed: int = 0,
    shuffle: bool = True,
) -> Dict[str, Any]:
    """Group-wise holdout split (train/test) given per-sample group labels.

    This keeps all samples from the same group together.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    uniq = _unique_preserve_order(groups)
    if len(uniq) < 2:
        raise ValueError(f"Need at least 2 unique groups, got {len(uniq)}")

    uniq_arr = np.array(uniq, dtype=object)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq_arr)

    n_tr_g = int(round(train_ratio * len(uniq_arr)))
    # ensure both sides are non-empty
    n_tr_g = max(1, min(n_tr_g, len(uniq_arr) - 1))

    train_groups = uniq_arr[:n_tr_g].tolist()
    test_groups = uniq_arr[n_tr_g:].tolist()

    test_set = set(test_groups)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for i, g in enumerate(groups):
        if g in test_set:
            test_idx.append(i)
        else:
            train_idx.append(i)

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_groups": train_groups,
        "test_groups": test_groups,
    }


def make_split(
    *,
    split_mode: str = "random",
    n_samples: Optional[int] = None,
    groups: Optional[Sequence[Hashable]] = None,
    train_ratio: float = 0.8,
    seed: int = 0,
    n_splits: int = 3,
    fold_idx: int = 0,
    shuffle_groups: bool = True,
) -> Dict[str, Any]:
    """Unified helper for common splitting modes.

    Supported split_mode values:
      - "random": sample-wise random split
      - "group_holdout": group-wise holdout split
      - "group_kfold": group-wise K-fold split

    Aliases (for convenience / backward-compat):
      - "subject" -> "group_holdout"
      - "kfold_subject" -> "group_kfold"

    Notes:
      - For group-based modes, pass `groups` (length N, group label per sample).
      - For random mode, pass `n_samples`.
      - If `n_samples` is None and `groups` is provided, n_samples=len(groups).
    """
    aliases = {
        "subject": "group_holdout",
        "kfold_subject": "group_kfold",
    }
    mode = aliases.get(split_mode, split_mode)
    if mode not in {"random", "group_holdout", "group_kfold"}:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    if n_samples is None and groups is not None:
        n_samples = len(groups)

    if mode == "random":
        if n_samples is None:
            raise ValueError("n_samples is required for split_mode='random'")
        return make_random_split(n_samples=n_samples, train_ratio=train_ratio, seed=seed)

    if groups is None:
        raise ValueError(f"groups is required for split_mode='{split_mode}'")

    if mode == "group_holdout":
        return make_group_holdout_split(
            groups=groups,
            train_ratio=train_ratio,
            seed=seed,
            shuffle=shuffle_groups,
        )

    # mode == "group_kfold"
    return make_group_kfold_split(
        groups=groups,
        n_splits=n_splits,
        fold_idx=fold_idx,
        seed=seed,
        shuffle=shuffle_groups,
    )


def make_loaders_from_split(
    dataset: "torch.utils.data.Dataset",
    split: Dict[str, Any],
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    shuffle_train: bool = True,
) -> Tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """Build train/test DataLoaders from a split dict with train_idx/test_idx.

    Kept here so different datasets can share the exact same loader construction.
    """
    # Local imports to avoid importing torch when only doing index splits.
    import torch
    from torch.utils.data import DataLoader, Subset

    dl_train = DataLoader(
        Subset(dataset, split["train_idx"]),
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    dl_test = DataLoader(
        Subset(dataset, split["test_idx"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dl_train, dl_test


def _unique_preserve_order(items: Sequence[Hashable]) -> List[Hashable]:
    # dict preserves insertion order in Python 3.7+
    return list(dict.fromkeys(items))


def make_kfold_folds(
    items: Sequence[Hashable],
    n_splits: int,
    seed: int = 0,
    shuffle: bool = True,
) -> List[List[Hashable]]:
    """Split *unique* items into `n_splits` folds.

    Args:
        items: Sequence of (possibly repeated) items. Duplicates are removed while
            preserving first-seen order.
        n_splits: Number of folds.
        seed: RNG seed used when `shuffle=True`.
        shuffle: Whether to shuffle unique items before splitting.

    Returns:
        folds: List of length `n_splits`, each element is a list of items.

    Raises:
        ValueError: if n_splits < 2 or if unique item count < n_splits.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    uniq = _unique_preserve_order(items)
    if len(uniq) < n_splits:
        raise ValueError(
            f"Need at least n_splits={n_splits} unique items, got {len(uniq)}"
        )

    uniq_arr = np.array(uniq, dtype=object)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq_arr)

    parts = np.array_split(uniq_arr, n_splits)
    return [p.tolist() for p in parts]


def kfold_train_test_from_folds(
    folds: Sequence[Sequence[Hashable]],
    fold_idx: int,
) -> Tuple[List[Hashable], List[Hashable]]:
    """Return (train_items, test_items) for a given `fold_idx`."""
    n_splits = len(folds)
    if n_splits < 2:
        raise ValueError("folds must contain at least 2 folds")
    if not (0 <= fold_idx < n_splits):
        raise ValueError(f"fold_idx must be in [0, {n_splits - 1}], got {fold_idx}")

    test_items = list(folds[fold_idx])
    train_items: List[Hashable] = []
    for i in range(n_splits):
        if i == fold_idx:
            continue
        train_items.extend(list(folds[i]))
    return train_items, test_items


def make_group_kfold_split(
    groups: Sequence[Hashable],
    n_splits: int,
    fold_idx: int,
    seed: int = 0,
    shuffle: bool = True,
) -> Dict[str, Any]:
    """Group-wise K-fold split for sample indices.

    This is a lightweight alternative to sklearn's GroupKFold.

    Args:
        groups: Length-N sequence of group labels (e.g., subject IDs) per sample.
        n_splits: Number of folds.
        fold_idx: Which fold is used as the test fold.
        seed: RNG seed used when `shuffle=True`.
        shuffle: Whether to shuffle unique groups before splitting.

    Returns:
        A dict with keys:
            - train_idx, test_idx: List[int]
            - train_groups, test_groups: List[Hashable]
            - fold_idx, n_splits: int

    Notes:
        - All samples from the same group will be assigned to the same fold.
        - Fold sizes are balanced by *number of groups*, not number of samples.
    """
    folds = make_kfold_folds(groups, n_splits=n_splits, seed=seed, shuffle=shuffle)
    train_groups, test_groups = kfold_train_test_from_folds(folds, fold_idx=fold_idx)

    test_set = set(test_groups)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for i, g in enumerate(groups):
        if g in test_set:
            test_idx.append(i)
        else:
            train_idx.append(i)

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_groups": train_groups,
        "test_groups": test_groups,
        "fold_idx": int(fold_idx),
        "n_splits": int(n_splits),
    }

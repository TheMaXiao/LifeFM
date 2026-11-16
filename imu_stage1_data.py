from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import torch


MOTIONSENSE_ACTIVITY_MAP = {
    "dws": "downstairs",
    "ups": "upstairs",
    "wlk": "walking",
    "jog": "jogging",
    "sit": "sitting",
    "std": "standing",
}

WISDM_ACTIVITY_CODE_TO_NAME = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "teeth",
    "H": "soup",
    "I": "chips",
    "J": "pasta",
    "K": "drinking",
    "L": "sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding",
}

WISDM_GROUP6_ORDER = [
    "locomotion",
    "stairs",
    "static",
    "eat_drink",
    "sports",
    "upper_body",
]

WISDM_18_TO_GROUP6 = {
    "walking": "locomotion",
    "jogging": "locomotion",
    "stairs": "stairs",
    "sitting": "static",
    "standing": "static",
    "soup": "eat_drink",
    "chips": "eat_drink",
    "pasta": "eat_drink",
    "drinking": "eat_drink",
    "sandwich": "eat_drink",
    "kicking": "sports",
    "catch": "sports",
    "dribbling": "sports",
    "typing": "upper_body",
    "teeth": "upper_body",
    "writing": "upper_body",
    "clapping": "upper_body",
    "folding": "upper_body",
}

HHAR_ACTIVITY_NAMES = {
    "bike": "biking",
    "sit": "sitting",
    "stand": "standing",
    "walk": "walking",
    "stairsup": "upstairs",
    "stairsdown": "downstairs",
}

REALWORLDHAR_ACTIVITY_NAMES = {
    "walking": "walking",
    "running": "running",
    "sitting": "sitting",
    "standing": "standing",
    "lying": "lying",
    "climbingup": "upstairs",
    "climbingdown": "downstairs",
    "jumping": "jumping",
}

PAMAP2_CORE12_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
PAMAP2_CORE12_NAMES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    24: "rope_jumping",
}
PAMAP2_IMU6_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
PAMAP2_PLACEMENT_BLOCK_STARTS = {
    "hand": 3,
    "chest": 20,
    "ankle": 37,
}
PAMAP2_IMU6_RELATIVE_COLS = [4, 5, 6, 7, 8, 9]

MHEALTH_ACTIVITY_NAMES = {
    1: "standing_still",
    2: "sitting_relaxing",
    3: "lying_down",
    4: "walking",
    5: "climbing_stairs",
    6: "waist_bends_forward",
    7: "frontal_arm_elevation",
    8: "knees_bending",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jump_front_back",
}

OPPORTUNITY_LOCOMOTION_LABELS = {
    1: "stand",
    2: "walk",
    4: "sit",
    5: "lie",
}

OPPORTUNITY_GESTURE_LABELS = {
    404505: "close_dishwasher",
    404508: "close_drawer_3",
    404511: "close_drawer_2",
    404516: "close_door_1",
    404517: "close_door_2",
    404519: "close_drawer_1",
    404520: "close_fridge",
    405506: "toggle_switch",
    406505: "open_dishwasher",
    406508: "open_drawer_3",
    406511: "open_drawer_2",
    406516: "open_door_1",
    406517: "open_door_2",
    406519: "open_drawer_1",
    406520: "open_fridge",
    407521: "drink_from_cup",
    408512: "clean_table",
}


@dataclass
class IMUBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    label_names: list[str]
    sampling_rate_hz: int
    target_len: int
    channels: list[str]

    @property
    def n_classes(self) -> int:
        return int(len(self.label_names))

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])


def _cache_root(root: str | Path | None = None) -> Path:
    if root is None:
        root = Path("/home/xiaoma/foundation_model/pulseppg/kd_sweep_results/imu_preliminary/cache")
    return Path(root)


def _npz_to_bundle(path: Path, name: str) -> IMUBundle:
    payload = np.load(path, allow_pickle=True)
    return IMUBundle(
        name=name,
        X=payload["X"].astype(np.float32),
        y=np.asarray(payload["y"]),
        groups=payload["groups"].astype(np.int64),
        label_names=[str(x) for x in payload["label_names"].tolist()],
        sampling_rate_hz=int(payload["sampling_rate_hz"]),
        target_len=int(payload["target_len"]),
        channels=[str(x) for x in payload["channels"].tolist()],
    )


def _save_bundle(path: Path, bundle: IMUBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=bundle.X.astype(np.float32),
        y=np.asarray(bundle.y),
        groups=bundle.groups.astype(np.int64),
        label_names=np.asarray(bundle.label_names, dtype=object),
        sampling_rate_hz=np.asarray(bundle.sampling_rate_hz, dtype=np.int64),
        target_len=np.asarray(bundle.target_len, dtype=np.int64),
        channels=np.asarray(bundle.channels, dtype=object),
    )


def _rename_bundle(bundle: IMUBundle, name: str) -> IMUBundle:
    return IMUBundle(
        name=str(name),
        X=bundle.X,
        y=bundle.y,
        groups=bundle.groups,
        label_names=list(bundle.label_names),
        sampling_rate_hz=int(bundle.sampling_rate_hz),
        target_len=int(bundle.target_len),
        channels=list(bundle.channels),
    )


def _canonical_label_name(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return text.strip("_")


def _load_activity_label_map(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                activity_id = int(parts[0])
            except Exception:
                continue
            mapping[int(activity_id)] = _canonical_label_name(" ".join(parts[1:]))
    return mapping


def _segment_to_windows(
    segment: np.ndarray,
    *,
    win_len: int,
    stride: int,
    target_len: int,
    allow_short_segment: bool = False,
) -> np.ndarray:
    if segment.ndim != 2:
        raise ValueError(f"Expected segment with shape [time, channels], got {segment.shape}")
    n_steps = int(segment.shape[0])
    n_channels = int(segment.shape[1])
    if n_steps <= 0:
        return np.empty((0, n_channels, int(target_len)), dtype=np.float32)
    if n_steps < int(win_len):
        if not bool(allow_short_segment):
            return np.empty((0, n_channels, int(target_len)), dtype=np.float32)
        return _resample_window(segment.T, int(target_len))[None, ...].astype(np.float32)
    return _window_sequence(segment.T, win_len=int(win_len), stride=int(stride), target_len=int(target_len))


def _resample_windows_batch(windows: np.ndarray, target_len: int) -> np.ndarray:
    if int(windows.shape[-1]) == int(target_len):
        return windows.astype(np.float32, copy=False)
    if int(windows.shape[0]) == 0:
        return windows.astype(np.float32, copy=False)
    return np.stack([_resample_window(window, int(target_len)) for window in windows], axis=0).astype(np.float32)


def _zscore_per_window(windows: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = windows.mean(axis=-1, keepdims=True)
    std = windows.std(axis=-1, keepdims=True)
    return ((windows - mean) / (std + float(eps))).astype(np.float32)


def _resample_window(window: np.ndarray, target_len: int) -> np.ndarray:
    if int(window.shape[-1]) == int(target_len):
        return window.astype(np.float32, copy=False)
    src_x = np.linspace(0.0, 1.0, num=int(window.shape[-1]), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=int(target_len), dtype=np.float32)
    out = np.stack([np.interp(dst_x, src_x, ch).astype(np.float32) for ch in window], axis=0)
    return out


def _hr_from_ecg_window(
    ecg_window: np.ndarray,
    *,
    fs: float,
    hr_min: float = 30.0,
    hr_max: float = 220.0,
    prominence: float = 0.5,
) -> float:
    window = np.asarray(ecg_window, dtype=np.float32)
    if window.ndim == 1:
        window = window[:, None]
    if window.ndim != 2 or int(window.shape[0]) < 3:
        return float("nan")

    min_distance = max(1, int(round(0.3 * float(fs))))
    hr_estimates: list[float] = []
    for lead_idx in range(int(window.shape[1])):
        lead = window[:, lead_idx]
        if not np.isfinite(lead).all():
            continue
        lead = lead - float(np.mean(lead))
        lead_std = float(np.std(lead))
        if lead_std < 1e-8:
            continue
        lead = lead / (lead_std + 1e-8)

        best_valid_hrs: np.ndarray | None = None
        best_peak_count = -1
        for signed in (lead, -lead):
            peaks, _ = find_peaks(signed, distance=min_distance, prominence=float(prominence))
            if int(len(peaks)) < 2:
                continue
            rr = np.diff(peaks).astype(np.float32) / float(fs)
            hrs = 60.0 / np.clip(rr, 1e-8, None)
            valid_hrs = hrs[(hrs >= float(hr_min)) & (hrs <= float(hr_max))]
            if int(len(valid_hrs)) > best_peak_count:
                best_peak_count = int(len(valid_hrs))
                best_valid_hrs = valid_hrs
        if best_valid_hrs is not None and int(len(best_valid_hrs)) > 0:
            hr_estimates.append(float(np.median(best_valid_hrs)))

    if not hr_estimates:
        return float("nan")
    return float(np.median(np.asarray(hr_estimates, dtype=np.float32)))


def _window_start_indices(total_len: int, *, win_len: int, stride: int) -> list[int]:
    if int(total_len) < int(win_len):
        return []
    return list(range(0, int(total_len) - int(win_len) + 1, int(stride)))


def _window_sequence(
    x: np.ndarray,
    *,
    win_len: int,
    stride: int,
    target_len: int,
) -> np.ndarray:
    total_len = int(x.shape[-1])
    starts = _window_start_indices(total_len, win_len=int(win_len), stride=int(stride))
    if not starts:
        return np.empty((0, int(x.shape[0]), int(target_len)), dtype=np.float32)
    windows = [_resample_window(x[:, s : s + int(win_len)], int(target_len)) for s in starts]
    if not windows:
        return np.empty((0, int(x.shape[0]), int(target_len)), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)


def build_motionsense_bundle(
    *,
    cache_dir: str | Path | None = None,
    zip_path: str | Path = "/home/xiaoma/foundation_model/imu_stage1/data/raw/motionsense/motion-sense-master/data/A_DeviceMotion_data.zip",
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    source_fs: int = 50,
    target_len: int = 500,
    normalize: bool = True,
) -> IMUBundle:
    cache_path = _cache_root(cache_dir) / (
        f"motionsense_phone_imu6_win{int(round(win_sec * 10))}_"
        f"stride{int(round(stride_sec * 10))}_len{int(target_len)}.npz"
    )
    if cache_path.exists():
        return _npz_to_bundle(cache_path, "motionsense")

    win_len = int(round(float(win_sec) * int(source_fs)))
    stride = int(round(float(stride_sec) * int(source_fs)))
    cols = [
        "userAcceleration.x",
        "userAcceleration.y",
        "userAcceleration.z",
        "rotationRate.x",
        "rotationRate.y",
        "rotationRate.z",
    ]
    channels = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    label_names = sorted(set(MOTIONSENSE_ACTIVITY_MAP.values()))
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = sorted(
            name
            for name in zf.namelist()
            if name.startswith("A_DeviceMotion_data/")
            and name.endswith(".csv")
            and "__MACOSX" not in name
        )
        for name in names:
            parts = Path(name).parts
            if len(parts) < 3:
                continue
            folder = parts[1]
            stem = Path(parts[2]).stem
            activity_key = folder.split("_")[0]
            activity_name = MOTIONSENSE_ACTIVITY_MAP.get(activity_key)
            if activity_name is None:
                continue
            try:
                subject_id = int(stem.split("_")[1])
            except Exception:
                continue
            with zf.open(name, "r") as fp:
                df = pd.read_csv(io.BytesIO(fp.read()))
            missing = [c for c in cols if c not in df.columns]
            if missing:
                continue
            arr = df[cols].to_numpy(dtype=np.float32).T
            windows = _window_sequence(arr, win_len=win_len, stride=stride, target_len=int(target_len))
            if windows.shape[0] == 0:
                continue
            x_parts.append(windows)
            y_parts.append(np.full(windows.shape[0], label_to_idx[activity_name], dtype=np.int64))
            g_parts.append(np.full(windows.shape[0], int(subject_id), dtype=np.int64))

    if not x_parts:
        raise RuntimeError("No MotionSense windows were built.")

    X = np.concatenate(x_parts, axis=0)
    if normalize:
        X = _zscore_per_window(X)
    bundle = IMUBundle(
        name="motionsense",
        X=X.astype(np.float32),
        y=np.concatenate(y_parts, axis=0),
        groups=np.concatenate(g_parts, axis=0),
        label_names=label_names,
        sampling_rate_hz=int(source_fs),
        target_len=int(target_len),
        channels=channels,
    )
    _save_bundle(cache_path, bundle)
    return bundle


def _read_wisdm_sensor_file(path: Path) -> pd.DataFrame:
    cols = ["subject", "activity", "timestamp", "x", "y", "z"]
    df = pd.read_csv(
        path,
        header=None,
        names=cols,
        sep=",",
        engine="python",
    )
    df["z"] = df["z"].astype(str).str.rstrip(";")
    for col in ("subject", "timestamp", "x", "y", "z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["activity"] = df["activity"].astype(str).str.strip()
    df = df.dropna()
    df["subject"] = df["subject"].astype(np.int64)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    return df


def _merge_wisdm_phone_subject(acc_path: Path, gyro_path: Path, *, tick_ns: int) -> pd.DataFrame:
    acc = _read_wisdm_sensor_file(acc_path)
    gyro = _read_wisdm_sensor_file(gyro_path)

    for df in (acc, gyro):
        df["tbin"] = np.rint(df["timestamp"].to_numpy(dtype=np.float64) / float(tick_ns)).astype(np.int64)

    acc = (
        acc.groupby(["subject", "activity", "tbin"], as_index=False)[["x", "y", "z"]]
        .mean()
        .rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})
    )
    gyro = (
        gyro.groupby(["subject", "activity", "tbin"], as_index=False)[["x", "y", "z"]]
        .mean()
        .rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})
    )
    merged = acc.merge(gyro, how="inner", on=["subject", "activity", "tbin"])
    merged = merged.sort_values(["activity", "tbin"]).reset_index(drop=True)
    return merged


def build_wisdm_phone_bundle(
    *,
    cache_dir: str | Path | None = None,
    root_dir: str | Path = "/home/xiaoma/foundation_model/imu_stage1/data/raw/wisdm/wisdm-dataset/wisdm-dataset",
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    source_fs: int = 20,
    target_len: int = 500,
    normalize: bool = True,
) -> IMUBundle:
    cache_path = _cache_root(cache_dir) / (
        f"wisdm_phone_imu6_win{int(round(win_sec * 10))}_"
        f"stride{int(round(stride_sec * 10))}_len{int(target_len)}.npz"
    )
    if cache_path.exists():
        return _npz_to_bundle(cache_path, "wisdm")

    win_len = int(round(float(win_sec) * int(source_fs)))
    stride = int(round(float(stride_sec) * int(source_fs)))
    tick_ns = int(round(1e9 / int(source_fs)))
    channels = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    label_names = [WISDM_ACTIVITY_CODE_TO_NAME[k] for k in sorted(WISDM_ACTIVITY_CODE_TO_NAME.keys())]
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}

    raw_root = Path(root_dir) / "raw" / "phone"
    accel_dir = raw_root / "accel"
    gyro_dir = raw_root / "gyro"
    accel_files = sorted(p for p in accel_dir.glob("data_*_accel_phone.txt") if p.is_file())

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []

    for acc_path in accel_files:
        subject_id = acc_path.stem.split("_")[1]
        gyro_path = gyro_dir / f"data_{subject_id}_gyro_phone.txt"
        if not gyro_path.exists():
            continue

        merged = _merge_wisdm_phone_subject(acc_path, gyro_path, tick_ns=tick_ns)
        if merged.empty:
            continue
        subject = int(merged["subject"].iloc[0])

        for activity_code, act_df in merged.groupby("activity", sort=False):
            activity_name = WISDM_ACTIVITY_CODE_TO_NAME.get(str(activity_code))
            if activity_name is None:
                continue
            act_df = act_df.sort_values("tbin").reset_index(drop=True)
            times = act_df["tbin"].to_numpy(dtype=np.int64)
            values = act_df[channels].to_numpy(dtype=np.float32)
            if len(values) < win_len:
                continue
            split_points = np.where(np.diff(times) > 1)[0] + 1
            segment_edges = np.concatenate(([0], split_points, [len(values)]))
            for seg_lo, seg_hi in zip(segment_edges[:-1], segment_edges[1:]):
                seg = values[seg_lo:seg_hi]
                if len(seg) < win_len:
                    continue
                windows = _window_sequence(seg.T, win_len=win_len, stride=stride, target_len=int(target_len))
                if windows.shape[0] == 0:
                    continue
                x_parts.append(windows)
                y_parts.append(np.full(windows.shape[0], label_to_idx[activity_name], dtype=np.int64))
                g_parts.append(np.full(windows.shape[0], subject, dtype=np.int64))

    if not x_parts:
        raise RuntimeError("No WISDM phone windows were built.")

    X = np.concatenate(x_parts, axis=0)
    if normalize:
        X = _zscore_per_window(X)
    bundle = IMUBundle(
        name="wisdm",
        X=X.astype(np.float32),
        y=np.concatenate(y_parts, axis=0),
        groups=np.concatenate(g_parts, axis=0),
        label_names=label_names,
        sampling_rate_hz=int(source_fs),
        target_len=int(target_len),
        channels=channels,
    )
    _save_bundle(cache_path, bundle)
    return bundle


def build_wisdm_phone_group6_bundle(
    *,
    cache_dir: str | Path | None = None,
    root_dir: str | Path = "/home/xiaoma/foundation_model/imu_stage1/data/raw/wisdm/wisdm-dataset/wisdm-dataset",
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    source_fs: int = 20,
    target_len: int = 500,
    normalize: bool = True,
) -> IMUBundle:
    base = build_wisdm_phone_bundle(
        cache_dir=cache_dir,
        root_dir=root_dir,
        win_sec=float(win_sec),
        stride_sec=float(stride_sec),
        source_fs=int(source_fs),
        target_len=int(target_len),
        normalize=bool(normalize),
    )
    group_to_idx = {name: idx for idx, name in enumerate(WISDM_GROUP6_ORDER)}
    y_group = np.empty_like(base.y)
    for old_idx, label_name in enumerate(base.label_names):
        group_name = WISDM_18_TO_GROUP6[str(label_name)]
        y_group[base.y == old_idx] = group_to_idx[group_name]
    return IMUBundle(
        name="wisdm_group6",
        X=base.X,
        y=y_group.astype(np.int64),
        groups=base.groups,
        label_names=list(WISDM_GROUP6_ORDER),
        sampling_rate_hz=int(base.sampling_rate_hz),
        target_len=int(base.target_len),
        channels=list(base.channels),
    )


def _mat_scalar_str(value: Any, default: str = "") -> str:
    if value is None:
        return str(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return str(default)
    return str(arr.reshape(-1)[0]).strip()


def _mat_scalar_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return int(default)
    item = arr.reshape(-1)[0]
    try:
        return int(item)
    except Exception:
        try:
            return int(float(item))
        except Exception:
            return int(default)


def build_usc_had_bundle(
    *,
    cache_dir: str | Path | None = None,
    root_dir: str | Path = "/home/xiaoma/foundation_model/imu_stage1/data/raw/uschad/USC-HAD",
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    source_fs: int = 100,
    target_len: int = 500,
    normalize: bool = True,
) -> IMUBundle:
    cache_path = _cache_root(cache_dir) / (
        f"usc_had_imu6_win{int(round(win_sec * 10))}_"
        f"stride{int(round(stride_sec * 10))}_len{int(target_len)}.npz"
    )
    if cache_path.exists():
        return _npz_to_bundle(cache_path, "usc_had")

    from scipy.io import loadmat

    win_len = int(round(float(win_sec) * int(source_fs)))
    stride = int(round(float(stride_sec) * int(source_fs)))
    root = Path(root_dir)
    files = sorted(root.glob("Subject*/a*t*.mat"))
    if not files:
        raise RuntimeError(f"No USC-HAD MAT files found under {root}.")

    entries: list[tuple[int, int, str, np.ndarray]] = []
    activity_names: dict[int, str] = {}
    for path in files:
        payload = loadmat(path)
        readings = payload.get("sensor_readings")
        if readings is None:
            continue
        readings = np.asarray(readings, dtype=np.float32)
        if readings.ndim != 2 or readings.shape[1] < 6:
            continue
        subject = int(path.parent.name.replace("Subject", ""))
        activity_id = _mat_scalar_int(payload.get("activity_number"), default=0)
        if activity_id <= 0:
            try:
                activity_id = int(path.stem.split("t")[0].replace("a", ""))
            except Exception:
                continue
        activity_name = _mat_scalar_str(payload.get("activity"), default=f"activity_{activity_id}")
        activity_names[int(activity_id)] = str(activity_name)
        entries.append((subject, int(activity_id), str(activity_name), readings[:, :6]))

    if not entries:
        raise RuntimeError("No valid USC-HAD sequences were loaded.")

    ordered_ids = sorted(activity_names)
    label_names = [activity_names[idx] for idx in ordered_ids]
    label_to_idx = {idx: i for i, idx in enumerate(ordered_ids)}

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    for subject, activity_id, _, readings in entries:
        windows = _window_sequence(readings.T, win_len=win_len, stride=stride, target_len=int(target_len))
        if windows.shape[0] == 0:
            continue
        x_parts.append(windows)
        y_parts.append(np.full(windows.shape[0], label_to_idx[int(activity_id)], dtype=np.int64))
        g_parts.append(np.full(windows.shape[0], int(subject), dtype=np.int64))

    if not x_parts:
        raise RuntimeError("No USC-HAD windows were built.")

    X = np.concatenate(x_parts, axis=0)
    if normalize:
        X = _zscore_per_window(X)
    bundle = IMUBundle(
        name="usc_had",
        X=X.astype(np.float32),
        y=np.concatenate(y_parts, axis=0),
        groups=np.concatenate(g_parts, axis=0),
        label_names=label_names,
        sampling_rate_hz=int(source_fs),
        target_len=int(target_len),
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    )
    _save_bundle(cache_path, bundle)
    return bundle


def build_pamap2_bundle(
    *,
    cache_dir: str | Path | None = None,
    root_dir: str | Path = "/home/xiaoma/foundation_model/imu_stage1/data/raw/pamap2/PAMAP2_Dataset/PAMAP2_Dataset",
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    source_fs: int = 100,
    target_len: int = 500,
    normalize: bool = True,
    placement: str = "chest",
    label_mode: str = "activity12",
    min_hr_samples_per_window: int = 5,
) -> IMUBundle:
    placement_key = str(placement).lower()
    block_start = PAMAP2_PLACEMENT_BLOCK_STARTS.get(placement_key)
    if block_start is None:
        raise ValueError(
            f"Unsupported PAMAP2 placement '{placement}'. Expected one of {sorted(PAMAP2_PLACEMENT_BLOCK_STARTS)}"
        )
    label_mode_key = str(label_mode).lower()
    if label_mode_key not in {"activity12", "hr"}:
        raise ValueError(f"Unsupported PAMAP2 label_mode '{label_mode}'. Expected one of ['activity12', 'hr']")
    cache_stem = f"pamap2_{placement_key}_imu6_win{int(round(win_sec * 10))}_stride{int(round(stride_sec * 10))}_len{int(target_len)}"
    if label_mode_key == "hr":
        cache_stem += f"_hrmin{int(min_hr_samples_per_window)}"
    cache_path = _cache_root(cache_dir) / (
        f"{cache_stem}.npz"
    )
    if cache_path.exists():
        if label_mode_key == "activity12":
            bundle_name = "pamap2" if placement_key == "chest" else f"pamap2_{placement_key}"
        else:
            bundle_name = "pamap2_hr" if placement_key == "chest" else f"pamap2_{placement_key}_hr"
        return _npz_to_bundle(cache_path, bundle_name)

    win_len = int(round(float(win_sec) * int(source_fs)))
    stride = int(round(float(stride_sec) * int(source_fs)))
    protocol_dir = Path(root_dir) / "Protocol"
    files = sorted(protocol_dir.glob("subject*.dat"))
    if not files:
        raise RuntimeError(f"No PAMAP2 protocol files found under {protocol_dir}.")

    label_to_idx = {activity_id: idx for idx, activity_id in enumerate(PAMAP2_CORE12_IDS)}
    label_names = [f"activity_{activity_id}" for activity_id in PAMAP2_CORE12_IDS]
    usecols = [0, 1, *[block_start + offset for offset in PAMAP2_IMU6_RELATIVE_COLS]]
    col_names = ["timestamp", "activity_id", *PAMAP2_IMU6_CHANNELS]
    if label_mode_key == "hr":
        usecols = [0, 1, 2, *[block_start + offset for offset in PAMAP2_IMU6_RELATIVE_COLS]]
        col_names = ["timestamp", "activity_id", "heart_rate", *PAMAP2_IMU6_CHANNELS]
    required_cols = ["timestamp", "activity_id", *PAMAP2_IMU6_CHANNELS]

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    for path in files:
        subject = int(path.stem.replace("subject", ""))
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=col_names,
            usecols=usecols,
            na_values=["NaN", "nan"],
            engine="python",
        )
        df = df.dropna(subset=required_cols)
        if df.empty:
            continue
        df["activity_id"] = df["activity_id"].astype(np.int64)
        df = df[df["activity_id"].isin(label_to_idx)]
        if df.empty:
            continue

        for activity_id, act_df in df.groupby("activity_id", sort=False):
            times = act_df["timestamp"].to_numpy(dtype=np.float64)
            values = act_df[PAMAP2_IMU6_CHANNELS].to_numpy(dtype=np.float32)
            if len(values) < win_len:
                continue
            hr_values = None
            if label_mode_key == "hr":
                hr_values = act_df["heart_rate"].to_numpy(dtype=np.float32)
            split_points = np.where(np.diff(times) > (1.5 / float(source_fs)))[0] + 1
            segment_edges = np.concatenate(([0], split_points, [len(values)]))
            for seg_lo, seg_hi in zip(segment_edges[:-1], segment_edges[1:]):
                seg = values[seg_lo:seg_hi]
                if len(seg) < win_len:
                    continue
                starts = _window_start_indices(len(seg), win_len=win_len, stride=stride)
                if not starts:
                    continue
                if label_mode_key == "activity12":
                    windows = _window_sequence(seg.T, win_len=win_len, stride=stride, target_len=int(target_len))
                    if windows.shape[0] == 0:
                        continue
                    x_parts.append(windows)
                    y_parts.append(np.full(windows.shape[0], label_to_idx[int(activity_id)], dtype=np.int64))
                    g_parts.append(np.full(windows.shape[0], int(subject), dtype=np.int64))
                    continue

                assert hr_values is not None
                seg_hr = hr_values[seg_lo:seg_hi]
                hr_windows: list[np.ndarray] = []
                hr_targets: list[float] = []
                for start in starts:
                    hr_window = seg_hr[start : start + win_len]
                    n_hr = int(np.isfinite(hr_window).sum())
                    if n_hr < int(min_hr_samples_per_window):
                        continue
                    hr_targets.append(float(np.nanmean(hr_window.astype(np.float32))))
                    hr_windows.append(_resample_window(seg[start : start + win_len].T, int(target_len)))
                if not hr_windows:
                    continue
                x_parts.append(np.stack(hr_windows, axis=0).astype(np.float32))
                y_parts.append(np.asarray(hr_targets, dtype=np.float32))
                g_parts.append(np.full(len(hr_targets), int(subject), dtype=np.int64))

    if not x_parts:
        raise RuntimeError("No PAMAP2 windows were built.")

    X = np.concatenate(x_parts, axis=0)
    if normalize:
        X = _zscore_per_window(X)
    if label_mode_key == "activity12":
        bundle_name = "pamap2" if placement_key == "chest" else f"pamap2_{placement_key}"
        bundle_label_names = label_names
    else:
        bundle_name = "pamap2_hr" if placement_key == "chest" else f"pamap2_{placement_key}_hr"
        bundle_label_names = ["heart_rate_bpm"]
    bundle = IMUBundle(
        name=bundle_name,
        X=X.astype(np.float32),
        y=np.concatenate(y_parts, axis=0),
        groups=np.concatenate(g_parts, axis=0),
        label_names=bundle_label_names,
        sampling_rate_hz=int(source_fs),
        target_len=int(target_len),
        channels=list(PAMAP2_IMU6_CHANNELS),
    )
    _save_bundle(cache_path, bundle)
    return bundle


def build_ucihar_bundle(
    *,
    cache_dir: str | Path | None = None,
    root_dir: str | Path = "/home/xiaoma/foundation_model/imu_stage1/data/raw/ucihar/UCI HAR Dataset/UCI HAR Dataset",
    source_fs: int = 50,
    target_len: int = 500,
    normalize: bool = True,
) -> IMUBundle:
    cache_path = _cache_root(cache_dir) / f"ucihar_phone_imu6_fixed128_len{int(target_len)}.npz"
    if cache_path.exists():
        return _npz_to_bundle(cache_path, "ucihar")

    root = Path(root_dir)
    label_map = _load_activity_label_map(root / "activity_labels.txt")
    if not label_map:
        raise RuntimeError(f"Failed to load UCI HAR activity labels from {root / 'activity_labels.txt'}")

    signal_bases = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
    ]
    label_ids = sorted(label_map)
    label_to_idx = {activity_id: idx for idx, activity_id in enumerate(label_ids)}

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    for split in ("train", "test"):
        split_root = root / split
        sig_root = split_root / "Inertial Signals"
        channels = [np.atleast_2d(np.loadtxt(sig_root / f"{base}_{split}.txt", dtype=np.float32)) for base in signal_bases]
        X_split = np.stack(channels, axis=1).astype(np.float32)
        X_split = _resample_windows_batch(X_split, int(target_len))
        y_raw = np.asarray(np.loadtxt(split_root / f"y_{split}.txt", dtype=np.int64)).reshape(-1)
        groups = np.asarray(np.loadtxt(split_root / f"subject_{split}.txt", dtype=np.int64)).reshape(-1)
        if int(X_split.shape[0]) != int(len(y_raw)) or int(len(y_raw)) != int(len(groups)):
            raise RuntimeError(f"UCI HAR split {split} has inconsistent sample counts")
        y_parts.append(np.asarray([label_to_idx[int(label)] for label in y_raw], dtype=np.int64))
        x_parts.append(X_split)
        g_parts.append(groups.astype(np.int64))

    X = np.concatenate(x_parts, axis=0)
    if normalize:
        X = _zscore_per_window(X)
    bundle = IMUBundle(
        name="ucihar",
        X=X.astype(np.float32),
        y=np.concatenate(y_parts, axis=0),
        groups=np.concatenate(g_parts, axis=0),
        label_names=[label_map[idx] for idx in label_ids],
        sampling_rate_hz=int(source_fs),
        target_len=int(target_len),
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    )
    _save_bundle(cache_path, bundle)
    return bundle


def build_hhar_bundle(
    *,
    cache_dir: str | Path | None = None,
    root_dir: str | Path = "/home/xiaoma/foundation_model/imu_stage1/data/raw/hhar/Activity recognition exp/Activity recognition exp",
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    source_fs: int = 200,
    target_len: int = 500,
    normalize: bool = True,
    merge_tolerance_ms: int = 20,
) -> IMUBundle:
    cache_path = _cache_root(cache_dir) / (
        f"hhar_phone_imu6_win{int(round(win_sec * 10))}_"
        f"stride{int(round(stride_sec * 10))}_len{int(target_len)}.npz"
    )
    if cache_path.exists():
        return _npz_to_bundle(cache_path, "hhar")

    win_len = int(round(float(win_sec) * int(source_fs)))
    stride = int(round(float(stride_sec) * int(source_fs)))
    gap_ms = max(250, int(round((1000.0 / float(source_fs)) * 50.0)))
    root = Path(root_dir)
    acc_path = root / "Phones_accelerometer.csv"
    gyro_path = root / "Phones_gyroscope.csv"
    if not acc_path.exists() or not gyro_path.exists():
        raise RuntimeError(f"HHAR phone CSVs not found under {root}.")

    usecols = ["Arrival_Time", "x", "y", "z", "User", "Device", "gt"]
    dtypes = {
        "Arrival_Time": np.int64,
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
        "User": "category",
        "Device": "category",
        "gt": "category",
    }
    acc = pd.read_csv(acc_path, usecols=usecols, dtype=dtypes).dropna(subset=usecols)
    gyro = pd.read_csv(gyro_path, usecols=usecols, dtype=dtypes).dropna(subset=usecols)
    for df in (acc, gyro):
        df["User"] = df["User"].astype(str)
        df["Device"] = df["Device"].astype(str)
        df["gt"] = df["gt"].astype(str).str.strip().str.lower()
    acc = acc[acc["gt"].isin(HHAR_ACTIVITY_NAMES)]
    gyro = gyro[gyro["gt"].isin(HHAR_ACTIVITY_NAMES)]

    label_ids = list(HHAR_ACTIVITY_NAMES)
    label_to_idx = {label_id: idx for idx, label_id in enumerate(label_ids)}
    users = sorted(set(acc["User"]) & set(gyro["User"]))
    user_to_group = {user: idx for idx, user in enumerate(users)}

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    key_cols = ["User", "Device", "gt"]
    gyro_groups: dict[tuple[str, str, str], pd.DataFrame] = {}
    for key, grp in gyro.groupby(key_cols, sort=False, observed=True):
        user, device, label_id = (str(key[0]), str(key[1]), str(key[2]))
        if label_id not in label_to_idx:
            continue
        gyro_groups[(user, device, label_id)] = (
            grp.groupby("Arrival_Time", as_index=False, sort=True)[["x", "y", "z"]]
            .mean()
            .sort_values("Arrival_Time")
        )

    for key, grp in acc.groupby(key_cols, sort=False, observed=True):
        user, device, label_id = (str(key[0]), str(key[1]), str(key[2]))
        if label_id not in label_to_idx:
            continue
        gyro_df = gyro_groups.get((user, device, label_id))
        if gyro_df is None or gyro_df.empty:
            continue
        acc_df = (
            grp.groupby("Arrival_Time", as_index=False, sort=True)[["x", "y", "z"]]
            .mean()
            .sort_values("Arrival_Time")
        )
        merged = pd.merge_asof(
            acc_df,
            gyro_df,
            on="Arrival_Time",
            direction="nearest",
            tolerance=int(merge_tolerance_ms),
            suffixes=("_acc", "_gyro"),
        ).dropna()
        if merged.empty:
            continue
        times = merged["Arrival_Time"].to_numpy(dtype=np.float64)
        values = merged[["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"]].to_numpy(dtype=np.float32)
        split_points = np.where(np.diff(times) > gap_ms)[0] + 1
        segment_edges = np.concatenate(([0], split_points, [len(values)]))
        for seg_lo, seg_hi in zip(segment_edges[:-1], segment_edges[1:]):
            segment = values[seg_lo:seg_hi]
            windows = _segment_to_windows(
                segment,
                win_len=win_len,
                stride=stride,
                target_len=int(target_len),
                allow_short_segment=True,
            )
            if int(windows.shape[0]) == 0:
                continue
            x_parts.append(windows)
            y_parts.append(np.full(windows.shape[0], label_to_idx[label_id], dtype=np.int64))
            g_parts.append(np.full(windows.shape[0], user_to_group[user], dtype=np.int64))

    if not x_parts:
        raise RuntimeError("No HHAR windows were built.")

    X = np.concatenate(x_parts, axis=0)
    if normalize:
        X = _zscore_per_window(X)
    bundle = IMUBundle(
        name="hhar",
        X=X.astype(np.float32),
        y=np.concatenate(y_parts, axis=0),
        groups=np.concatenate(g_parts, axis=0),
        label_names=[HHAR_ACTIVITY_NAMES[label_id] for label_id in label_ids],
        sampling_rate_hz=int(source_fs),
        target_len=int(target_len),
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    )
    _save_bundle(cache_path, bundle)
    return bundle


def build_hapt_bundle(
    *,
    cache_dir: str | Path | None = None,
    root_dir: str | Path = "/home/xiaoma/foundation_model/imu_stage1/data/raw/hapt",
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    source_fs: int = 50,
    target_len: int = 500,
    normalize: bool = True,
) -> IMUBundle:
    cache_path = _cache_root(cache_dir) / (
        f"hapt_phone_imu6_win{int(round(win_sec * 10))}_"
        f"stride{int(round(stride_sec * 10))}_len{int(target_len)}.npz"
    )
    if cache_path.exists():
        return _npz_to_bundle(cache_path, "hapt")

    win_len = int(round(float(win_sec) * int(source_fs)))
    stride = int(round(float(stride_sec) * int(source_fs)))
    root = Path(root_dir)
    raw_root = root / "RawData"
    label_map = _load_activity_label_map(root / "activity_labels.txt")
    if not label_map:
        raise RuntimeError(f"Failed to load HAPT activity labels from {root / 'activity_labels.txt'}")
    label_ids = sorted(label_map)
    label_to_idx = {activity_id: idx for idx, activity_id in enumerate(label_ids)}

    labels = pd.read_csv(
        raw_root / "labels.txt",
        sep=r"\s+",
        header=None,
        names=["exp", "user", "activity_id", "start", "end"],
        engine="python",
    )
    if labels.empty:
        raise RuntimeError("No HAPT segments were loaded from labels.txt")

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    for (exp_id, user_id), seg_df in labels.groupby(["exp", "user"], sort=False):
        acc_path = raw_root / f"acc_exp{int(exp_id):02d}_user{int(user_id):02d}.txt"
        gyro_path = raw_root / f"gyro_exp{int(exp_id):02d}_user{int(user_id):02d}.txt"
        if not acc_path.exists() or not gyro_path.exists():
            continue
        acc = np.atleast_2d(np.loadtxt(acc_path, dtype=np.float32))
        gyro = np.atleast_2d(np.loadtxt(gyro_path, dtype=np.float32))
        n_rows = min(int(acc.shape[0]), int(gyro.shape[0]))
        if n_rows <= 0:
            continue
        seq = np.concatenate([acc[:n_rows, :3], gyro[:n_rows, :3]], axis=1).astype(np.float32)
        for row in seg_df.itertuples(index=False):
            activity_id = int(row.activity_id)
            if activity_id not in label_to_idx:
                continue
            start_idx = max(0, int(row.start))
            stop_idx = min(n_rows, int(row.end) + 1)
            if stop_idx <= start_idx:
                continue
            segment = seq[start_idx:stop_idx]
            windows = _segment_to_windows(
                segment,
                win_len=win_len,
                stride=stride,
                target_len=int(target_len),
                allow_short_segment=True,
            )
            if int(windows.shape[0]) == 0:
                continue
            x_parts.append(windows)
            y_parts.append(np.full(windows.shape[0], label_to_idx[activity_id], dtype=np.int64))
            g_parts.append(np.full(windows.shape[0], int(user_id), dtype=np.int64))

    if not x_parts:
        raise RuntimeError("No HAPT windows were built.")

    X = np.concatenate(x_parts, axis=0)
    if normalize:
        X = _zscore_per_window(X)
    bundle = IMUBundle(
        name="hapt",
        X=X.astype(np.float32),
        y=np.concatenate(y_parts, axis=0),
        groups=np.concatenate(g_parts, axis=0),
        label_names=[label_map[idx] for idx in label_ids],
        sampling_rate_hz=int(source_fs),
        target_len=int(target_len),
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    )
    _save_bundle(cache_path, bundle)
    return bundle


def _read_realworldhar_sensor_from_zip(
    zip_path: Path,
    member_name: str,
) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name, "r") as fp:
            df = pd.read_csv(io.BytesIO(fp.read()))
    required = ["attr_time", "attr_x", "attr_y", "attr_z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing RealWorldHAR columns {missing} in {zip_path}:{member_name}")
    out = df[required].copy()
    out["attr_time"] = pd.to_numeric(out["attr_time"], errors="coerce")
    out[["attr_x", "attr_y", "attr_z"]] = out[["attr_x", "attr_y", "attr_z"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    out = out.dropna().sort_values("attr_time")
    out["attr_time"] = out["attr_time"].astype(np.int64)
    return out.reset_index(drop=True)


def _find_realworldhar_layout(root: Path) -> tuple[str, dict[int, Path], Path | None]:
    candidates = [root]
    nested_root = root / "realworldhar"
    if nested_root != root:
        candidates.append(nested_root)

    for candidate in candidates:
        if (candidate / "proband1").exists():
            subject_roots = {
                int(path.name.replace("proband", "")): path
                for path in sorted(candidate.glob("proband*"))
                if path.is_dir() and path.name.replace("proband", "").isdigit()
            }
            if subject_roots:
                return "archive", subject_roots, None

    for candidate in candidates:
        probe_root = candidate / "realworldhar_probe"
        if probe_root.exists():
            return "probe", {1: probe_root}, probe_root

    for candidate in candidates:
        direct_subject_roots = {
            int(path.name[1:]): path
            for path in sorted(candidate.glob("s*"))
            if path.is_dir() and path.name[1:].isdigit()
        }
        if direct_subject_roots:
            return "direct", direct_subject_roots, None

    raise RuntimeError(
        "RealWorldHAR data not found. Expected extracted proband*/ directories under the dataset root, "
        "subject folders like s1/, or a probe folder under /home/xiaoma/foundation_model/imu_stage1/data/raw."
    )


def _iter_realworldhar_subject_activity_pairs(
    layout: str,
    subject_roots: dict[int, Path],
    probe_root: Path | None,
) -> list[tuple[int, str, Path, str, Path, str]]:
    pairs: list[tuple[int, str, Path, str, Path, str]] = []
    if layout == "archive":
        for subject_id, subject_root in sorted(subject_roots.items()):
            data_root = subject_root / "data"
            if not data_root.exists():
                continue
            acc_files = {m.group(1): path for path in data_root.glob("acc_*_csv.zip") if (m := re.match(r"acc_(.+)_csv\.zip$", path.name))}
            gyr_files = {m.group(1): path for path in data_root.glob("gyr_*_csv.zip") if (m := re.match(r"gyr_(.+)_csv\.zip$", path.name))}
            for activity in sorted(set(acc_files) & set(gyr_files)):
                pairs.append(
                    (
                        int(subject_id),
                        str(activity),
                        acc_files[activity],
                        f"acc_{activity}_waist.csv",
                        gyr_files[activity],
                        f"Gyroscope_{activity}_waist.csv",
                    )
                )
        return pairs

    if layout == "probe" and probe_root is not None:
        acc_zip = probe_root / "acc_walking_s1.zip"
        gyr_zip = probe_root / "gyr_walking_s1.zip"
        if acc_zip.exists() and gyr_zip.exists():
            pairs.append((1, "walking", acc_zip, "acc_walking_waist.csv", gyr_zip, "Gyroscope_walking_waist.csv"))
        return pairs

    for subject_id, subject_root in sorted(subject_roots.items()):
        acc_files = {m.group(1): path for path in subject_root.glob("acc_*_csv.zip") if (m := re.match(r"acc_(.+)_csv\.zip$", path.name))}
        gyr_files = {m.group(1): path for path in subject_root.glob("gyr_*_csv.zip") if (m := re.match(r"gyr_(.+)_csv\.zip$", path.name))}
        for activity in sorted(set(acc_files) & set(gyr_files)):
            pairs.append(
                (
                    int(subject_id),
                    str(activity),
                    acc_files[activity],
                    f"acc_{activity}_waist.csv",
                    gyr_files[activity],
                    f"Gyroscope_{activity}_waist.csv",
                )
            )
    return pairs


def build_realworldhar_bundle(
    *,
    cache_dir: str | Path | None = None,
    root_dir: str | Path = "/home/xiaoma/foundation_model/imu_stage1/data/raw",
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    source_fs: int = 50,
    target_len: int = 500,
    normalize: bool = True,
    merge_tolerance_ms: int = 25,
) -> IMUBundle:
    cache_path = _cache_root(cache_dir) / (
        f"realworldhar_waist_imu6_win{int(round(win_sec * 10))}_"
        f"stride{int(round(stride_sec * 10))}_len{int(target_len)}.npz"
    )
    if cache_path.exists():
        return _npz_to_bundle(cache_path, "realworldhar")

    win_len = int(round(float(win_sec) * int(source_fs)))
    stride = int(round(float(stride_sec) * int(source_fs)))
    gap_ms = max(250, int(round((1000.0 / float(source_fs)) * 50.0)))
    root = Path(root_dir)
    layout, subject_roots, probe_root = _find_realworldhar_layout(root)
    pairs = _iter_realworldhar_subject_activity_pairs(layout, subject_roots, probe_root)
    if not pairs:
        raise RuntimeError(f"No RealWorldHAR sensor archives found under {root}")

    label_ids = list(REALWORLDHAR_ACTIVITY_NAMES)
    label_to_idx = {label_id: idx for idx, label_id in enumerate(label_ids)}

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []

    for subject_id, activity, acc_zip, acc_member, gyr_zip, gyr_member in pairs:
        if activity not in label_to_idx:
            continue
        try:
            acc_df = _read_realworldhar_sensor_from_zip(acc_zip, acc_member)
            gyr_df = _read_realworldhar_sensor_from_zip(gyr_zip, gyr_member)
        except KeyError:
            continue
        if acc_df.empty or gyr_df.empty:
            continue
        acc_df = (
            acc_df.groupby("attr_time", as_index=False, sort=True)[["attr_x", "attr_y", "attr_z"]]
            .mean()
            .rename(columns={"attr_x": "acc_x", "attr_y": "acc_y", "attr_z": "acc_z"})
        )
        gyr_df = (
            gyr_df.groupby("attr_time", as_index=False, sort=True)[["attr_x", "attr_y", "attr_z"]]
            .mean()
            .rename(columns={"attr_x": "gyro_x", "attr_y": "gyro_y", "attr_z": "gyro_z"})
        )
        merged = pd.merge_asof(
            acc_df,
            gyr_df,
            on="attr_time",
            direction="nearest",
            tolerance=int(merge_tolerance_ms),
        ).dropna()
        if merged.empty:
            continue
        times = merged["attr_time"].to_numpy(dtype=np.float64)
        values = merged[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float32)
        split_points = np.where(np.diff(times) > gap_ms)[0] + 1
        segment_edges = np.concatenate(([0], split_points, [len(values)]))
        for seg_lo, seg_hi in zip(segment_edges[:-1], segment_edges[1:]):
            segment = values[seg_lo:seg_hi]
            windows = _segment_to_windows(
                segment,
                win_len=win_len,
                stride=stride,
                target_len=int(target_len),
                allow_short_segment=True,
            )
            if int(windows.shape[0]) == 0:
                continue
            x_parts.append(windows)
            y_parts.append(np.full(windows.shape[0], label_to_idx[activity], dtype=np.int64))
            g_parts.append(np.full(windows.shape[0], int(subject_id), dtype=np.int64))

    if not x_parts:
        raise RuntimeError("No RealWorldHAR windows were built.")

    X = np.concatenate(x_parts, axis=0)
    if normalize:
        X = _zscore_per_window(X)
    bundle = IMUBundle(
        name="realworldhar",
        X=X.astype(np.float32),
        y=np.concatenate(y_parts, axis=0),
        groups=np.concatenate(g_parts, axis=0),
        label_names=[REALWORLDHAR_ACTIVITY_NAMES[label_id] for label_id in label_ids],
        sampling_rate_hz=int(source_fs),
        target_len=int(target_len),
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    )
    _save_bundle(cache_path, bundle)
    return bundle


def build_mhealth_bundle(
    *,
    cache_dir: str | Path | None = None,
    root_dir: str | Path = "/home/xiaoma/foundation_model/imu_stage1/data/raw/mhealth/MHEALTHDATASET",
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    source_fs: int = 50,
    target_len: int = 500,
    normalize: bool = True,
    label_mode: str = "activity12",
) -> IMUBundle:
    label_mode_key = str(label_mode).lower()
    if label_mode_key not in {"activity12", "hr"}:
        raise ValueError(f"Unsupported MHEALTH label_mode '{label_mode}'. Expected one of ['activity12', 'hr']")

    cache_stem = (
        f"mhealth_rarm_imu6_win{int(round(win_sec * 10))}_"
        f"stride{int(round(stride_sec * 10))}_len{int(target_len)}"
    )
    if label_mode_key == "hr":
        cache_stem += "_hr"
    cache_path = _cache_root(cache_dir) / f"{cache_stem}.npz"
    if cache_path.exists():
        return _npz_to_bundle(cache_path, "mhealth" if label_mode_key == "activity12" else "mhealth_hr")

    win_len = int(round(float(win_sec) * int(source_fs)))
    stride = int(round(float(stride_sec) * int(source_fs)))
    root = Path(root_dir)
    files = sorted(root.glob("mHealth_subject*.log"))
    if not files:
        raise RuntimeError(f"No MHEALTH subject logs found under {root}.")

    label_ids = sorted(MHEALTH_ACTIVITY_NAMES)
    label_to_idx = {activity_id: idx for idx, activity_id in enumerate(label_ids)}

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    for path in files:
        try:
            subject = int(path.stem.replace("mHealth_subject", ""))
        except Exception:
            continue
        arr = np.genfromtxt(path, dtype=np.float32)
        arr = np.atleast_2d(arr)
        if int(arr.shape[1]) < 24:
            continue
        values = arr[:, 14:20].astype(np.float32)
        ecg = arr[:, 3:5].astype(np.float32)
        labels = arr[:, 23].astype(np.int64)
        valid = np.isfinite(values).all(axis=1) & np.isfinite(labels)
        if label_mode_key == "hr":
            valid &= np.isfinite(ecg).all(axis=1)
        values = values[valid]
        ecg = ecg[valid]
        labels = labels[valid]
        if int(len(values)) == 0:
            continue

        split_points = np.where(np.diff(labels) != 0)[0] + 1
        segment_edges = np.concatenate(([0], split_points, [len(values)]))
        for seg_lo, seg_hi in zip(segment_edges[:-1], segment_edges[1:]):
            label_id = int(labels[seg_lo])
            if label_id <= 0 or label_id not in label_to_idx:
                continue
            segment = values[seg_lo:seg_hi]
            if label_mode_key == "activity12":
                windows = _segment_to_windows(
                    segment,
                    win_len=win_len,
                    stride=stride,
                    target_len=int(target_len),
                    allow_short_segment=True,
                )
                if int(windows.shape[0]) == 0:
                    continue
                x_parts.append(windows)
                y_parts.append(np.full(windows.shape[0], label_to_idx[label_id], dtype=np.int64))
                g_parts.append(np.full(windows.shape[0], int(subject), dtype=np.int64))
                continue

            seg_ecg = ecg[seg_lo:seg_hi]
            starts = _window_start_indices(int(len(segment)), win_len=win_len, stride=stride)
            hr_windows: list[np.ndarray] = []
            hr_targets: list[float] = []
            if starts:
                for start in starts:
                    ecg_window = seg_ecg[start : start + win_len]
                    hr_target = _hr_from_ecg_window(ecg_window, fs=float(source_fs))
                    if not np.isfinite(hr_target):
                        continue
                    hr_windows.append(_resample_window(segment[start : start + win_len].T, int(target_len)))
                    hr_targets.append(float(hr_target))
            else:
                hr_target = _hr_from_ecg_window(seg_ecg, fs=float(source_fs))
                if np.isfinite(hr_target):
                    hr_windows.append(_resample_window(segment.T, int(target_len)))
                    hr_targets.append(float(hr_target))
            if not hr_windows:
                continue
            x_parts.append(np.stack(hr_windows, axis=0).astype(np.float32))
            y_parts.append(np.asarray(hr_targets, dtype=np.float32))
            g_parts.append(np.full(len(hr_targets), int(subject), dtype=np.int64))

    if not x_parts:
        raise RuntimeError("No MHEALTH windows were built.")

    X = np.concatenate(x_parts, axis=0)
    if normalize:
        X = _zscore_per_window(X)
    if label_mode_key == "activity12":
        bundle_name = "mhealth"
        bundle_label_names = [MHEALTH_ACTIVITY_NAMES[idx] for idx in label_ids]
    else:
        bundle_name = "mhealth_hr"
        bundle_label_names = ["heart_rate_bpm"]
    bundle = IMUBundle(
        name=bundle_name,
        X=X.astype(np.float32),
        y=np.concatenate(y_parts, axis=0),
        groups=np.concatenate(g_parts, axis=0),
        label_names=bundle_label_names,
        sampling_rate_hz=int(source_fs),
        target_len=int(target_len),
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    )
    _save_bundle(cache_path, bundle)
    return bundle


def build_opportunity_bundle(
    *,
    task: str,
    cache_dir: str | Path | None = None,
    root_dir: str | Path = "/home/xiaoma/foundation_model/imu_stage1/data/raw/opportunity/OpportunityUCIDataset/dataset",
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    source_fs: int = 30,
    target_len: int = 500,
    normalize: bool = True,
) -> IMUBundle:
    task_name = str(task)
    if task_name == "locomotion4":
        label_map = OPPORTUNITY_LOCOMOTION_LABELS
        label_col = 243
    elif task_name == "gesture17":
        label_map = OPPORTUNITY_GESTURE_LABELS
        label_col = 249
    else:
        raise KeyError(f"Unsupported Opportunity task: {task_name}")

    cache_path = _cache_root(cache_dir) / (
        f"opportunity_back_{task_name}_imu6_win{int(round(win_sec * 10))}_"
        f"stride{int(round(stride_sec * 10))}_len{int(target_len)}.npz"
    )
    if cache_path.exists():
        return _npz_to_bundle(cache_path, f"opportunity_{task_name}")

    win_len = int(round(float(win_sec) * int(source_fs)))
    stride = int(round(float(stride_sec) * int(source_fs)))
    root = Path(root_dir)
    files = sorted(root.glob("S*-*.dat"))
    if not files:
        raise RuntimeError(f"No Opportunity .dat files found under {root}.")

    ordered_label_ids = sorted(label_map)
    label_to_idx = {activity_id: idx for idx, activity_id in enumerate(ordered_label_ids)}
    usecols = [0, 37, 38, 39, 40, 41, 42, label_col]
    col_names = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "label_id"]
    gap_ms = max(2, int(round((1000.0 / float(source_fs)) * 2.5)))

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    for path in files:
        try:
            subject = int(path.stem.split("-", 1)[0].replace("S", ""))
        except Exception:
            continue
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=col_names,
            usecols=usecols,
            na_values=["NaN", "nan"],
            engine="python",
        )
        df = df.dropna(subset=col_names)
        if df.empty:
            continue
        df["label_id"] = df["label_id"].astype(np.int64)
        df = df[df["label_id"].isin(label_to_idx)]
        if df.empty:
            continue

        times = df["timestamp"].to_numpy(dtype=np.float64)
        labels = df["label_id"].to_numpy(dtype=np.int64)
        values = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float32)
        split_points = np.where((np.diff(labels) != 0) | (np.diff(times) > gap_ms))[0] + 1
        segment_edges = np.concatenate(([0], split_points, [len(values)]))
        for seg_lo, seg_hi in zip(segment_edges[:-1], segment_edges[1:]):
            label_id = int(labels[seg_lo])
            segment = values[seg_lo:seg_hi]
            windows = _segment_to_windows(
                segment,
                win_len=win_len,
                stride=stride,
                target_len=int(target_len),
                allow_short_segment=True,
            )
            if int(windows.shape[0]) == 0:
                continue
            x_parts.append(windows)
            y_parts.append(np.full(windows.shape[0], label_to_idx[label_id], dtype=np.int64))
            g_parts.append(np.full(windows.shape[0], int(subject), dtype=np.int64))

    if not x_parts:
        raise RuntimeError(f"No Opportunity windows were built for task {task_name}.")

    X = np.concatenate(x_parts, axis=0)
    if normalize:
        X = _zscore_per_window(X)
    bundle = IMUBundle(
        name=f"opportunity_{task_name}",
        X=X.astype(np.float32),
        y=np.concatenate(y_parts, axis=0),
        groups=np.concatenate(g_parts, axis=0),
        label_names=[label_map[idx] for idx in ordered_label_ids],
        sampling_rate_hz=int(source_fs),
        target_len=int(target_len),
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    )
    _save_bundle(cache_path, bundle)
    return bundle


def build_eval_bundles(
    *,
    cache_dir: str | Path | None = None,
    target_len: int = 500,
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    task_ids: list[str] | None = None,
) -> dict[str, IMUBundle]:
    if task_ids is None:
        task_ids = ["motionsense_activity6", "wisdm_phone_activity18"]

    bundles: dict[str, IMUBundle] = {}
    for task_id in dict.fromkeys(str(t) for t in task_ids):
        if task_id == "motionsense_activity6":
            bundle = build_motionsense_bundle(
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
            )
        elif task_id == "wisdm_phone_activity18":
            bundle = build_wisdm_phone_bundle(
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
            )
        elif task_id == "wisdm_phone_activity6":
            bundle = build_wisdm_phone_group6_bundle(
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
            )
        elif task_id == "usc_had_activity12":
            bundle = build_usc_had_bundle(
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
            )
        elif task_id in {
            "pamap2_activity12",
            "pamap2_hand_activity12",
            "pamap2_chest_activity12",
            "pamap2_ankle_activity12",
            "pamap2_hr",
            "pamap2_hand_hr",
            "pamap2_chest_hr",
            "pamap2_ankle_hr",
        }:
            placement = "chest"
            if task_id == "pamap2_hand_activity12":
                placement = "hand"
            elif task_id == "pamap2_ankle_activity12":
                placement = "ankle"
            elif task_id == "pamap2_hand_hr":
                placement = "hand"
            elif task_id == "pamap2_ankle_hr":
                placement = "ankle"
            label_mode = "activity12"
            if task_id in {"pamap2_hr", "pamap2_hand_hr", "pamap2_chest_hr", "pamap2_ankle_hr"}:
                label_mode = "hr"
            bundle = build_pamap2_bundle(
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
                placement=placement,
                label_mode=label_mode,
            )
        elif task_id == "ucihar_activity6":
            bundle = build_ucihar_bundle(
                cache_dir=cache_dir,
                target_len=int(target_len),
            )
        elif task_id == "hhar_activity6":
            bundle = build_hhar_bundle(
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
            )
        elif task_id == "hapt_activity12":
            bundle = build_hapt_bundle(
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
            )
        elif task_id == "mhealth_activity12":
            bundle = build_mhealth_bundle(
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
            )
        elif task_id == "mhealth_hr":
            bundle = build_mhealth_bundle(
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
                label_mode="hr",
            )
        elif task_id == "opportunity_locomotion4":
            bundle = build_opportunity_bundle(
                task="locomotion4",
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
            )
        elif task_id == "opportunity_gesture17":
            bundle = build_opportunity_bundle(
                task="gesture17",
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
            )
        elif task_id == "realworldhar_activity8":
            bundle = build_realworldhar_bundle(
                cache_dir=cache_dir,
                target_len=int(target_len),
                win_sec=float(win_sec),
                stride_sec=float(stride_sec),
            )
        else:
            raise KeyError(f"Unsupported IMU eval task: {task_id}")
        bundles[str(task_id)] = _rename_bundle(bundle, str(task_id))
    return bundles


def summarize_bundles(bundles: dict[str, IMUBundle]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, bundle in bundles.items():
        labels, counts = np.unique(bundle.y, return_counts=True)
        rows.append(
            {
                "dataset": str(name),
                "n_samples": int(bundle.n_samples),
                "n_subjects": int(np.unique(bundle.groups).size),
                "n_classes": int(bundle.n_classes),
                "target_len": int(bundle.target_len),
                "channels": int(bundle.X.shape[1]),
                "min_class_n": int(counts.min()),
                "max_class_n": int(counts.max()),
                "label_names": ",".join(bundle.label_names),
            }
        )
    return pd.DataFrame(rows)


def build_public_pool(
    bundles: dict[str, IMUBundle],
    *,
    total_n: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not bundles:
        raise ValueError("No IMU bundles provided for public pool construction.")
    rng = np.random.default_rng(int(seed))
    names = list(bundles.keys())
    base = int(math.floor(int(total_n) / max(1, len(names))))
    targets = {name: base for name in names}
    for name in names[: int(total_n) - base * len(names)]:
        targets[name] += 1

    sampled_x: list[np.ndarray] = []
    sampled_sources: list[np.ndarray] = []
    actual_counts: dict[str, int] = {}
    remainders: list[tuple[str, np.ndarray]] = []

    for name in names:
        bundle = bundles[name]
        total = int(bundle.n_samples)
        want = min(int(targets[name]), total)
        idx = rng.choice(total, size=want, replace=False) if want > 0 else np.empty((0,), dtype=np.int64)
        sampled_x.append(bundle.X[idx])
        sampled_sources.append(np.full(want, str(name), dtype=object))
        actual_counts[name] = int(want)
        remain_mask = np.ones(total, dtype=bool)
        remain_mask[idx] = False
        remain_idx = np.where(remain_mask)[0]
        remainders.append((name, remain_idx))

    current = int(sum(actual_counts.values()))
    need_more = int(total_n) - current
    if need_more > 0:
        for name, remain_idx in remainders:
            if need_more <= 0:
                break
            if len(remain_idx) == 0:
                continue
            take = min(int(need_more), int(len(remain_idx)))
            pick = rng.choice(remain_idx, size=take, replace=False)
            sampled_x.append(bundles[name].X[pick])
            sampled_sources.append(np.full(take, str(name), dtype=object))
            actual_counts[name] = int(actual_counts.get(name, 0) + take)
            need_more -= int(take)

    X = np.concatenate(sampled_x, axis=0).astype(np.float32)
    src = np.concatenate(sampled_sources, axis=0)
    perm = rng.permutation(len(X))
    X = X[perm]
    src = src[perm]
    meta = {
        "target_total_n": int(total_n),
        "actual_total_n": int(len(X)),
        "seed": int(seed),
        "source_counts": {str(k): int(v) for k, v in actual_counts.items()},
        "channels": list(next(iter(bundles.values())).channels),
        "target_len": int(next(iter(bundles.values())).target_len),
        "sources": src.tolist(),
    }
    return torch.from_numpy(X), meta

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _default_wfppg_base_dir() -> str:
	# repo_root/pulseppg/data/datasets/wfppg
	return os.path.join(os.path.dirname(__file__), "pulseppg", "data", "datasets", "wfppg")


class WFPPGNpyHRDataset(Dataset):
	"""Loads processed WF-PPG heart-rate regression dataset from numpy files.

	Exported by: pulseppg/data/process/WFPPG.py

	Expected files under root_dir:
	  - X_ppg_{hz}Hz.npy        -> (N, 1, L) float32
	  - y_hr_{hz}Hz.npy         -> (N,) float32 BPM
	  - subject_{hz}Hz.npy      -> (N,) str subject id (e.g., "01")

	Optional:
	  - session_{hz}Hz.npy      -> (N,) int session id
	  - win_start_{hz}Hz.npy    -> (N,) int start index (samples) in the session's uniform grid

	split:
	  - "all" | "full" | "total" loads unsplit file names above
	  - otherwise loads legacy split-prefixed names (e.g., train_X_ppg_50Hz.npy)

	Returns:
	  - (x, y) by default
	  - (x, y, meta) if return_meta=True
	"""

	def __init__(
		self,
		root_dir: str | Path,
		split: str = "full",
		hz: int = 50,
		mmap: bool = True,
		return_meta: bool = False,
	):
		super().__init__()
		self.root_dir = Path(root_dir)
		self.split = str(split)
		self.hz = int(hz)
		self.return_meta = bool(return_meta)

		mode = "r" if mmap else None
		if split in ("all", "full", "total"):
			x_name = f"X_ppg_{self.hz}Hz.npy"
			y_name = f"y_hr_{self.hz}Hz.npy"
			s_name = f"subject_{self.hz}Hz.npy"
			sess_name = f"session_{self.hz}Hz.npy"
			start_name = f"win_start_{self.hz}Hz.npy"
		else:
			x_name = f"{split}_X_ppg_{self.hz}Hz.npy"
			y_name = f"{split}_y_hr_{self.hz}Hz.npy"
			s_name = f"{split}_subject_{self.hz}Hz.npy"
			sess_name = f"{split}_session_{self.hz}Hz.npy"
			start_name = f"{split}_win_start_{self.hz}Hz.npy"

		self.X = np.load(self.root_dir / x_name, mmap_mode=mode)
		self.y = np.load(self.root_dir / y_name, mmap_mode=mode)

		self.subject = np.load(self.root_dir / s_name, mmap_mode=mode)
		self.subject = np.asarray(self.subject).astype(str)

		sess_path = self.root_dir / sess_name
		self.session: Optional[np.ndarray]
		if sess_path.exists():
			self.session = np.asarray(np.load(sess_path, mmap_mode=mode)).astype(np.int64)
		else:
			self.session = None

		start_path = self.root_dir / start_name
		self.win_start: Optional[np.ndarray]
		if start_path.exists():
			self.win_start = np.asarray(np.load(start_path, mmap_mode=mode)).astype(np.int64)
		else:
			self.win_start = None

		if self.X.ndim == 4:
			# some pipelines save (N,1,1,L)
			self.X = self.X[:, 0, :, :]

		assert self.X.ndim == 3 and self.X.shape[1] == 1, f"Expected (N,1,L), got {self.X.shape}"
		assert len(self.X) == len(self.y) == len(self.subject), (
			f"Length mismatch: X={len(self.X)} y={len(self.y)} subject={len(self.subject)}"
		)
		if self.session is not None and len(self.session) != len(self.X):
			raise ValueError(f"Length mismatch: session={len(self.session)} vs X={len(self.X)}")
		if self.win_start is not None and len(self.win_start) != len(self.X):
			raise ValueError(f"Length mismatch: win_start={len(self.win_start)} vs X={len(self.X)}")

		self.L = int(self.X.shape[-1])
		self.sec_per_sample = float(self.L) / float(self.hz)

	def __len__(self):
		return int(self.X.shape[0])

	def __getitem__(self, idx: int):
		x_np = np.asarray(self.X[idx])
		if not x_np.flags.writeable:
			x_np = x_np.copy()
		x = torch.from_numpy(x_np).float()  # (1, L)

		y = torch.tensor(float(self.y[idx]), dtype=torch.float32)
		if not self.return_meta:
			return x, y

		meta: Dict[str, Any] = {
			"split": self.split,
			"hz": self.hz,
			"sec": self.sec_per_sample,
			"subject": self.subject[idx],
		}
		if self.session is not None:
			meta["session"] = int(self.session[idx])
		if self.win_start is not None:
			meta["win_start"] = int(self.win_start[idx])
		return x, y, meta


def make_wfppg_dataset(
	base_dir: str | None = None,
	ppg_source: str = "wrist",
	split: str = "full",
	hz: int = 50,
	mmap: bool = True,
	return_meta: bool = False,
) -> WFPPGNpyHRDataset:
	"""Convenience wrapper to build a processed WF-PPG HR dataset.

	Expects files in:
	  base_dir/{ppg_source}/X_ppg_{hz}Hz.npy
	  base_dir/{ppg_source}/y_hr_{hz}Hz.npy
	  base_dir/{ppg_source}/subject_{hz}Hz.npy
	"""
	if ppg_source not in ("wrist", "finger"):
		raise ValueError("ppg_source must be 'wrist' or 'finger'")
	if base_dir is None:
		base_dir = _default_wfppg_base_dir()
	root = os.path.join(str(base_dir), ppg_source)
	return WFPPGNpyHRDataset(root, split=split, hz=hz, mmap=mmap, return_meta=return_meta)


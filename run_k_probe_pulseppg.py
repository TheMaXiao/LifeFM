#!/usr/bin/env python
"""
K-Probe (PulsePPG): Fair full-D vs subspace comparison
=======================================================

Reuse the proven NormWear k_probe logic, but replace:
- Teacher: PulsePPG FM teacher
- Student: small ResNet1D students
- Full dimension: D_FULL (typically 512)

Fairness rule:
- Student output dimension is always D_FULL
- Only KD loss space changes:
  - full-D: MSE(z_s, z_t)
  - subspace-k: MSE(z_s @ V_k^T, z_t @ V_k^T)

Default probe: k in {50, 512}

Usage examples:
    conda activate pulseppg
    CUDA_VISIBLE_DEVICES=0 python -u run_k_probe_pulseppg.py

    # Quick smoke test
    CUDA_VISIBLE_DEVICES=0 python -u run_k_probe_pulseppg.py --epochs 2 --n-data 1000 --max-samples 5000 --students BF10_NB14
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from downstream_eval import run_all_tasks, set_fp16, set_gpu_probe

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

SEED = 20260621
KD_LR = 1e-4
KD_BS = 64
EVAL_HZ = 50

STUDENT_CONFIGS = {
    "BF10_NB14": dict(base_filters=10, n_block=14),
    "BF20_NB8": dict(base_filters=20, n_block=8),
    "BF16_NB12": dict(base_filters=16, n_block=12),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-data", type=int, default=10000)
    p.add_argument("--k-values", type=int, nargs="+", default=[50, 512])
    p.add_argument("--students", type=str, nargs="+", default=list(STUDENT_CONFIGS.keys()))
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--out-dir", type=str, default="kd_sweep_results/pulseppg_preliminary/exp_k_probe_fair")
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   help="Optional subset of downstream tasks to run")
    p.add_argument("--include-sportppg", action="store_true",
                   help="Include sportppg_activity task (default: off)")
    p.add_argument("--gpu-probe", action="store_true",
                   help="Enable GPU-accelerated linear probes")
    return p.parse_args()


def _unwrap_model_output(out):
    if isinstance(out, dict):
        for key in ("feat", "features", "embedding", "emb", "rep", "z", "proj", "logits"):
            if key in out and torch.is_tensor(out[key]):
                out = out[key]
                break
        else:
            out = next(v for v in out.values() if torch.is_tensor(v))
    if not torch.is_tensor(out):
        raise TypeError(f"Unexpected output type: {type(out)}")
    return out


def train_subspace_mse(student, ds, V_k_t, epochs, dev):
    student = student.to(dev).train()
    for m in student.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d)):
            m.p = 0.0
    ld = DataLoader(ds, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=KD_LR, weight_decay=0.0)
    sc = torch.amp.GradScaler(device="cuda")
    V = V_k_t.to(dev)

    for _ in range(epochs):
        for x_b, t_proj in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            t_proj = t_proj.to(dev, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b).float()
                z_s_proj = z_s @ V.T
                loss = F.mse_loss(z_s_proj, t_proj)
            sc.scale(loss).backward()
            sc.step(opt)
            sc.update()

    student.eval()
    return student


def train_full_mse(student, ds, epochs, dev):
    student = student.to(dev).train()
    for m in student.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d)):
            m.p = 0.0
    ld = DataLoader(ds, batch_size=KD_BS, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=KD_LR, weight_decay=0.0)
    sc = torch.amp.GradScaler(device="cuda")

    for _ in range(epochs):
        for x_b, z_t in ld:
            x_b = x_b.to(dev, non_blocking=True)
            if x_b.ndim == 2:
                x_b = x_b.unsqueeze(1)
            z_t = z_t.to(dev, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                z_s = student(x_b).float()
                loss = F.mse_loss(z_s, z_t)
            sc.scale(loss).backward()
            sc.step(opt)
            sc.update()

    student.eval()
    return student


class DaliaPublicNPYDataset(Dataset):
    def __init__(self, root, target_len=500, hz=50):
        root = Path(root)
        parts = []
        for split in ["train", "val", "test"]:
            p = root / f"{split}_X_ppg_{hz}Hz.npy"
            if p.exists():
                parts.append(np.load(str(p), mmap_mode="r"))
        self.data = np.concatenate(parts, axis=0).astype(np.float32)
        L = self.data.shape[-1]
        if L < target_len:
            self.data = np.pad(
                self.data,
                [(0, 0)] * (self.data.ndim - 1) + [(0, target_len - L)],
                mode="constant",
            )
        self.data = self.data[..., :target_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return torch.as_tensor(x, dtype=torch.float32)


class SDBPublicNPYDataset(Dataset):
    def __init__(self, root, hz=50):
        root = Path(root)
        parts = []
        for split in ["train", "val", "test"]:
            p = root / f"{split}_X_ppg_{hz}Hz.npy"
            if p.exists():
                parts.append(np.load(str(p), mmap_mode="r"))
        self.data = np.concatenate(parts, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return torch.as_tensor(x, dtype=torch.float32)


class PPGNPYDataset(Dataset):
    def __init__(self, ppg_path, target_len=500):
        from scipy.signal import resample_poly

        ppg = np.load(ppg_path, mmap_mode="r")
        n, L = ppg.shape
        if L != target_len:
            self.data = resample_poly(ppg, up=2, down=5, axis=1).astype(np.float32)
        else:
            self.data = ppg.astype(np.float32)
        self.data = self.data[:, :target_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return torch.as_tensor(x, dtype=torch.float32)


def load_pulseppg_teacher(dev):
    from pulseppg.experiments.configs.MotifDist_expconfigs import allmotifdist_expconfigs
    from pulseppg.experiments.configs.PulsePPG_expconfigs import allpulseppg_expconfigs
    from pulseppg.utils.imports import import_model

    all_cfgs = {**allmotifdist_expconfigs, **allpulseppg_expconfigs}
    out_root = Path("pulseppg/experiments/out")
    cands = [
        (p.name, (p / "checkpoint_best.pkl").stat().st_mtime)
        for p in out_root.glob("*")
        if p.is_dir() and (p / "checkpoint_best.pkl").exists()
    ]
    if not cands:
        raise FileNotFoundError("No pulseppg checkpoint found under pulseppg/experiments/out")

    config_name = sorted(cands, key=lambda x: x[1], reverse=True)[0][0]
    cfg = all_cfgs[config_name]
    cfg.set_rundir(config_name)
    wrap = import_model(cfg, reload_ckpt=False)
    teacher = wrap.net.to(dev).eval()

    ckpt_path = out_root / config_name / "checkpoint_best.pkl"
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "net", "model"]:
            if k in ckpt:
                state = ckpt[k]
                break
        if state is None and sum(isinstance(v, torch.Tensor) for v in ckpt.values()) >= len(ckpt) // 2:
            state = ckpt

    if state is not None:
        cleaned = {}
        for k, v in state.items():
            nk = k
            for pref in ("module.", "net.", "model.", "encoder.", "backbone."):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            cleaned[nk] = v
        teacher.load_state_dict(cleaned, strict=False)

    teacher = teacher.to(dev).eval()
    return teacher, config_name


def make_student_builder(dev, d_full):
    from pulseppg.nets.ResNet1D.ResNet1D_Net import Net as ResNet1DNet

    def build_student(seed, student_cfg):
        torch.manual_seed(seed)
        np.random.seed(seed)
        enc = ResNet1DNet(
            in_channels=1,
            base_filters=student_cfg["base_filters"],
            kernel_size=11,
            stride=2,
            groups=1,
            n_block=student_cfg["n_block"],
            finalpool="max",
            downsample_gap=2,
            increasefilter_gap=4,
            use_bn=True,
            use_do=True,
            verbose=False,
        )
        with torch.inference_mode():
            ed = enc(torch.randn(2, 1, 500)).shape[-1]
        proj = nn.Linear(ed, d_full, bias=False)
        nn.init.kaiming_normal_(proj.weight)
        return nn.Sequential(enc, proj).to(dev)

    return build_student


def build_mixed_pool(teacher, dev, n_total, sources_filter=None, pool_seed=None):
    """Build a KD pool of `n_total` windows.

    sources_filter: optional iterable of source names in {"dalia","sdb","pulsedb"}.
                    If None, uses all available sources (current default behaviour).
                    If a single source is given (e.g. ["pulsedb"]), n_total
                    windows are drawn from that source only.
    pool_seed:      optional int; if given, the per-source random subsample uses
                    this seed instead of the global SEED. Used for stress tests
                    that probe variance across pool subsamples.
    """
    ds_dalia = DaliaPublicNPYDataset("pulseppg/data/datasets/dalia", target_len=500)
    ds_sdb = SDBPublicNPYDataset("pulseppg/data/datasets/sdb")

    sources = [("dalia", ds_dalia), ("sdb", ds_sdb)]

    pulsedb_ppg = Path("/home/xiaoma/datasets/ppg/PulseDB/subset_3M_h5_ppg.npy")
    if pulsedb_ppg.exists():
        ds_pulsedb = PPGNPYDataset(str(pulsedb_ppg), target_len=500)
        sources.append(("pulsedb", ds_pulsedb))
    else:
        print("  [WARN] PulseDB not found, using 2-source mix (dalia + sdb)")

    if sources_filter is not None:
        sf = set(sources_filter)
        sources = [(n, d) for (n, d) in sources if n in sf]
        if not sources:
            raise ValueError(f"No sources matched filter {sources_filter}")
        print(f"  [pool] sources_filter active: keeping {[n for n,_ in sources]}")

    n_per = max(1, n_total // len(sources))
    rng = np.random.default_rng(SEED if pool_seed is None else int(pool_seed))

    all_x, all_z = [], []
    print("\n-- Extracting mixed embeddings --")
    for name, ds in sources:
        n_use = min(n_per, len(ds))
        idx = np.sort(rng.choice(len(ds), n_use, replace=False))
        loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False, num_workers=0)
        xp, zp = [], []
        with torch.inference_mode(), torch.cuda.amp.autocast():
            for batch in loader:
                x = batch.to(dev).float()
                if x.ndim == 2:
                    x = x.unsqueeze(1)
                z = _unwrap_model_output(teacher(x)).float()
                if z.ndim > 2:
                    z = z.flatten(1)
                xp.append(x.cpu())
                zp.append(z.cpu().numpy().astype(np.float32))
        x_cat = torch.cat(xp)
        z_cat = np.concatenate(zp)
        all_x.append(x_cat)
        all_z.append(z_cat)
        print(f"  {name}: {len(z_cat)}")

    X_all = torch.cat(all_x)
    Z_all = np.concatenate(all_z)
    print(f"  Total: X={tuple(X_all.shape)}, Z={Z_all.shape}")
    return X_all, Z_all


def metric_is_higher_better(metric_name: str) -> bool:
    return metric_name != "mae"


def main():
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  K-Probe (PulsePPG): fair full-D vs subspace")
    print("=" * 70)
    print(f"  Device: {dev}")
    print(f"  n_data={args.n_data}, epochs={args.epochs}, n_seeds={args.n_seeds}")
    print(f"  k_values={args.k_values}")

    if dev.type != "cuda":
        raise RuntimeError("This script expects CUDA for practical runtime.")

    set_fp16(True)
    if args.gpu_probe:
        set_gpu_probe(str(dev))

    print("\n-- Loading PulsePPG teacher --")
    teacher, teacher_name = load_pulseppg_teacher(dev)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {teacher_name}, params={n_teacher:,}")

    X_all, Z_all = build_mixed_pool(teacher, dev, args.n_data)
    d_full = int(Z_all.shape[1])
    print(f"  D_FULL={d_full}")

    if d_full not in args.k_values:
        print(f"  [INFO] adding full dimension k={d_full}")
        args.k_values = sorted(set(args.k_values + [d_full]))

    build_student = make_student_builder(dev, d_full)

    print("\n-- Student sizes --")
    selected_students = [s for s in args.students if s in STUDENT_CONFIGS]
    for sname in selected_students:
        s = build_student(0, STUDENT_CONFIGS[sname])
        n_params = sum(p.numel() for p in s.parameters())
        print(f"  {sname}: {n_params:,} params, ratio={n_teacher / max(1, n_params):.1f}x")
        del s
    gc.collect()
    torch.cuda.empty_cache()

    data_dirs = {
        "wesad": "pulseppg/data/datasets/wesad",
        "sdb": "pulseppg/data/datasets/sdb",
        "wfppg": "pulseppg/data/datasets/wfppg",
        "dalia": "pulseppg/data/datasets/dalia",
        "ppgbp": "pulseppg/data/datasets/ppgbp",
        "mimicperform": "pulseppg/data/datasets/mimicperform",
        "ppgarrhythmia": "pulseppg/data/datasets/ppgarrhythmia",
        "earset": "pulseppg/data/datasets/earset",
    }

    if args.include_sportppg:
        data_dirs["sportppg"] = {
            "data_root": "/home/xiaoma/datasets/ppg/PPG-Sport-F5B0/dataset/",
            "gt_root": "/home/xiaoma/datasets/ppg/PPG-Sport-F5B0/cleaned_gt/",
        }

    teacher_cache = out_dir / "teacher_baseline.json"
    if teacher_cache.exists():
        print("\n-- Loading cached teacher baseline --")
        teacher_baseline = json.loads(teacher_cache.read_text())
    else:
        print("\n-- Computing teacher baseline --")
        t_results = run_all_tasks(
            teacher,
            str(dev),
            data_dirs,
            tasks=args.tasks,
            hz=EVAL_HZ,
            skip_missing=True,
            max_samples=args.max_samples,
        )
        teacher_baseline = {
            t: {"value": float(m["primary"]), "metric": m["primary_metric"]}
            for t, m in t_results.items()
        }
        teacher_cache.write_text(json.dumps(teacher_baseline, indent=2))
        print(f"  Cached teacher baseline: {len(teacher_baseline)} tasks")

    eval_tasks = list(teacher_baseline.keys())

    def eval_student(student):
        student.eval()
        stu_results = run_all_tasks(
            student,
            str(dev),
            data_dirs,
            tasks=eval_tasks,
            hz=EVAL_HZ,
            skip_missing=True,
            max_samples=args.max_samples,
        )

        rets = {}
        for tname, m in stu_results.items():
            ref = teacher_baseline[tname]
            stu_val = float(m["primary"])
            if metric_is_higher_better(ref["metric"]):
                ret = stu_val / ref["value"] if ref["value"] != 0 else np.nan
            else:
                ret = ref["value"] / stu_val if stu_val != 0 else np.nan
            rets[tname] = float(ret)

        vals = np.array(list(rets.values()), dtype=np.float64)
        return float(np.nanmean(vals)), float(np.nanmin(vals)), rets

    ckpt_path = out_dir / "k_probe_pulseppg_results.csv"
    all_rows = []
    done = set()
    if ckpt_path.exists():
        df_ck = pd.read_csv(ckpt_path)
        all_rows = df_ck.to_dict("records")
        done = {
            (r["student"], int(r["n_data"]), int(r["epochs"]), int(r["k"]), int(r["seed"]))
            for r in all_rows
        }
        print(f"\n-- Resume mode: {len(done)} finished runs --")

    rng_sub = np.random.default_rng(SEED + 42)
    if args.n_data < len(Z_all):
        idx_sub = np.sort(rng_sub.choice(len(Z_all), args.n_data, replace=False))
    else:
        idx_sub = np.arange(len(Z_all))

    X_sub = X_all[idx_sub]
    Z_sub = Z_all[idx_sub]

    mu_sub = Z_sub.mean(axis=0)
    Zc_sub = (Z_sub - mu_sub).astype(np.float64)
    _, _, Vt_sub = np.linalg.svd(Zc_sub, full_matrices=False)
    Vt_sub = Vt_sub.astype(np.float32)
    del Zc_sub

    print("\n-- Running probe --")
    run_count = 0
    for sname in selected_students:
        scfg = STUDENT_CONFIGS[sname]
        print(f"\n{'═' * 60}")
        print(f"  Student: {sname}")
        print(f"{'═' * 60}")

        k_results = {}
        for k in sorted(set(args.k_values)):
            if k > d_full:
                print(f"  skip k={k}: k > D_FULL ({d_full})")
                continue

            if k == d_full:
                ds_train = TensorDataset(X_sub, torch.as_tensor(Z_sub, dtype=torch.float32))
                V_k_t = None
            else:
                V_k = Vt_sub[:k]
                Z_proj = Z_sub @ V_k.T
                ds_train = TensorDataset(X_sub, torch.as_tensor(Z_proj, dtype=torch.float32))
                V_k_t = torch.as_tensor(V_k, dtype=torch.float32, device=dev)

            k_rets = []
            for si in range(args.n_seeds):
                key = (sname, args.n_data, args.epochs, k, si)
                if key in done:
                    prev = [
                        r
                        for r in all_rows
                        if r["student"] == sname
                        and int(r["n_data"]) == args.n_data
                        and int(r["epochs"]) == args.epochs
                        and int(r["k"]) == k
                        and int(r["seed"]) == si
                    ]
                    if prev:
                        k_rets.append(float(prev[0]["avg_ret"]))
                        print(f"  k={k:>3d} seed={si}: avg_ret={prev[0]['avg_ret']:.4f} [cached]")
                    continue

                t0 = time.time()
                stu = build_student(77 + si * 1000, scfg)

                if k == d_full:
                    stu = train_full_mse(stu, ds_train, epochs=args.epochs, dev=dev)
                else:
                    stu = train_subspace_mse(stu, ds_train, V_k_t, epochs=args.epochs, dev=dev)

                avg_ret, floor_ret, per_task = eval_student(stu)
                elapsed = time.time() - t0

                row = {
                    "student": sname,
                    "n_data": args.n_data,
                    "epochs": args.epochs,
                    "k": k,
                    "seed": si,
                    "avg_ret": avg_ret,
                    "floor_ret": floor_ret,
                    "elapsed_s": elapsed,
                }
                for tname, ret in per_task.items():
                    row[f"ret_{tname}"] = ret

                all_rows.append(row)
                done.add(key)
                k_rets.append(avg_ret)
                run_count += 1

                print(f"  k={k:>3d} seed={si}: avg_ret={avg_ret:.4f}, floor={floor_ret:.4f} [{elapsed:.0f}s]")

                pd.DataFrame(all_rows).to_csv(ckpt_path, index=False)
                del stu
                gc.collect()
                torch.cuda.empty_cache()

            k_results[k] = k_rets

        print("\n  ┌─ Summary")
        for k in sorted(k_results.keys()):
            vals = k_results[k]
            if vals:
                print(f"  │  k={k:>3d}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        if 50 in k_results and d_full in k_results and k_results[50] and k_results[d_full]:
            gap = np.mean(k_results[50]) - np.mean(k_results[d_full])
            print(f"  │  Δ(k=50 - k={d_full}) = {gap:+.4f} ({gap * 100:+.2f}%)")
        print("  └─")

    print(f"\n{'=' * 70}")
    print(f"  Grand Summary ({run_count} new runs)")
    print(f"{'=' * 70}")

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        summary = df.groupby(["student", "n_data", "epochs", "k"])["avg_ret"].agg(["mean", "std"]).reset_index()
        summary.columns = ["student", "n_data", "epochs", "k", "mean", "std"]

        for (sname, n_data, n_ep), grp in summary.groupby(["student", "n_data", "epochs"]):
            r50 = grp[grp.k == 50]["mean"].values
            rf = grp[grp.k == d_full]["mean"].values
            r50_s = f"{r50[0]:.4f}" if len(r50) else "N/A"
            rf_s = f"{rf[0]:.4f}" if len(rf) else "N/A"
            gap = (r50[0] - rf[0]) * 100 if len(r50) and len(rf) else np.nan
            print(
                f"  {sname:12s} n={int(n_data):>5d} ep={int(n_ep):>2d} "
                f"k50={r50_s} k{d_full}={rf_s} Δ50v{d_full}={gap:+.2f}%"
            )

    print(f"\n  Results saved to: {ckpt_path}")
    print("  Done!")


if __name__ == "__main__":
    main()

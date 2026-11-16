"""ECGFounder -> Net1D same-arch KD.

Source-free embedding-level KD using ECGFounder (12-lead, 1024-D) as teacher and
smaller Net1D students as students.

The current script supports a compact ECGFounder size sweep in the ~0.5M-2M
range plus the original BF8 / BF32 / BF4 anchors.

Adaptive K: smallest k such that cumulative singular value energy >= 90% of pool.

Eval: ECGFounder-native 13-task benchmark on 10s windows @ 500Hz.
Retention = student / teacher on the higher-better metric (`AUROC` for
classification; `-MAE` mirror for regression).
"""
from __future__ import annotations
import os, sys, time, json, gc, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/home/xiaoma/foundation_model/ecg")
sys.path.insert(0, "/home/xiaoma/foundation_model/ecg/ECGFounder")
from ecgfounder_teacher import ECGFounderTeacher, ECGFOUNDER_DIM
from net1d import Net1D

warnings.filterwarnings("ignore")

# ─────────────────────────── constants ─────────────────────────────────────
N_LEADS, FS, T = 12, 500, 5000        # ECGFounder native: 10s @ 500Hz
TEACHER_DIM = ECGFOUNDER_DIM          # 1024
KD_LR, KD_BS, KD_EPOCHS = 1e-4, 16, 30
ENERGY_THRESHOLD = 0.90               # adaptive-K target

PTBXL_DIR = "/home/xiaoma/foundation_model/ecg/data/ptbxl_processed"
CHAP_DIR  = "/home/xiaoma/foundation_model/ecg/data/chapman_processed"
CINC_DIR  = "/home/xiaoma/foundation_model/ecg/data/cinc2017_processed_v2"
DEFAULT_POOL = f"{PTBXL_DIR}/signals_500hz.npy"   # (21799, 12, 5000) — 10s native

# 13-task multi-dataset benchmark.  All signals are 12-lead x 5000 @ 500Hz.
# Each entry: (name, kind, root_dir, label_file, valid_file_or_None)
# kind in {cls, bin, reg}
TASKS_13 = [
    ("super5",            "cls", PTBXL_DIR, "labels_super_5cls.npy",   None),
    ("afib_bin",          "bin", PTBXL_DIR, "labels_afib_bin.npy",     None),
    ("rhythm12",          "cls", PTBXL_DIR, "labels_rhythm_12cls.npy", None),
    ("sub23",             "cls", PTBXL_DIR, "labels_sub_Ncls.npy",     None),
    ("norm_bin",          "bin", PTBXL_DIR, "labels_norm_bin.npy",     "valid_norm_bin.npy"),
    ("mi_bin",            "bin", PTBXL_DIR, "labels_mi_bin.npy",       "valid_mi_bin.npy"),
    ("sttc_bin",          "bin", PTBXL_DIR, "labels_sttc_bin.npy",     "valid_sttc_bin.npy"),
    ("cd_bin",            "bin", PTBXL_DIR, "labels_cd_bin.npy",       "valid_cd_bin.npy"),
    ("sex_bin",           "bin", PTBXL_DIR, "labels_sex_bin.npy",      "valid_sex_bin.npy"),
    ("age_reg",           "reg", PTBXL_DIR, "labels_age_reg.npy",      "valid_age_reg.npy"),
    ("chap_rhythm4",      "cls", CHAP_DIR,  "labels_rhythm4.npy",      None),
    ("chap_abnormal_bin", "bin", CHAP_DIR,  "labels_abnormal_bin.npy", None),
    ("cinc_af_bin",       "bin", CINC_DIR,  "labels_af_bin.npy",       None),
]

STUDENT_CFGS = {
    # ~1M-param same-arch student: 32.5x compression vs 30.82M ECGFounder teacher.
    "BF8_NB7":  dict(base_filters=8,
                     filter_list=[8, 24, 24, 64, 64, 160, 160],
                     m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                     kernel_size=16, stride=2, groups_width=8,
                     feat_dim=160),
    # ~0.58M-param same-arch student.
    "BF6_NB7":  dict(base_filters=6,
                     filter_list=[6, 18, 18, 48, 48, 120, 120],
                     m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                     kernel_size=16, stride=2, groups_width=2,
                     feat_dim=120),
    # ~1.50M-param same-arch student.
    "BF10_NB7": dict(base_filters=10,
                     filter_list=[10, 30, 30, 80, 80, 200, 200],
                     m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                     kernel_size=16, stride=2, groups_width=5,
                     feat_dim=200),
    # ~1.92M-param same-arch student.
    "BF12_NB7": dict(base_filters=12,
                     filter_list=[12, 36, 36, 96, 96, 240, 240],
                     m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                     kernel_size=16, stride=2, groups_width=2,
                     feat_dim=240),
    # Mid-size (kept for reference): 8.96M, 3.4x compression.
    "BF32_NB7": dict(base_filters=32,
                     filter_list=[32, 80, 80, 192, 192, 512, 512],
                     m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                     kernel_size=16, stride=2, groups_width=16,
                     feat_dim=512),
    # Larger mid-size follow-up to test when the KD optimum moves past ~107.
    "BF40_NB7": dict(base_filters=40,
                     filter_list=[40, 100, 100, 240, 240, 640, 640],
                     m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                     kernel_size=16, stride=2, groups_width=20,
                     feat_dim=640),
    # Tiny ~234K-param student: 131x compression vs ECGFounder.
    "BF4_NB7":  dict(base_filters=4,
                     filter_list=[4, 8, 8, 24, 24, 64, 64],
                     m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                     kernel_size=16, stride=2, groups_width=4,
                     feat_dim=64),
}

# ─────────────────────────── normalization ────────────────────────────────
def zscore_batch(x: np.ndarray) -> np.ndarray:
    """Global z-score per recording (matches ECGFounder dataset.py)."""
    if x.ndim == 3:
        mu = x.mean(axis=(1, 2), keepdims=True)
        sd = x.std(axis=(1, 2), keepdims=True) + 1e-8
    else:
        mu = x.mean(); sd = x.std() + 1e-8
    return (x - mu) / sd

# ─────────────────────────── student model ────────────────────────────────
def build_student(cfg_name: str, seed: int) -> tuple[nn.Module, int]:
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = STUDENT_CFGS[cfg_name]
    enc = Net1D(
        in_channels=N_LEADS, base_filters=cfg["base_filters"], ratio=1,
        filter_list=cfg["filter_list"], m_blocks_list=cfg["m_blocks_list"],
        kernel_size=cfg["kernel_size"], stride=cfg["stride"],
        groups_width=cfg["groups_width"],
        verbose=False, use_bn=False, use_do=False,
        n_classes=150,                 # placeholder; we use deep_features
        return_features=True,
    )
    proj = nn.Linear(cfg["feat_dim"], TEACHER_DIM, bias=False)

    class StudentNet(nn.Module):
        def __init__(self, enc, proj):
            super().__init__()
            self.enc = enc
            self.proj = proj
        def forward(self, x):
            _, feat = self.enc(x)        # feat: (B, feat_dim)
            return self.proj(feat)       # (B, 1024)

    return StudentNet(enc, proj), cfg["feat_dim"]


def save_student_ckpt(model: nn.Module, ckpt_path: Path, *, student_name: str,
                      method: str, seed: int, k: int, lam: float,
                      feat_dim: int, student_cfg: dict, teacher_dim: int):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "student": student_name,
        "method": method,
        "seed": seed,
        "k": k,
        "lam": lam,
        "feat_dim": feat_dim,
        "teacher_dim": teacher_dim,
        "student_cfg": student_cfg,
        "state_dict": model.state_dict(),
    }, str(ckpt_path))

# ─────────────────────────── pool & teacher embeddings ────────────────────
def build_pool(pool_npy: str, teacher: ECGFounderTeacher, n: int, seed: int,
               cache_dir: Path):
    cache_x = cache_dir / "pool_X.npy"
    cache_z = cache_dir / "pool_Z.npy"
    if cache_x.exists() and cache_z.exists():
        X = np.load(cache_x); Z = np.load(cache_z)
        print(f"[pool] cached: X={X.shape} Z={Z.shape}")
        return X, Z
    X_all = np.load(pool_npy, mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X_all), min(n, len(X_all)), replace=False))
    X_raw = np.asarray(X_all[idx], dtype=np.float32)
    X = zscore_batch(X_raw)
    print(f"[pool] sampled {X.shape} (z-scored); embedding...")
    bs = 16
    Z = np.zeros((X.shape[0], TEACHER_DIM), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(X), bs):
        Z[i:i+bs] = teacher(torch.from_numpy(X[i:i+bs])).cpu().numpy()
        if (i // bs) % 50 == 0:
            print(f"  pool {i}/{len(X)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[pool] embedded in {time.time()-t0:.0f}s")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X); np.save(cache_z, Z)
    return X, Z

def adaptive_k(Z: np.ndarray, energy_thresh: float = ENERGY_THRESHOLD) -> tuple[int, np.ndarray]:
    """Pick smallest k such that cumulative energy of (Zc^T Zc) eigvals >= threshold."""
    Zc = Z - Z.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    energy = (S ** 2).cumsum() / (S ** 2).sum()
    k = int(np.searchsorted(energy, energy_thresh) + 1)
    print(f"[adaptive_k] D={Z.shape[1]}, k={k} retains {energy[k-1]*100:.2f}% energy "
          f"(top-5 sv: {S[:5].round(2)})")
    return k, torch.from_numpy(Vt[:k]).float()

# ─────────────────────────── KD losses ─────────────────────────────────────
def _rkd_pdist(z, eps=1e-12):
    dot = z @ z.T
    sq  = dot.diagonal()
    d2  = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2 * dot).clamp(min=eps)
    return d2.sqrt()

def _rkd_distance(z_s, z_t):
    td = _rkd_pdist(z_t); sd = _rkd_pdist(z_s)
    td = td / (td.mean() + 1e-8); sd = sd / (sd.mean() + 1e-8)
    B = z_t.shape[0]
    mask = torch.triu(torch.ones(B, B, dtype=torch.bool, device=z_t.device), diagonal=1)
    return F.huber_loss(sd[mask], td[mask], delta=1.0)

def _rkd_angle(z_s, z_t):
    td = F.normalize(z_t.unsqueeze(0) - z_t.unsqueeze(1), p=2, dim=-1)
    sd = F.normalize(z_s.unsqueeze(0) - z_s.unsqueeze(1), p=2, dim=-1)
    cos_t = torch.bmm(td, td.permute(0, 2, 1))
    cos_s = torch.bmm(sd, sd.permute(0, 2, 1))
    return F.huber_loss(cos_s, cos_t, delta=1.0)


def kd_loss(z_s, z_t, method, V_k=None, lam=0.1, **hp):
    """Dispatch all KD methods. z_s, z_t: (B, D) fp32."""
    eps = 1e-7
    if method == "full_mse":
        return F.mse_loss(z_s, z_t)
    if method == "subspace_mse":
        return F.mse_loss(z_s @ V_k.T, z_t @ V_k.T)
    if method == "ns_kd":
        sub = F.mse_loss(z_s @ V_k.T, z_t @ V_k.T)
        ns  = (z_s - (z_s @ V_k.T) @ V_k).pow(2).mean()
        return sub + lam * ns
    if method == "rkd":
        return hp.get("mu_d", 25.0) * _rkd_distance(z_s, z_t) + \
               hp.get("mu_a", 50.0) * _rkd_angle(z_s, z_t)
    if method == "pkt":
        # PKT: pairwise cosine similarity → probability → KL(p_t || p_s)
        zs = F.normalize(z_s, dim=1, eps=eps); zt = F.normalize(z_t, dim=1, eps=eps)
        ss = (zs @ zs.T + 1.0) * 0.5; st = (zt @ zt.T + 1.0) * 0.5
        ss = ss - torch.diag(torch.diag(ss)); st = st - torch.diag(torch.diag(st))
        ss = ss / (ss.sum(1, keepdim=True) + eps); st = st / (st.sum(1, keepdim=True) + eps)
        return (st * (torch.log(st + eps) - torch.log(ss + eps))).sum(1).mean()
    if method == "sp":
        # SP: row-normalised batch Gram match
        gs = F.normalize(z_s @ z_s.T, p=2, dim=1)
        gt = F.normalize(z_t @ z_t.T, p=2, dim=1)
        return F.mse_loss(gs, gt)
    if method == "crd":
        # batch-only InfoNCE, symmetric
        tau = hp.get("tau", 0.07)
        zs = F.normalize(z_s, dim=1); zt = F.normalize(z_t, dim=1)
        logits = zs @ zt.T / tau
        labels = torch.arange(z_s.shape[0], device=z_s.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    if method == "seed":
        # batch-as-queue: KL(softmax(t-t sims) || softmax(s-t sims))
        tau_s = hp.get("tau_s", 0.1); tau_t = hp.get("tau_t", 0.04)
        zs = F.normalize(z_s, dim=1); zt = F.normalize(z_t, dim=1)
        ss = zs @ zt.T / tau_s; st = zt @ zt.T / tau_t
        B = z_s.shape[0]
        mask = torch.eye(B, device=z_s.device, dtype=torch.bool)
        ss = ss.masked_fill(mask, -1e9); st = st.masked_fill(mask, -1e9)
        return -(F.softmax(st, dim=1) * F.log_softmax(ss, dim=1)).sum(1).mean()
    if method == "disco":
        # cos-loss + InfoNCE
        tau = hp.get("tau_disco", 0.2)
        zs = F.normalize(z_s, dim=1); zt = F.normalize(z_t, dim=1)
        cos_l = 1.0 - (zs * zt).sum(1).mean()
        logits = zs @ zt.T / tau
        labels = torch.arange(z_s.shape[0], device=z_s.device)
        nce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        return cos_l + nce
    if method == "norm":
        # L2-normalised feature MSE + L2-normalised centred feature MSE
        zs1 = F.normalize(z_s, dim=1); zt1 = F.normalize(z_t, dim=1)
        zs2 = F.normalize(z_s - z_s.mean(0, keepdim=True), dim=1)
        zt2 = F.normalize(z_t - z_t.mean(0, keepdim=True), dim=1)
        return F.mse_loss(zs1, zt1) + F.mse_loss(zs2, zt2)
    if method == "lower_bound":
        # never trained; sentinel — caller skips training
        return torch.tensor(0.0, device=z_s.device)
    raise ValueError(method)

# ─────────────────────────── training ─────────────────────────────────────
def train_one(method: str, student_cfg: str, seed: int, X: np.ndarray, Z: np.ndarray,
              V_k: torch.Tensor, lam: float, device: str, epochs: int = KD_EPOCHS,
              lr: float = KD_LR, bs: int = KD_BS):
    full, feat_dim = build_student(student_cfg, seed)
    full.to(device)
    n_params = sum(p.numel() for p in full.parameters())
    print(f"[student] {student_cfg} feat={feat_dim} params={n_params/1e6:.2f}M  method={method}")

    if method == "lower_bound":
        full.eval()
        return full, feat_dim

    V_k_d = V_k.to(device)
    Xt = torch.from_numpy(X).float()
    Zt = torch.from_numpy(Z).float()
    ld = DataLoader(TensorDataset(Xt, Zt), batch_size=bs, shuffle=True,
                    drop_last=True, num_workers=0, pin_memory=True)

    opt = torch.optim.AdamW(full.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    full.train()
    for ep in range(epochs):
        tot, n = 0., 0
        for xb, zb in ld:
            xb = xb.to(device, non_blocking=True); zb = zb.to(device, non_blocking=True)
            zs = full(xb)
            loss = kd_loss(zs, zb, method, V_k=V_k_d, lam=lam)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(full.parameters(), 5.0); opt.step()
            tot += float(loss); n += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"   ep{ep:02d} loss={tot/max(n,1):.5f}", flush=True)
    return full, feat_dim

# ─────────────────────────── PTB-XL eval ──────────────────────────────────
def load_task(task: str):
    for name, kind, root, lab_fn, valid_fn in TASKS_13:
        if name == task:
            sigs = np.load(f"{root}/signals_500hz.npy", mmap_mode="r")
            fold = np.load(f"{root}/strat_fold.npy")
            y = np.load(f"{root}/{lab_fn}")
            if y.ndim > 1: y = y.argmax(axis=1)
            if valid_fn is not None:
                valid = np.load(f"{root}/{valid_fn}").astype(bool)
            else:
                valid = np.isfinite(y) if kind == "reg" else (y >= 0)
            # drop records whose signals contain NaN/Inf (chapman has 97 such)
            sig_valid_path = f"{root}/_sig_valid.npy"
            if os.path.exists(sig_valid_path):
                sig_valid = np.load(sig_valid_path).astype(bool)
            else:
                print(f"[load_task] computing signal-finite mask for {root} ...")
                sig_valid = np.zeros(len(sigs), dtype=bool)
                for i in range(0, len(sigs), 1024):
                    chunk = np.asarray(sigs[i:i+1024], dtype=np.float32)
                    sig_valid[i:i+len(chunk)] = np.isfinite(chunk).reshape(len(chunk), -1).all(axis=1)
                np.save(sig_valid_path, sig_valid)
                print(f"[load_task] {root}: {sig_valid.sum()}/{len(sigs)} finite")
            valid = valid & sig_valid
            assert len(sigs) == len(fold) == len(y), f"{task}: shape mismatch"
            return sigs, y, fold, valid, kind
    raise ValueError(f"unknown task: {task}")

def load_ptbxl_task(task: str):  # back-compat
    sigs, y, fold, valid, _ = load_task(task)
    return sigs, y, fold, valid

def split_by_fold(idx_pool: np.ndarray, fold: np.ndarray):
    tr = idx_pool[np.isin(fold[idx_pool], list(range(1, 9)))]
    va = idx_pool[fold[idx_pool] == 9]
    te = idx_pool[fold[idx_pool] == 10]
    return tr, va, te

@torch.no_grad()
def encode_indices(encoder, sigs, idx, device, bs=16):
    out = []
    encoder.eval()
    for i in range(0, len(idx), bs):
        chunk = idx[i:i+bs]
        x = zscore_batch(np.asarray(sigs[chunk], dtype=np.float32))
        x = torch.from_numpy(x).to(device)
        z = encoder(x)
        if z.ndim == 3: z = z.mean(-1)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, 0)

def cls_probe(X_tr, y_tr, X_va, y_va, X_te, y_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, roc_auc_score
    sc = StandardScaler().fit(np.concatenate([X_tr, X_va]))
    X_tv = sc.transform(np.concatenate([X_tr, X_va]))
    y_tv = np.concatenate([y_tr, y_va])
    best, best_C = -1, 1.0
    for C in [0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=300, n_jobs=4).fit(X_tv, y_tv)
        v = f1_score(y_va, clf.predict(sc.transform(X_va)), average="macro", zero_division=0)
        if v > best: best, best_C = v, C
    clf = LogisticRegression(C=best_C, max_iter=300, n_jobs=4).fit(X_tv, y_tv)
    Xte_s = sc.transform(X_te); pred = clf.predict(Xte_s)
    f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))
    try:
        proba = clf.predict_proba(Xte_s)
        if proba.shape[1] == 2:
            auroc = float(roc_auc_score(y_te, proba[:, 1]))
        else:
            auroc = float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro"))
    except Exception:
        auroc = float("nan")
    return f1, auroc

def reg_probe(X_tr, y_tr, X_va, y_va, X_te, y_te):
    """Ridge probe; returns (mae, neg_mae) so retention math (higher=better)
    can use auroc_* uniformly across cls and reg tasks (negmae_s/negmae_t)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    sc = StandardScaler().fit(np.concatenate([X_tr, X_va]))
    X_tv = sc.transform(np.concatenate([X_tr, X_va]))
    y_tv = np.concatenate([y_tr, y_va])
    best, best_a = -np.inf, 1.0
    for a in [0.1, 1.0, 10.0]:
        clf = Ridge(alpha=a).fit(X_tv, y_tv)
        v = -mean_absolute_error(y_va, clf.predict(sc.transform(X_va)))
        if v > best: best, best_a = v, a
    clf = Ridge(alpha=best_a).fit(X_tv, y_tv)
    mae = float(mean_absolute_error(y_te, clf.predict(sc.transform(X_te))))
    return mae, -mae

def evaluate_encoder(encoder, device, prefix="", task_list=None):
    if task_list is None:
        task_list = [t[0] for t in TASKS_13]
    res = {}
    for task in task_list:
        sigs, y, fold, valid, kind = load_task(task)
        idx_pool = np.where(valid)[0]
        tr_i, va_i, te_i = split_by_fold(idx_pool, fold)
        if min(len(tr_i), len(va_i), len(te_i)) == 0:
            print(f"  [eval]{prefix}  {task:18s}  SKIP (empty split)", flush=True)
            continue
        E_tr = encode_indices(encoder, sigs, tr_i, device)
        E_va = encode_indices(encoder, sigs, va_i, device)
        E_te = encode_indices(encoder, sigs, te_i, device)
        if kind == "reg":
            mae, neg_mae = reg_probe(E_tr, y[tr_i], E_va, y[va_i], E_te, y[te_i])
            res["f1_" + task]    = mae       # primary stored under f1_*
            res["auroc_" + task] = neg_mae   # higher-better mirror for ret
            print(f"  [eval]{prefix}  {task:18s}  MAE={mae:.4f}  (n_tr={len(tr_i)})", flush=True)
        else:
            f1, auroc = cls_probe(E_tr, y[tr_i], E_va, y[va_i], E_te, y[te_i])
            res["f1_" + task] = f1
            res["auroc_" + task] = auroc
            print(f"  [eval]{prefix}  {task:18s}  f1={f1:.4f}  AUROC={auroc:.4f}  (n_tr={len(tr_i)})", flush=True)
    return res

def evaluate_teacher(teacher, device):
    """Wrap teacher to a uniform .__call__ interface for evaluate_encoder."""
    class W(nn.Module):
        def __init__(self, t): super().__init__(); self.t = t
        def forward(self, x): return self.t(x)
        def eval(self): return self
    return evaluate_encoder(W(teacher), device, prefix=" [teacher]")


def method_completed(existing_df: pd.DataFrame, ckpt_dir: Path, method: str, seed: int) -> bool:
    if existing_df.empty:
        return False
    has_row = ((existing_df["method"] == method) & (existing_df["seed"] == seed)).any()
    ckpt_ok = (ckpt_dir / f"{method}_seed{seed}.pt").exists()
    return bool(has_row and ckpt_ok)

# ─────────────────────────── main ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="BF8_NB7", choices=list(STUDENT_CFGS))
    ap.add_argument("--methods", nargs="+",
                    default=["full_mse", "subspace_mse", "ns_kd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--pool-n", type=int, default=5000)
    ap.add_argument("--energy-threshold", type=float, default=ENERGY_THRESHOLD)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--exp-suffix", default="")
    ap.add_argument("--csv-name", default="ns_kd_results.csv",
                    help="filename of results CSV (use distinct names when running "
                         "multiple processes against the same out-dir in parallel).")
    ap.add_argument("--out-dir",
                    default="/home/xiaoma/foundation_model/pulseppg/kd_sweep_results/ecg_preliminary")
    ap.add_argument("--ckpt-subdir", default="checkpoints",
                    help="checkpoint subdir under each experiment dir")
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.out_dir) / f"exp_ecgfounder_{args.student}{args.exp_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / args.ckpt_subdir
    print(f"[exp_dir] {out_dir}")

    teacher = ECGFounderTeacher(n_leads=12, device=device)
    print(f"[teacher] ECGFounder loaded: {teacher.n_params()/1e6:.2f}M params, D={TEACHER_DIM}")

    # Pool + teacher embeddings (cached)
    X, Z = build_pool(DEFAULT_POOL, teacher, n=args.pool_n, seed=0, cache_dir=out_dir)

    # Adaptive K
    k, V_k = adaptive_k(Z, args.energy_threshold)

    # Teacher baseline (cached)
    tb_path = out_dir / "teacher_baseline.json"
    if tb_path.exists():
        teacher_res = json.loads(tb_path.read_text())
        print(f"[teacher_baseline] cached: {teacher_res}")
    else:
        print(f"\n=== teacher baseline (ECGFounder linear probe) ===")
        teacher_res = evaluate_teacher(teacher, device)
        tb_path.write_text(json.dumps(teacher_res, indent=2))

    # KD sweep
    rows = []
    csv_path = out_dir / args.csv_name
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        rows = existing.to_dict("records")
        print(f"[resume] {len(rows)} existing rows")
    else:
        existing = pd.DataFrame()

    for method in args.methods:
        for seed in args.seeds:
            mname = method
            if method_completed(existing, ckpt_dir, mname, seed):
                print(f"[skip] {mname} seed{seed}"); continue
            print(f"\n=== {mname}  seed={seed}  k={k} ===")
            t0 = time.time()
            full, feat_dim = train_one(method, args.student, seed, X, Z, V_k=V_k,
                                       lam=args.lam, device=device, epochs=args.epochs)
            dt_train = time.time() - t0
            print(f"  trained in {dt_train:.0f}s; evaluating...")
            eval_res = evaluate_encoder(full, device, prefix=f" [{mname} s{seed}]")
            ckpt_path = ckpt_dir / f"{mname}_seed{seed}.pt"
            save_student_ckpt(
                full, ckpt_path,
                student_name=args.student,
                method=mname,
                seed=seed,
                k=k,
                lam=args.lam,
                feat_dim=feat_dim,
                student_cfg=STUDENT_CFGS[args.student],
                teacher_dim=TEACHER_DIM,
            )
            print(f"  checkpoint -> {ckpt_path}")
            row = {"method": mname, "seed": seed, "k": k, "lam": args.lam,
                   "student": args.student, "train_s": dt_train, **eval_res}
            for tk, tv in teacher_res.items():
                if tk.startswith("auroc_"):
                    sk = tk.replace("auroc_", "ret_auroc_")
                elif tk.startswith("f1_"):
                    sk = tk.replace("f1_", "ret_f1_")
                else:
                    continue
                sv = eval_res.get(tk, float("nan"))
                if not np.isnan(sv) and tv > 0:
                    row[sk] = sv / tv
            auroc_rets = [v for kk, v in row.items()
                          if kk.startswith("ret_auroc_") and isinstance(v, float) and not np.isnan(v)]
            row["avg_ret"]   = float(np.mean(auroc_rets)) if auroc_rets else float("nan")
            row["floor_ret"] = float(np.min(auroc_rets)) if auroc_rets else float("nan")
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"  -> avg_ret(AUROC)={row['avg_ret']:.4f}  floor_ret={row['floor_ret']:.4f}")
            del full; torch.cuda.empty_cache(); gc.collect()

    print(f"\n[done] {len(rows)} rows -> {csv_path}")
    print(pd.DataFrame(rows)[["method","seed","k","avg_ret","floor_ret"]].to_string(index=False))

if __name__ == "__main__":
    main()

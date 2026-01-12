import os
import random
import zipfile
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import h5py
import pandas as pd
from tqdm import tqdm
import json
import warnings


DEVICE = torch.device(os.getenv("DEVICE", "cpu"))
print("Device:", DEVICE)

# === USER INPUT: Set dataset roots (either edit here OR export environment variables) ===
TRAIN_ROOT = os.getenv("TRAIN_ROOT", "./data/train")          # <-- USER INPUT
PRETEST_ROOT = os.getenv("PRETEST_ROOT", "./data/pretest")    # <-- USER INPUT
FINAL_EVAL_ROOT = os.getenv("FINAL_EVAL_ROOT", "./data/final")  # <-- USER INPUT
FINAL_RESULT_DIR = os.getenv("FINAL_RESULT_DIR", "./outputs/Result")  # <-- USER INPUT

STATS_CACHE_PATH = os.getenv("STATS_CACHE_PATH", "./artifacts/global_norm_stats.json")  # <-- USER INPUT (optional)
FORCE_RECOMPUTE_STATS = os.getenv("FORCE_RECOMPUTE_STATS", "0").strip() == "1"          # <-- USER INPUT (optional)

TRAIN_MATERIALS = ["3C90", "3C94", "3E6", "3F4", "77", "78", "N27", "N30", "N49", "N87"]
PRETEST_MATERIALS = ["3C90", "3C94", "3E6", "3F4", "77", "78", "N27", "N30", "N49", "N87", "N87-2"]
MATERIAL_ALIAS = {"N87-2": "N87"}

FREQS = [1, 2, 3, 4, 5, 6, 7]
FREQ_TO_IDX = {f: i for i, f in enumerate(FREQS)}

SEQ_LEN = int(os.getenv("SEQ_LEN", "80"))
STRIDE = int(os.getenv("STRIDE", "20"))

MAX_WINDOWS_PER_TRIPLET = int(os.getenv("MAX_WINDOWS_PER_TRIPLET", "2500"))
SAMPLE_WINDOWS_PER_TRIPLET = int(os.getenv("SAMPLE_WINDOWS_PER_TRIPLET", "900"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))
BASE_LR = float(os.getenv("BASE_LR", "2e-3"))
LAMBDA_ENE = float(os.getenv("LAMBDA_ENE", "0.20"))

WEIGHT_3F4 = float(os.getenv("WEIGHT_3F4", "3.0"))
WEIGHT_N49 = float(os.getenv("WEIGHT_N49", "1.4"))

MAX_WINDOWS_EVAL = int(os.getenv("MAX_WINDOWS_EVAL", "3000"))
METRIC_EPS = 1e-12

REQUIRED_COLUMNS = [
    "material", "segment", "n_windows",
    "rmse", "rel_err", "err_seq", "err_ene",
    "file_used"
]

# === USER INPUT: Set checkpoint paths (either edit here OR export environment variables) ===
START_CHECKPOINT = os.getenv("START_CHECKPOINT", "./artifacts/baseline_model.pt")  # <-- USER INPUT
FINETUNE_OUT = os.getenv("FINETUNE_OUT", "./artifacts/finetuned_model.pt")         # <-- USER INPUT

EXTRA_EPOCHS = int(os.getenv("EXTRA_EPOCHS", "2"))
LR_DECAY_MULT = float(os.getenv("LR_DECAY_MULT", "0.3"))

FINAL_MATERIALS = ["A", "B", "C", "D", "E"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_1d_csv(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) == 0:
            return None
    except OSError:
        return None
    try:
        arr = pd.read_csv(path, header=None).values.squeeze()
    except Exception:
        return None
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return arr if arr.size > 0 else None


def save_stats(stats: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved global normalization stats -> {path}")


def load_stats(path: str) -> dict:
    with open(path, "r") as f:
        stats = json.load(f)
    print(f"Loaded global normalization stats <- {path}")
    return stats


def compute_global_normalisation(root: str, materials: List[str], freqs: List[int]) -> dict:
    all_B, all_H, all_dB = [], [], []

    print("Computing global normalization stats...")
    for mat in tqdm(materials, desc="Training materials"):
        mat_dir = os.path.join(root, mat)
        if not os.path.isdir(mat_dir):
            continue
        for f in freqs:
            B = load_1d_csv(os.path.join(mat_dir, f"{mat}_{f}_B.csv"))
            H = load_1d_csv(os.path.join(mat_dir, f"{mat}_{f}_H.csv"))
            if B is None or H is None:
                continue
            n = min(len(B), len(H))
            B = B[:n]
            H = H[:n]
            dBdt = np.gradient(B).astype(np.float32)

            all_B.append(B)
            all_H.append(H)
            all_dB.append(dBdt)

    if not all_B:
        raise RuntimeError("No training CSVs found to compute normalization stats. Check TRAIN_ROOT.")

    Bc = np.concatenate(all_B)
    Hc = np.concatenate(all_H)
    dBc = np.concatenate(all_dB)

    return {
        "B_mean": float(Bc.mean()),
        "B_std": float(Bc.std() + 1e-8),
        "H_mean": float(Hc.mean()),
        "H_std": float(Hc.std() + 1e-8),
        "dB_mean": float(dBc.mean()),
        "dB_std": float(dBc.std() + 1e-8),
    }


def _find_BH_keys(keys: List[str]) -> Tuple[Optional[str], Optional[str]]:
    low = {k: k.lower() for k in keys}
    b_pref = ["b", "b_seq", "bwave", "b_wave", "bfield", "bfield_seq", "b_field"]
    h_pref = ["h", "h_seq", "hwave", "h_wave", "hfield", "hfield_seq", "h_field"]

    bkey = None
    hkey = None

    for bp in b_pref:
        for k in keys:
            if low[k] == bp:
                bkey = k
                break
        if bkey is not None:
            break

    for hp in h_pref:
        for k in keys:
            if low[k] == hp:
                hkey = k
                break
        if hkey is not None:
            break

    if bkey is None:
        b_cand = [k for k in keys if ("b" in low[k]) and ("db" not in low[k])]
        bkey = sorted(b_cand)[0] if b_cand else None
    if hkey is None:
        h_cand = [k for k in keys if ("h" in low[k]) and ("dh" not in low[k])]
        hkey = sorted(h_cand)[0] if h_cand else None

    return bkey, hkey


def iter_segments_from_h5(f: h5py.File) -> List[Tuple[Union[str, int], np.ndarray, np.ndarray]]:
    segs: List[Tuple[Union[str, int], np.ndarray, np.ndarray]] = []
    keys = list(f.keys())

    group_names = [k for k in keys if isinstance(f[k], h5py.Group)]
    if group_names:
        for gname in sorted(group_names):
            grp = f[gname]
            gkeys = list(grp.keys())
            bkey, hkey = _find_BH_keys(gkeys)
            if bkey is None or hkey is None:
                continue
            B = np.asarray(grp[bkey][...], dtype=np.float32).reshape(-1)
            H = np.asarray(grp[hkey][...], dtype=np.float32).reshape(-1)
            segs.append((gname, B, H))
        return segs

    bkey, hkey = _find_BH_keys(keys)
    if (
        bkey is not None
        and hkey is not None
        and isinstance(f[bkey], h5py.Dataset)
        and isinstance(f[hkey], h5py.Dataset)
    ):
        B_ds = f[bkey]
        H_ds = f[hkey]
        if B_ds.ndim == 1 and H_ds.ndim == 1:
            segs.append(
                (
                    "root",
                    np.asarray(B_ds[...], dtype=np.float32).reshape(-1),
                    np.asarray(H_ds[...], dtype=np.float32).reshape(-1),
                )
            )
            return segs
        if B_ds.ndim == 2 and H_ds.ndim == 2:
            n_seg = B_ds.shape[0]
            for i in range(n_seg):
                segs.append(
                    (
                        int(i),
                        np.asarray(B_ds[i], dtype=np.float32).reshape(-1),
                        np.asarray(H_ds[i], dtype=np.float32).reshape(-1),
                    )
                )
            return segs

    ds_names = [k for k in keys if isinstance(f[k], h5py.Dataset)]
    for name in sorted(ds_names):
        ds = f[name]
        arr = np.asarray(ds[...], dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[1] == 2:
                segs.append((name, arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)))
            elif arr.shape[0] == 2:
                segs.append((name, arr[0, :].reshape(-1), arr[1, :].reshape(-1)))

    return segs


def make_windows_deltaH_training_energyWIN(
    B: np.ndarray,
    H: np.ndarray,
    stats: dict,
    max_windows: int,
    sample_windows: Optional[int],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    B = np.asarray(B, dtype=np.float32).reshape(-1)
    H = np.asarray(H, dtype=np.float32).reshape(-1)
    n = min(len(B), len(H))
    B, H = B[:n], H[:n]
    if n <= SEQ_LEN + 1:
        return None

    dBdt = np.gradient(B).astype(np.float32)

    Bn = (B - stats["B_mean"]) / stats["B_std"]
    Hn = (H - stats["H_mean"]) / stats["H_std"]
    dBn = (dBdt - stats["dB_mean"]) / stats["dB_std"]

    X, y, Hwin, dBwin, Hprev = [], [], [], [], []
    count = 0
    for start in range(0, n - SEQ_LEN - 1, STRIDE):
        end = start + SEQ_LEN
        X.append(np.stack([Bn[start:end], Hn[start:end], dBn[start:end]], axis=-1))
        y.append(Hn[end] - Hn[end - 1])
        Hwin.append(H[start:end])
        dBwin.append(dBdt[start:end])
        Hprev.append(H[end - 1])

        count += 1
        if max_windows is not None and count >= max_windows:
            break

    if not X:
        return None

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    Hwin = np.asarray(Hwin, dtype=np.float32)
    dBwin = np.asarray(dBwin, dtype=np.float32)
    Hprev = np.asarray(Hprev, dtype=np.float32)

    if sample_windows is not None and len(X) > sample_windows:
        idx = np.random.choice(len(X), size=sample_windows, replace=False)
        X, y, Hwin, dBwin, Hprev = X[idx], y[idx], Hwin[idx], dBwin[idx], Hprev[idx]

    return X, y, Hwin, dBwin, Hprev


def build_eval_windows_and_energy(
    B: np.ndarray,
    H: np.ndarray,
    stats: dict,
    max_windows: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    B = np.asarray(B, dtype=np.float32).reshape(-1)
    H = np.asarray(H, dtype=np.float32).reshape(-1)
    n = min(len(B), len(H))
    B, H = B[:n], H[:n]
    if n <= SEQ_LEN + 1:
        return None

    dBdt = np.gradient(B).astype(np.float32)
    E_total = float(np.sum(dBdt * H))

    Bn = (B - stats["B_mean"]) / stats["B_std"]
    Hn = (H - stats["H_mean"]) / stats["H_std"]
    dBn = (dBdt - stats["dB_mean"]) / stats["dB_std"]

    X, y_delta, H_prev_phys, dB_targets = [], [], [], []
    count = 0
    for start in range(0, n - SEQ_LEN - 1, STRIDE):
        end = start + SEQ_LEN
        X.append(np.stack([Bn[start:end], Hn[start:end], dBn[start:end]], axis=-1))
        y_delta.append(Hn[end] - Hn[end - 1])
        H_prev_phys.append(H[end - 1])
        dB_targets.append(dBdt[end])
        count += 1
        if max_windows is not None and count >= max_windows:
            break

    if not X:
        return None

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y_delta, dtype=np.float32),
        np.asarray(H_prev_phys, dtype=np.float32),
        np.asarray(dB_targets, dtype=np.float32),
        E_total,
    )


class GlobalLSTM(nn.Module):
    def __init__(
        self,
        n_materials: int,
        n_freqs: int,
        input_dim: int = 3,
        hidden_dim: int = 32,
        mat_emb_dim: int = 4,
        freq_emb_dim: int = 2,
        num_layers: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.mat_emb = nn.Embedding(n_materials, mat_emb_dim)
        self.freq_emb = nn.Embedding(n_freqs, freq_emb_dim)
        self.fc = nn.Linear(hidden_dim + mat_emb_dim + freq_emb_dim, 1)

    def forward(self, x, mat_idx, freq_idx):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        feat = torch.cat([last, self.mat_emb(mat_idx), self.freq_emb(freq_idx)], dim=-1)
        return self.fc(feat).squeeze(-1)


def _material_weights_tensor(mat_idx: torch.Tensor) -> torch.Tensor:
    w = torch.ones_like(mat_idx, dtype=torch.float32)
    idx_3f4 = TRAIN_MATERIALS.index("3F4")
    idx_n49 = TRAIN_MATERIALS.index("N49")
    w = torch.where(mat_idx == idx_3f4, torch.tensor(WEIGHT_3F4, dtype=torch.float32), w)
    w = torch.where(mat_idx == idx_n49, torch.tensor(WEIGHT_N49, dtype=torch.float32), w)
    return w


def build_training_dataset(stats: dict):
    print("Building training dataset...")

    X_list, y_list, mat_list, freq_list, Hwin_list, dBwin_list, Hprev_list = [], [], [], [], [], [], []

    for mat in tqdm(TRAIN_MATERIALS, desc="Dataset materials"):
        mat_dir = os.path.join(TRAIN_ROOT, mat)
        if not os.path.isdir(mat_dir):
            continue
        mat_idx = TRAIN_MATERIALS.index(mat)

        for f in FREQS:
            B_path = os.path.join(mat_dir, f"{mat}_{f}_B.csv")
            H_path = os.path.join(mat_dir, f"{mat}_{f}_H.csv")
            B = load_1d_csv(B_path)
            H = load_1d_csv(H_path)
            if B is None or H is None:
                continue

            out = make_windows_deltaH_training_energyWIN(
                B, H, stats,
                max_windows=MAX_WINDOWS_PER_TRIPLET,
                sample_windows=SAMPLE_WINDOWS_PER_TRIPLET
            )
            if out is None:
                continue

            X, y, Hwin, dBwin, Hprev = out
            X_list.append(X)
            y_list.append(y)
            Hwin_list.append(Hwin)
            dBwin_list.append(dBwin)
            Hprev_list.append(Hprev)

            mat_list.append(np.full((len(X),), mat_idx, dtype=np.int64))
            freq_list.append(np.full((len(X),), FREQ_TO_IDX[f], dtype=np.int64))

    if not X_list:
        raise RuntimeError("No training windows were created. Check TRAIN_ROOT CSV structure.")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    Hwin_all = np.concatenate(Hwin_list, axis=0)
    dBwin_all = np.concatenate(dBwin_list, axis=0)
    Hprev_all = np.concatenate(Hprev_list, axis=0)
    mat_all = np.concatenate(mat_list, axis=0)
    freq_all = np.concatenate(freq_list, axis=0)

    print(f"Training windows: {len(X_all):,}")

    X_t = torch.from_numpy(X_all).to(DEVICE)
    y_t = torch.from_numpy(y_all).to(DEVICE)
    Hwin_t = torch.from_numpy(Hwin_all).to(DEVICE)
    dBwin_t = torch.from_numpy(dBwin_all).to(DEVICE)
    Hprev_t = torch.from_numpy(Hprev_all).to(DEVICE)
    mat_t = torch.from_numpy(mat_all).to(DEVICE)
    freq_t = torch.from_numpy(freq_all).to(DEVICE)

    idx = torch.randperm(len(X_t))
    return (X_t[idx], y_t[idx], Hwin_t[idx], dBwin_t[idx], Hprev_t[idx], mat_t[idx], freq_t[idx])


def finetune_from_checkpoint(stats: dict, checkpoint_path: str, out_path: str) -> str:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    X_t, y_t, Hwin_t, dBwin_t, Hprev_t, mat_t, freq_t = build_training_dataset(stats)

    model = GlobalLSTM(n_materials=len(TRAIN_MATERIALS), n_freqs=len(FREQS)).to(DEVICE)
    try:
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.train()

    finetune_lr = BASE_LR * LR_DECAY_MULT
    opt = torch.optim.Adam(model.parameters(), lr=finetune_lr)

    n = len(X_t)
    steps_per_epoch = (n + BATCH_SIZE - 1) // BATCH_SIZE
    H_std = float(stats["H_std"])

    print(f"Fine-tuning from {checkpoint_path}")
    print(f"LR: {BASE_LR} -> {finetune_lr} | Extra epochs: {EXTRA_EPOCHS}")

    for ep in range(1, EXTRA_EPOCHS + 1):
        running = 0.0
        pbar = tqdm(range(steps_per_epoch), desc=f"FT Epoch {ep}/{EXTRA_EPOCHS}", leave=False)
        for step in pbar:
            s = step * BATCH_SIZE
            e = min(n, (step + 1) * BATCH_SIZE)

            xb = X_t[s:e]
            yb = y_t[s:e]
            Hwinb = Hwin_t[s:e]
            dBwinb = dBwin_t[s:e]
            Hprevb = Hprev_t[s:e]
            mb = mat_t[s:e]
            fb = freq_t[s:e]

            w = _material_weights_tensor(mb).to(DEVICE)

            opt.zero_grad()
            pred_delta_norm = model(xb, mb, fb)

            loss_dh = (pred_delta_norm - yb) ** 2

            delta_pred_phys = pred_delta_norm * H_std
            H_end_pred = Hprevb + delta_pred_phys

            E_true = (dBwinb * Hwinb).sum(dim=1)
            Hwin_pred = Hwinb.clone()
            Hwin_pred[:, -1] = H_end_pred
            E_pred = (dBwinb * Hwin_pred).sum(dim=1)

            denom = (E_true.detach() ** 2).mean().clamp_min(1e-6)
            loss_ene = ((E_pred - E_true) ** 2) / denom

            loss = (w * (loss_dh + LAMBDA_ENE * loss_ene)).mean()

            loss.backward()
            opt.step()

            running += float(loss.item())
            pbar.set_postfix(loss=running / (step + 1))

        print(f"FT Epoch {ep}: avg_loss={running / steps_per_epoch:.6f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved fine-tuned model -> {out_path}")
    return out_path


def eval_pretest(stats: dict, model_path: str) -> None:
    model = GlobalLSTM(n_materials=len(TRAIN_MATERIALS), n_freqs=len(FREQS)).to(DEVICE)

    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=DEVICE)

    model.load_state_dict(state)
    model.eval()

    all_rows: List[Dict[str, Any]] = []

    for mat in tqdm(PRETEST_MATERIALS, desc="Pretest materials"):
        mat_lookup = MATERIAL_ALIAS.get(mat, mat)
        if mat_lookup not in TRAIN_MATERIALS:
            continue
        mat_idx_val = TRAIN_MATERIALS.index(mat_lookup)

        true_path = os.path.join(PRETEST_ROOT, mat, f"{mat}_Testing_True.h5")
        pad_path = os.path.join(PRETEST_ROOT, mat, f"{mat}_Testing_Padded.h5")
        h5_path = true_path if os.path.exists(true_path) else pad_path
        if not os.path.exists(h5_path):
            continue

        with h5py.File(h5_path, "r") as f:
            segs = iter_segments_from_h5(f)
            if not segs:
                print(f"[{mat}] No readable segments in {os.path.basename(h5_path)}")
                continue

            for seg_id, B, H in tqdm(segs, desc=f"{mat} segments", leave=False):
                out = build_eval_windows_and_energy(B, H, stats, max_windows=MAX_WINDOWS_EVAL)
                if out is None:
                    continue
                X, y_delta, H_prev_phys, dB_targets, E_total = out

                X_t = torch.from_numpy(X).to(DEVICE)
                mat_idx = torch.full((len(X_t),), mat_idx_val, dtype=torch.long, device=DEVICE)

                preds = []
                with torch.no_grad():
                    for fi in range(len(FREQS)):
                        freq_idx = torch.full((len(X_t),), fi, dtype=torch.long, device=DEVICE)
                        preds.append(model(X_t, mat_idx, freq_idx).cpu().numpy())

                delta_pred_norm = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)

                delta_pred_phys = delta_pred_norm * stats["H_std"]
                H_pred = H_prev_phys + delta_pred_phys

                delta_true_phys = y_delta * stats["H_std"]
                H_true = H_prev_phys + delta_true_phys

                rmse = float(np.sqrt(np.mean((H_pred - H_true) ** 2)))

                rms = float(np.sqrt(np.mean(H_true ** 2)))
                rel_err = float(rmse / (rms + METRIC_EPS) * 100.0)

                mae = float(np.mean(np.abs(H_pred - H_true)))
                err_seq = float(mae / (rms + METRIC_EPS) * 100.0)

                E_pred = float(np.sum(dB_targets * H_pred))
                E_true = float(np.sum(dB_targets * H_true))
                err_ene = float(abs(E_pred - E_true) / (abs(E_total) + METRIC_EPS) * 100.0)

                row = {
                    "material": str(mat),
                    "segment": seg_id if isinstance(seg_id, (int, str)) else str(seg_id),
                    "n_windows": int(len(X)),
                    "rmse": float(rmse),
                    "rel_err": float(rel_err),
                    "err_seq": float(err_seq),
                    "err_ene": float(err_ene),
                    "file_used": str(os.path.basename(h5_path)),
                }
                row = {k: row[k] for k in REQUIRED_COLUMNS}
                all_rows.append(row)

    if not all_rows:
        print("No results produced. Check PRETEST_ROOT and file layouts.")
        return

    df = pd.DataFrame(all_rows)[REQUIRED_COLUMNS]

    metric_cols = ["rmse", "rel_err", "err_seq", "err_ene"]
    finite_mask = np.isfinite(df[metric_cols].to_numpy()).all(axis=1)
    bad = int((~finite_mask).sum())
    if bad > 0:
        warnings.warn(f"Skipped {bad} row(s) with NaN/Inf in metrics for summary aggregation.")
    df_valid = df.loc[finite_mask].copy()

    summary = (
        df_valid
        .groupby("material", as_index=False)[metric_cols]
        .agg("mean")
        .sort_values("material")
        .reset_index(drop=True)
    )

    assert list(df.columns) == REQUIRED_COLUMNS, f"Column order mismatch: {list(df.columns)}"
    print("Validation OK: per-segment DataFrame has required columns in correct order.")

    print("PRETEST SUMMARY (avg per material)")
    for r in summary.to_dict("records"):
        print(json.dumps(r, ensure_ascii=False))


def _load_model_for_inference(stats: dict, model_path: str) -> GlobalLSTM:
    model = GlobalLSTM(n_materials=len(TRAIN_MATERIALS), n_freqs=len(FREQS)).to(DEVICE)
    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _material_folder(letter: str) -> str:
    return f"Material {letter}"


def generate_final_submission(stats: dict, model_path: str, out_dir: str) -> None:
    model = _load_model_for_inference(stats, model_path)
    os.makedirs(out_dir, exist_ok=True)

    param_count = _count_parameters(model)
    model_bytes = os.path.getsize(model_path) if os.path.exists(model_path) else 0

    param_csv_path = os.path.join(out_dir, "Parameter_Size.csv")
    with open(param_csv_path, "w") as f:
        f.write("num_parameters,model_file_bytes\n")
        f.write(f"{param_count},{model_bytes}\n")

    testing_root = os.path.join(FINAL_EVAL_ROOT, "Testing")
    if not os.path.isdir(testing_root):
        raise FileNotFoundError(f"Final evaluation Testing folder not found: {testing_root}")

    B_mean = float(stats["B_mean"])
    B_std = float(stats["B_std"])
    H_mean = float(stats["H_mean"])
    H_std_norm = float(stats["H_std"])
    dB_mean = float(stats["dB_mean"])
    dB_std = float(stats["dB_std"])

    mat_idx_val = 0
    mat_idx = torch.full((1,), mat_idx_val, dtype=torch.long, device=DEVICE)

    for letter in FINAL_MATERIALS:
        mat_dir = os.path.join(testing_root, _material_folder(letter))
        padded_path = os.path.join(mat_dir, f"{letter}_Padded_H_seq.csv")
        b_path = os.path.join(mat_dir, f"{letter}_True_B_seq.csv")
        t_path = os.path.join(mat_dir, f"{letter}_True_T.csv")

        if not os.path.exists(padded_path):
            raise FileNotFoundError(f"Missing padded H file: {padded_path}")
        if not os.path.exists(b_path):
            raise FileNotFoundError(f"Missing B file: {b_path}")
        if not os.path.exists(t_path):
            raise FileNotFoundError(f"Missing T file: {t_path}")

        padded_H = pd.read_csv(padded_path, header=None)
        B_df = pd.read_csv(b_path, header=None)
        T_df = pd.read_csv(t_path, header=None)

        if padded_H.shape != B_df.shape:
            raise RuntimeError(f"Shape mismatch for material {letter}: H{padded_H.shape} vs B{B_df.shape}")
        if T_df.shape[0] != padded_H.shape[0]:
            raise RuntimeError(f"Row mismatch for material {letter}: T{T_df.shape} vs H{padded_H.shape}")

        H_out = padded_H.copy()

        for i in tqdm(range(H_out.shape[0]), desc=f"Final submit {letter}", leave=False):
            B_seq = B_df.iloc[i].to_numpy(dtype=np.float32).reshape(-1)
            H_seq = H_out.iloc[i].to_numpy(dtype=np.float32).reshape(-1)

            nan_mask = np.isnan(H_seq)
            if not nan_mask.any():
                continue

            H_seq_filled = H_seq.copy()
            first_nan = int(np.argmax(nan_mask))

            if first_nan == 0:
                H_seq_filled[0] = 0.0
                first_nan = 1

            dBdt_full = np.gradient(B_seq).astype(np.float32)

            while first_nan < len(H_seq_filled):
                start = max(0, first_nan - SEQ_LEN)
                end = start + SEQ_LEN
                if end > len(H_seq_filled):
                    break

                B_win = B_seq[start:end]
                H_win = H_seq_filled[start:end]

                if np.isnan(H_win).any():
                    break

                dB_win = dBdt_full[start:end]

                Bn = (B_win - B_mean) / B_std
                Hn = (H_win - H_mean) / H_std_norm
                dBn = (dB_win - dB_mean) / dB_std

                x = np.stack([Bn, Hn, dBn], axis=-1).astype(np.float32)
                x_t = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

                preds = []
                with torch.no_grad():
                    for fi in range(len(FREQS)):
                        freq_idx = torch.full((1,), fi, dtype=torch.long, device=DEVICE)
                        preds.append(float(model(x_t, mat_idx, freq_idx).cpu().item()))
                delta_pred_norm = float(np.mean(preds))

                H_prev_phys = float(H_win[-1])
                delta_pred_phys = delta_pred_norm * H_std_norm
                H_next = H_prev_phys + delta_pred_phys

                next_pos = end
                if next_pos >= len(H_seq_filled):
                    break

                if not np.isnan(H_seq_filled[next_pos]):
                    first_nan = next_pos + 1
                    continue

                H_seq_filled[next_pos] = H_next
                first_nan = next_pos + 1

            H_out.iloc[i] = H_seq_filled

        out_path = os.path.join(out_dir, f"{letter}_Pred_H_seq.csv")
        H_out.to_csv(out_path, header=False, index=False)

    zip_path = os.path.join(os.path.dirname(out_dir), "Result.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(out_dir)):
            if fn.endswith(".csv"):
                fullp = os.path.join(out_dir, fn)
                zf.write(fullp, arcname=os.path.join("Result", fn))

    print(f"Submission archive created: {zip_path}")
    print(f"Result folder: {out_dir}")


def main() -> None:
    set_seed(int(os.getenv("SEED", "42")))

    if FORCE_RECOMPUTE_STATS or not os.path.exists(STATS_CACHE_PATH):
        stats = compute_global_normalisation(TRAIN_ROOT, TRAIN_MATERIALS, FREQS)
        save_stats(stats, STATS_CACHE_PATH)
    else:
        stats = load_stats(STATS_CACHE_PATH)

    finetuned_path = finetune_from_checkpoint(stats, START_CHECKPOINT, FINETUNE_OUT)
    eval_pretest(stats, finetuned_path)
    generate_final_submission(stats, finetuned_path, FINAL_RESULT_DIR)


if __name__ == "__main__":
    main()

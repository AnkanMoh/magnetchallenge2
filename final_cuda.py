import os
import random
import json
import zipfile
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

TRAIN_ROOT = "/kaggle/input/magnet-dataset"
FINAL_EVAL_ROOT = "/kaggle/input/final-evaluation"
OUT_ROOT = "/kaggle/working"
STATS_CACHE_PATH = os.path.join(OUT_ROOT, "global_norm_stats.json")
RESULT_DIR = os.path.join(OUT_ROOT, "Result")

TRAIN_MATERIALS = ["3C90", "3C94", "3E6", "3F4", "77", "78", "N27", "N30", "N49", "N87"]
FREQS = [1,2,3,4,5,6,7]
FREQ_TO_IDX = {f:i for i,f in enumerate(FREQS)}
FINAL_MATERIALS = ["A","B","C","D","E"]

SEQ_LEN = 80
STRIDE = 20
MAX_WINDOWS_PER_TRIPLET = 2500
SAMPLE_WINDOWS_PER_TRIPLET = 900

BATCH_SIZE = 256
BASE_LR = 2e-3
LR_DECAY_MULT = 0.3
EXTRA_EPOCHS = 2
LAMBDA_ENE = 0.20

WEIGHT_3F4 = 3.0
WEIGHT_N49 = 1.4

TRAIN_FROM_SCRATCH_IF_NO_CKPT = True
SCRATCH_EPOCHS = 10
SCRATCH_OUT = os.path.join(OUT_ROOT, "base_model.pt")

CHECKPOINT_BASENAME = "deltaH_global_lstm_energyWIN_ep10.pt"
FINETUNE_OUT = os.path.join(OUT_ROOT, "finetuned_model.pt")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_1d_csv(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        arr = pd.read_csv(path, header=None).values.squeeze()
    except Exception:
        return None
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return arr if arr.size > 0 else None

def save_stats(stats: dict, path: str):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved global normalization stats -> {path}")

def load_stats(path: str) -> dict:
    with open(path, "r") as f:
        stats = json.load(f)
    print(f"Loaded global normalization stats <- {path}")
    return stats

def resolve_checkpoint_path(basename: str) -> Optional[str]:
    p = os.path.join(OUT_ROOT, basename)
    if os.path.exists(p):
        return p
    for root, _, files in os.walk("/kaggle/input"):
        if basename in files:
            return os.path.join(root, basename)
    pts = []
    for base in [OUT_ROOT, "/kaggle/input"]:
        for root, _, files in os.walk(base):
            for fn in files:
                if fn.endswith(".pt"):
                    full = os.path.join(root, fn)
                    pts.append((os.path.getmtime(full), full))
    if pts:
        pts.sort(reverse=True)
        print("[ckpt] basename not found; using newest .pt:", pts[0][1])
        return pts[0][1]
    return None

def compute_global_normalisation(root, materials, freqs):
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
            B, H = B[:n], H[:n]
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

def make_windows_deltaH_training_energyWIN(B: np.ndarray, H: np.ndarray, stats: dict,
                                          max_windows: int, sample_windows: Optional[int]):
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

class GlobalLSTM(nn.Module):
    def __init__(self, n_materials: int, n_freqs: int,
                 input_dim: int = 3, hidden_dim: int = 32,
                 mat_emb_dim: int = 4, freq_emb_dim: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
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
    w = torch.where(mat_idx == idx_3f4, torch.tensor(WEIGHT_3F4, dtype=torch.float32, device=mat_idx.device), w)
    w = torch.where(mat_idx == idx_n49, torch.tensor(WEIGHT_N49, dtype=torch.float32, device=mat_idx.device), w)
    return w

def build_training_dataset(stats: dict):
    X_list, y_list, mat_list, freq_list, Hwin_list, dBwin_list, Hprev_list = [], [], [], [], [], [], []
    for mat in tqdm(TRAIN_MATERIALS, desc="Dataset materials"):
        mat_dir = os.path.join(TRAIN_ROOT, mat)
        if not os.path.isdir(mat_dir):
            continue
        mat_idx = TRAIN_MATERIALS.index(mat)
        for f in FREQS:
            B = load_1d_csv(os.path.join(mat_dir, f"{mat}_{f}_B.csv"))
            H = load_1d_csv(os.path.join(mat_dir, f"{mat}_{f}_H.csv"))
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
            X_list.append(X); y_list.append(y)
            Hwin_list.append(Hwin); dBwin_list.append(dBwin); Hprev_list.append(Hprev)
            mat_list.append(np.full((len(X),), mat_idx, dtype=np.int64))
            freq_list.append(np.full((len(X),), FREQ_TO_IDX[f], dtype=np.int64))
    if not X_list:
        raise RuntimeError("No training windows created. Check TRAIN_ROOT structure.")
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    Hwin_all = np.concatenate(Hwin_list, axis=0)
    dBwin_all = np.concatenate(dBwin_list, axis=0)
    Hprev_all = np.concatenate(Hprev_list, axis=0)
    mat_all = np.concatenate(mat_list, axis=0)
    freq_all = np.concatenate(freq_list, axis=0)
    idx = np.random.permutation(len(X_all))
    X_all = X_all[idx]; y_all = y_all[idx]
    Hwin_all = Hwin_all[idx]; dBwin_all = dBwin_all[idx]; Hprev_all = Hprev_all[idx]
    mat_all = mat_all[idx]; freq_all = freq_all[idx]
    X_t = torch.from_numpy(X_all).to(DEVICE)
    y_t = torch.from_numpy(y_all).to(DEVICE)
    Hwin_t = torch.from_numpy(Hwin_all).to(DEVICE)
    dBwin_t = torch.from_numpy(dBwin_all).to(DEVICE)
    Hprev_t = torch.from_numpy(Hprev_all).to(DEVICE)
    mat_t = torch.from_numpy(mat_all).to(DEVICE)
    freq_t = torch.from_numpy(freq_all).to(DEVICE)
    return X_t, y_t, Hwin_t, dBwin_t, Hprev_t, mat_t, freq_t

def train_from_scratch(stats: dict, out_path: str, epochs: int):
    print("[train] Training from scratch because no checkpoint was found.")
    X_t, y_t, Hwin_t, dBwin_t, Hprev_t, mat_t, freq_t = build_training_dataset(stats)
    model = GlobalLSTM(len(TRAIN_MATERIALS), len(FREQS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    H_std = float(stats["H_std"])
    n = len(X_t)
    steps = (n + BATCH_SIZE - 1) // BATCH_SIZE
    model.train()
    for ep in range(1, epochs + 1):
        running = 0.0
        pbar = tqdm(range(steps), desc=f"Scratch Epoch {ep}/{epochs}", leave=False)
        for step in pbar:
            s = step * BATCH_SIZE
            e = min(n, (step + 1) * BATCH_SIZE)
            xb = X_t[s:e]; yb = y_t[s:e]
            Hwinb = Hwin_t[s:e]; dBwinb = dBwin_t[s:e]; Hprevb = Hprev_t[s:e]
            mb = mat_t[s:e]; fb = freq_t[s:e]
            w = _material_weights_tensor(mb)
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
            pbar.set_postfix(loss=running/(step+1))
        print(f"[train] Epoch {ep}: avg_loss={running/steps:.6f}")
    torch.save(model.state_dict(), out_path)
    print("[train] Saved scratch checkpoint ->", out_path)
    return out_path

def finetune_from_checkpoint(stats: dict, checkpoint_path: str, out_path: str) -> str:
    X_t, y_t, Hwin_t, dBwin_t, Hprev_t, mat_t, freq_t = build_training_dataset(stats)
    model = GlobalLSTM(len(TRAIN_MATERIALS), len(FREQS)).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.train()
    finetune_lr = BASE_LR * LR_DECAY_MULT
    opt = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    n = len(X_t)
    steps = (n + BATCH_SIZE - 1) // BATCH_SIZE
    H_std = float(stats["H_std"])
    print(f"[finetune] From {checkpoint_path}")
    print(f"[finetune] LR {BASE_LR} -> {finetune_lr}, epochs={EXTRA_EPOCHS}")
    for ep in range(1, EXTRA_EPOCHS + 1):
        running = 0.0
        pbar = tqdm(range(steps), desc=f"FT Epoch {ep}/{EXTRA_EPOCHS}", leave=False)
        for step in pbar:
            s = step * BATCH_SIZE
            e = min(n, (step + 1) * BATCH_SIZE)
            xb = X_t[s:e]; yb = y_t[s:e]
            Hwinb = Hwin_t[s:e]; dBwinb = dBwin_t[s:e]; Hprevb = Hprev_t[s:e]
            mb = mat_t[s:e]; fb = freq_t[s:e]
            w = _material_weights_tensor(mb)
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
            pbar.set_postfix(loss=running/(step+1))
        print(f"[finetune] Epoch {ep}: avg_loss={running/steps:.6f}")
    torch.save(model.state_dict(), out_path)
    print("[finetune] Saved fine-tuned ->", out_path)
    return out_path

def resolve_testing_root(final_eval_root: str) -> str:
    candidates = []
    for root, dirs, files in os.walk(final_eval_root):
        if all(os.path.isdir(os.path.join(root, f"Material {c}")) for c in FINAL_MATERIALS):
            candidates.append(root)
    if candidates:
        candidates.sort(key=lambda x: len(x))
        return candidates[0]
    for root, dirs, _ in os.walk(final_eval_root):
        for d in dirs:
            if d == "Testing":
                return os.path.join(root, d)
    raise FileNotFoundError(f"Could not locate Testing root under: {final_eval_root}")

def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

def _interp_prefix(h: np.ndarray, end: int) -> np.ndarray:
    x = h[:end].copy()
    if end <= 1:
        return x
    nans = np.isnan(x)
    if not nans.any():
        return x
    idx = np.arange(end)
    good = ~nans
    if good.sum() == 0:
        return x
    if good.sum() == 1:
        x[nans] = x[good][0]
        return x
    x[nans] = np.interp(idx[nans], idx[good], x[good])
    return x

def _build_padded_window(arr: np.ndarray, k: int, L: int) -> Tuple[np.ndarray, bool]:
    if k <= 0:
        base = float(arr[0]) if arr.size > 0 and np.isfinite(arr[0]) else 0.0
        return np.full((L,), base, dtype=np.float32), True
    start = k - L
    if start >= 0:
        return arr[start:k].astype(np.float32, copy=True), False
    need = -start
    base = float(arr[0]) if arr.size > 0 and np.isfinite(arr[0]) else 0.0
    pad = np.full((need,), base, dtype=np.float32)
    w = np.concatenate([pad, arr[0:k].astype(np.float32, copy=True)], axis=0)
    return w, True

@torch.no_grad()
def _predict_delta_norm(model: nn.Module, x_win: np.ndarray) -> float:
    x_t = torch.from_numpy(x_win.astype(np.float32)).unsqueeze(0).to(DEVICE)
    mat_idx = torch.tensor([0], dtype=torch.long, device=DEVICE)
    preds = []
    for fi in range(len(FREQS)):
        freq_idx = torch.tensor([fi], dtype=torch.long, device=DEVICE)
        preds.append(float(model(x_t, mat_idx, freq_idx).detach().cpu().item()))
    return float(np.mean(preds))

def _fill_row_robust(model: nn.Module, stats: dict, B_seq: np.ndarray, H_seq: np.ndarray) -> Tuple[np.ndarray, bool, int]:
    B_seq = B_seq.astype(np.float32, copy=True)
    H_seq = H_seq.astype(np.float32, copy=True)
    n = min(len(B_seq), len(H_seq))
    B_seq = B_seq[:n]
    H_seq = H_seq[:n]
    dB = np.gradient(B_seq).astype(np.float32)

    B_mean = float(stats["B_mean"]); B_std = float(stats["B_std"])
    H_mean = float(stats["H_mean"]); H_std = float(stats["H_std"])
    dB_mean = float(stats["dB_mean"]); dB_std = float(stats["dB_std"])

    if not np.isnan(H_seq).any():
        return H_seq, False, 0

    idxs0 = np.where(np.isnan(H_seq))[0]
    k0 = int(idxs0[0])
    short_prefix_used = False
    forced_adv = 0

    if k0 == 0:
        j = 1
        while j < n and np.isnan(H_seq[j]):
            j += 1
        base = float(H_seq[j]) if j < n and np.isfinite(H_seq[j]) else 0.0
        H_seq[0] = base
        forced_adv += 1

    max_steps = n * 4
    steps = 0

    while steps < max_steps:
        steps += 1
        nan_mask = np.isnan(H_seq)
        if not nan_mask.any():
            break
        k = int(np.where(nan_mask)[0][0])

        if k == 0:
            H_seq[0] = 0.0
            forced_adv += 1
            continue

        prefix = _interp_prefix(H_seq, k)
        H_seq[:k] = prefix

        H_win, shH = _build_padded_window(H_seq, k, SEQ_LEN)
        B_win, shB = _build_padded_window(B_seq, k, SEQ_LEN)
        dB_win, shdB = _build_padded_window(dB, k, SEQ_LEN)
        if shH or shB or shdB:
            short_prefix_used = True

        if np.isnan(H_win).any():
            last = H_seq[k-1] if np.isfinite(H_seq[k-1]) else (H_win[-1] if np.isfinite(H_win[-1]) else 0.0)
            H_seq[k] = float(last)
            forced_adv += 1
            continue

        x = np.stack([
            (B_win - B_mean) / B_std,
            (H_win - H_mean) / H_std,
            (dB_win - dB_mean) / dB_std
        ], axis=-1).astype(np.float32)

        delta_norm = _predict_delta_norm(model, x)
        delta_phys = float(delta_norm * H_std)

        last_valid = H_seq[k-1]
        if not np.isfinite(last_valid):
            last_valid = float(H_win[-1]) if np.isfinite(H_win[-1]) else 0.0

        new_h = float(last_valid + delta_phys)
        if not np.isfinite(new_h):
            new_h = float(last_valid)

        H_seq[k] = new_h

        if np.isnan(H_seq[k]):
            H_seq[k] = float(last_valid)
            forced_adv += 1

    return H_seq, short_prefix_used, forced_adv

def generate_final_submission(stats: dict, model_path: str):
    os.makedirs(RESULT_DIR, exist_ok=True)

    model = GlobalLSTM(len(TRAIN_MATERIALS), len(FREQS)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with open(os.path.join(RESULT_DIR, "Parameter_Size.csv"), "w") as f:
        f.write("num_parameters,model_file_bytes\n")
        f.write(f"{count_parameters(model)},{os.path.getsize(model_path)}\n")

    testing_root = resolve_testing_root(FINAL_EVAL_ROOT)
    print("Using Testing root:", testing_root)

    diag = {}

    for letter in FINAL_MATERIALS:
        mat_dir = os.path.join(testing_root, f"Material {letter}")
        padded_path = os.path.join(mat_dir, f"{letter}_Padded_H_seq.csv")
        b_path = os.path.join(mat_dir, f"{letter}_True_B_seq.csv")

        if not os.path.exists(padded_path):
            raise FileNotFoundError(f"Missing: {padded_path}")
        if not os.path.exists(b_path):
            raise FileNotFoundError(f"Missing: {b_path}")

        H_pad = pd.read_csv(padded_path, header=None)
        B_df = pd.read_csv(b_path, header=None)

        if H_pad.shape != B_df.shape:
            raise RuntimeError(f"Shape mismatch in {letter}: H{H_pad.shape} vs B{B_df.shape}")

        rows_total = int(H_pad.shape[0])
        rows_filled = 0
        rows_shortpad = 0
        rows_still_nan = 0
        forced_total = 0

        H_out = H_pad.copy()

        for i in tqdm(range(rows_total), desc=f"Final submit {letter}"):
            B_seq = B_df.iloc[i].to_numpy(dtype=np.float32).reshape(-1)
            H_seq = H_out.iloc[i].to_numpy(dtype=np.float32).reshape(-1)

            if not np.isnan(H_seq).any():
                rows_filled += 1
                continue

            H_filled, short_used, forced = _fill_row_robust(model, stats, B_seq, H_seq)
            H_out.iloc[i] = H_filled

            if short_used:
                rows_shortpad += 1
            forced_total += int(forced)
            if np.isnan(H_filled).any():
                rows_still_nan += 1
            else:
                rows_filled += 1

        out_path = os.path.join(RESULT_DIR, f"{letter}_Pred_H_seq.csv")
        H_out.to_csv(out_path, header=False, index=False)

        diag[letter] = {
            "rows_total": rows_total,
            "rows_fully_filled": int(rows_filled),
            "rows_short_prefix_padding": int(rows_shortpad),
            "rows_still_with_nans": int(rows_still_nan),
            "forced_advances_total": int(forced_total),
            "padded_src": padded_path
        }

    with open(os.path.join(OUT_ROOT, "final_eval_diag.json"), "w") as f:
        json.dump(diag, f, indent=2)

    for k in sorted(diag.keys()):
        v = diag[k]
        print(f"[{k}] total={v['rows_total']} filled={v['rows_fully_filled']} short_prefix_pad={v['rows_short_prefix_padding']} still_nan_rows={v['rows_still_with_nans']} forced_adv_total={v['forced_advances_total']}")
        if v["rows_still_with_nans"] > 0:
            print(f"[{k}] WARNING: rows still contain NaNs after fill")

    zip_path = os.path.join(OUT_ROOT, "Result.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(RESULT_DIR)):
            if fn.endswith(".csv"):
                zf.write(os.path.join(RESULT_DIR, fn), arcname=os.path.join("Result", fn))

    print("Submission ready:", zip_path)
    print("Result folder:", RESULT_DIR)
    print("Diagnostics: /kaggle/working/final_eval_diag.json")

def main():
    set_seed(42)

    if not os.path.exists(STATS_CACHE_PATH):
        stats = compute_global_normalisation(TRAIN_ROOT, TRAIN_MATERIALS, FREQS)
        save_stats(stats, STATS_CACHE_PATH)
    else:
        stats = load_stats(STATS_CACHE_PATH)

    ckpt = resolve_checkpoint_path(CHECKPOINT_BASENAME)

    if ckpt is None:
        if not TRAIN_FROM_SCRATCH_IF_NO_CKPT:
            raise FileNotFoundError(
                "No checkpoint found under /kaggle/input or /kaggle/working.\n"
                "Attach dataset containing your .pt file OR set TRAIN_FROM_SCRATCH_IF_NO_CKPT=True."
            )
        ckpt = train_from_scratch(stats, SCRATCH_OUT, SCRATCH_EPOCHS)

    print("Using checkpoint:", ckpt)

    finetuned_path = finetune_from_checkpoint(stats, ckpt, FINETUNE_OUT)

    generate_final_submission(stats, finetuned_path)

if __name__ == "__main__":
    main()

"""
data_util.py
Field-based LOFO splits for CNN, CNN-LSTM, ResNet / ViT and tabular baselines.

Example
-------
splits = get_data_splits(df, Path("/data/imagery"), model="cnn_lstm")
for Xtr, ytr, Xte, yte, fld in splits:
    ...
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib  import Path
from typing    import List, Tuple, Dict

import numpy as np
import pandas as pd
import rasterio
from skimage.transform import resize
from numpy.lib.stride_tricks import sliding_window_view
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample

# ────────────── constants (edit if you change tiling) ─────────
BANDS, PATCH, STRIDE, VALID = 10, 32, 16, 0.60

# ───────────────────── tiling helpers ─────────────────────────
def _tile(img: np.ndarray) -> np.ndarray:
    """(C,H,W) → (N,C,P,P) with ≥VALID non-zero pixels."""
    c, h, w = img.shape
    if h < PATCH or w < PATCH:                          # upscale tiny chips
        s = max(PATCH / h, PATCH / w)
        img = np.stack(
            [resize(img[i], (int(h * s), int(w * s)),
                    order=1, mode="reflect", preserve_range=True)
             for i in range(c)], axis=0)

    win = sliding_window_view(img, (PATCH, PATCH), axis=(1, 2))[:, ::STRIDE, ::STRIDE]
    H, W = win.shape[1:3]
    tiles = win.transpose(1, 2, 0, 3, 4).reshape(H * W, c, PATCH, PATCH)
    keep  = (tiles != 0).mean(axis=(1, 2, 3)) >= VALID
    return tiles[keep]


def _read_tif(tif: Path) -> np.ndarray:
    with rasterio.open(tif) as src:
        img = src.read(range(2, 2 + BANDS))             # bands 2-11
    return np.nan_to_num(img, nan=0.).astype(np.float32)

# ─────────────────── per-field sequence ──────────────────────
def _field_sequence(rows: pd.DataFrame, root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Npatch,T,C,H,W) + (Npatch,) for ONE field."""
    blocks = []
    for p in rows.sort_values("start_date").image_path:
        arr = _tile(_read_tif(root / p.replace("./Yield_GEE_S2_ByField/", "")))
        if arr.size:
            blocks.append(arr)

    if not blocks:
        return np.empty((0, 0, BANDS, PATCH, PATCH)), np.empty((0,))

    n_min = min(len(b) for b in blocks)                 # align patches
    seq   = np.stack([b[:n_min] for b in blocks], axis=1)
    y     = np.repeat(rows.iloc[0]["yield"], n_min).astype(np.float32)
    return seq, y

# ────────────────────── public API ───────────────────────────
@dataclass
class Split:  # only used internally; final return is list of tuples
    Xtr: np.ndarray; ytr: np.ndarray
    Xte: np.ndarray; yte: np.ndarray
    field: str

# ───────────────────────── get_data_splits ─────────────────────────
def get_data_splits(
    df: pd.DataFrame,
    root: Path,
    model: str = "cnn",
    date_range: Tuple[str, str] | None = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Return LOFO splits ready for 'cnn', 'cnn_lstm', 'resnet'/'vit', or 'tabular'."""
    if date_range:
        s, e = date_range
        df = df[(df.end_date >= s) & (df.start_date <= e)]

    # ---- preprocess every field once -----------------------------------
    field_data = {
        f: _field_sequence(rows, root)          # (Npatch,T,C,H,W) , (Npatch,)
        for f, rows in df.groupby("field_name")
    }

    # ---- helpers -------------------------------------------------------
    _pad_seq = lambda s, Tmax: np.pad(
        s, ((0, 0), (0, Tmax - s.shape[1]), (0, 0), (0, 0), (0, 0))
    )

    def _flatten_seq(seq: np.ndarray) -> Tuple[np.ndarray, int]:
        """(N,T,C,H,W)->(N*T,C,H,W) + return T."""
        if not seq.size:
            return np.empty((0, BANDS, PATCH, PATCH)), 0
        N, T = seq.shape[:2]
        flat = seq.transpose(1, 0, 2, 3, 4).reshape(-1, BANDS, PATCH, PATCH)
        return flat, T

    _mean_feat = lambda s: (
        s.mean((2, 3)).reshape(-1, BANDS) if s.size else np.empty((0, BANDS))
    )

    # ---- build splits --------------------------------------------------
    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []

    for test_f, (X_te, y_te_base) in field_data.items():
        # training = concat over remaining fields
        others = [v for f, v in field_data.items() if f != test_f and v[0].size]

        # ---------------- CNN-LSTM ----------------
        if model.endswith("lstm"):
            Tmax = max((v[0].shape[1] for v in others), default=0)
            X_tr = np.concatenate([_pad_seq(v[0], Tmax) for v in others], axis=0) \
                   if others else np.empty((0, 0, BANDS, PATCH, PATCH))
            y_tr = np.concatenate([v[1] for v in others]) if others else np.empty((0,))

            if X_te.size and X_te.shape[1] < Tmax:          # pad test seq
                X_te = _pad_seq(X_te, Tmax)
            y_te = y_te_base                                # already (Npatch,)

        # ---------------- Plain CNN / ResNet / ViT --------
        elif model in {"cnn", "resnet", "vit"}:
            X_tr_blocks, y_tr_blocks = [], []
            for seq, y_vec in others:
                flat, T = _flatten_seq(seq)
                X_tr_blocks.append(flat)
                y_tr_blocks.append(np.repeat(y_vec, T))
            X_tr = np.concatenate(X_tr_blocks) if X_tr_blocks else \
                   np.empty((0, BANDS, PATCH, PATCH))
            y_tr = np.concatenate(y_tr_blocks) if y_tr_blocks else np.empty((0,))

            X_te, T_te = _flatten_seq(X_te)
            y_te = np.repeat(y_te_base, T_te)

        # ---------------- Tabular ------------------------
        elif model == "tabular":
            X_tr = np.vstack([_mean_feat(v[0]) for v in others]) if others else np.empty((0, BANDS))
            y_tr = np.concatenate([np.repeat(v[1], v[0].shape[1]) for v in others]) \
                   if others else np.empty((0,))
            X_te = _mean_feat(X_te)
            y_te = np.repeat(y_te_base, X_te.shape[0])

        else:
            raise ValueError(f"Unsupported model type: {model}")

        splits.append((X_tr, y_tr, X_te, y_te, test_f))

    return splits


# ─────────────────────── metrics  (unchanged) ────────────────
def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))
def compute_metrics(y_true, y_pred):
    return dict(rmse=rmse(y_true, y_pred),
                mae=float(mean_absolute_error(y_true, y_pred)),
                r2=float(r2_score(y_true, y_pred)))

def paired_bootstrap(y, p1, p2, n=1000):
    diffs = [(rmse(y[idx], p1[idx]) - rmse(y[idx], p2[idx]))
             for idx in (resample(np.arange(len(y)), replace=True) for _ in range(n))]
    p = (np.sum(np.array(diffs) > 0) + 1) / (n + 1)
    return p, diffs
"""
data_util.py  ·  field-based LOFO splits

Exports
-------
get_data_splits(...) → List[(X_tr, y_tr, X_te, y_te, field)]
compute_metrics(...) , paired_bootstrap(...)
"""

from __future__ import annotations
from pathlib import Path
from typing  import List, Tuple, Dict
from dataclasses import dataclass
import numpy as np, pandas as pd, rasterio
from skimage.transform import resize
from numpy.lib.stride_tricks import sliding_window_view
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
import torch

torch.manual_seed(42)  # for reproducibility
np.random.seed(42)     # for reproducibility

# ─────────── GLOBALS (edit if you change tiling) ────────────
BANDS, PATCH, STRIDE, VALID = 10, 32, 16, 0.60

# ─────────────────── low-level tiling utils ─────────────────
def _tile(img: np.ndarray) -> np.ndarray:
    """(C,H,W) → (N,C,P,P) with ≥VALID non-nodata pixels."""
    c, h, w = img.shape
    if h < PATCH or w < PATCH:           # upscale very small chips
        scale = max(PATCH / h, PATCH / w)
        img   = np.stack([resize(img[i],
                                 (int(h * scale), int(w * scale)),
                                 order=1, mode="reflect", preserve_range=True)
                          for i in range(c)], axis=0)
    win   = sliding_window_view(img, (PATCH, PATCH), axis=(1, 2))[:, ::STRIDE, ::STRIDE]
    H, W  = win.shape[1:3]
    tiles = win.transpose(1, 2, 0, 3, 4).reshape(H * W, c, PATCH, PATCH)
    keep  = (tiles != 0).mean(axis=(1, 2, 3)) >= VALID
    return tiles[keep]

def _read_tif(tif: Path, bands: bool = True) -> np.ndarray:
    with rasterio.open(tif) as src:
        arr = src.read(range(2, 2 + BANDS) if bands else None)           # Sentinel-2 bands 2-11
    return np.nan_to_num(arr, nan=0.).astype(np.float32)

# ───────────── per-field sequence constructor ───────────────
def _field_sequence(rows: pd.DataFrame, root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Npatch, T, C, H, W) tensor + patch-level yield vector for ONE field."""
    blocks = [_tile(_read_tif(root / p.replace("./Yield_GEE_S2_ByField/", "")))
              for p in rows.sort_values("start_date").filename]
    blocks = [b for b in blocks if b.size]            # drop empty
    if not blocks:
        return np.empty((0, 0, BANDS, PATCH, PATCH)), np.empty((0,))
    n_min  = min(len(b) for b in blocks)              # align patches
    seq    = np.stack([b[:n_min] for b in blocks], axis=1)
    y      = np.repeat(rows.iloc[0]["yield"], n_min).astype(np.float32)
    return seq, y

# ───────────── helpers reused by all branches ───────────────
_pad_seq   = lambda s, Tmax: np.pad(
    s, ((0, 0), (0, Tmax - s.shape[1]), (0, 0), (0, 0), (0, 0)))

def _flatten_seq(seq: np.ndarray) -> Tuple[np.ndarray, int]:
    """(N,T,C,H,W) → (N*T,C,H,W) and return T."""
    if not seq.size:
        return np.empty((0, BANDS, PATCH, PATCH)), 0
    N, T = seq.shape[:2]
    flat = seq.transpose(1, 0, 2, 3, 4).reshape(-1, BANDS, PATCH, PATCH)
    return flat, T

# classic indices (Sentinel-2 bands: 2=Blue,3=Green,4=Red,5-7=Red-edge, 8=NIR, 11=SWIR1)
def _indices(mean_bands: np.ndarray) -> np.ndarray:
    b = mean_bands  # convenience
    # b[0]=Blue(2), b[1]=Green(3), b[2]=Red(4), b[3]=RE1(5), b[4]=RE2(6), b[5]=RE3(7), b[6]=NIR(8), b[7]=RE4(8A), b[8]=SWIR1(11), b[9]=SWIR2(12)
    ndvi   = (b[6] - b[2]) / (b[6] + b[2] + 1e-6)  # (NIR - Red) / (NIR + Red)
    evi    = 2.5 * (b[6] - b[2]) / (b[6] + 6*b[2] - 7.5*b[0] + 1)  # 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1)
    ndwi   = (b[1] - b[6]) / (b[1] + b[6] + 1e-6)  # (Green - NIR) / (Green + NIR)
    msi    = b[8] / (b[6] + 1e-6)                  # SWIR1 / NIR
    rendvi = (b[6] - b[4]) / (b[6] + b[4] + 1e-6)  # (NIR - RE2) / (NIR + RE2)
    return np.array([ndvi, evi, ndwi, msi, rendvi], dtype=b.dtype)

_mean_feat = lambda s: (
    np.hstack([m := s.mean((2, 3)).reshape(-1, BANDS),          # bands
               np.apply_along_axis(_indices, 1, m)])            # + 5 indices
    if s.size else np.empty((0, BANDS + 5))
)

# ─────────── branch builders (clean & reusable) ──────────────
def _make_lstm_split(others, X_te, y_te):
    Tmax = max((v[0].shape[1] for v in others), default=0)
    Xtr  = np.concatenate([_pad_seq(v[0], Tmax) for v in others], axis=0) \
           if others else np.empty((0, 0, BANDS, PATCH, PATCH))
    ytr  = np.concatenate([v[1] for v in others]) if others else np.empty((0,))
    if X_te.size and X_te.shape[1] < Tmax:
        X_te = _pad_seq(X_te, Tmax)
    return Xtr, ytr, X_te, y_te

def _make_patchcnn_split(others, X_te, y_te_base):
    Xb, yb = [], []
    for seq, yvec in others:
        flat, T = _flatten_seq(seq)
        Xb.append(flat);  yb.append(np.repeat(yvec, T))
    Xtr = np.concatenate(Xb) if Xb else np.empty((0, BANDS, PATCH, PATCH))
    ytr = np.concatenate(yb) if yb else np.empty((0,))
    X_te, T_te = _flatten_seq(X_te)
    y_te       = np.repeat(y_te_base, T_te)
    return Xtr, ytr, X_te, y_te

def _make_xgb_split(
    others:      List[Tuple[np.ndarray, np.ndarray]],   # (seq, yvec)
    others_tmp:  List[np.ndarray | None],               # temp vectors
    X_te_img:    np.ndarray,
    y_te_base:   np.ndarray,
    temp_te:     np.ndarray | None,
):
    feats_tr, y_tr = [], []

    # ---------- TRAIN ----------
    for (seq, yvec), t_vec in zip(others, others_tmp):
        flat_ref, T = _flatten_seq(seq)                 # (Np*T, …)
        band_feat   = _mean_feat(flat_ref)              # (Np*T, 15)
        n_patch     = seq.shape[0]                      # patches / date

        # --- align temperature vector length to T -------
        if t_vec is not None and t_vec.size:
            if t_vec.size > T:          # too many dates → trim head
                t_vec = t_vec[:T]
            elif t_vec.size < T:        # too few → pad last known value
                pad = np.full(T - t_vec.size, t_vec[-1], t_vec.dtype)
                t_vec = np.concatenate([t_vec, pad])

            temp_feat = np.repeat(t_vec, n_patch).reshape(-1, 1)
        else:
            temp_feat = np.zeros((band_feat.shape[0], 1), band_feat.dtype)

        feats_tr.append(np.hstack([band_feat, temp_feat]))
        y_tr.append(np.repeat(yvec, T))

    Xtr = np.vstack(feats_tr) if feats_tr else np.empty((0, 16))
    ytr = np.concatenate(y_tr) if y_tr else np.empty((0,))

    # ---------- TEST / VAL ----------
    flat_ref_te, Tt = _flatten_seq(X_te_img)
    band_feat_te    = _mean_feat(flat_ref_te)
    n_patch_te      = X_te_img.shape[0]

    if temp_te is not None and temp_te.size:
        if temp_te.size > Tt:
            temp_te = temp_te[:Tt]
        elif temp_te.size < Tt:
            temp_te = np.pad(temp_te, (0, Tt - temp_te.size), mode="edge")
        temp_feat_te = np.repeat(temp_te, n_patch_te).reshape(-1, 1)
    else:
        temp_feat_te = np.zeros((band_feat_te.shape[0], 1), band_feat_te.dtype)

    X_te = np.hstack([band_feat_te, temp_feat_te])
    y_te = np.repeat(y_te_base, Tt)
    
    print(f"Xtr shape: {Xtr.shape}, ytr shape: {ytr.shape}, "
          f"X_te shape: {X_te.shape}, y_te shape: {y_te.shape}")  # debug info

    return Xtr, ytr, X_te, y_te


# ------------------------------------------------------------------
# Read a temperature GeoTIFF → single mean value
def _mean_temp_tif(path: Path) -> float:
    return float(_read_tif(path, bands=False).mean())

# Build {field → 1-D temperature vector length T} aligned to S2 dates
def _load_field_temps(
    temp_df: pd.DataFrame | None,
    df_img:  pd.DataFrame,
    root:    Path,
) -> Dict[str, np.ndarray]:

    if temp_df is None or True:
        return {}

    temp_root  = root / "Yield_GEE_TEMP_ByField"
    field_temp: Dict[str, np.ndarray] = {}

    for fld, rows_img in df_img.groupby("field_name"):
        dates_img = rows_img.sort_values("start_date")["start_date"].values
        rows_t    = temp_df[temp_df.field_name == fld]

        # map each available date → mean temperature
        date2val = {
            r.start_date: _mean_temp_tif(
                temp_root / r.filename.replace("./Yield_GEE_TEMP_ByField/", "")
            )
            for _, r in rows_t.iterrows()
        }

        vec = np.array([date2val.get(d, np.nan) for d in dates_img], np.float32)

        # linearly interpolate missing dates
        if np.isnan(vec).any():
            vec = (
                pd.Series(vec)
                .interpolate("linear", limit_direction="both")
                .to_numpy(np.float32)
            )

        field_temp[fld] = vec            # shape (T,)
    return field_temp

# ───────────────── get_data_splits (clean) ───────────────────
def get_data_splits(
    df: pd.DataFrame,
    temp_df: pd.DataFrame | None,
    root: Path,
    model: str = "cnn",
    date_range: Tuple[str, str] | None = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Return LOFO splits for 'cnn', 'cnn_lstm', 'resnet', 'vit', or 'xgb'."""
    if date_range:
        s, e = date_range
        df = df[(df.end_date >= s) & (df.start_date <= e)]

    # ---- build reflectance sequences -----------------------------------
    field_data = {
        f: _field_sequence(rows, root / "Yield_GEE_S2_ByField")
        for f, rows in df.groupby("field_name")
    }

    # ---- load per-field temperature vectors ----------------------------
    field_temp = _load_field_temps(temp_df, df, root)

    # ---- assemble LOFO splits ------------------------------------------
    splits: list[Tuple[np.ndarray, ...]] = []
    
    for test_f, (X_te_img, y_te_base) in field_data.items():
        if test_f == "Field8":
            continue

        # -------- training sets --------
        others_pairs = [
            (field_data[f], field_temp.get(f))
            for f in field_data
            if f != test_f and field_data[f][0].size
        ]

        others      = [p[0] for p in others_pairs]   # (seq, yvec)
        others_tmp  = [p[1] for p in others_pairs]   # temp vectors or None

        test_tmp = field_temp.get(test_f)            # may be None

        # -------- branch selection -----
        if model.endswith("lstm"):
            Xtr, ytr, X_te, y_te = _make_lstm_split(others, X_te_img, y_te_base)

        elif model in {"cnn", "resnet", "vit"}:
            Xtr, ytr, X_te, y_te = _make_patchcnn_split(others, X_te_img, y_te_base)

        elif model in {"xgb", "linear"}:
            Xtr, ytr, X_te, y_te = _make_xgb_split(
                others, others_tmp, X_te_img, y_te_base, test_tmp
            )

        else:
            raise ValueError(f"Unsupported model: {model}")

        splits.append((Xtr, ytr, X_te, y_te, test_f))

    return splits

# ───────────────────────── metrics utils ─────────────────────
def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))
def compute_metrics(y_true, y_pred):
    return dict(rmse=rmse(y_true, y_pred),
                mae=float(mean_absolute_error(y_true, y_pred)),
                r2=float(r2_score(y_true, y_pred)))

def paired_bootstrap(y, p1, p2, n=1000):
    idxs  = (resample(np.arange(len(y)), replace=True) for _ in range(n))
    diff  = [rmse(y[i], p1[i]) - rmse(y[i], p2[i]) for i in idxs]
    p_val = (np.sum(np.array(diff) > 0) + 1) / (n + 1)
    return p_val, diff

# ---------------------------------------------------------------------
# vegetation-index helper (vectorised, Torch)
# ---------------------------------------------------------------------
def _veg_indices(x: torch.Tensor) -> torch.Tensor:
    """
    x : (B, 10, H, W) Sentinel-2 bands 2–12 (we drop band-1 & 9).
    returns (B, 5)  [NDVI, EVI, NDWI, MSI, RE-NDVI]   each averaged over H×W.
    """
    B, _, H, W = x.shape
    b = x.mean((-2, -1))                    # (B,10) spectral means

    blue, green, red   = b[:, 0], b[:, 1], b[:, 2]
    nir, swir1         = b[:, 6], b[:, 8]
    re2                = b[:, 4]

    ndvi   = (nir - red) / (nir + red + 1e-6)
    evi    = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    ndwi   = (green - nir) / (green + nir + 1e-6)
    msi    = swir1 / (nir + 1e-6)
    rendvi = (nir - re2) / (nir + re2 + 1e-6)

    return torch.stack([ndvi, evi, ndwi, msi, rendvi], dim=1)   # (B,5)

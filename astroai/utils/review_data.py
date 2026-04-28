"""Data loading and metric helpers for review analysis."""

from __future__ import annotations

import pickle
import re
from typing import Iterable

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from astroai.tools.utils import create_circular_mask, set_wcs


def load_dataset(path: str):
    if path.endswith(".pickle"):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=True, encoding="latin1", fix_imports=True).flat[0]
    raise ValueError(f"Unsupported dataset extension: {path}")


def guess_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of columns found: {list(candidates)}")
    return None


def parse_zenith(irf_name: str) -> str:
    txt = str(irf_name).lower()
    match = re.search(r"z(20|40|60)", txt)
    return f"z{match.group(1)}" if match else "z?"


def parse_nbs(irf_name: str) -> str:
    txt = str(irf_name)
    if "_N_" in txt:
        return "N"
    if "_S_" in txt:
        return "S"
    return "B"


def add_common_meta(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    irf_col = guess_col(out, ["irf", "IRF"], required=False)
    if irf_col is None:
        out["irf"] = "unknown"
        irf_col = "irf"

    out["zenith_tag"] = out[irf_col].apply(parse_zenith)
    out["nbs_tag"] = out[irf_col].apply(parse_nbs)

    theta_col = guess_col(
        out,
        ["theta", "Theta", "offset", "source_offset", "src_offset", "alpha"],
        required=False,
    )
    out["theta_tag"] = out[theta_col] if theta_col else np.nan

    snr_col = guess_col(out, ["snr", "SNR"], required=False)
    if snr_col is None:
        ex_col = guess_col(out, ["excess", "counts_excess"], required=False)
        off_col = guess_col(out, ["counts_off", "off", "background"], required=False)
        if ex_col and off_col:
            ex = out[ex_col].to_numpy(dtype=float)
            off = out[off_col].to_numpy(dtype=float)
            out["snr_tag"] = ex / np.sqrt(np.maximum(ex + off, 1e-12))
        else:
            out["snr_tag"] = np.nan
    else:
        out["snr_tag"] = out[snr_col]
    return out


def make_quantile_bins(series, n: int = 4, label: str = "q"):
    values = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return pd.Series([], dtype=str)
    n_eff = max(1, min(n, values.nunique()))
    qcats = pd.qcut(values, q=n_eff, duplicates="drop")
    labels = [
        f"{label}{idx + 1}: {interval.left:.3g}-{interval.right:.3g}"
        for idx, interval in enumerate(qcats.cat.categories)
    ]
    mapper = dict(zip(qcats.cat.categories, labels))
    return qcats.map(mapper)


def build_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group_nbs"] = out["nbs_tag"]
    out["group_zenith"] = pd.Categorical(
        out["zenith_tag"], categories=["z20", "z40", "z60", "z?"], ordered=True
    )
    out["group_snr"] = make_quantile_bins(out["snr_tag"], n=4, label="SNR")
    out["group_theta"] = make_quantile_bins(out["theta_tag"], n=4, label="theta")
    return out


def compute_cleaner_metrics(
    test_noisy,
    test_clean,
    cleaner_pred,
    cleaner_meta: pd.DataFrame,
    point_ref: float,
    pixelsize,
    radius_pix: float,
) -> pd.DataFrame:
    sum_residual_std = []
    sum_residual_cnn = []
    sum_on_std = []
    sum_on_cnn = []

    for noisy, clean, pred, row in zip(test_noisy, test_clean, cleaner_pred, cleaner_meta.itertuples()):
        res_std = noisy - clean
        res_cnn = noisy - pred
        sum_residual_std.append(float(np.sum(res_std)))
        sum_residual_cnn.append(float(np.sum(res_cnn)))

        wcs = set_wcs(
            point_ra=row.point_ra,
            point_dec=row.point_dec,
            point_ref=point_ref,
            pixelsize=pixelsize,
        )
        x_src, y_src = wcs.world_to_pixel(
            SkyCoord(row.source_ra, row.source_dec, unit="deg", frame="icrs")
        )

        height, width = clean.shape[:2]
        mask = create_circular_mask(height, width, center=(y_src, x_src), radius=radius_pix)
        masked_std = clean.copy()
        masked_std[~mask] = 0
        masked_cnn = pred.copy()
        masked_cnn[~mask] = 0
        sum_on_std.append(float(np.sum(masked_std)))
        sum_on_cnn.append(float(np.sum(masked_cnn)))

    out = cleaner_meta.copy()
    out["sum_residual_gammapy"] = sum_residual_std
    out["sum_residual_cnn"] = sum_residual_cnn
    out["sum_residual_diff"] = out["sum_residual_gammapy"] - out["sum_residual_cnn"]
    out["sum_on_gammapy"] = sum_on_std
    out["sum_on_cnn"] = sum_on_cnn
    out["sum_on_diff"] = out["sum_on_gammapy"] - out["sum_on_cnn"]
    return out


def compute_regressor_metrics(
    reg_meta: pd.DataFrame,
    reg_pred,
    point_ref: float,
    pixelsize,
    binning: float,
) -> pd.DataFrame:
    err_deg = []
    for pred, row in zip(reg_pred, reg_meta.itertuples()):
        wcs = set_wcs(
            point_ra=row.point_ra,
            point_dec=row.point_dec,
            point_ref=point_ref,
            pixelsize=pixelsize,
        )
        found_sky = wcs.pixel_to_world(pred[0] * binning, pred[1] * binning)
        true_sky = SkyCoord(row.source_ra, row.source_dec, unit="deg", frame="icrs")
        err_deg.append(float(true_sky.separation(found_sky).degree))
    out = reg_meta.copy()
    out["err_deg"] = err_deg
    return out


def align_meta_with_test_split(meta_df: pd.DataFrame, train_size: int, test_size: int) -> pd.DataFrame:
    seed_start = train_size + 1
    seed_stop = seed_start + test_size
    out = meta_df[meta_df["seed"].between(seed_start, seed_stop - 1)].copy()
    out = out.sort_values("seed").reset_index(drop=True)
    if len(out) != test_size:
        out = meta_df.iloc[train_size : train_size + test_size].copy().reset_index(drop=True)
    return out

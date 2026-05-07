#!/usr/bin/env python3
# *******************************************************************************
# Copyright (C) 2024 Ambra Di Piano
#
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *******************************************************************************

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TIMING_ORDER = [
    "t_irf_reduce",
    "t_prepare",
    "t_dataset_read",
    "t_analysis_total",
    "t_setup",
    "t_counts_map",
    "t_blindsearch",
    "t_photometry",
    "t_preparation",
    "t_model_load",
    "t_preprocess",
    "t_cleaner",
    "t_regressor",
    "t_decode",
    "t_total",
]

SETUP_REFERENCE_STEPS = ["t_irf_reduce", "t_model_load"]
GP_MEANBAR_KEYS = ["t_irf_reduce", "t_preparation", "t_blindsearch", "t_photometry", "t_total"]
CNN_MEANBAR_KEYS = ["t_model_load", "t_preprocess", "t_cleaner", "t_regressor", "t_decode", "t_cleaner_metrics", "t_total"]


def load_table(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=r"\s+", header=0)


def get_timing_columns(df):
    return [c for c in df.columns if c.startswith("t_")]


def save_total_hist(gp, cnn, outdir):
    if "t_total" not in gp.columns or "t_total" not in cnn.columns:
        return

    gp_total = gp["t_total"].dropna().values
    cnn_total = cnn["t_total"].dropna().values
    if len(gp_total) == 0 or len(cnn_total) == 0:
        return

    xmin = min(gp_total.min(), cnn_total.min())
    xmax = max(gp_total.max(), cnn_total.max())
    bins = np.linspace(xmin, xmax, 16)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(gp_total, bins=bins, alpha=0.35, color="#2E86AB", edgecolor="#1B4F72", linewidth=1.2, label="gammapy")
    ax.hist(cnn_total, bins=bins, alpha=0.35, color="#F18F01", edgecolor="#9C640C", linewidth=1.2, label="cnn")
    ax.axvline(np.mean(gp_total), color="#1B4F72", linestyle="--", linewidth=1.8)
    ax.axvline(np.mean(cnn_total), color="#9C640C", linestyle="--", linewidth=1.8)
    ax.text(np.mean(gp_total), ax.get_ylim()[1] * 0.90, "mean gp", color="#1B4F72", fontsize=10)
    ax.text(np.mean(cnn_total), ax.get_ylim()[1] * 0.82, "mean cnn", color="#9C640C", fontsize=10)
    ax.set_title("Total Runtime Distribution", fontsize=13)
    ax.set_xlabel("total runtime [s]")
    ax.set_ylabel("samples")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "timing_total_hist.png"))
    plt.close(fig)


def save_mean_bar(gp, cnn, outdir):
    gp_cols = get_timing_columns(gp)
    cnn_cols = get_timing_columns(cnn)
    gp_ordered = [c for c in GP_MEANBAR_KEYS if c in gp_cols]
    cnn_ordered = [c for c in CNN_MEANBAR_KEYS if c in cnn_cols]
    if len(gp_ordered) == 0 and len(cnn_ordered) == 0:
        return

    h = 0.38
    fig, (ax_gp, ax_cnn) = plt.subplots(2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [1, 1]})

    # GAMMAPY subplot
    y_gp = np.arange(len(gp_ordered))
    gp_mean = np.array([gp[c].mean() for c in gp_ordered], dtype=float)
    gp_std = np.array([gp[c].std() for c in gp_ordered], dtype=float)
    gp_plot = np.nan_to_num(gp_mean, nan=0.0)
    gp_err = np.nan_to_num(gp_std, nan=0.0)
    ax_gp.barh(y_gp, gp_plot, xerr=gp_err, capsize=4, height=0.65, color="#2E86AB")
    ax_gp.set_yticks(y_gp)
    ax_gp.set_yticklabels(gp_ordered)
    ax_gp.set_xlabel("mean runtime [s]")
    ax_gp.set_title("Gammapy Mean Runtime by Timing Step", fontsize=12)
    ax_gp.grid(axis="x", alpha=0.3)
    ax_gp.set_xscale("log")
    ax_gp.set_xlim(left=1e-3)
    setup_idx_gp = [i for i, c in enumerate(gp_ordered) if c in SETUP_REFERENCE_STEPS]
    if len(setup_idx_gp) > 0:
        ax_gp.axhspan(min(setup_idx_gp) - 0.5, max(setup_idx_gp) + 0.5, color="#F6F6F6", zorder=0)
    ax_gp.invert_yaxis()

    # CNN subplot
    y_cnn = np.arange(len(cnn_ordered))
    cnn_mean = np.array([cnn[c].mean() for c in cnn_ordered], dtype=float)
    cnn_std = np.array([cnn[c].std() for c in cnn_ordered], dtype=float)
    cnn_plot = np.nan_to_num(cnn_mean, nan=0.0)
    cnn_err = np.nan_to_num(cnn_std, nan=0.0)
    ax_cnn.barh(y_cnn, cnn_plot, xerr=cnn_err, capsize=4, height=0.65, color="#F18F01")
    ax_cnn.set_yticks(y_cnn)
    ax_cnn.set_yticklabels(cnn_ordered)
    ax_cnn.set_xlabel("mean runtime [s]")
    ax_cnn.set_title("CNN Mean Runtime by Timing Step", fontsize=12)
    ax_cnn.grid(axis="x", alpha=0.3)
    ax_cnn.set_xscale("log")
    ax_cnn.set_xlim(left=1e-3)
    setup_idx_cnn = [i for i, c in enumerate(cnn_ordered) if c in SETUP_REFERENCE_STEPS]
    if len(setup_idx_cnn) > 0:
        ax_cnn.axhspan(min(setup_idx_cnn) - 0.5, max(setup_idx_cnn) + 0.5, color="#F6F6F6", zorder=0)
    ax_cnn.invert_yaxis()

    fig.suptitle("Mean Runtime by Timing Step (separate pipelines)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "timing_mean_bar.png"))
    plt.close(fig)


def save_total_scatter(gp, cnn, outdir):
    if "seed" not in gp.columns or "seed" not in cnn.columns:
        return
    if "t_total" not in gp.columns or "t_total" not in cnn.columns:
        return

    g = gp[["seed", "t_total"]].rename(columns={"t_total": "t_total_gp"})
    c = cnn[["seed", "t_total"]].rename(columns={"t_total": "t_total_cnn"})
    m = pd.merge(g, c, on="seed", how="inner")
    if len(m) == 0:
        return

    x = m["t_total_gp"].to_numpy()
    y = m["t_total_cnn"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=45, color="#2E86AB", edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.set_xlabel("gammapy t_total [s]")
    ax.set_ylabel("cnn t_total [s]")
    ax.set_title("Total Runtime: CNN vs Gammapy", fontsize=13)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "timing_total_scatter.png"))
    plt.close(fig)

def get_step_mean(df, cols):
    existing = [c for c in cols if c in df.columns]
    if len(existing) == 0:
        return np.nan
    return np.nansum([df[c].mean() for c in existing])

def get_step_series(df, cols):
    existing = [c for c in cols if c in df.columns]
    if len(existing) == 0:
        return np.array([])
    data = df[existing].copy().astype(float)
    return np.nansum(data.to_numpy(), axis=1)


def save_simplified_bar(gp, cnn, outdir):
    # 1-to-1 conceptual mapping between gammapy and cnn timing steps
    labels = ["prep_once_ref", "data_prep", "localization", "photometry", "total"]
    gp_vals = [
        get_step_mean(gp, ["t_irf_reduce"]),
        get_step_mean(gp, ["t_preparation"]),
        get_step_mean(gp, ["t_blindsearch"]),
        get_step_mean(gp, ["t_photometry"]),
        get_step_mean(gp, ["t_total"]),
    ]
    cnn_vals = [
        get_step_mean(cnn, ["t_model_load"]),
        get_step_mean(cnn, ["t_preprocess", "t_cleaner"]),
        get_step_mean(cnn, ["t_regressor", "t_decode"]),
        get_step_mean(cnn, ["t_cleaner_metrics"]),
        get_step_mean(cnn, ["t_total"]),
    ]
    gp_std = [
        np.nanstd(get_step_series(gp, ["t_irf_reduce"])),
        np.nanstd(get_step_series(gp, ["t_preparation"])),
        np.nanstd(get_step_series(gp, ["t_blindsearch"])),
        np.nanstd(get_step_series(gp, ["t_photometry"])),
        np.nanstd(get_step_series(gp, ["t_total"])),
    ]
    cnn_std = [
        np.nanstd(get_step_series(cnn, ["t_model_load"])),
        np.nanstd(get_step_series(cnn, ["t_preprocess", "t_cleaner"])),
        np.nanstd(get_step_series(cnn, ["t_regressor", "t_decode"])),
        np.nanstd(get_step_series(cnn, ["t_cleaner_metrics"])),
        np.nanstd(get_step_series(cnn, ["t_total"])),
    ]

    y = np.arange(len(labels))
    h = 0.38
    gp_plot = np.nan_to_num(gp_vals, nan=0.0)
    cnn_plot = np.nan_to_num(cnn_vals, nan=0.0)
    gp_err = np.nan_to_num(gp_std, nan=0.0)
    cnn_err = np.nan_to_num(cnn_std, nan=0.0)
    fig, (ax_rt, ax_ref) = plt.subplots(
        2, 1, figsize=(10, 8.5), gridspec_kw={"height_ratios": [4, 1.8]}
    )

    # on-the-fly runtime steps
    idx_rt = [1, 2, 3, 4]
    y_rt = np.arange(len(idx_rt))
    labels_rt = [labels[i] for i in idx_rt]
    gp_rt = [gp_plot[i] for i in idx_rt]
    cnn_rt = [cnn_plot[i] for i in idx_rt]
    gp_rt_err = [gp_err[i] for i in idx_rt]
    cnn_rt_err = [cnn_err[i] for i in idx_rt]

    ax_rt.barh(y_rt - h / 2, gp_rt, xerr=gp_rt_err, capsize=4, height=h, color="#2E86AB", label="gammapy")
    ax_rt.barh(y_rt + h / 2, cnn_rt, xerr=cnn_rt_err, capsize=4, height=h, color="#F18F01", label="cnn")
    ax_rt.set_yticks(y_rt)
    ax_rt.set_yticklabels(labels_rt)
    ax_rt.set_xlabel("mean runtime [s]")
    ax_rt.set_title("On-the-fly Benchmark (mean +/- std)", fontsize=12)
    ax_rt.grid(axis="x", alpha=0.3)
    ax_rt.set_xscale("log")
    ax_rt.set_xlim(left=1e-3)
    ax_rt.invert_yaxis()
    ax_rt.legend(loc="lower right")

    # setup reference step
    idx_ref = [0]
    y_ref = np.arange(len(idx_ref))
    labels_ref = [labels[i] for i in idx_ref]
    gp_ref = [gp_plot[i] for i in idx_ref]
    cnn_ref = [cnn_plot[i] for i in idx_ref]
    gp_ref_err = [gp_err[i] for i in idx_ref]
    cnn_ref_err = [cnn_err[i] for i in idx_ref]

    ax_ref.barh(y_ref - h / 2, gp_ref, xerr=gp_ref_err, capsize=4, height=h, color="#2E86AB")
    ax_ref.barh(y_ref + h / 2, cnn_ref, xerr=cnn_ref_err, capsize=4, height=h, color="#F18F01")
    ax_ref.set_yticks(y_ref)
    ax_ref.set_yticklabels(labels_ref)
    ax_ref.set_xlabel("mean runtime [s]")
    ax_ref.set_title("Setup Reference", fontsize=12)
    ax_ref.grid(axis="x", alpha=0.3)
    ax_ref.set_xscale("log")
    ax_ref.set_xlim(left=1e-3)
    ax_ref.invert_yaxis()
    ax_ref.text(
        0.98,
        0.08,
        "once-per-night\nreference only",
        transform=ax_ref.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgray",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="lightgray"),
    )

    fig.suptitle("Simplified Step-by-Step Benchmark", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "timing_simplified_bar.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--gp", type=str, default="benchmark/time_gp_10.txt", help="gammapy timing file")
    parser.add_argument("--cnn", type=str, default="benchmark/time_cnn_10.txt", help="cnn timing file")
    parser.add_argument("-o", "--outdir", type=str, default="benchmark", help="output directory for timing plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gp = load_table(args.gp)
    cnn = load_table(args.cnn)

    save_total_hist(gp=gp, cnn=cnn, outdir=args.outdir)
    save_mean_bar(gp=gp, cnn=cnn, outdir=args.outdir)
    save_total_scatter(gp=gp, cnn=cnn, outdir=args.outdir)
    save_simplified_bar(gp=gp, cnn=cnn, outdir=args.outdir)


if __name__ == "__main__":
    main()

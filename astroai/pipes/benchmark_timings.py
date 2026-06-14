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
from matplotlib.patches import Patch

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
GP_GRANULAR_KEYS = ["t_preparation", "t_setup", "t_counts_map", "t_blindsearch", "t_photometry"]
CNN_GRANULAR_KEYS = ["t_counts_map", "t_prepare", "t_preprocess", "t_cleaner", "t_regressor", "t_decode", "t_cleaner_metrics"]


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
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "timing_total_hist.png"))
    plt.close(fig)


def pretty_step_label(key):
    label = key.replace("t_", "").replace("_", " ")
    if label == "decode":
        return "decoding"
    return label


def is_nullish_timing(series, eps=1e-12):
    vals = pd.to_numeric(series, errors="coerce")
    if vals.isna().all():
        return True
    return np.nanmax(np.abs(vals.to_numpy(dtype=float))) <= eps


def filter_visible_steps(labels, means, stds, eps=1e-12, hide_zero=True):
    kept_labels = []
    kept_means = []
    kept_stds = []
    for lab, mean, std in zip(labels, means, stds):
        if np.isnan(mean):
            continue
        if hide_zero and abs(mean) <= eps:
            continue
        kept_labels.append(lab)
        kept_means.append(mean)
        kept_stds.append(0.0 if np.isnan(std) else std)
    return kept_labels, np.array(kept_means, dtype=float), np.array(kept_stds, dtype=float)


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


def cnn_prep_cols(df):
    cols = []
    if "t_counts_map" in df.columns:
        cols.append("t_counts_map")
    if "t_prepare" in df.columns:
        cols.append("t_prepare")
    if len(cols) == 0 and "t_preprocess" in df.columns:
        cols.append("t_preprocess")
    return cols


def save_merged_bar_panels(gp, cnn, outdir, include_breakdown=False):
    # panel 1: ahead-of-nightly reference
    gp_irf_mean = get_step_mean(gp, ["t_irf_reduce"])
    cnn_load_mean = get_step_mean(cnn, ["t_model_load"])
    gp_irf_std = np.nanstd(get_step_series(gp, ["t_irf_reduce"]))
    cnn_load_std = np.nanstd(get_step_series(cnn, ["t_model_load"]))

    # panel 2: simplified mapping
    labels = ["preparation", "analysis", "total"]
    gp_vals = [
        get_step_mean(gp, ["t_preparation"]),
        get_step_mean(gp, ["t_blindsearch", "t_photometry"]),
        get_step_mean(gp, ["t_total"]),
    ]
    cnn_vals = [
        get_step_mean(cnn, cnn_prep_cols(cnn) + ["t_cleaner"]),
        get_step_mean(cnn, ["t_regressor", "t_decode", "t_cleaner_metrics"]),
        get_step_mean(cnn, ["t_total"]),
    ]
    gp_std = [
        np.nanstd(get_step_series(gp, ["t_preparation"])),
        np.nanstd(get_step_series(gp, ["t_blindsearch", "t_photometry"])),
        np.nanstd(get_step_series(gp, ["t_total"])),
    ]
    cnn_std = [
        np.nanstd(get_step_series(cnn, cnn_prep_cols(cnn) + ["t_cleaner"])),
        np.nanstd(get_step_series(cnn, ["t_regressor", "t_decode", "t_cleaner_metrics"])),
        np.nanstd(get_step_series(cnn, ["t_total"])),
    ]
    gp_plot = np.nan_to_num(gp_vals, nan=0.0)
    cnn_plot = np.nan_to_num(cnn_vals, nan=0.0)
    gp_err = np.nan_to_num(gp_std, nan=0.0)
    cnn_err = np.nan_to_num(cnn_std, nan=0.0)

    # panel 3: cnn granular breakdown
    cnn_labels = ["counts map", "encoding", "cleaner", "regressor", "decoding", "photometry"]
    cnn_cols_preparation = ["t_prepare"] if "t_prepare" in cnn.columns else ["t_preprocess"]
    cnn_groups = [
        ["t_counts_map"],             # counts map extraction (dl3 -> dl4)
        cnn_cols_preparation,         # encoding / preparation
        ["t_cleaner"],                # cleaner
        ["t_regressor"],              # regressor
        ["t_decode"],                 # decoding
        ["t_cleaner_metrics"],        # photometry proxy
    ]
    cnn_gr_mean = np.array([get_step_mean(cnn, cols) for cols in cnn_groups], dtype=float)
    cnn_gr_std = np.array([np.nanstd(get_step_series(cnn, cols)) for cols in cnn_groups], dtype=float)
    cnn_labels, cnn_gr_mean, cnn_gr_std = filter_visible_steps(cnn_labels, cnn_gr_mean, cnn_gr_std, hide_zero=True)

    # panel 4: gammapy granular breakdown
    gp_labels = ["prepare", "dataset read", "setup", "counts map", "blindsearch", "photometry"]
    gp_groups = [
        ["t_prepare"],        # prepare
        ["t_dataset_read"],   # dataset read
        ["t_setup"],          # setup
        ["t_counts_map"],     # counts map
        ["t_blindsearch"],    # blindsearch
        ["t_photometry"],     # photometry
    ]
    gp_gr_mean = np.array([get_step_mean(gp, cols) for cols in gp_groups], dtype=float)
    gp_gr_std = np.array([np.nanstd(get_step_series(gp, cols)) for cols in gp_groups], dtype=float)
    gp_labels, gp_gr_mean, gp_gr_std = filter_visible_steps(gp_labels, gp_gr_mean, gp_gr_std, hide_zero=True)

    if include_breakdown:
        fig, axs = plt.subplots(2, 2, figsize=(14, 11))
        ax0, ax1 = axs[0, 0], axs[0, 1]
        ax2, ax3 = axs[1, 0], axs[1, 1]
    else:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        ax0, ax1 = axs[0], axs[1]
    h = 0.38

    # 1) ahead of nightly activity
    panel1_labels = ["reduce irf", "load model"]
    panel1_vals = np.nan_to_num(np.array([gp_irf_mean, cnn_load_mean], dtype=float), nan=0.0)
    panel1_err = np.nan_to_num(np.array([gp_irf_std, cnn_load_std], dtype=float), nan=0.0)
    panel1_colors = ["#2E86AB", "#F18F01"]
    y0 = np.arange(len(panel1_labels))
    ax0.barh(y0, panel1_vals, xerr=panel1_err, capsize=4, height=0.6, color=panel1_colors)
    ax0.set_yticks(y0)
    ax0.set_yticklabels(panel1_labels)
    ax0.set_xlabel("mean runtime [s]")
    ax0.set_title("1) Ahead of Nightly Activity", fontsize=12)
    ax0.grid(axis="x", alpha=0.3)
    ax0.invert_yaxis()
    ax0.legend(
        handles=[
            Patch(facecolor="#2E86AB", label="gammapy"),
            Patch(facecolor="#F18F01", label="cnn"),
        ],
        loc=0,
    )

    # 2) simplified comparison
    y1 = np.arange(len(labels))
    ax1.barh(y1 - h / 2, gp_plot, xerr=gp_err, capsize=4, height=h, color="#2E86AB", label="gammapy")
    ax1.barh(y1 + h / 2, cnn_plot, xerr=cnn_err, capsize=4, height=h, color="#F18F01", label="cnn")
    ax1.set_yticks(y1)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("mean runtime [s]")
    ax1.set_title("2) Pipeline Comparison", fontsize=12)
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_yaxis()
    ax1.legend(loc=0)

    if include_breakdown:
        # 3) cnn granular
        y2 = np.arange(len(cnn_labels))
        ax2.barh(y2, cnn_gr_mean, xerr=cnn_gr_std, capsize=4, height=0.65, color="#F18F01")
        ax2.set_yticks(y2)
        ax2.set_yticklabels(cnn_labels)
        ax2.set_xlabel("mean runtime [s]")
        ax2.set_title("3) CNN Breakdown", fontsize=12)
        ax2.grid(axis="x", alpha=0.3)
        ax2.invert_yaxis()
        ax2.legend(handles=[Patch(facecolor="#F18F01", label="cnn")], loc=0)

        # 4) gammapy granular
        y3 = np.arange(len(gp_labels))
        ax3.barh(y3, gp_gr_mean, xerr=gp_gr_std, capsize=4, height=0.65, color="#2E86AB")
        ax3.set_yticks(y3)
        ax3.set_yticklabels(gp_labels)
        ax3.set_xlabel("mean runtime [s]")
        ax3.set_title("4) Gammapy Breakdown", fontsize=12)
        ax3.grid(axis="x", alpha=0.3)
        ax3.invert_yaxis()
        ax3.legend(handles=[Patch(facecolor="#2E86AB", label="gammapy")], loc=0)

    fig.suptitle("Merged Timing Bars", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "timing_bar_panels.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--gp", type=str, default="benchmark/time_gp_10.txt", help="gammapy timing file")
    parser.add_argument("--cnn", type=str, default="benchmark/time_cnn_10.txt", help="cnn timing file")
    parser.add_argument("-o", "--outdir", type=str, default="benchmark", help="output directory for timing plots")
    parser.add_argument("--with-breakdown", action="store_true", help="include panels 3 and 4 in timing_bar_panels plot")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gp = load_table(args.gp)
    cnn = load_table(args.cnn)

    save_total_hist(gp=gp, cnn=cnn, outdir=args.outdir)
    save_merged_bar_panels(gp=gp, cnn=cnn, outdir=args.outdir, include_breakdown=args.with_breakdown)
    save_total_scatter(gp=gp, cnn=cnn, outdir=args.outdir)


if __name__ == "__main__":
    main()

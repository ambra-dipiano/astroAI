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


def load_table(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=r"\s+", header=0)


def get_timing_columns(df):
    return [c for c in df.columns if c.startswith("t_")]


def save_total_hist(gp, cnn, outdir):
    if "t_total" not in gp.columns or "t_total" not in cnn.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(gp["t_total"], bins=20, alpha=0.6, label="gammapy")
    ax.hist(cnn["t_total"], bins=20, alpha=0.6, label="cnn")
    ax.set_xlabel("total runtime [s]")
    ax.set_ylabel("samples")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "timing_total_hist.png"))
    plt.close(fig)


def save_mean_bar(gp, cnn, outdir):
    gp_cols = get_timing_columns(gp)
    cnn_cols = get_timing_columns(cnn)
    cols = sorted(set(gp_cols + cnn_cols))
    if len(cols) == 0:
        return

    gp_mean = [gp[c].mean() if c in gp.columns else np.nan for c in cols]
    cnn_mean = [cnn[c].mean() if c in cnn.columns else np.nan for c in cols]

    x = np.arange(len(cols))
    w = 0.4
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, gp_mean, width=w, label="gammapy")
    ax.bar(x + w / 2, cnn_mean, width=w, label="cnn")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_ylabel("mean runtime [s]")
    ax.grid(axis="y")
    ax.legend()
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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(m["t_total_gp"], m["t_total_cnn"], s=20)
    vmin = min(m["t_total_gp"].min(), m["t_total_cnn"].min())
    vmax = max(m["t_total_gp"].max(), m["t_total_cnn"].max())
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1)
    ax.set_xlabel("gammapy t_total [s]")
    ax.set_ylabel("cnn t_total [s]")
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "timing_total_scatter.png"))
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


if __name__ == "__main__":
    main()

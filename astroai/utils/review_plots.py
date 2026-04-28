"""Plot helpers for review analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FS = 12


def _annotate_bar_counts(ax, bars, counts) -> None:
    ymax = ax.get_ylim()[1]
    pad = 0.02 * ymax if ymax > 0 else 0.1
    for bar, count in zip(bars, counts):
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        ax.text(x, y + pad, f"n={int(count)}", ha="center", va="bottom", fontsize=FS - 2)


def _grouped_hist(ax, values, groups, title: str, xlabel: str, bins: int = 30) -> None:
    val = pd.Series(values).astype(float)
    grp = pd.Series(groups)
    mask = np.isfinite(val) & grp.notna()
    val = val[mask]
    grp = grp[mask]
    labels = list(pd.unique(grp))
    data = [val[grp == lab].to_numpy() for lab in labels]
    legend = [f"{lab} (n={len(chunk)})" for lab, chunk in zip(labels, data)]
    ax.hist(data, bins=bins, histtype="step", density=False, label=legend)
    ax.set_title(title, fontsize=FS)
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel("Number of samples [count]", fontsize=FS)
    ax.grid(alpha=0.3)
    if len(legend) > 0:
        ax.legend(fontsize=FS - 3)


def _save_fig(fig, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_cleaner_grouped_hists(cleaner_metrics: pd.DataFrame, output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    _grouped_hist(
        axs[0, 0],
        cleaner_metrics["sum_residual_diff"],
        cleaner_metrics["group_snr"],
        "Cleaner residual diff vs SNR bins",
        "sum residual (Gammapy - CNN)",
    )
    _grouped_hist(
        axs[0, 1],
        cleaner_metrics["sum_residual_diff"],
        cleaner_metrics["group_nbs"],
        "Cleaner residual diff vs NBS",
        "sum residual (Gammapy - CNN)",
    )
    _grouped_hist(
        axs[1, 0],
        cleaner_metrics["sum_residual_diff"],
        cleaner_metrics["group_zenith"],
        "Cleaner residual diff vs zenith",
        "sum residual (Gammapy - CNN)",
    )
    _grouped_hist(
        axs[1, 1],
        cleaner_metrics["sum_residual_diff"],
        cleaner_metrics["group_theta"],
        "Cleaner residual diff vs theta bins",
        "sum residual (Gammapy - CNN)",
    )
    p1 = output_dir / "cleaner_residual_grouped_hist.png"
    _save_fig(fig, p1)
    paths.append(p1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    _grouped_hist(
        axs[0, 0],
        cleaner_metrics["sum_on_diff"],
        cleaner_metrics["group_snr"],
        "Cleaner ON diff vs SNR bins",
        "sum ON (Gammapy - CNN)",
    )
    _grouped_hist(
        axs[0, 1],
        cleaner_metrics["sum_on_diff"],
        cleaner_metrics["group_nbs"],
        "Cleaner ON diff vs NBS",
        "sum ON (Gammapy - CNN)",
    )
    _grouped_hist(
        axs[1, 0],
        cleaner_metrics["sum_on_diff"],
        cleaner_metrics["group_zenith"],
        "Cleaner ON diff vs zenith",
        "sum ON (Gammapy - CNN)",
    )
    _grouped_hist(
        axs[1, 1],
        cleaner_metrics["sum_on_diff"],
        cleaner_metrics["group_theta"],
        "Cleaner ON diff vs theta bins",
        "sum ON (Gammapy - CNN)",
    )
    p2 = output_dir / "cleaner_on_grouped_hist.png"
    _save_fig(fig, p2)
    paths.append(p2)
    return paths


def plot_regressor_grouped_hists(reg_metrics: pd.DataFrame, output_dir: Path) -> Path:
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    _grouped_hist(
        axs[0, 0],
        reg_metrics["err_deg"],
        reg_metrics["group_snr"],
        "Regressor error vs SNR bins",
        "angular separation (deg)",
    )
    _grouped_hist(
        axs[0, 1],
        reg_metrics["err_deg"],
        reg_metrics["group_nbs"],
        "Regressor error vs NBS",
        "angular separation (deg)",
    )
    _grouped_hist(
        axs[1, 0],
        reg_metrics["err_deg"],
        reg_metrics["group_zenith"],
        "Regressor error vs zenith",
        "angular separation (deg)",
    )
    _grouped_hist(
        axs[1, 1],
        reg_metrics["err_deg"],
        reg_metrics["group_theta"],
        "Regressor error vs theta bins",
        "angular separation (deg)",
    )
    path = output_dir / "regressor_error_grouped_hist.png"
    _save_fig(fig, path)
    return path


def plot_resource_comparison(step_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    out_paths: List[Path] = []
    per_step = step_df[step_df["group"].isin(["gammapy", "cnn"])].copy()
    if per_step.empty:
        return out_paths
    per_step["stage"] = np.where(
        per_step["step"].str.contains("cleaner", case=False, na=False),
        "cleaner",
        np.where(
            per_step["step"].str.contains("regressor", case=False, na=False),
            "regressor",
            "other",
        ),
    )
    per_step["items_num"] = pd.to_numeric(per_step["items"], errors="coerce")

    # Dataset cycles are counted stage-wise from items; each stage should have one true cycle count.
    stage_cycles = (
        per_step[per_step["stage"].isin(["cleaner", "regressor"])]
        .groupby(["group", "stage"], as_index=False)["items_num"]
        .max()
    )
    total_cycles = stage_cycles.groupby("group", as_index=False)["items_num"].sum()
    total_cycles = total_cycles.rename(columns={"items_num": "n_cycles"})

    group_runtime = per_step.groupby("group", as_index=False)["duration_s"].sum()
    group_ram = per_step.groupby("group", as_index=False)["ram_peak_mb"].max()
    cmp = group_runtime.merge(group_ram, on="group", how="left").merge(total_cycles, on="group", how="left")
    cmp["duration_per_cycle_s"] = cmp["duration_s"] / cmp["n_cycles"].replace(0, np.nan)
    cmp["throughput_items_s"] = cmp["n_cycles"] / cmp["duration_s"].replace(0, np.nan)
    cmp = cmp[cmp["group"].isin(["gammapy", "cnn"])].copy()
    if cmp.empty:
        return out_paths

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("duration_per_cycle_s", "Duration per cycle"),
        ("ram_peak_mb", "RAM peak"),
        ("throughput_items_s", "Throughput"),
    ]
    for ax, (col, title) in zip(axs, metrics):
        vals = cmp[col].fillna(0.0).to_numpy()
        bars = ax.bar(cmp["group"], vals, color=["#1f77b4", "#ff7f0e"])
        _annotate_bar_counts(ax, bars, cmp["n_cycles"].fillna(0).to_numpy())
        ax.set_title(f"{title} (label: N cycles)")
        if col == "duration_per_cycle_s":
            ax.set_ylabel("Duration [s/item]")
        elif col == "ram_peak_mb":
            ax.set_ylabel("RAM peak [MB]")
        else:
            ax.set_ylabel("Throughput [items/s]")
        ax.set_xlabel("Pipeline")
        ax.grid(alpha=0.3, axis="y")
    p = output_dir / "gammapy_vs_cnn_resources.png"
    _save_fig(fig, p)
    out_paths.append(p)

    # Removed the old step-duration plot because it was misleading for this benchmark.
    return out_paths

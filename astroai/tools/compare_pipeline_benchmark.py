#!/usr/bin/env python3
"""Compare CNN vs Gammapy benchmark runs on per-item pipeline timings."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_BY_TASK: Dict[str, str] = {
    "seed_setup": "input_setup",
    "observation_setup": "input_setup",
    "dataset_read": "input_preparation",
    "dl3_to_counts_map": "input_preparation",
    "core_analysis": "analysis",
    "cnn_inference": "analysis",
    "gammapy_pipeline_call": "analysis",
    "prepare_output": "setup_once",
    "load_configuration": "setup_once",
    "model_loading": "setup_once",
    "irf_lookup": "setup_once",
    "irf_reduction": "setup_once",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two benchmark_pipeline runs with per-item averages."
    )
    parser.add_argument("--cnn-run-dir", type=str, required=True, help="Path to CNN benchmark run root.")
    parser.add_argument(
        "--gammapy-run-dir",
        type=str,
        required=True,
        help="Path to Gammapy benchmark run root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="astroai/review/pipeline-benchmark-comparison",
        help="Comparison output root; writes data/ and plots/.",
    )
    return parser.parse_args()


def _load_task_table(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "data", "task_timings.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing task timings file: {path}")
    df = pd.read_csv(path)
    df["meta_dict"] = df["meta"].fillna("{}").map(lambda x: json.loads(x) if isinstance(x, str) and x else {})
    df["seed"] = df["meta_dict"].map(lambda m: m.get("seed"))
    df["stage"] = df["task"].map(lambda t: STAGE_BY_TASK.get(t, "other"))
    return df


def _load_summary(run_dir: str) -> dict:
    path = os.path.join(run_dir, "data", "summary.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing summary file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _per_item_stats(df: pd.DataFrame) -> pd.DataFrame:
    item_df = df[df["seed"].notna()].copy()
    return (
        item_df.groupby(["task", "stage"], as_index=False)
        .agg(
            n_cycles=("seed", "nunique"),
            n_samples=("duration_s", "count"),
            mean_duration_s=("duration_s", "mean"),
            std_duration_s=("duration_s", "std"),
            p50_duration_s=("duration_s", "median"),
        )
        .sort_values(["stage", "task"])
    )


def _setup_once_stats(df: pd.DataFrame) -> pd.DataFrame:
    once_df = df[df["seed"].isna()].copy()
    return (
        once_df.groupby(["task", "stage"], as_index=False)
        .agg(
            n_calls=("duration_s", "count"),
            total_duration_s=("duration_s", "sum"),
            mean_duration_s=("duration_s", "mean"),
        )
        .sort_values(["stage", "task"])
    )


def _stage_per_item(df: pd.DataFrame) -> pd.DataFrame:
    item_df = df[df["seed"].notna()].copy()
    by_stage_seed = (
        item_df.groupby(["stage", "seed"], as_index=False)["duration_s"].sum()
    )
    return (
        by_stage_seed.groupby("stage", as_index=False)
        .agg(
            n_cycles=("seed", "nunique"),
            mean_duration_s=("duration_s", "mean"),
            std_duration_s=("duration_s", "std"),
        )
        .sort_values("stage")
    )


def _plot_shared_task_comparison(cnn: pd.DataFrame, gp: pd.DataFrame, outpath: str) -> None:
    merged = cnn.merge(
        gp,
        on=["task", "stage"],
        suffixes=("_cnn", "_gammapy"),
        how="inner",
    )
    if merged.empty:
        return
    merged = merged.sort_values(["stage", "task"]).reset_index(drop=True)
    x = np.arange(len(merged))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(merged) * 0.8), 5))
    b1 = ax.bar(x - width / 2, merged["mean_duration_s_cnn"], width, label="CNN")
    b2 = ax.bar(x + width / 2, merged["mean_duration_s_gammapy"], width, label="Gammapy")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["task"], rotation=30, ha="right")
    ax.set_ylabel("Mean duration per analysis item [s/item]")
    ax.set_xlabel("Comparable task")
    ax.set_title("Per-item task timing comparison (shared tasks only)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for bars, n_col in [(b1, "n_cycles_cnn"), (b2, "n_cycles_gammapy")]:
        for bar, n in zip(bars, merged[n_col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.01,
                f"N={int(n)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def _plot_stage_comparison(cnn_stage: pd.DataFrame, gp_stage: pd.DataFrame, outpath: str) -> None:
    merged = cnn_stage.merge(gp_stage, on="stage", suffixes=("_cnn", "_gammapy"), how="outer").fillna(0.0)
    if merged.empty:
        return
    x = np.arange(len(merged))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, merged["mean_duration_s_cnn"], width, label="CNN")
    b2 = ax.bar(x + width / 2, merged["mean_duration_s_gammapy"], width, label="Gammapy")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["stage"], rotation=20, ha="right")
    ax.set_ylabel("Mean duration per analysis item [s/item]")
    ax.set_xlabel("Benchmark stage")
    ax.set_title("Per-item stage timing comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for bars, n_col in [(b1, "n_cycles_cnn"), (b2, "n_cycles_gammapy")]:
        for bar, n in zip(bars, merged[n_col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.01 if bar.get_height() > 0 else 0.01,
                f"N={int(n)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def _plot_once_comparison(cnn_once: pd.DataFrame, gp_once: pd.DataFrame, outpath: str) -> None:
    cnn_total = cnn_once["total_duration_s"].sum() if not cnn_once.empty else 0.0
    gp_total = gp_once["total_duration_s"].sum() if not gp_once.empty else 0.0
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["cnn", "gammapy"], [cnn_total, gp_total], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("One-time setup total [s]")
    ax.set_xlabel("Pipeline")
    ax.set_title("One-time setup cost comparison")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, [cnn_total, gp_total]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value * 1.01 if value > 0 else 0.01, f"{value:.2f}s", ha="center", va="bottom")
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_data = os.path.join(args.output_dir, "data")
    out_plots = os.path.join(args.output_dir, "plots")
    os.makedirs(out_data, exist_ok=True)
    os.makedirs(out_plots, exist_ok=True)

    cnn_tasks = _load_task_table(args.cnn_run_dir)
    gp_tasks = _load_task_table(args.gammapy_run_dir)
    cnn_summary = _load_summary(args.cnn_run_dir)
    gp_summary = _load_summary(args.gammapy_run_dir)

    cnn_per_item = _per_item_stats(cnn_tasks)
    gp_per_item = _per_item_stats(gp_tasks)
    cnn_once = _setup_once_stats(cnn_tasks)
    gp_once = _setup_once_stats(gp_tasks)
    cnn_stage = _stage_per_item(cnn_tasks)
    gp_stage = _stage_per_item(gp_tasks)

    cnn_per_item.to_csv(os.path.join(out_data, "cnn_per_item_task_stats.csv"), index=False)
    gp_per_item.to_csv(os.path.join(out_data, "gammapy_per_item_task_stats.csv"), index=False)
    cnn_once.to_csv(os.path.join(out_data, "cnn_setup_once_stats.csv"), index=False)
    gp_once.to_csv(os.path.join(out_data, "gammapy_setup_once_stats.csv"), index=False)
    cnn_stage.to_csv(os.path.join(out_data, "cnn_per_item_stage_stats.csv"), index=False)
    gp_stage.to_csv(os.path.join(out_data, "gammapy_per_item_stage_stats.csv"), index=False)

    shared = cnn_per_item.merge(
        gp_per_item,
        on=["task", "stage"],
        suffixes=("_cnn", "_gammapy"),
        how="inner",
    )
    if not shared.empty:
        shared["speedup_gammapy_over_cnn"] = shared["mean_duration_s_gammapy"] / shared["mean_duration_s_cnn"]
    shared.to_csv(os.path.join(out_data, "shared_per_item_task_comparison.csv"), index=False)

    cycle_cmp = pd.DataFrame(
        [
            {
                "pipeline": "cnn",
                "work_items": cnn_summary.get("work_items"),
                "total_runtime_s": cnn_summary.get("total_runtime_s"),
                "runtime_per_item_s": (
                    float(cnn_summary["total_runtime_s"]) / float(cnn_summary["work_items"])
                    if cnn_summary.get("work_items")
                    else None
                ),
            },
            {
                "pipeline": "gammapy",
                "work_items": gp_summary.get("work_items"),
                "total_runtime_s": gp_summary.get("total_runtime_s"),
                "runtime_per_item_s": (
                    float(gp_summary["total_runtime_s"]) / float(gp_summary["work_items"])
                    if gp_summary.get("work_items")
                    else None
                ),
            },
        ]
    )
    cycle_cmp.to_csv(os.path.join(out_data, "pipeline_runtime_per_item.csv"), index=False)

    _plot_shared_task_comparison(cnn_per_item, gp_per_item, os.path.join(out_plots, "shared_task_duration_per_item.png"))
    _plot_stage_comparison(cnn_stage, gp_stage, os.path.join(out_plots, "stage_duration_per_item.png"))
    _plot_once_comparison(cnn_once, gp_once, os.path.join(out_plots, "setup_once_total.png"))

    print(f"Comparison outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Run ZALL cleaner/regressor benchmark pipeline from terminal."""

from __future__ import annotations

import argparse
import os

import pandas as pd
import tensorflow as tf

from astroai.utils.review_data import (
    add_common_meta,
    align_meta_with_test_split,
    build_group_columns,
    compute_cleaner_metrics,
    compute_regressor_metrics,
    load_dataset,
)
from astroai.utils.review_plots import plot_resource_comparison
from astroai.utils.review_profile import ExecutionProfiler
from astroai.tools.utils import split_noisy_dataset, split_regression_dataset


def parse_args() -> argparse.Namespace:
    root = "astroai"
    default_data_root = os.path.join(os.path.expanduser("~"), "E4", "irf_random", "crab")
    parser = argparse.ArgumentParser(
        description="DEPRECATED: dataset-style review benchmark. Prefer pipes benchmark + compare_pipeline_benchmark."
    )
    parser.add_argument("--root", type=str, default=root, help="astroAI package root path.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=default_data_root,
        help="Root path containing cleaner/regressor datasets and .dat files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(root, "review"),
        help="Root output directory; results are saved in data/ and plots/ subfolders.",
    )
    parser.add_argument("--binning", type=int, default=200, help="Map binning.")
    parser.add_argument("--split", type=int, default=80, help="Train/test split percentage.")
    parser.add_argument("--cleaner-table", default="cleaner_5sgm_expALL.pickle")
    parser.add_argument("--cleaner-model", default="cleaner_zALL.keras")
    parser.add_argument("--regressor-table", default="regressor_5sgm_xy_flip.pickle")
    parser.add_argument("--regressor-model", default="regressor_zALL.keras")
    parser.add_argument("--point-ref", type=float, default=100)
    parser.add_argument("--radius-deg", type=float, default=0.2)
    parser.add_argument("--pixelsize-deg", type=float, default=0.025)
    return parser.parse_args()


def save_tables(
    data_dir: str,
    cleaner_metrics: pd.DataFrame,
    reg_metrics: pd.DataFrame,
    step_table: pd.DataFrame,
    total_table: pd.DataFrame,
) -> None:
    os.makedirs(data_dir, exist_ok=True)
    cleaner_metrics.to_csv(os.path.join(data_dir, "cleaner_metrics.csv"), index=False)
    reg_metrics.to_csv(os.path.join(data_dir, "regressor_metrics.csv"), index=False)
    step_table.to_csv(os.path.join(data_dir, "profiling_steps.csv"), index=False)
    total_table.to_csv(os.path.join(data_dir, "profiling_total.csv"), index=False)

    summary_rows = []
    for group in ["gammapy", "cnn"]:
        subset = step_table[step_table["group"] == group]
        if subset.empty:
            continue
        summary_rows.append(
            {
                "group": group,
                "duration_s_sum": subset["duration_s"].sum(),
                "duration_s_mean": subset["duration_s"].mean(),
                "ram_peak_mb_max": subset["ram_peak_mb"].max(),
                "throughput_items_s_mean": subset["throughput_items_s"].dropna().mean(),
            }
        )
    pd.DataFrame(summary_rows).to_csv(os.path.join(data_dir, "gammapy_vs_cnn_summary.csv"), index=False)


def main() -> int:
    args = parse_args()
    print("WARNING: run_review_benchmark is deprecated. Use pipes benchmark + compare_pipeline_benchmark.", flush=True)
    output_root = args.output_dir
    data_dir = os.path.join(output_root, "data")
    plots_dir = os.path.join(output_root, "plots")
    profiler = ExecutionProfiler()

    pixelsize = args.pixelsize_deg
    radius_pix = args.radius_deg / pixelsize

    profiler.begin_step("load_cleaner_data")
    cleaner_ds = load_dataset(os.path.join(args.data_root, args.cleaner_table))
    cleaner_info = pd.read_csv(
        os.path.join(args.data_root, args.cleaner_table.replace(".pickle", ".dat")),
        sep=" ",
        header=0,
    ).sort_values(by=["seed"])
    profiler.end_step("load_cleaner_data", group="cnn")

    profiler.begin_step("split_cleaner_data")
    train_clean, train_noisy, test_clean, test_noisy = split_noisy_dataset(
        cleaner_ds, split=args.split, reshape=True, binning=args.binning
    )
    profiler.end_step("split_cleaner_data", group="cnn", items=len(test_noisy))

    profiler.begin_step("cleaner_cnn_inference")
    cleaner_model = tf.keras.models.load_model(os.path.join(args.root, "models", "crta_models", args.cleaner_model))
    cleaner_pred = cleaner_model.predict(test_noisy, verbose=0)
    profiler.end_step("cleaner_cnn_inference", group="cnn", items=len(test_noisy))

    profiler.begin_step("prepare_cleaner_meta")
    cleaner_meta = align_meta_with_test_split(cleaner_info, len(train_noisy), len(test_noisy))
    cleaner_meta = add_common_meta(cleaner_meta)
    profiler.end_step("prepare_cleaner_meta", group="gammapy", items=len(cleaner_meta))

    profiler.begin_step("compute_cleaner_metrics")
    cleaner_metrics = compute_cleaner_metrics(
        test_noisy=test_noisy,
        test_clean=test_clean,
        cleaner_pred=cleaner_pred,
        cleaner_meta=cleaner_meta,
        point_ref=args.point_ref,
        pixelsize=pixelsize,
        radius_pix=radius_pix,
    )
    cleaner_metrics = build_group_columns(cleaner_metrics)
    profiler.sample_step("compute_cleaner_metrics")
    profiler.end_step("compute_cleaner_metrics", group="gammapy", items=len(cleaner_metrics))

    profiler.begin_step("load_regressor_data")
    reg_ds = load_dataset(os.path.join(args.data_root, args.regressor_table))
    reg_info = pd.read_csv(
        os.path.join(args.data_root, args.regressor_table.replace(".pickle", ".dat")),
        sep=" ",
        header=0,
    ).sort_values(by=["seed"])
    profiler.end_step("load_regressor_data", group="cnn")

    profiler.begin_step("split_regressor_data")
    train_data, _, test_data, _ = split_regression_dataset(
        reg_ds, split=args.split, reshape=True, binning=args.binning
    )
    profiler.end_step("split_regressor_data", group="cnn", items=len(test_data))

    profiler.begin_step("regressor_cnn_inference")
    reg_model = tf.keras.models.load_model(os.path.join(args.root, "models", "crta_models", args.regressor_model))
    reg_pred = reg_model.predict(test_data, verbose=0)
    profiler.end_step("regressor_cnn_inference", group="cnn", items=len(test_data))

    profiler.begin_step("prepare_regressor_meta")
    reg_meta = align_meta_with_test_split(reg_info, len(train_data), len(test_data))
    reg_meta = add_common_meta(reg_meta)
    profiler.end_step("prepare_regressor_meta", group="gammapy", items=len(reg_meta))

    profiler.begin_step("compute_regressor_metrics")
    reg_metrics = compute_regressor_metrics(
        reg_meta=reg_meta,
        reg_pred=reg_pred,
        point_ref=args.point_ref,
        pixelsize=pixelsize,
        binning=args.binning,
    )
    reg_metrics = build_group_columns(reg_metrics)
    profiler.sample_step("compute_regressor_metrics")
    profiler.end_step("compute_regressor_metrics", group="gammapy", items=len(reg_metrics))

    profiler.begin_step("make_resource_plots")
    step_table = profiler.to_step_table()
    os.makedirs(plots_dir, exist_ok=True)
    plot_resource_comparison(step_table, plots_dir)
    profiler.end_step("make_resource_plots", group="overall")

    step_table = profiler.to_step_table()
    total_table = profiler.execution_summary()
    save_tables(data_dir, cleaner_metrics, reg_metrics, step_table, total_table)

    print(f"Benchmark completed. Outputs written to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


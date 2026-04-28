#!/usr/bin/env python3
"""Create science-side degradation plots from saved result tables."""

from __future__ import annotations

import argparse
import os

import pandas as pd

from astroai.utils.review_data import build_group_columns
from astroai.utils.review_plots import (
    plot_cleaner_grouped_hists,
    plot_regressor_grouped_hists,
)


def parse_args() -> argparse.Namespace:
    default_root = os.path.join("astroai", "review")
    parser = argparse.ArgumentParser(
        description="Generate cleaner/regressor science plots from saved CSV metrics."
    )
    parser.add_argument(
        "--review-dir",
        type=str,
        default=default_root,
        help="Root review directory containing data/ and plots/ subfolders.",
    )
    return parser.parse_args()


def _ensure_groups(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"group_snr", "group_nbs", "group_zenith", "group_theta"}
    if needed.issubset(df.columns):
        return df
    return build_group_columns(df)


def main() -> int:
    args = parse_args()
    review_dir = args.review_dir
    data_dir = os.path.join(review_dir, "data")
    plots_dir = os.path.join(review_dir, "plots")
    cleaner_path = os.path.join(data_dir, "cleaner_metrics.csv")
    regressor_path = os.path.join(data_dir, "regressor_metrics.csv")

    if not os.path.isfile(cleaner_path):
        raise FileNotFoundError(f"Missing file: {cleaner_path}")
    if not os.path.isfile(regressor_path):
        raise FileNotFoundError(f"Missing file: {regressor_path}")

    cleaner_metrics = _ensure_groups(pd.read_csv(cleaner_path))
    reg_metrics = _ensure_groups(pd.read_csv(regressor_path))

    os.makedirs(plots_dir, exist_ok=True)
    plot_cleaner_grouped_hists(cleaner_metrics, plots_dir)
    plot_regressor_grouped_hists(reg_metrics, plots_dir)

    print(f"Science plots saved in: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Data preprocessing

After you have compiled the configuration file following the commentary in [template_cnn.yml](../conf/template_cnn.yml) you can execute the data preprocessing pipeline.

```bash
python preprocess_ds.py -f ../conf/your_conf_file.yml
```

To pre-clean the processed dataset using a trained cnn-cleaner you trained or that was provided by this project.

```bash
python preclean_ds.py -f ../conf/your_conf_file.yml
```

Alternatively you can submit a job to [Slurm](https://slurm.schedmd.com/documentation.html) if available to you.

```bash
python slurmjobs.py -f ../conf/your_conf_file.yml -s preprocess_ds
```

or 

```bash
python slurmjobs.py -f ../conf/your_conf_file.yml -s preclean_ds
```

## Review analysis scripts

### 1) Benchmark pipeline run (resources)

Run the (deprecated) dataset-style review benchmark:

```bash
python -m astroai.tools.run_review_benchmark
```

This script saves by default under:

`astroai/review`

with:
- `data/` for CSV and tables
- `plots/` for figures

and produces:
- `cleaner_metrics.csv`
- `regressor_metrics.csv`
- `profiling_steps.csv`
- `profiling_total.csv`
- `gammapy_vs_cnn_summary.csv`
- resource comparison plot (`gammapy_vs_cnn_resources.png`)

Note: this script is deprecated for performance benchmarking. Use pipeline benchmarking from `astroai/pipes` with:
- `python ../tools/benchmark_pipeline.py ...`
- `python ../tools/compare_pipeline_benchmark.py ...`

### 2) Science plots from saved results

Create SNR/zenith/NBS/theta science plots from already saved CSV results (no pipeline execution):

```bash
python -m astroai.tools.plot_review_science
```

This reproduces the science-side grouped histograms from:
- `cleaner_metrics.csv`
- `regressor_metrics.csv`

You can override inputs/outputs from CLI, for example:

```bash
python -m astroai.tools.run_review_benchmark \
  --data-root "$HOME/E4/irf_random/crab" \
  --output-dir "astroai/review"
```

```bash
python -m astroai.tools.plot_review_science \
  --review-dir "astroai/review"
```

Implementation note: reusable function-only helpers used by this script are located in `astroai/utils` (`review_data.py`, `review_plots.py`, `review_profile.py`).


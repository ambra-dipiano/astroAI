# Reference pipeline

After you have compiled the configuration file following the commentary in [template_gp.yml](../conf/template_gp.yml) you can run the reference real-time pipeline for comparison.

```bash
python run_gammapy.py -f ../conf/your_conf_file.yml
```

Alternatively you can submit a job to [Slurm](https://slurm.schedmd.com/documentation.html) if available to you.

```bash
python slurmjobs.py -f ../conf/your_conf_file.yml -p gammapy
```

## Benchmarking pipelines

To benchmark pipeline latency, throughput, CPU/RAM/GPU usage, and per-task timings you can use:

```bash
python ../tools/benchmark_pipeline.py \
  --name cnn_make_map \
  --command "python run_cnn.py -f ../conf/your_conf_file.yml" \
  --work-items 100
```

For the reference pipeline (full run with map-making):

```bash
python ../tools/benchmark_pipeline.py \
  --name gammapy_make_map \
  --command "python run_gammapy.py -f ../conf/your_conf_file.yml" \
  --work-items 100
```

Each benchmark run writes:
- `summary.json` with end-to-end and core runtime metrics
- `task_timings.csv` with per-task durations from internal task markers
- `resource_samples.json` with sampled CPU/RAM/GPU time series
- `stdout.log` and `stderr.log` from the pipeline command

All these files are under `<run>/data` and plots can be written under `<run>/plots`.

Common task markers used for direct CNN/Gammapy mapping:
- `load_configuration`
- `prepare_output`
- `seed_setup`
- `observation_setup`
- `dataset_read`
- `core_analysis`

Pipeline-specific markers:
- CNN: `model_loading`, `cnn_inference`
- Gammapy: `irf_lookup`, `irf_reduction`, `dl3_to_counts_map`

Recommended comparison modes:
- Make-map mode (full end-to-end): use names `cnn_make_map` and `gammapy_make_map` with no exclusions.
- Skip-map mode (core-only on prepared inputs): use names `cnn_skip_map` and `gammapy_skip_map`.
  - CNN: `--exclude-task model_loading --exclude-task dataset_read`
  - Gammapy: `--exclude-task irf_reduction --exclude-task irf_lookup --exclude-task dl3_to_counts_map`

## Compare benchmark runs correctly

To compare pipelines with **per-analysis-item average timing** (not cumulative dataset timing), run:

```bash
python ../tools/compare_pipeline_benchmark.py \
  --cnn-run-dir ../review/benchmarks/cnn_make_map_YYYYMMDDTHHMMSSZ \
  --gammapy-run-dir ../review/benchmarks/gammapy_make_map_YYYYMMDDTHHMMSSZ \
  --output-dir ../review/pipeline-benchmark-comparison
```

This script separates:
- `setup_once` tasks (no `seed` marker): one-time initialization cost
- per-item tasks (`seed` present): averaged as seconds per analysis item (`s/item`)

Outputs:
- `data/shared_per_item_task_comparison.csv`
- `data/*_per_item_task_stats.csv`
- `data/*_setup_once_stats.csv`
- `plots/shared_task_duration_per_item.png`
- `plots/stage_duration_per_item.png`
- `plots/setup_once_total.png`


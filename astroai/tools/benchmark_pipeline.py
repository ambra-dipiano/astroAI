#!/usr/bin/env python3
"""Generic benchmarking utility for astroAI pipelines."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple


MARKER_PREFIX = "ASTROAI_BENCHMARK "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an arbitrary pipeline command.")
    parser.add_argument("--name", required=True, help="Benchmark name used in output files.")
    parser.add_argument("--command", required=True, help="Command to execute (quoted string).")
    parser.add_argument(
        "--output-dir",
        default="review/benchmarks",
        help="Root directory where benchmark run folders are created.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds for CPU/RAM/GPU snapshots.",
    )
    parser.add_argument(
        "--work-items",
        type=float,
        default=None,
        help="Total items processed (events/observations) for throughput estimation.",
    )
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=[],
        help="Task name to exclude from core runtime. Repeatable.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra env variable in KEY=VALUE format. Repeatable.",
    )
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Disable ASTROAI_BENCHMARK_MARKERS injection.",
    )
    return parser.parse_args()


def has_nvidia_smi() -> bool:
    return subprocess.run(
        ["bash", "-lc", "command -v nvidia-smi >/dev/null 2>&1"], check=False
    ).returncode == 0


def read_proc_metrics(pid: int) -> Optional[Tuple[float, int]]:
    stat_path = f"/proc/{pid}/stat"
    status_path = f"/proc/{pid}/status"
    if not os.path.isfile(stat_path) or not os.path.isfile(status_path):
        return None

    with open(stat_path, "r", encoding="utf-8") as handle:
        stat_parts = handle.read().split()
    # utime and stime are columns 14 and 15 in /proc/<pid>/stat (1-indexed)
    total_ticks = float(stat_parts[13]) + float(stat_parts[14])

    rss_kb = 0
    with open(status_path, "r", encoding="utf-8") as handle:
        status_lines = handle.read().splitlines()
    for line in status_lines:
        if line.startswith("VmRSS:"):
            rss_kb = int(line.split()[1])
            break
    return total_ticks, rss_kb


def read_total_cpu_ticks() -> float:
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        first = handle.readline().strip().split()
    # Sum all CPU states from "cpu ..."
    return float(sum(int(x) for x in first[1:]))


def sample_gpu() -> Optional[Dict[str, float]]:
    if not has_nvidia_smi():
        return None
    cmd = (
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits"
    )
    out = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, check=False)
    if out.returncode != 0 or not out.stdout.strip():
        return None
    first_gpu = out.stdout.splitlines()[0]
    util, mem_used, mem_total = [x.strip() for x in first_gpu.split(",")]
    return {
        "utilization_pct": float(util),
        "memory_used_mb": float(mem_used),
        "memory_total_mb": float(mem_total),
    }


def parse_marker(line: str) -> Optional[Dict]:
    if not line.startswith(MARKER_PREFIX):
        return None
    raw = line[len(MARKER_PREFIX) :].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def build_task_rows(markers: List[Dict]) -> List[Dict]:
    open_tasks: Dict[str, List[float]] = {}
    rows: List[Dict] = []
    for marker in markers:
        event = marker.get("event")
        task = marker.get("task")
        timestamp = marker.get("timestamp")
        meta = marker.get("meta", {})
        if not task or not isinstance(timestamp, (int, float)):
            continue
        if event == "TASK_START":
            open_tasks.setdefault(task, []).append(float(timestamp))
        elif event == "TASK_END":
            start_list = open_tasks.get(task, [])
            if not start_list:
                continue
            start_ts = start_list.pop(0)
            rows.append(
                {
                    "task": task,
                    "start_ts": start_ts,
                    "end_ts": float(timestamp),
                    "duration_s": float(timestamp) - start_ts,
                    "meta": meta,
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.output_dir, f"{args.name}_{timestamp}")
    data_dir = os.path.join(out_dir, "data")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    stdout_log = os.path.join(data_dir, "stdout.log")
    stderr_log = os.path.join(data_dir, "stderr.log")
    samples_json = os.path.join(data_dir, "resource_samples.json")
    tasks_csv = os.path.join(data_dir, "task_timings.csv")
    summary_json = os.path.join(data_dir, "summary.json")

    env = os.environ.copy()
    if not args.no_markers:
        env["ASTROAI_BENCHMARK_MARKERS"] = "1"
    for item in args.env:
        if "=" not in item:
            raise ValueError(f"Invalid --env value: {item}")
        key, value = item.split("=", 1)
        env[key] = value

    proc = subprocess.Popen(
        ["bash", "-lc", args.command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )

    markers: List[Dict] = []
    lock = threading.Lock()

    def consume_stream(stream, destination: str) -> None:
        with open(destination, "w", encoding="utf-8") as handle:
            for line in iter(stream.readline, ""):
                handle.write(line)
                marker = parse_marker(line)
                if marker:
                    with lock:
                        markers.append(marker)
            stream.close()

    stdout_thread = threading.Thread(target=consume_stream, args=(proc.stdout, stdout_log))
    stderr_thread = threading.Thread(target=consume_stream, args=(proc.stderr, stderr_log))
    stdout_thread.start()
    stderr_thread.start()

    cpu_samples: List[float] = []
    ram_samples_mb: List[float] = []
    gpu_samples: List[Dict[str, float]] = []

    prev_proc = read_proc_metrics(proc.pid)
    prev_total = read_total_cpu_ticks()
    start_time = time.perf_counter()

    while proc.poll() is None:
        time.sleep(args.sample_interval)
        proc_metrics = read_proc_metrics(proc.pid)
        total_ticks = read_total_cpu_ticks()
        if proc_metrics and prev_proc:
            delta_proc = proc_metrics[0] - prev_proc[0]
            delta_total = total_ticks - prev_total
            cpu_pct = (100.0 * delta_proc / delta_total) if delta_total > 0 else 0.0
            cpu_samples.append(cpu_pct)
            ram_samples_mb.append(proc_metrics[1] / 1024.0)
            prev_proc = proc_metrics
            prev_total = total_ticks
        gpu = sample_gpu()
        if gpu:
            gpu_samples.append(gpu)

    end_time = time.perf_counter()
    stdout_thread.join()
    stderr_thread.join()

    task_rows = build_task_rows(markers)
    excluded = set(args.exclude_task)
    excluded_time = sum(row["duration_s"] for row in task_rows if row["task"] in excluded)
    total_runtime = end_time - start_time
    core_runtime = total_runtime - excluded_time

    throughput_full = None
    throughput_core = None
    if args.work_items and total_runtime > 0:
        throughput_full = args.work_items / total_runtime
    if args.work_items and core_runtime > 0:
        throughput_core = args.work_items / core_runtime

    summary = {
        "name": args.name,
        "command": args.command,
        "return_code": proc.returncode,
        "total_runtime_s": total_runtime,
        "core_runtime_s": core_runtime,
        "excluded_task_time_s": excluded_time,
        "excluded_tasks": sorted(excluded),
        "work_items": args.work_items,
        "throughput_full_items_per_s": throughput_full,
        "throughput_core_items_per_s": throughput_core,
        "cpu_avg_pct": (sum(cpu_samples) / len(cpu_samples)) if cpu_samples else None,
        "cpu_peak_pct": max(cpu_samples) if cpu_samples else None,
        "ram_peak_mb": max(ram_samples_mb) if ram_samples_mb else None,
        "gpu_avg_util_pct": (
            sum(s["utilization_pct"] for s in gpu_samples) / len(gpu_samples)
            if gpu_samples
            else None
        ),
        "gpu_peak_util_pct": (
            max(s["utilization_pct"] for s in gpu_samples) if gpu_samples else None
        ),
        "gpu_peak_mem_mb": (
            max(s["memory_used_mb"] for s in gpu_samples) if gpu_samples else None
        ),
        "output_dir": str(out_dir),
        "data_dir": str(data_dir),
        "plots_dir": str(plots_dir),
    }

    with open(samples_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cpu_percent_samples": cpu_samples,
                "ram_mb_samples": ram_samples_mb,
                "gpu_samples": gpu_samples,
                "markers": markers,
            },
            handle,
            indent=2,
        )

    with open(tasks_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "start_ts", "end_ts", "duration_s", "meta"])
        writer.writeheader()
        for row in task_rows:
            writer.writerow(
                {
                    "task": row["task"],
                    "start_ts": row["start_ts"],
                    "end_ts": row["end_ts"],
                    "duration_s": row["duration_s"],
                    "meta": json.dumps(row["meta"], sort_keys=True),
                }
            )

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

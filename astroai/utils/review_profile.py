"""Profiling helpers for review analysis scripts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _read_process_rss_mb() -> float:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return 0.0
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            rss_kb = float(line.split()[1])
            return rss_kb / 1024.0
    return 0.0


@dataclass
class StepRecord:
    step: str
    group: str
    duration_s: float
    ram_start_mb: float
    ram_end_mb: float
    ram_peak_mb: float
    items: Optional[int]
    throughput_items_s: Optional[float]


class ExecutionProfiler:
    """Collect timing, memory and throughput information per step."""

    def __init__(self) -> None:
        self._records: List[StepRecord] = []
        self._step_start: Dict[str, float] = {}
        self._step_start_ram: Dict[str, float] = {}
        self._step_peak_ram: Dict[str, float] = {}
        self._exec_start = time.perf_counter()
        self._exec_start_ram = _read_process_rss_mb()

    def begin_step(self, step: str) -> None:
        self._step_start[step] = time.perf_counter()
        ram_now = _read_process_rss_mb()
        self._step_start_ram[step] = ram_now
        self._step_peak_ram[step] = ram_now

    def sample_step(self, step: str) -> None:
        if step in self._step_peak_ram:
            self._step_peak_ram[step] = max(self._step_peak_ram[step], _read_process_rss_mb())

    def end_step(self, step: str, group: str, items: Optional[int] = None) -> None:
        end_t = time.perf_counter()
        end_ram = _read_process_rss_mb()
        start_t = self._step_start.pop(step)
        start_ram = self._step_start_ram.pop(step)
        peak_ram = max(self._step_peak_ram.pop(step), end_ram)
        duration = end_t - start_t
        throughput = None
        if items is not None and items > 0 and duration > 0:
            throughput = float(items) / duration
        self._records.append(
            StepRecord(
                step=step,
                group=group,
                duration_s=duration,
                ram_start_mb=start_ram,
                ram_end_mb=end_ram,
                ram_peak_mb=peak_ram,
                items=items,
                throughput_items_s=throughput,
            )
        )

    def to_step_table(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self._records])

    def execution_summary(self) -> pd.DataFrame:
        end_t = time.perf_counter()
        end_ram = _read_process_rss_mb()
        total_duration = end_t - self._exec_start
        overall_peak_ram = end_ram
        if self._records:
            overall_peak_ram = max(overall_peak_ram, max(r.ram_peak_mb for r in self._records))
        total_items = 0
        for rec in self._records:
            if rec.items is not None:
                total_items += rec.items
        throughput = None
        if total_items > 0 and total_duration > 0:
            throughput = float(total_items) / total_duration

        return pd.DataFrame(
            [
                {
                    "step": "TOTAL",
                    "group": "overall",
                    "duration_s": total_duration,
                    "ram_start_mb": self._exec_start_ram,
                    "ram_end_mb": end_ram,
                    "ram_peak_mb": overall_peak_ram,
                    "items": total_items if total_items > 0 else None,
                    "throughput_items_s": throughput,
                }
            ]
        )

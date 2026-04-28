"""Helpers to emit structured benchmark markers from pipelines."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def benchmark_markers_enabled() -> bool:
    """Return True when benchmark marker emission is enabled."""
    return os.environ.get("ASTROAI_BENCHMARK_MARKERS", "0") == "1"


def emit_benchmark_marker(event: str, task: str, **metadata: Any) -> None:
    """Emit a single JSON marker line to stdout.

    The line starts with a fixed prefix to make downstream parsing robust.
    """
    if not benchmark_markers_enabled():
        return

    payload: Dict[str, Any] = {
        "event": event,
        "task": task,
        "timestamp": time.perf_counter(),
    }
    if metadata:
        payload["meta"] = metadata
    print(f"ASTROAI_BENCHMARK {json.dumps(payload, sort_keys=True)}", flush=True)


class BenchmarkTask:
    """Simple context manager for START/END task marker pairs."""

    def __init__(self, task: str, **metadata: Any):
        self.task = task
        self.metadata = metadata

    def __enter__(self) -> "BenchmarkTask":
        emit_benchmark_marker("TASK_START", self.task, **self.metadata)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> Optional[bool]:
        outcome = "error" if exc is not None else "ok"
        emit_benchmark_marker("TASK_END", self.task, outcome=outcome)
        return None

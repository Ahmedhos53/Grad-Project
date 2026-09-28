"""Measure bounded dashboard resource use without retaining media.

The runner starts the existing dashboard in a subprocess with a frame limit,
samples process CPU/RSS/thread count using psutil, and writes only aggregate
resource values plus whitelisted lifecycle markers. Automatic screenshots are
disabled for this measurement; transcript recording is always disabled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "src" / "video" / "driver_visual_prototype.py"


def _aggregate_process_stats(process: psutil.Process) -> tuple[float, float, int]:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    cpu_seconds = 0.0
    rss = 0
    threads = 0
    for item in processes:
        try:
            cpu_times = item.cpu_times()
            cpu_seconds += float(cpu_times.user + cpu_times.system)
            rss += int(item.memory_info().rss)
            threads += int(item.num_threads())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return cpu_seconds, rss / (1024.0 * 1024.0), threads


def _lifecycle_markers(output: str) -> dict[str, bool]:
    return {
        "audio_started": "Audio pipeline listening:" in output,
        "telemetry_started": "Telemetry replay ready:" in output,
        "bounded_run_started": "Bounded smoke run:" in output,
        "bounded_run_stopped": "Maximum frame limit reached" in output,
        "clean_interrupt_message": "Shutting down CabInspector cleanly" in output,
        "traceback_present": "Traceback (most recent call last)" in output,
    }


def _gpu_snapshot() -> dict[str, object] | None:
    """Read one aggregate NVIDIA GPU sample without touching media or model data."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 4:
        return None
    try:
        return {
            "name": fields[0],
            "utilization_percent": float(fields[1]),
            "memory_used_mb": float(fields[2]),
            "memory_total_mb": float(fields[3]),
        }
    except ValueError:
        return None


def _summarize_gpu(samples: list[dict[str, object]]) -> dict[str, object]:
    if not samples:
        return {"available": False, "samples": 0}
    utilization = [float(sample["utilization_percent"]) for sample in samples]
    memory_used = [float(sample["memory_used_mb"]) for sample in samples]
    return {
        "available": True,
        "name": samples[0]["name"],
        "samples": len(samples),
        "utilization_percent_mean": round(sum(utilization) / len(utilization), 2),
        "utilization_percent_max": round(max(utilization), 2),
        "memory_used_mb_mean": round(sum(memory_used) / len(memory_used), 2),
        "memory_used_mb_max": round(max(memory_used), 2),
        "memory_total_mb": round(float(samples[0]["memory_total_mb"]), 2),
    }


def run_measurement(
    *,
    frames: int,
    audio: bool,
    telemetry: bool,
    trip: int,
    speed: float,
    model: str,
    whisper_model: str,
    sample_interval: float,
    runtime_metrics_path: Path | None = None,
) -> dict[str, object]:
    if frames < 1:
        raise ValueError("frames must be positive")
    if sample_interval <= 0.0:
        raise ValueError("sample_interval must be positive")

    environment = os.environ.copy()
    environment.update(
        {
            "CABINSPECTOR_MAX_FRAMES": str(frames),
            "CABINSPECTOR_ENABLE_AUTO_SCREENSHOTS": "0",
            "CABINSPECTOR_STORE_TRANSCRIPTS": "0",
            "CABINSPECTOR_USE_AUDIO": "1" if audio else "0",
            "CABINSPECTOR_USE_TELEMETRY_REPLAY": "1" if telemetry else "0",
            "CABINSPECTOR_TELEMETRY_REPLAY_TRIP": str(trip),
            "CABINSPECTOR_TELEMETRY_REPLAY_SPEED": str(speed),
            "CABINSPECTOR_TELEMETRY_REPLAY_MODEL": model,
            "CABINSPECTOR_WHISPER_MODEL": whisper_model,
        }
    )
    if runtime_metrics_path is not None:
        runtime_metrics_path.unlink(missing_ok=True)
        environment["CABINSPECTOR_RUNTIME_METRICS_PATH"] = str(runtime_metrics_path.resolve())
    command = [sys.executable, str(APP_PATH)]
    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    process = psutil.Process(child.pid)
    samples: list[dict[str, float | int]] = []
    previous_cpu_seconds, _, _ = _aggregate_process_stats(process)
    previous_sample_at = time.perf_counter()
    gpu_samples: list[dict[str, object]] = []
    while child.poll() is None:
        time.sleep(sample_interval)
        sampled_at = time.perf_counter()
        cpu_seconds, rss_mb, threads = _aggregate_process_stats(process)
        elapsed = max(sampled_at - previous_sample_at, 1e-9)
        cpu = max(0.0, (cpu_seconds - previous_cpu_seconds) / elapsed * 100.0)
        samples.append(
            {
                "elapsed_seconds": round(sampled_at - started, 3),
                "cpu_percent": round(cpu, 2),
                "rss_mb": round(rss_mb, 2),
                "threads": threads,
            }
        )
        gpu_sample = _gpu_snapshot()
        if gpu_sample is not None:
            gpu_samples.append(gpu_sample)
        previous_cpu_seconds = cpu_seconds
        previous_sample_at = sampled_at
    output, _ = child.communicate()
    duration = time.perf_counter() - started
    runtime_metrics: dict[str, object] | None = None
    if runtime_metrics_path is not None and runtime_metrics_path.is_file():
        try:
            runtime_metrics = json.loads(runtime_metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            runtime_metrics = None
    cpu_values = [float(sample["cpu_percent"]) for sample in samples]
    rss_values = [float(sample["rss_mb"]) for sample in samples]
    thread_values = [int(sample["threads"]) for sample in samples]
    lifecycle = _lifecycle_markers(output)
    completed_frames = frames if lifecycle["bounded_run_stopped"] else None
    return {
        "status": "bounded_dashboard_resource_measurement",
        "frames_requested": frames,
        "audio_enabled": audio,
        "telemetry_enabled": telemetry,
        "telemetry_model": model if telemetry else "",
        "telemetry_trip": trip if telemetry else 0,
        "telemetry_speed": speed if telemetry else 0.0,
        "whisper_model": whisper_model if audio else "",
        "automatic_screenshots_enabled": False,
        "transcript_recording_enabled": False,
        "exit_code": child.returncode,
        "wall_time_seconds": round(duration, 3),
        "frames_completed": completed_frames,
        "effective_fps": round(completed_frames / duration, 3) if completed_frames else None,
        "samples": len(samples),
        "cpu_percent_mean": round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else 0.0,
        "cpu_percent_max": round(max(cpu_values), 2) if cpu_values else 0.0,
        "rss_mb_mean": round(sum(rss_values) / len(rss_values), 2) if rss_values else 0.0,
        "rss_mb_max": round(max(rss_values), 2) if rss_values else 0.0,
        "threads_max": max(thread_values) if thread_values else 0,
        "gpu": _summarize_gpu(gpu_samples),
        "runtime_metrics": runtime_metrics,
        "lifecycle": lifecycle,
        "limitations": [
            "Resource samples cover only this bounded local run and are hardware/configuration dependent.",
            "The runner records aggregate process statistics, not media or transcript content.",
            "Effective FPS includes model startup and display-loop overhead; it is not a per-model latency measurement.",
            "Per-stage/model and queue metrics are present only when the dashboard writes its explicit runtime metrics file.",
            "An exit code of zero is authoritative only for bounded runs; live interactive use remains unbounded.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--audio", action="store_true", help="enable the live audio pipeline")
    parser.add_argument("--telemetry", action="store_true", help="enable public PRIMUS replay")
    parser.add_argument("--trip", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--model", choices=("primus", "random_forest"), default="primus")
    parser.add_argument("--whisper-model", default="large-v3-turbo")
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "combined_resource_evaluation.json",
    )
    args = parser.parse_args()
    try:
        runtime_metrics_path = args.output.with_suffix(args.output.suffix + ".runtime.json")
        report = run_measurement(
            frames=args.frames,
            audio=args.audio,
            telemetry=args.telemetry,
            trip=args.trip,
            speed=args.speed,
            model=args.model,
            whisper_model=args.whisper_model,
            sample_interval=args.sample_interval,
            runtime_metrics_path=runtime_metrics_path,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluation written: {args.output}")
    print(f"Exit code: {report['exit_code']}")
    print(f"Effective FPS: {report['effective_fps']}")
    print(f"RSS max MB: {report['rss_mb_max']}")
    print(f"CPU max percent: {report['cpu_percent_max']}")
    print(f"GPU samples: {report['gpu']['samples']}")
    print(f"Traceback present: {report['lifecycle']['traceback_present']}")
    return 0 if report["exit_code"] == 0 and not report["lifecycle"]["traceback_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

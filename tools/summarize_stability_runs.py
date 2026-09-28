"""Summarize repeated privacy-safe bounded dashboard resource runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_RUNS = (
    EVALUATION_DIR / "combined_resource_stability_900.json",
    EVALUATION_DIR / "combined_resource_stability_900_repeat2.json",
    EVALUATION_DIR / "combined_resource_stability_900_repeat3.json",
)
DEFAULT_OUTPUT = EVALUATION_DIR / "combined_resource_stability_summary.json"


def _range(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "max": round(max(values), 3),
    }


def summarize(paths: tuple[Path, ...]) -> dict[str, object]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    clean_runs = [
        report
        for report in reports
        if report.get("exit_code") == 0
        and not report.get("lifecycle", {}).get("traceback_present", True)
        and report.get("frames_completed") == report.get("frames_requested") == 900
        and report.get("lifecycle", {}).get("audio_started") is True
        and report.get("lifecycle", {}).get("telemetry_started") is True
    ]
    durations = [float(report["wall_time_seconds"]) for report in reports]
    fps = [float(report["effective_fps"]) for report in reports]
    rss_peak = [float(report["rss_mb_max"]) for report in reports]
    rss_mean = [float(report["rss_mb_mean"]) for report in reports]
    cpu_mean = [float(report["cpu_percent_mean"]) for report in reports]
    cpu_peak = [float(report["cpu_percent_max"]) for report in reports]
    thread_peak = [float(report["threads_max"]) for report in reports]
    gpu_reports = [report.get("gpu", {}) for report in reports]
    gpu_util_mean = [float(gpu["utilization_percent_mean"]) for gpu in gpu_reports if gpu.get("available")]
    gpu_util_peak = [float(gpu["utilization_percent_max"]) for gpu in gpu_reports if gpu.get("available")]
    vram_peak = [float(gpu["memory_used_mb_max"]) for gpu in gpu_reports if gpu.get("available")]
    runtime_reports = [report.get("runtime_metrics") for report in reports]
    runtime_present = [runtime for runtime in runtime_reports if isinstance(runtime, dict)]
    runtime_stage_names = sorted(
        {
            stage
            for runtime in runtime_present
            for stage in (runtime.get("stages", {}) if isinstance(runtime.get("stages", {}), dict) else {})
        }
    )
    runtime_no_media = all(
        runtime.get("media_saved") is False and runtime.get("transcript_text_saved") is False
        for runtime in runtime_present
    )
    runtime_queue_drops = []
    runtime_whisper_completions = []
    runtime_yamnet_means = []
    runtime_yamnet_maxima = []
    runtime_whisper_means = []
    runtime_whisper_maxima = []
    for runtime in runtime_present:
        gauges = runtime.get("gauges", {}) if isinstance(runtime.get("gauges", {}), dict) else {}
        runtime_queue_drops.append(
            sum(
                int(value)
                for name, value in gauges.items()
                if str(name).endswith("queue_drops") and isinstance(value, (int, float))
            )
        )
        runtime_whisper_completions.append(int(gauges.get("audio.transcription.completed_count", 0)))
        runtime_yamnet_means.append(float(gauges.get("audio.yamnet.inference_mean_ms", 0.0)))
        runtime_yamnet_maxima.append(float(gauges.get("audio.yamnet.inference_max_ms", 0.0)))
        runtime_whisper_means.append(float(gauges.get("audio.transcription.inference_mean_ms", 0.0)))
        runtime_whisper_maxima.append(float(gauges.get("audio.transcription.inference_max_ms", 0.0)))

    stage_mean_ranges: dict[str, dict[str, float]] = {}
    stage_p95_ranges: dict[str, dict[str, float]] = {}
    for stage in runtime_stage_names:
        mean_values = [
            float(runtime.get("stages", {}).get(stage, {}).get("mean_ms", 0.0))
            for runtime in runtime_present
            if isinstance(runtime.get("stages", {}), dict) and stage in runtime.get("stages", {})
        ]
        p95_values = [
            float(runtime.get("stages", {}).get(stage, {}).get("p95_ms", 0.0))
            for runtime in runtime_present
            if isinstance(runtime.get("stages", {}), dict) and stage in runtime.get("stages", {})
        ]
        if mean_values:
            stage_mean_ranges[stage] = _range(mean_values)
        if p95_values:
            stage_p95_ranges[stage] = _range(p95_values)

    return {
        "status": "repeated_bounded_dashboard_stability_summary",
        "run_count": len(reports),
        "run_files": [str(path) for path in paths],
        "frames_requested_per_run": [report.get("frames_requested") for report in reports],
        "clean_runs": len(clean_runs),
        "all_runs_clean": len(clean_runs) == len(reports),
        "audio_and_telemetry_started_all_runs": all(
            report.get("lifecycle", {}).get("audio_started") is True
            and report.get("lifecycle", {}).get("telemetry_started") is True
            for report in reports
        ),
        "runtime_metrics_present_all_runs": len(runtime_present) == len(reports),
        "runtime_metrics_no_media_all_runs": len(runtime_present) == len(reports) and runtime_no_media,
        "runtime_stage_names": runtime_stage_names,
        "runtime_queue_drops_per_run": runtime_queue_drops,
        "runtime_stage_mean_ms": stage_mean_ranges,
        "runtime_stage_p95_ms": stage_p95_ranges,
        "runtime_yamnet_inference_mean_ms": _range(runtime_yamnet_means),
        "runtime_yamnet_inference_max_ms": _range(runtime_yamnet_maxima),
        "runtime_whisper_completions_per_run": runtime_whisper_completions,
        "runtime_whisper_inference_mean_ms": _range(runtime_whisper_means),
        "runtime_whisper_inference_max_ms": _range(runtime_whisper_maxima),
        "wall_time_seconds": _range(durations),
        "effective_fps": _range(fps),
        "rss_mean_mb": _range(rss_mean),
        "rss_peak_mb": _range(rss_peak),
        "cpu_mean_percent": _range(cpu_mean),
        "cpu_peak_percent": _range(cpu_peak),
        "threads_peak": _range(thread_peak),
        "gpu": {
            "available_all_runs": len(gpu_util_mean) == len(reports),
            "utilization_mean_percent": _range(gpu_util_mean) if gpu_util_mean else None,
            "utilization_peak_percent": _range(gpu_util_peak) if gpu_util_peak else None,
            "vram_peak_mb": _range(vram_peak) if vram_peak else None,
            "memory_total_mb": [gpu.get("memory_total_mb") for gpu in gpu_reports],
        },
        "privacy_flags": {
            "automatic_screenshots_enabled": False,
            "transcript_recording_enabled": False,
            "raw_media_saved": False,
        },
        "thermal_data_available": False,
        "per_model_latency_instrumented": len(runtime_present) == len(reports),
        "queue_or_drop_counts_instrumented": len(runtime_present) == len(reports),
        "limitations": [
            "Runs were sequential bounded observations on one laptop, not a thermal or multi-hardware study.",
            "Variation includes model startup, cache, scheduler, and GPU sampling effects.",
            "Runtime metrics cover dashboard stages, YOLO scheduling, audio queues, YAMNet, and observed Whisper work; they do not cover every internal or telemetry-startup path.",
            "Whisper latency is zero in runs with no completed utterance, so it must be interpreted with its completion counts.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run", type=Path, action="append", dest="runs")
    args = parser.parse_args()
    paths = tuple(args.runs or DEFAULT_RUNS)
    try:
        result = summarize(paths)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Could not summarize stability runs: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Summary written: {args.output}")
    print(json.dumps(result, indent=2))
    return 0 if result["all_runs_clean"] and result["runtime_metrics_present_all_runs"] and result["runtime_metrics_no_media_all_runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

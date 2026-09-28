"""Live YAMNet/LiteRT audio-event detection for CabInspector.

The detector deliberately produces contextual evidence rather than driver
violations.  It does not record raw microphone audio and keeps only the latest
classification state in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread
from typing import Iterable, Mapping
import os
import time
import zipfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "audio" / "yamnet_audio_classifier_with_metadata.tflite"
)

SAMPLE_RATE = 16_000
CHANNELS = 1
BLOCK_SIZE = 1_600
YAMNET_WINDOW_SAMPLES = 15_600
YAMNET_HOP_SAMPLES = 7_800
STALE_AFTER_MS = 3_000

# These labels are the exact English category names used by the official YAMNet
# metadata.  Unknown labels are safely ignored.
YAMNET_CATEGORY_MAP: Mapping[str, str] = {
    "speech": "speech",
    "conversation": "speech",
    "narration, monologue": "speech",
    "shout": "raised_voice",
    "yell": "raised_voice",
    "screaming": "raised_voice",
    "vehicle horn, car horn, honking": "horn",
    "siren": "siren",
    "music": "music",
    "silence": "quiet",
}


@dataclass(frozen=True)
class AudioState:
    """Latest stable audio evidence exposed to the video/fusion loop."""

    classifier_available: bool = False
    top_label: str = "Unavailable"
    top_confidence: float = 0.0
    speech_active: bool = False
    raised_voice_active: bool = False
    horn_active: bool = False
    siren_active: bool = False
    music_active: bool = False
    quiet_active: bool = False
    timestamp_ms: int = 0
    age_ms: int = 0
    error: str = "Audio disabled"

    @classmethod
    def unavailable(cls, error: str = "Audio disabled") -> "AudioState":
        return cls(error=error)


@dataclass(frozen=True)
class AudioSmoothingConfig:
    thresholds: Mapping[str, float]
    confirm_windows: int = 2
    release_windows: int = 3

    @classmethod
    def default(cls) -> "AudioSmoothingConfig":
        return cls(
            thresholds={
                "speech": 0.35,
                "raised_voice": 0.30,
                "horn": 0.30,
                "siren": 0.30,
                "music": 0.35,
                "quiet": 0.50,
            }
        )


def map_yamnet_categories(
    categories: Iterable[tuple[str, float]],
) -> dict[str, float]:
    """Collapse YAMNet labels into CabInspector's small event vocabulary."""

    mapped: dict[str, float] = {}
    for label, score in categories:
        category = YAMNET_CATEGORY_MAP.get((label or "").strip().casefold())
        if category is None:
            continue
        mapped[category] = max(mapped.get(category, 0.0), float(score))
    return mapped


def load_yamnet_labels(model_path: str | Path) -> list[str]:
    """Read the label list packaged in the official metadata-enabled model."""
    with zipfile.ZipFile(model_path) as model_package:
        labels = model_package.read("yamnet_label_list.txt").decode("utf-8")
    return [label.strip() for label in labels.splitlines() if label.strip()]


class AudioStateSmoother:
    """Converts per-window classification scores into stable boolean evidence."""

    def __init__(self, config: AudioSmoothingConfig | None = None):
        self.config = config or AudioSmoothingConfig.default()
        self._confirm = {name: 0 for name in self.config.thresholds}
        self._release = {name: 0 for name in self.config.thresholds}
        self._active = {name: False for name in self.config.thresholds}

    def update(
        self,
        category_scores: Mapping[str, float],
        *,
        top_label: str,
        top_confidence: float,
        timestamp_ms: int,
    ) -> AudioState:
        for name, threshold in self.config.thresholds.items():
            above_threshold = category_scores.get(name, 0.0) >= threshold
            if above_threshold:
                self._confirm[name] += 1
                self._release[name] = 0
                if self._confirm[name] >= self.config.confirm_windows:
                    self._active[name] = True
            else:
                self._confirm[name] = 0
                if self._active[name]:
                    self._release[name] += 1
                    if self._release[name] >= self.config.release_windows:
                        self._active[name] = False
                        self._release[name] = 0

        return AudioState(
            classifier_available=True,
            top_label=top_label or "No relevant event",
            top_confidence=round(float(top_confidence), 4),
            speech_active=self._active.get("speech", False),
            raised_voice_active=self._active.get("raised_voice", False),
            horn_active=self._active.get("horn", False),
            siren_active=self._active.get("siren", False),
            music_active=self._active.get("music", False),
            quiet_active=self._active.get("quiet", False),
            timestamp_ms=timestamp_ms,
            error="",
        )


class YamNetAudioDetector:
    """Optional non-blocking microphone source backed by MediaPipe YAMNet.

    Audio callbacks only enqueue PCM data. A worker thread runs LiteRT/YAMNet
    inference and updates the latest :class:`AudioState`.
    No raw audio is written to disk.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        device: str | int | None = None,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        smoother: AudioStateSmoother | None = None,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._smoother = smoother or AudioStateSmoother()
        self._lock = RLock()
        self._latest_state = AudioState.unavailable()
        self._queue: Queue[np.ndarray] = Queue(maxsize=12)
        self._stop_event = Event()
        self._worker: Thread | None = None
        self._stream = None
        self._interpreter = None
        self._input_index = None
        self._output_index = None
        self._labels: list[str] = []
        self._processing_started = False
        self._queue_drops = 0
        self._max_queue_depth = 0
        self._inference_count = 0
        self._inference_total_ms = 0.0
        self._inference_max_ms = 0.0

    @staticmethod
    def list_input_devices() -> list[dict[str, object]]:
        """Return audio input devices without starting a capture stream."""
        try:
            import sounddevice as sd
        except ImportError:
            return []

        devices = []
        for index, device in enumerate(sd.query_devices()):
            if device.get("max_input_channels", 0) > 0:
                devices.append({"index": index, "name": device["name"]})
        return devices

    def start(self, *, capture_microphone: bool = True) -> bool:
        """Start YAMNet processing, optionally opening a microphone stream.

        ``capture_microphone=False`` allows :class:`SpeechAnalysisPipeline` to
        feed the same microphone samples to YAMNet and transcription, avoiding
        competing audio-device streams.
        """
        if self._processing_started:
            return True

        if not self.model_path.exists():
            self._set_unavailable(f"YAMNet model not found: {self.model_path}")
            return False

        try:
            from ai_edge_litert.interpreter import Interpreter

            self._labels = load_yamnet_labels(self.model_path)
            self._interpreter = Interpreter(model_path=str(self.model_path), num_threads=1)
            self._interpreter.allocate_tensors()
            self._input_index = self._interpreter.get_input_details()[0]["index"]
            self._output_index = self._interpreter.get_output_details()[0]["index"]
            self._stop_event.clear()
            self._worker = Thread(target=self._run_worker, name="yamnet-audio", daemon=True)
            self._worker.start()
            self._processing_started = True
            if capture_microphone:
                import sounddevice as sd

                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=CHANNELS,
                    dtype="float32",
                    blocksize=self.block_size,
                    device=self.device,
                    callback=self._on_audio_input,
                )
                self._stream.start()
            self._set_latest(
                AudioState(
                    classifier_available=True,
                    top_label="Listening",
                    error="",
                )
            )
            return True
        except Exception as exc:
            self.stop()
            self._set_unavailable(f"Audio unavailable: {exc}")
            return False

    def stop(self) -> None:
        """Stop capture and release the model without persisting audio."""
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

        self._interpreter = None
        self._input_index = None
        self._output_index = None
        self._labels = []
        self._processing_started = False

    def get_latest_state(self, *, stale_after_ms: int = STALE_AFTER_MS) -> AudioState:
        with self._lock:
            state = self._latest_state

        if not state.classifier_available or state.timestamp_ms <= 0:
            return state

        age_ms = max(0, int(time.monotonic() * 1000) - state.timestamp_ms)
        if age_ms > stale_after_ms:
            return replace(state, age_ms=age_ms, error="Audio result is stale")
        return replace(state, age_ms=age_ms)

    def runtime_statistics(self) -> dict[str, int | float]:
        with self._lock:
            count = self._inference_count
            return {
                "queue_drops": self._queue_drops,
                "queue_depth": self._queue.qsize(),
                "max_queue_depth": self._max_queue_depth,
                "inference_count": count,
                "inference_mean_ms": round(self._inference_total_ms / count, 4) if count else 0.0,
                "inference_max_ms": round(self._inference_max_ms, 4),
            }

    def _on_audio_input(self, indata, frames, _time_info, status) -> None:
        if status:
            self._set_unavailable(f"Microphone status: {status}")
            return

        self.feed_audio(indata)

    def feed_audio(self, samples) -> None:
        """Queue one PCM chunk from a shared capture service for YAMNet."""
        samples = np.asarray(samples, dtype=np.float32).copy()
        if samples.ndim == 2 and samples.shape[1] > 1:
            samples = samples.mean(axis=1, dtype=np.float32)
        elif samples.ndim == 2:
            samples = samples[:, 0]

        try:
            self._queue.put_nowait(samples)
            with self._lock:
                self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
        except Full:
            # Prefer current audio over old audio when the system is overloaded.
            with self._lock:
                self._queue_drops += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(samples)
                with self._lock:
                    self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
            except (Empty, Full):
                pass

    def _run_worker(self) -> None:
        try:
            audio_buffer = np.empty(0, dtype=np.float32)

            while not self._stop_event.is_set():
                try:
                    samples = self._queue.get(timeout=0.2)
                except Empty:
                    continue

                if self._interpreter is None:
                    break
                audio_buffer = np.concatenate((audio_buffer, samples.astype(np.float32, copy=False)))
                while len(audio_buffer) >= YAMNET_WINDOW_SAMPLES:
                    self._classify_window(audio_buffer[:YAMNET_WINDOW_SAMPLES])
                    audio_buffer = audio_buffer[YAMNET_HOP_SAMPLES:]
        except Exception as exc:
            self._set_unavailable(f"Audio inference failed: {exc}")

    def _classify_window(self, samples: np.ndarray) -> None:
        if self._interpreter is None or self._input_index is None or self._output_index is None:
            return

        started = time.perf_counter()
        try:
            self._interpreter.set_tensor(self._input_index, samples)
            self._interpreter.invoke()
            scores = self._interpreter.get_tensor(self._output_index)[0]
            if len(scores) != len(self._labels):
                raise RuntimeError(
                    f"YAMNet label count ({len(self._labels)}) does not match output ({len(scores)})"
                )

            top_index = int(np.argmax(scores))
            categories = list(zip(self._labels, scores.tolist()))
            state = self._smoother.update(
                map_yamnet_categories(categories),
                top_label=self._labels[top_index],
                top_confidence=float(scores[top_index]),
                timestamp_ms=int(time.monotonic() * 1000),
            )
            self._set_latest(state)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._inference_count += 1
                self._inference_total_ms += elapsed_ms
                self._inference_max_ms = max(self._inference_max_ms, elapsed_ms)

    def _set_unavailable(self, message: str) -> None:
        self._set_latest(AudioState.unavailable(message))

    def _set_latest(self, state: AudioState) -> None:
        with self._lock:
            self._latest_state = state


def audio_enabled_from_environment() -> bool:
    """Read the opt-in runtime flag without importing MediaPipe at module load."""
    return os.environ.get("CABINSPECTOR_USE_AUDIO", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

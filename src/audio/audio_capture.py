"""One privacy-safe microphone capture service shared by audio models."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import RLock
import time

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    timestamp_ms: int


class AudioCaptureService:
    """Captures 16 kHz mono audio and exposes a bounded in-memory queue only."""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        sample_rate: int = 16_000,
        block_size: int = 1_600,
        queue_size: int = 30,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._queue: Queue[AudioChunk] = Queue(maxsize=queue_size)
        self._stream = None
        self._lock = RLock()
        self._error = ""
        self._input_callbacks = 0
        self._queue_drops = 0
        self._max_queue_depth = 0

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def start(self) -> bool:
        if self._stream is not None:
            return True
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.block_size,
                device=self.device,
                callback=self._on_audio_input,
            )
            self._stream.start()
            return True
        except Exception as exc:
            with self._lock:
                self._error = f"Microphone unavailable: {exc}"
            self.stop()
            return False

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.clear()

    def get(self, timeout: float = 0.2) -> AudioChunk | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return

    def runtime_statistics(self) -> dict[str, int]:
        with self._lock:
            return {
                "input_callbacks": self._input_callbacks,
                "queue_drops": self._queue_drops,
                "max_queue_depth": self._max_queue_depth,
                "queue_depth": self._queue.qsize(),
            }

    def _on_audio_input(self, indata, _frames, _time_info, status) -> None:
        with self._lock:
            self._input_callbacks += 1
        if status:
            with self._lock:
                self._error = f"Microphone status: {status}"
            return
        samples = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
        chunk = AudioChunk(samples=samples, timestamp_ms=int(time.monotonic() * 1000))
        try:
            self._queue.put_nowait(chunk)
            with self._lock:
                self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
        except Full:
            # Keep current audio when inference is slower than capture.
            with self._lock:
                self._queue_drops += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk)
                with self._lock:
                    self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
            except (Empty, Full):
                pass

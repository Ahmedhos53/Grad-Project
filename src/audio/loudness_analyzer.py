"""Session-calibrated speech loudness analysis without retaining audio."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np


EPSILON = 1e-8


@dataclass(frozen=True)
class LoudnessState:
    rms: float = 0.0
    dbfs: float = -120.0
    normal_reference_dbfs: float = -30.0
    voice_level: str = "QUIET"
    loud_voice_active: bool = False
    clipping_detected: bool = False
    calibrated: bool = False
    timestamp_ms: int = 0


class LoudnessAnalyzer:
    """Classify voice loudness relative to a consented normal-speaking reference."""

    def __init__(
        self,
        *,
        default_normal_dbfs: float = -30.0,
        low_delta_db: float = -7.0,
        high_delta_db: float = 7.0,
        very_high_delta_db: float = 14.0,
        confirm_chunks: int = 4,
        release_chunks: int = 6,
    ):
        self.default_normal_dbfs = default_normal_dbfs
        self.low_delta_db = low_delta_db
        self.high_delta_db = high_delta_db
        self.very_high_delta_db = very_high_delta_db
        self.confirm_chunks = confirm_chunks
        self.release_chunks = release_chunks
        self._normal_reference_dbfs: float | None = None
        self._loud_confirm = 0
        self._quiet_release = 0
        self._loud_active = False

    @staticmethod
    def rms_dbfs(samples: np.ndarray) -> tuple[float, float]:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return 0.0, -120.0
        rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float32))))
        return rms, 20.0 * math.log10(max(rms, EPSILON))

    @property
    def calibrated(self) -> bool:
        return self._normal_reference_dbfs is not None

    @property
    def normal_reference_dbfs(self) -> float:
        return self._normal_reference_dbfs or self.default_normal_dbfs

    def calibrate(self, normal_speech_samples: np.ndarray) -> float:
        """Set a normal-speaking reference from a consented calibration sample."""
        _, reference = self.rms_dbfs(normal_speech_samples)
        self._normal_reference_dbfs = reference
        self._loud_confirm = 0
        self._quiet_release = 0
        self._loud_active = False
        return reference

    def update(self, samples: np.ndarray, *, speech_active: bool) -> LoudnessState:
        rms, dbfs = self.rms_dbfs(samples)
        timestamp_ms = int(time.monotonic() * 1000)
        clipping = bool(np.any(np.abs(np.asarray(samples)) >= 0.99))

        if not speech_active:
            self._loud_confirm = 0
            self._quiet_release += 1
            if self._quiet_release >= self.release_chunks:
                self._loud_active = False
            return LoudnessState(
                rms=rms,
                dbfs=round(dbfs, 2),
                normal_reference_dbfs=self.normal_reference_dbfs,
                voice_level="QUIET",
                loud_voice_active=False,
                clipping_detected=clipping,
                calibrated=self.calibrated,
                timestamp_ms=timestamp_ms,
            )

        self._quiet_release = 0
        voice_level = self._classify_voiced_level(dbfs)
        if voice_level in {"HIGH", "VERY HIGH"}:
            self._loud_confirm += 1
            if self._loud_confirm >= self.confirm_chunks:
                self._loud_active = True
        else:
            self._loud_confirm = 0
            self._quiet_release += 1
            if self._quiet_release >= self.release_chunks:
                self._loud_active = False

        return LoudnessState(
            rms=rms,
            dbfs=round(dbfs, 2),
            normal_reference_dbfs=self.normal_reference_dbfs,
            voice_level=voice_level,
            loud_voice_active=self._loud_active,
            clipping_detected=clipping,
            calibrated=self.calibrated,
            timestamp_ms=timestamp_ms,
        )

    def _classify_voiced_level(self, dbfs: float) -> str:
        """Return one of LOW, NORMAL, HIGH, or VERY HIGH for voiced audio."""
        reference = self.normal_reference_dbfs
        if dbfs < reference + self.low_delta_db:
            return "LOW"
        if dbfs < reference + self.high_delta_db:
            return "NORMAL"
        if dbfs < reference + self.very_high_delta_db:
            return "HIGH"
        return "VERY HIGH"

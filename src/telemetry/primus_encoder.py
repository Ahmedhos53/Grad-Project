"""Isolated inference support for the externally pretrained PRIMUS IMU encoder.

This module intentionally does not map embeddings to CabInspector driving-event labels and is
not imported by the dashboard or replay path.  It mirrors only the published IMU encoder
architecture so the official checkpoint can be evaluated without installing the source
repository's training stack or loading its unrelated video/text state at runtime.

Source architecture: Nokia Bell Labs PRIMUS, BSD-3-Clause-Clear source code.
Checkpoint provenance: ``models/telemetry/pretrained/primus/PROVENANCE.md``.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMUS_CHECKPOINT = PROJECT_ROOT / "models" / "telemetry" / "pretrained" / "primus" / "best_model.ckpt"
DEFAULT_PRIMUS_ENCODER_STATE = (
    PROJECT_ROOT / "models" / "telemetry" / "pretrained" / "primus" / "imu_encoder_state.pt"
)
PRIMUS_CHANNELS = 6
PRIMUS_TARGET_HZ = 200
PRIMUS_WINDOW_SECONDS = 5.0
PRIMUS_SAMPLES_PER_WINDOW = int(PRIMUS_TARGET_HZ * PRIMUS_WINDOW_SECONDS)
PRIMUS_EMBEDDING_SIZE = 512


def _torch():
    """Import PyTorch lazily so normal telemetry modules keep their existing startup cost."""
    import torch

    return torch


def resample_imu_window(
    inputs: np.ndarray,
    *,
    target_samples: int = PRIMUS_SAMPLES_PER_WINDOW,
) -> np.ndarray:
    """Resample one normalized-time IMU window to the PRIMUS input length.

    This preserves the six-channel order and endpoint values.  It is suitable only for a
    structural encoder proof-of-life using existing event windows.  It does *not* create a
    deployment-valid five-second telemetry window; that needs raw timestamped sensor data and
    a separately evaluated fixed-window protocol.
    """
    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != PRIMUS_CHANNELS:
        raise ValueError(f"Expected one IMU window shaped (6, samples), got {values.shape}")
    if values.shape[1] < 2:
        raise ValueError("At least two samples are required for resampling")
    if target_samples < 2:
        raise ValueError("target_samples must be at least 2")
    if not np.isfinite(values).all():
        raise ValueError("IMU inputs must be finite")

    source_time = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float64)
    target_time = np.linspace(0.0, 1.0, target_samples, dtype=np.float64)
    return np.stack(
        [np.interp(target_time, source_time, channel) for channel in values],
        axis=0,
    ).astype(np.float32)


class PrimusIMUEncoder:
    """Inference-only wrapper for PRIMUS's published six-channel Conv1D/GRU encoder."""

    def __init__(self, *, embedding_size: int = PRIMUS_EMBEDDING_SIZE) -> None:
        torch = _torch()
        nn = torch.nn

        class Block(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, adaptive: bool = False):
                super().__init__()
                pool = (
                    nn.AdaptiveAvgPool1d(output_size=32)
                    if adaptive
                    else nn.MaxPool1d(kernel_size=3)
                )
                self.net = nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=2,
                        bias=False,
                    ),
                    pool,
                )

            def forward(self, batch):
                return self.net(batch)

        class Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.GroupNorm(2, PRIMUS_CHANNELS),
                    Block(PRIMUS_CHANNELS, 32, 10),
                    Block(32, 32, 5),
                    Block(32, 32, 5, adaptive=True),
                    nn.GroupNorm(4, 32),
                    nn.GRU(batch_first=True, input_size=32, hidden_size=embedding_size),
                )

            def forward(self, batch):
                return self.backbone(batch)[1][0]

        self._torch = torch
        self.embedding_size = embedding_size
        self.model = Encoder()
        self.load_seconds = 0.0

    def load_checkpoint(self, checkpoint_path: str | Path = DEFAULT_PRIMUS_CHECKPOINT) -> None:
        """Load only the official checkpoint's IMU backbone weights on CPU."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"PRIMUS checkpoint not found: {path}")

        started = perf_counter()
        payload = self._torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            raise ValueError("PRIMUS checkpoint does not contain a state_dict mapping")

        prefix = "imu_encoder."
        encoder_state = {
            key[len(prefix) :]: value
            for key, value in payload["state_dict"].items()
            if key.startswith(prefix + "backbone.")
        }
        if not encoder_state:
            raise ValueError("PRIMUS checkpoint does not contain imu_encoder backbone weights")
        result = self.model.load_state_dict(encoder_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                "PRIMUS backbone load mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        self.model.eval()
        self.load_seconds = perf_counter() - started

    def load_compact_state(self, state_path: str | Path = DEFAULT_PRIMUS_ENCODER_STATE) -> None:
        """Load the verified deployment-only IMU state exported from the source checkpoint."""

        path = Path(state_path)
        if not path.is_file():
            raise FileNotFoundError(f"Compact PRIMUS encoder state not found: {path}")
        started = perf_counter()
        payload = self._torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != "cabinspector_primus_imu_encoder_v1":
            raise ValueError("Unsupported compact PRIMUS encoder-state format")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Compact PRIMUS encoder state does not contain a state_dict")
        result = self.model.load_state_dict(state_dict, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                "Compact PRIMUS state mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        self.model.eval()
        self.load_seconds = perf_counter() - started

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Return one embedding per supplied `(batch, 6, samples)` input window."""
        values = np.asarray(inputs, dtype=np.float32)
        if values.ndim != 3 or values.shape[1] != PRIMUS_CHANNELS:
            raise ValueError(f"Expected PRIMUS inputs shaped (batch, 6, samples), got {values.shape}")
        if values.shape[2] < 64:
            raise ValueError("PRIMUS inputs require at least 64 samples")
        if not np.isfinite(values).all():
            raise ValueError("PRIMUS inputs must be finite")
        with self._torch.no_grad():
            embeddings = self.model(self._torch.from_numpy(values))
        return embeddings.cpu().numpy().astype(np.float32, copy=False)

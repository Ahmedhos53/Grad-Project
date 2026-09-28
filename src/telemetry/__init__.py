"""Telemetry data, training, and inference components for CabInspector.

The initial implementation uses only the public Driving Events Dataset.  It
does not automatically load or train on files in ``data/Telemetry/Recordings``.
"""

from .dataset_audit import audit_driving_events_dataset

__all__ = ["audit_driving_events_dataset"]

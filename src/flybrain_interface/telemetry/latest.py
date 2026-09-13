"""Bounded, non-blocking latest-value telemetry."""

from __future__ import annotations

from queue import Empty, Full, Queue

from flybrain_interface.contracts import TelemetryEvent


class LatestValueTelemetrySink:
    """Drop the oldest event instead of blocking the neural simulation."""

    def __init__(self, capacity: int = 1) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._events: Queue[TelemetryEvent] = Queue(maxsize=capacity)

    def publish(self, event: TelemetryEvent) -> None:
        try:
            self._events.put_nowait(event)
            return
        except Full:
            pass
        try:
            self._events.get_nowait()
        except Empty:
            pass
        try:
            self._events.put_nowait(event)
        except Full:
            # A concurrent producer won the slot; telemetry may be dropped.
            pass

    def drain_latest(self) -> TelemetryEvent | None:
        latest: TelemetryEvent | None = None
        while True:
            try:
                latest = self._events.get_nowait()
            except Empty:
                return latest

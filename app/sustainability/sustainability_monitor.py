"""
Sustainability monitoring with measured energy where possible.

Measurement strategy, in priority order (the first that succeeds wins, and
the report records which one produced the figures so they stay auditable):

  1. ``codecarbon`` — opt-in via ``PULSELEDGER_USE_CODECARBON=1``. Uses an
     ``OfflineEmissionsTracker`` (no network) that reads CPU/GPU/RAM power
     and applies a regional grid-intensity factor. Best for long-running
     work (training, NAS, batch) where its overhead is negligible.
  2. ``cpu-time`` — the always-on default. Integrates the process's *actual*
     CPU time (user+system) over the tracked interval against an effective
     per-core power draw, adds a small platform/RAM baseline for the elapsed
     wall-clock, then applies the regional grid factor. Scales with real work
     done, not merely with elapsed time.
  3. ``wall-clock`` — last-resort synthetic estimate proportional to elapsed
     time (the original compatibility behaviour), used only when ``psutil``
     is unavailable.

Each report carries ``method``, ``region`` and ``emissions_factor_kg_per_kwh``
so a consumer can explain exactly how a number was derived.
"""

import inspect
import os
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

# Approximate grid carbon intensity (kg CO2eq per kWh) by region. These are
# coarse public averages; override the active region with
# PULSELEDGER_GRID_REGION. For codecarbon's offline model the country is set
# separately via PULSELEDGER_COUNTRY_ISO.
GRID_INTENSITY_KG_PER_KWH: Dict[str, float] = {
    "US": 0.385,
    "EU": 0.255,
    "GB": 0.225,
    "IN": 0.708,
    "CA": 0.130,
    "AU": 0.510,
    "WORLD": 0.475,
}

# Effective per-CPU-second active power (W) and a platform/RAM baseline (W)
# attributed across the wall-clock interval. Both override-able so an operator
# can calibrate to their hardware.
_CORE_POWER_W = float(os.getenv("PULSELEDGER_CPU_CORE_WATTS", "15"))
_BASELINE_POWER_W = float(os.getenv("PULSELEDGER_BASELINE_WATTS", "5"))

_JOULES_PER_KWH = 3_600_000.0


def _region() -> str:
    return os.getenv("PULSELEDGER_GRID_REGION", "US").strip().upper()


def _grid_factor(region: str) -> float:
    return GRID_INTENSITY_KG_PER_KWH.get(
        region, GRID_INTENSITY_KG_PER_KWH["US"]
    )


def _codecarbon_enabled() -> bool:
    return os.getenv("PULSELEDGER_USE_CODECARBON", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class SustainabilityReport:
    total_energy_kwh: float
    total_emissions_kg: float
    duration_seconds: float
    experiment_id: str
    timestamp: str
    method: str
    region: str
    emissions_factor_kg_per_kwh: float


class SustainabilityMonitor:
    """Tracks energy/carbon per experiment with a measured-first strategy."""

    def __init__(self) -> None:
        self._active: Dict[str, Dict[str, Any]] = {}
        self._region = _region()
        self._grid_factor = _grid_factor(self._region)
        self._country_iso = (
            os.getenv("PULSELEDGER_COUNTRY_ISO", "USA").strip().upper()
        )

        # Probe psutil once so per-call cost is just a method lookup.
        try:
            import psutil  # noqa: F401

            self._psutil = psutil
        except Exception:  # pragma: no cover - environment dependent
            self._psutil = None

        self._use_codecarbon = _codecarbon_enabled()

    # -- lifecycle ---------------------------------------------------------

    def start_experiment_tracking(
        self, experiment_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        entry: Dict[str, Any] = {
            "start": datetime.now(),
            "cpu_seconds": self._process_cpu_seconds(),
            "tracker": None,
        }
        if self._use_codecarbon:
            entry["tracker"] = self._start_codecarbon()
        self._active[experiment_id] = entry
        return experiment_id

    def stop_experiment_tracking(self, experiment_id: str) -> Dict[str, Any]:
        entry = self._active.pop(experiment_id, None)
        now = datetime.now()
        if entry is None:
            # Unknown id: report a zero-cost, clearly-labelled result rather
            # than guessing.
            return self._report(experiment_id, 0.0, 0.0, 0.0, "unavailable")

        duration = max(0.0, (now - entry["start"]).total_seconds())

        # 1) codecarbon, if it was started for this experiment. It applies its
        #    own grid model, so derive the effective factor from its numbers
        #    to keep carbon == energy * factor true for every method.
        tracker = entry.get("tracker")
        if tracker is not None:
            measured = self._stop_codecarbon(tracker)
            if measured is not None:
                energy_kwh, emissions_kg = measured
                if energy_kwh > 0:
                    factor = emissions_kg / energy_kwh
                else:
                    factor, emissions_kg = self._grid_factor, 0.0
                return self._report(
                    experiment_id,
                    energy_kwh,
                    emissions_kg,
                    duration,
                    "codecarbon",
                    region=self._country_iso,
                    factor=factor,
                )

        # 2) cpu-time measurement via psutil.
        cpu_before = entry.get("cpu_seconds")
        cpu_now = self._process_cpu_seconds()
        if cpu_before is not None and cpu_now is not None:
            cpu_delta = max(0.0, cpu_now - cpu_before)
            joules = cpu_delta * _CORE_POWER_W + duration * _BASELINE_POWER_W
            energy_kwh = joules / _JOULES_PER_KWH
            emissions_kg = energy_kwh * self._grid_factor
            return self._report(
                experiment_id, energy_kwh, emissions_kg, duration, "cpu-time"
            )

        # 3) wall-clock fallback (no psutil available).
        energy_kwh = max(0.0, duration * 0.00003)
        emissions_kg = energy_kwh * self._grid_factor
        return self._report(
            experiment_id, energy_kwh, emissions_kg, duration, "wall-clock"
        )

    # -- helpers -----------------------------------------------------------

    def _process_cpu_seconds(self) -> Optional[float]:
        if self._psutil is None:
            return None
        try:
            times = self._psutil.Process().cpu_times()
            return float(times.user + times.system)
        except Exception:  # pragma: no cover - environment dependent
            return None

    def _start_codecarbon(self) -> Optional[Any]:
        try:
            from codecarbon import OfflineEmissionsTracker

            tracker = OfflineEmissionsTracker(
                country_iso_code=self._country_iso,
                save_to_file=False,
                log_level="error",
                # Rely on the start/stop final read rather than a periodic
                # sampling thread to keep per-call overhead low.
                measure_power_secs=3600,
                tracking_mode="process",
            )
            tracker.start()
            return tracker
        except Exception:  # pragma: no cover - optional dependency
            return None

    def _stop_codecarbon(self, tracker: Any) -> Optional[tuple]:
        try:
            emissions_kg = float(tracker.stop() or 0.0)
            data = getattr(tracker, "final_emissions_data", None)
            energy_kwh = float(getattr(data, "energy_consumed", 0.0) or 0.0)
            return energy_kwh, emissions_kg
        except Exception:  # pragma: no cover - optional dependency
            return None

    def _report(
        self,
        experiment_id: str,
        energy_kwh: float,
        emissions_kg: float,
        duration: float,
        method: str,
        region: Optional[str] = None,
        factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "experiment_id": experiment_id,
            "duration_seconds": duration,
            "energy_kwh": energy_kwh,
            "carbon_emissions": emissions_kg,
            "method": method,
            "region": region or self._region,
            "emissions_factor_kg_per_kwh": (
                factor if factor is not None else self._grid_factor
            ),
            "timestamp": datetime.now().isoformat(),
        }


def track_sustainability(func: Callable) -> Callable:
    """Decorator placeholder to keep backwards compatibility."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

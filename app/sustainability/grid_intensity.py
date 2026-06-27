"""
Grid carbon-intensity resolution: live where configured, static otherwise.

Converting energy (kWh) into carbon (kg CO2eq) needs a grid-intensity factor.
A static regional average is fine as a default, but the real grid mix swings
widely through the day (solar at noon vs. gas at night). When an operator
configures a live provider this module fetches the current intensity and
caches it; otherwise it returns the static regional factor.

Design notes:
  * Default provider is ``static`` — no network, no latency, no new deps.
  * Live lookups never block the caller: ``get()`` returns the cached (or
    static) value immediately and refreshes in a daemon thread when stale.
  * Networking uses the stdlib (``urllib``) with a short timeout, and any
    failure silently falls back to the last-known/static value.

Env:
  PULSELEDGER_GRID_PROVIDER          static (default) | electricitymaps | uk
  PULSELEDGER_ELECTRICITYMAPS_TOKEN  auth token for Electricity Maps
  PULSELEDGER_ELECTRICITYMAPS_ZONE   override the zone (e.g. US-CAL-CISO)
  PULSELEDGER_GRID_TTL_SECONDS       cache TTL (default 1800)
  PULSELEDGER_GRID_HTTP_TIMEOUT      per-request timeout seconds (default 2)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Approximate grid carbon intensity (kg CO2eq per kWh) by region. Coarse
# public averages used as the default and the fallback for live providers.
GRID_INTENSITY_KG_PER_KWH: Dict[str, float] = {
    "US": 0.385,
    "EU": 0.255,
    "GB": 0.225,
    "IN": 0.708,
    "CA": 0.130,
    "AU": 0.510,
    "WORLD": 0.475,
}

# Default region -> Electricity Maps zone. Override per-deployment with
# PULSELEDGER_ELECTRICITYMAPS_ZONE for a finer-grained zone (e.g. US-CAL-CISO).
REGION_TO_EM_ZONE: Dict[str, str] = {
    "US": "US",
    "EU": "DE",
    "GB": "GB",
    "IN": "IN",
    "CA": "CA",
    "AU": "AUS-NSW",
    "WORLD": "DE",
}


def static_factor(region: str) -> float:
    return GRID_INTENSITY_KG_PER_KWH.get(
        region, GRID_INTENSITY_KG_PER_KWH["US"]
    )


def _http_get_json(url: str, headers: Dict[str, str], timeout: float):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode("utf-8"))


def fetch_electricitymaps(
    region: str, timeout: float
) -> Optional[Tuple[float, str]]:
    """Latest carbon intensity (gCO2eq/kWh) for a zone, converted to kg/kWh."""
    token = os.getenv("PULSELEDGER_ELECTRICITYMAPS_TOKEN", "").strip()
    if not token:
        return None
    zone = os.getenv("PULSELEDGER_ELECTRICITYMAPS_ZONE", "").strip() or (
        REGION_TO_EM_ZONE.get(region, "DE")
    )
    url = (
        "https://api.electricitymap.org/v3/carbon-intensity/latest"
        f"?zone={zone}"
    )
    try:
        data = _http_get_json(url, {"auth-token": token}, timeout)
        grams = float(data["carbonIntensity"])
        return grams / 1000.0, f"electricitymaps:{zone}"
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Electricity Maps lookup failed: %s", exc)
        return None


def fetch_uk(timeout: float) -> Optional[Tuple[float, str]]:
    """Free UK Carbon Intensity API (no key); GB grid only, gCO2/kWh."""
    url = "https://api.carbonintensity.org.uk/intensity"
    try:
        data = _http_get_json(url, {"Accept": "application/json"}, timeout)
        grams = float(data["data"][0]["intensity"]["actual"])
        return grams / 1000.0, "carbonintensity.org.uk"
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("UK Carbon Intensity lookup failed: %s", exc)
        return None


class GridIntensityProvider:
    """Resolves a grid factor for a region, optionally from a live source.

    Reads its configuration from the environment at construction, so a fresh
    instance always reflects current settings (handy for tests).
    """

    def __init__(self) -> None:
        self._provider = (
            os.getenv("PULSELEDGER_GRID_PROVIDER", "static").strip().lower()
        )
        self._ttl = float(os.getenv("PULSELEDGER_GRID_TTL_SECONDS", "1800"))
        self._timeout = float(os.getenv("PULSELEDGER_GRID_HTTP_TIMEOUT", "2"))
        self._cache: Dict[str, Tuple[float, str, float]] = {}
        self._inflight: set = set()
        self._lock = threading.Lock()

    @property
    def is_live(self) -> bool:
        return self._provider in {"electricitymaps", "uk", "carbonintensity"}

    def get(self, region: str) -> Tuple[float, str]:
        """Return (factor_kg_per_kwh, source). Never blocks on the network."""
        static = (static_factor(region), f"static:{region}")
        if not self.is_live:
            return static

        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(region)
            fresh = cached is not None and (now - cached[2]) < self._ttl
            if fresh:
                return cached[0], cached[1]
            # Stale or cold: trigger a background refresh (once per region).
            if region not in self._inflight:
                self._inflight.add(region)
                threading.Thread(
                    target=self._refresh, args=(region,), daemon=True
                ).start()
            # Serve the last-known value, or static until the first refresh.
            if cached is not None:
                return cached[0], cached[1]
        return static

    def _refresh(self, region: str) -> None:
        try:
            result = self._fetch(region)
            if result is not None:
                with self._lock:
                    self._cache[region] = (
                        result[0],
                        result[1],
                        time.monotonic(),
                    )
        finally:
            with self._lock:
                self._inflight.discard(region)

    def _fetch(self, region: str) -> Optional[Tuple[float, str]]:
        if self._provider == "electricitymaps":
            return fetch_electricitymaps(region, self._timeout)
        if self._provider in {"uk", "carbonintensity"}:
            return fetch_uk(self._timeout)
        return None

"""Live grid-intensity provider: parsing, caching, and graceful fallback."""

from __future__ import annotations

import time

from app.sustainability import grid_intensity as gi


def test_static_is_default_and_never_touches_network(monkeypatch):
    monkeypatch.delenv("PULSELEDGER_GRID_PROVIDER", raising=False)

    def _boom(*a, **k):  # any network attempt is a failure here
        raise AssertionError("static provider must not hit the network")

    monkeypatch.setattr(gi, "_http_get_json", _boom)
    provider = gi.GridIntensityProvider()
    assert provider.is_live is False
    factor, source = provider.get("US")
    assert factor == gi.GRID_INTENSITY_KG_PER_KWH["US"]
    assert source == "static:US"


def test_electricitymaps_parsing_and_unit_conversion(monkeypatch):
    monkeypatch.setenv("PULSELEDGER_ELECTRICITYMAPS_TOKEN", "tok")
    monkeypatch.setenv("PULSELEDGER_ELECTRICITYMAPS_ZONE", "US-CAL-CISO")
    # API returns gCO2eq/kWh; we expect kg/kWh.
    monkeypatch.setattr(
        gi, "_http_get_json", lambda *a, **k: {"carbonIntensity": 250.0}
    )
    result = gi.fetch_electricitymaps("US", timeout=1.0)
    assert result is not None
    factor, source = result
    assert abs(factor - 0.250) < 1e-12
    assert source == "electricitymaps:US-CAL-CISO"


def test_electricitymaps_requires_token(monkeypatch):
    monkeypatch.delenv("PULSELEDGER_ELECTRICITYMAPS_TOKEN", raising=False)
    assert gi.fetch_electricitymaps("US", timeout=1.0) is None


def test_uk_parsing(monkeypatch):
    monkeypatch.setattr(
        gi,
        "_http_get_json",
        lambda *a, **k: {"data": [{"intensity": {"actual": 180}}]},
    )
    result = gi.fetch_uk(timeout=1.0)
    assert result == (0.180, "carbonintensity.org.uk")


def test_network_failure_falls_back_to_static(monkeypatch):
    monkeypatch.setenv("PULSELEDGER_GRID_PROVIDER", "uk")

    def _fail(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(gi, "_http_get_json", _fail)
    provider = gi.GridIntensityProvider()
    # First call serves static immediately and triggers a refresh that fails.
    factor, source = provider.get("GB")
    assert source == "static:GB"
    # Let the background refresh finish; it must not corrupt the cache.
    time.sleep(0.2)
    factor2, source2 = provider.get("GB")
    assert source2 == "static:GB"


def test_live_value_is_cached_after_refresh(monkeypatch):
    monkeypatch.setenv("PULSELEDGER_GRID_PROVIDER", "uk")
    monkeypatch.setattr(
        gi,
        "_http_get_json",
        lambda *a, **k: {"data": [{"intensity": {"actual": 120}}]},
    )
    provider = gi.GridIntensityProvider()
    # Cold call -> static + background refresh kicked off.
    assert provider.get("GB")[1] == "static:GB"
    # After the refresh lands, the cached live value is served.
    for _ in range(20):
        factor, source = provider.get("GB")
        if source != "static:GB":
            break
        time.sleep(0.05)
    assert source == "carbonintensity.org.uk"
    assert abs(factor - 0.120) < 1e-12

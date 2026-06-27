"""The sustainability monitor reports measured, method-tagged figures."""

from __future__ import annotations

from app.sustainability.sustainability_monitor import (
    SustainabilityMonitor,
    hardware_power_available,
    resolve_codecarbon_mode,
)


def _required_keys(report: dict) -> None:
    for key in (
        "energy_kwh",
        "carbon_emissions",
        "duration_seconds",
        "method",
        "region",
        "emissions_factor_kg_per_kwh",
    ):
        assert key in report, f"missing {key}"


def test_report_is_measured_and_tagged():
    monitor = SustainabilityMonitor()
    monitor.start_experiment_tracking("exp-1")
    # A little real CPU work so the cpu-time path has something to measure.
    total = sum(i * i for i in range(500_000))
    assert total > 0
    report = monitor.stop_experiment_tracking("exp-1")

    _required_keys(report)
    assert report["energy_kwh"] >= 0.0
    assert report["carbon_emissions"] >= 0.0
    # Carbon must be energy times the reported grid factor.
    expected = report["energy_kwh"] * report["emissions_factor_kg_per_kwh"]
    assert abs(report["carbon_emissions"] - expected) < 1e-12
    # On any machine with psutil this is a real measurement, not wall-clock.
    assert report["method"] in {"codecarbon", "cpu-time", "wall-clock"}


def test_energy_scales_with_cpu_work():
    """More CPU work should not report less energy than near-zero work."""
    monitor = SustainabilityMonitor()

    monitor.start_experiment_tracking("idle")
    light = monitor.stop_experiment_tracking("idle")

    monitor.start_experiment_tracking("busy")
    total = sum(i * i for i in range(3_000_000))
    assert total > 0
    busy = monitor.stop_experiment_tracking("busy")

    # When psutil is available the busy run must cost at least as much.
    if light["method"] == "cpu-time" and busy["method"] == "cpu-time":
        assert busy["energy_kwh"] >= light["energy_kwh"]


def test_unknown_experiment_is_labelled_not_guessed():
    monitor = SustainabilityMonitor()
    report = monitor.stop_experiment_tracking("never-started")
    _required_keys(report)
    assert report["method"] == "unavailable"
    assert report["energy_kwh"] == 0.0
    assert report["carbon_emissions"] == 0.0


def test_hardware_detection_returns_bool():
    # Must never raise on any platform; just reports availability.
    assert isinstance(hardware_power_available(), bool)


def test_codecarbon_mode_honours_explicit_override(monkeypatch):
    monkeypatch.setenv("PULSELEDGER_USE_CODECARBON", "1")
    assert resolve_codecarbon_mode() is True
    monkeypatch.setenv("PULSELEDGER_USE_CODECARBON", "off")
    assert resolve_codecarbon_mode() is False


def test_codecarbon_mode_auto_follows_hardware(monkeypatch):
    monkeypatch.delenv("PULSELEDGER_USE_CODECARBON", raising=False)
    # In 'auto' the decision must equal hardware availability.
    assert resolve_codecarbon_mode() == hardware_power_available()
    monkeypatch.setenv("PULSELEDGER_USE_CODECARBON", "auto")
    assert resolve_codecarbon_mode() == hardware_power_available()

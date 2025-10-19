"""
Energy Consumption Tracking System for Sustainable AI.

This module implements comprehensive energy monitoring for machine learning
training and inference, integrating CodeCarbon for real-time energy tracking,
GPU/CPU measurement, and detailed reporting capabilities.
"""

import time
import threading
import psutil
import torch
import numpy as np
import warnings
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from abc import ABC, abstractmethod

# Optional GPU monitoring
try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    warnings.warn("GPUtil not available. Install with: pip install GPUtil")

# CodeCarbon integration
try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker

    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    warnings.warn("CodeCarbon not available. Install with: pip install codecarbon")

try:
    from ..core.logging import get_logger, get_audit_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_logger, get_audit_logger

    # Create minimal implementations for testing
    class MockAuditLogger:
        def log_model_operation(self, **kwargs):
            pass

    def get_audit_logger():
        return MockAuditLogger()


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class EnergyConfig:
    """Configuration for energy consumption tracking."""

    # Tracking settings
    tracking_interval: float = 1.0  # seconds
    enable_gpu_tracking: bool = True
    enable_cpu_tracking: bool = True
    enable_codecarbon: bool = True

    # CodeCarbon settings
    project_name: str = "sustainable-credit-risk-ai"
    experiment_id: Optional[str] = None
    country_iso_code: str = "USA"  # Default to USA
    region: Optional[str] = None
    cloud_provider: Optional[str] = None
    cloud_region: Optional[str] = None

    # Output settings
    output_dir: str = "energy_logs"
    save_to_file: bool = True
    log_level: str = "INFO"

    # Measurement precision
    measurement_precision: int = 6  # decimal places

    # Aggregation settings
    enable_real_time_aggregation: bool = True
    aggregation_window: int = 60  # seconds


@dataclass
class EnergyMeasurement:
    """Container for energy measurement data."""

    timestamp: datetime
    cpu_energy_kwh: float
    gpu_energy_kwh: float
    total_energy_kwh: float
    cpu_utilization: float
    gpu_utilization: float
    memory_usage_gb: float
    gpu_memory_usage_gb: float
    power_draw_watts: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert measurement to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_energy_kwh": self.cpu_energy_kwh,
            "gpu_energy_kwh": self.gpu_energy_kwh,
            "total_energy_kwh": self.total_energy_kwh,
            "cpu_utilization": self.cpu_utilization,
            "gpu_utilization": self.gpu_utilization,
            "memory_usage_gb": self.memory_usage_gb,
            "gpu_memory_usage_gb": self.gpu_memory_usage_gb,
            "power_draw_watts": self.power_draw_watts,
        }


@dataclass
class EnergyReport:
    """Container for energy consumption report."""

    experiment_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    # Energy consumption
    total_energy_kwh: float
    cpu_energy_kwh: float
    gpu_energy_kwh: float

    # Carbon emissions (if CodeCarbon available)
    carbon_emissions_kg: Optional[float] = None
    carbon_intensity_gco2_kwh: Optional[float] = None

    # Resource utilization
    avg_cpu_utilization: float = 0.0
    avg_gpu_utilization: float = 0.0
    peak_memory_usage_gb: float = 0.0
    peak_gpu_memory_usage_gb: float = 0.0

    # Power consumption
    avg_power_draw_watts: float = 0.0
    peak_power_draw_watts: float = 0.0

    # Cost estimation (optional)
    estimated_cost_usd: Optional[float] = None
    energy_price_per_kwh: Optional[float] = None

    # Measurements
    measurements: List[EnergyMeasurement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "total_energy_kwh": self.total_energy_kwh,
            "cpu_energy_kwh": self.cpu_energy_kwh,
            "gpu_energy_kwh": self.gpu_energy_kwh,
            "carbon_emissions_kg": self.carbon_emissions_kg,
            "carbon_intensity_gco2_kwh": self.carbon_intensity_gco2_kwh,
            "avg_cpu_utilization": self.avg_cpu_utilization,
            "avg_gpu_utilization": self.avg_gpu_utilization,
            "peak_memory_usage_gb": self.peak_memory_usage_gb,
            "peak_gpu_memory_usage_gb": self.peak_gpu_memory_usage_gb,
            "avg_power_draw_watts": self.avg_power_draw_watts,
            "peak_power_draw_watts": self.peak_power_draw_watts,
            "estimated_cost_usd": self.estimated_cost_usd,
            "energy_price_per_kwh": self.energy_price_per_kwh,
            "num_measurements": len(self.measurements),
        }


class EnergyMonitor(ABC):
    """Abstract base class for energy monitoring implementations."""

    @abstractmethod
    def start_monitoring(self) -> None:
        """Start energy monitoring."""
        pass

    @abstractmethod
    def stop_monitoring(self) -> EnergyMeasurement:
        """Stop monitoring and return final measurement."""
        pass

    @abstractmethod
    def get_current_measurement(self) -> EnergyMeasurement:
        """Get current energy measurement."""
        pass


class SystemEnergyMonitor(EnergyMonitor):
    """System-level energy monitoring using psutil and GPUtil."""

    def __init__(self, config: EnergyConfig):
        self.config = config
        self.start_time = None
        self.baseline_energy = None

        # Initialize GPU monitoring if available
        self.gpu_available = (
            torch.cuda.is_available()
            and self.config.enable_gpu_tracking
            and GPUTIL_AVAILABLE
        )
        if self.gpu_available:
            try:
                self.gpus = GPUtil.getGPUs()
                if not self.gpus:
                    self.gpu_available = False
                    logger.warning("No GPUs detected despite CUDA availability")
            except Exception as e:
                self.gpu_available = False
                logger.warning(f"GPU monitoring disabled due to error: {e}")
        elif not GPUTIL_AVAILABLE:
            logger.warning("GPU monitoring disabled - GPUtil not available")

        logger.info(
            f"System energy monitor initialized - GPU tracking: {self.gpu_available}"
        )

    def start_monitoring(self) -> None:
        """Start energy monitoring."""
        self.start_time = datetime.now()
        self.baseline_energy = self._get_baseline_energy()
        logger.debug("System energy monitoring started")

    def stop_monitoring(self) -> EnergyMeasurement:
        """Stop monitoring and return final measurement."""
        if self.start_time is None:
            raise RuntimeError("Monitoring not started")

        final_measurement = self.get_current_measurement()
        logger.debug("System energy monitoring stopped")
        return final_measurement

    def get_current_measurement(self) -> EnergyMeasurement:
        """Get current energy measurement."""
        if self.start_time is None:
            raise RuntimeError("Monitoring not started")

        current_time = datetime.now()
        duration = (current_time - self.start_time).total_seconds()

        # CPU measurements
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Estimate CPU energy consumption
        # This is an approximation based on CPU utilization and typical power consumption
        cpu_power_watts = self._estimate_cpu_power(cpu_percent)
        cpu_energy_kwh = (cpu_power_watts * duration / 3600) / 1000

        # GPU measurements
        gpu_energy_kwh = 0.0
        gpu_utilization = 0.0
        gpu_memory_usage_gb = 0.0
        gpu_power_watts = 0.0

        if self.gpu_available:
            try:
                gpu_stats = self._get_gpu_stats()
                gpu_utilization = gpu_stats["utilization"]
                gpu_memory_usage_gb = gpu_stats["memory_used_gb"]
                gpu_power_watts = gpu_stats["power_watts"]
                gpu_energy_kwh = (gpu_power_watts * duration / 3600) / 1000
            except Exception as e:
                logger.warning(f"GPU measurement failed: {e}")

        total_energy_kwh = cpu_energy_kwh + gpu_energy_kwh
        total_power_watts = cpu_power_watts + gpu_power_watts

        return EnergyMeasurement(
            timestamp=current_time,
            cpu_energy_kwh=round(cpu_energy_kwh, self.config.measurement_precision),
            gpu_energy_kwh=round(gpu_energy_kwh, self.config.measurement_precision),
            total_energy_kwh=round(total_energy_kwh, self.config.measurement_precision),
            cpu_utilization=round(cpu_percent, 2),
            gpu_utilization=round(gpu_utilization, 2),
            memory_usage_gb=round(memory.used / (1024**3), 2),
            gpu_memory_usage_gb=round(gpu_memory_usage_gb, 2),
            power_draw_watts=round(total_power_watts, 2),
        )

    def _get_baseline_energy(self) -> Dict[str, float]:
        """Get baseline energy consumption."""
        # This would typically measure idle power consumption
        # For now, we'll use estimated baseline values
        return {
            "cpu_baseline_watts": 10.0,  # Typical idle CPU power
            "gpu_baseline_watts": (
                20.0 if self.gpu_available else 0.0
            ),  # Typical idle GPU power
        }

    def _estimate_cpu_power(self, cpu_percent: float) -> float:
        """Estimate CPU power consumption based on utilization."""
        # Typical CPU power consumption ranges from 10W (idle) to 100W (full load)
        # This is a rough approximation
        idle_power = 10.0
        max_power = 100.0
        return idle_power + (max_power - idle_power) * (cpu_percent / 100.0)

    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU statistics."""
        if not self.gpu_available:
            return {"utilization": 0.0, "memory_used_gb": 0.0, "power_watts": 0.0}

        try:
            if not GPUTIL_AVAILABLE:
                return {"utilization": 0.0, "memory_used_gb": 0.0, "power_watts": 0.0}

            # Refresh GPU list
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {"utilization": 0.0, "memory_used_gb": 0.0, "power_watts": 0.0}

            # Use first GPU for simplicity (can be extended for multi-GPU)
            gpu = gpus[0]

            # Get power consumption if available
            power_watts = 0.0
            try:
                # Try to get actual power draw (may not be available on all GPUs)
                if hasattr(gpu, "powerDraw") and gpu.powerDraw is not None:
                    power_watts = gpu.powerDraw
                else:
                    # Estimate based on utilization (typical GPU: 20W idle, 250W full load)
                    power_watts = 20.0 + (250.0 - 20.0) * (gpu.load / 100.0)
            except Exception:
                # Fallback estimation
                power_watts = 20.0 + (250.0 - 20.0) * (gpu.load / 100.0)

            return {
                "utilization": gpu.load * 100,  # Convert to percentage
                "memory_used_gb": gpu.memoryUsed / 1024,  # Convert MB to GB
                "power_watts": power_watts,
            }

        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return {"utilization": 0.0, "memory_used_gb": 0.0, "power_watts": 0.0}


class CodeCarbonEnergyMonitor(EnergyMonitor):
    """Energy monitoring using CodeCarbon library."""

    def __init__(self, config: EnergyConfig):
        if not CODECARBON_AVAILABLE:
            raise ImportError("CodeCarbon is required for CodeCarbonEnergyMonitor")

        self.config = config
        self.tracker = None
        self.start_time = None

        # Initialize CodeCarbon tracker
        self._initialize_tracker()

        logger.info("CodeCarbon energy monitor initialized")

    def _initialize_tracker(self):
        """Initialize CodeCarbon emissions tracker."""
        try:
            # Create output directory
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize tracker with configuration
            tracker_kwargs = {
                "project_name": self.config.project_name,
                "output_dir": str(output_dir),
                "log_level": self.config.log_level,
                "save_to_file": self.config.save_to_file,
                "country_iso_code": self.config.country_iso_code,
            }

            # Add optional parameters if provided
            if self.config.experiment_id:
                tracker_kwargs["experiment_id"] = self.config.experiment_id
            if self.config.region:
                tracker_kwargs["region"] = self.config.region
            if self.config.cloud_provider:
                tracker_kwargs["cloud_provider"] = self.config.cloud_provider
            if self.config.cloud_region:
                tracker_kwargs["cloud_region"] = self.config.cloud_region

            self.tracker = EmissionsTracker(**tracker_kwargs)

        except Exception as e:
            logger.error(f"Failed to initialize CodeCarbon tracker: {e}")
            # Fallback to offline tracker
            try:
                self.tracker = OfflineEmissionsTracker(
                    project_name=self.config.project_name,
                    country_iso_code=self.config.country_iso_code,
                )
                logger.info("Using offline CodeCarbon tracker as fallback")
            except Exception as e2:
                logger.error(f"Failed to initialize offline tracker: {e2}")
                raise

    def start_monitoring(self) -> None:
        """Start energy monitoring."""
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized")

        self.start_time = datetime.now()
        self.tracker.start()
        logger.debug("CodeCarbon monitoring started")

    def stop_monitoring(self) -> EnergyMeasurement:
        """Stop monitoring and return final measurement."""
        if self.tracker is None or self.start_time is None:
            raise RuntimeError("Monitoring not started")

        # Stop CodeCarbon tracking
        emissions_kg = self.tracker.stop()

        # Get final measurement
        final_measurement = self.get_current_measurement()

        # Add emissions data if available
        if emissions_kg is not None:
            # Store emissions data for later use in report generation
            self._last_emissions = emissions_kg

        logger.debug(
            f"CodeCarbon monitoring stopped - Emissions: {emissions_kg} kg CO2"
        )
        return final_measurement

    def get_current_measurement(self) -> EnergyMeasurement:
        """Get current energy measurement."""
        if self.start_time is None:
            raise RuntimeError("Monitoring not started")

        current_time = datetime.now()

        # Get system measurements (CodeCarbon doesn't provide real-time measurements)
        system_monitor = SystemEnergyMonitor(self.config)
        system_monitor.start_time = self.start_time  # Use same start time
        measurement = system_monitor.get_current_measurement()

        return measurement


class EnergyTracker:
    """Main energy tracking class that coordinates different monitoring approaches."""

    def __init__(self, config: Optional[EnergyConfig] = None):
        self.config = config or EnergyConfig()
        self.monitor = None
        self.measurements = []
        self.monitoring_thread = None
        self.stop_monitoring_flag = threading.Event()

        # Initialize appropriate monitor
        self._initialize_monitor()

        logger.info(f"Energy tracker initialized with {type(self.monitor).__name__}")

    def _initialize_monitor(self):
        """Initialize the appropriate energy monitor."""
        if self.config.enable_codecarbon and CODECARBON_AVAILABLE:
            try:
                self.monitor = CodeCarbonEnergyMonitor(self.config)
                return
            except Exception as e:
                logger.warning(
                    f"CodeCarbon monitor failed, falling back to system monitor: {e}"
                )

        # Fallback to system monitor
        self.monitor = SystemEnergyMonitor(self.config)

    @contextmanager
    def track(self, experiment_id: str = None):
        """Context manager for energy tracking."""
        experiment_id = (
            experiment_id or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        try:
            self.start_tracking(experiment_id)
            yield self
        finally:
            report = self.stop_tracking()
            logger.info(
                f"Energy tracking completed for {experiment_id}: "
                f"{report.total_energy_kwh:.6f} kWh"
            )

    def start_tracking(self, experiment_id: str = None) -> str:
        """Start energy tracking."""
        if self.monitor is None:
            raise RuntimeError("Monitor not initialized")

        experiment_id = (
            experiment_id or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.experiment_id = experiment_id

        # Clear previous measurements
        self.measurements.clear()

        # Start monitoring
        self.monitor.start_monitoring()

        # Start background measurement collection if enabled
        if self.config.enable_real_time_aggregation:
            self._start_background_monitoring()

        # Log start
        audit_logger.log_model_operation(
            user_id="system",
            model_id="energy_tracker",
            operation="start_tracking",
            success=True,
            details={"experiment_id": experiment_id},
        )

        logger.info(f"Energy tracking started for experiment: {experiment_id}")
        return experiment_id

    def stop_tracking(self) -> EnergyReport:
        """Stop energy tracking and generate report."""
        if self.monitor is None:
            raise RuntimeError("Monitor not initialized")

        # Stop background monitoring
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring_flag.set()
            self.monitoring_thread.join(timeout=5.0)

        # Get final measurement
        final_measurement = self.monitor.stop_monitoring()
        self.measurements.append(final_measurement)

        # Generate report
        report = self._generate_report()

        # Save report if configured
        if self.config.save_to_file:
            self._save_report(report)

        # Log completion
        audit_logger.log_model_operation(
            user_id="system",
            model_id="energy_tracker",
            operation="stop_tracking",
            success=True,
            details={
                "experiment_id": report.experiment_id,
                "total_energy_kwh": report.total_energy_kwh,
                "duration_seconds": report.duration_seconds,
                "carbon_emissions_kg": report.carbon_emissions_kg,
            },
        )

        logger.info(
            f"Energy tracking completed: {report.total_energy_kwh:.6f} kWh, "
            f"{report.duration_seconds:.2f}s"
        )

        return report

    def _start_background_monitoring(self):
        """Start background thread for continuous monitoring."""
        self.stop_monitoring_flag.clear()
        self.monitoring_thread = threading.Thread(
            target=self._background_monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.debug("Background monitoring thread started")

    def _background_monitoring_loop(self):
        """Background monitoring loop."""
        while not self.stop_monitoring_flag.is_set():
            try:
                measurement = self.monitor.get_current_measurement()
                self.measurements.append(measurement)

                # Wait for next measurement
                if self.stop_monitoring_flag.wait(self.config.tracking_interval):
                    break  # Stop flag was set

            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                # Continue monitoring despite errors
                if self.stop_monitoring_flag.wait(self.config.tracking_interval):
                    break

    def _generate_report(self) -> EnergyReport:
        """Generate energy consumption report."""
        if not self.measurements:
            raise RuntimeError("No measurements available")

        start_time = self.measurements[0].timestamp
        end_time = self.measurements[-1].timestamp
        duration = (end_time - start_time).total_seconds()

        # Calculate totals and averages
        total_energy = self.measurements[-1].total_energy_kwh
        cpu_energy = self.measurements[-1].cpu_energy_kwh
        gpu_energy = self.measurements[-1].gpu_energy_kwh

        # Calculate averages
        avg_cpu_util = np.mean([m.cpu_utilization for m in self.measurements])
        avg_gpu_util = np.mean([m.gpu_utilization for m in self.measurements])
        avg_power = np.mean([m.power_draw_watts for m in self.measurements])

        # Calculate peaks
        peak_memory = max([m.memory_usage_gb for m in self.measurements])
        peak_gpu_memory = max([m.gpu_memory_usage_gb for m in self.measurements])
        peak_power = max([m.power_draw_watts for m in self.measurements])

        # Get carbon emissions if available
        carbon_emissions = None
        carbon_intensity = None
        if isinstance(self.monitor, CodeCarbonEnergyMonitor):
            if hasattr(self.monitor, "_last_emissions"):
                carbon_emissions = self.monitor._last_emissions
                # Estimate carbon intensity
                if total_energy > 0:
                    carbon_intensity = (
                        carbon_emissions * 1000
                    ) / total_energy  # gCO2/kWh

        # Estimate cost (using average US electricity price)
        energy_price = 0.13  # USD per kWh (US average)
        estimated_cost = total_energy * energy_price

        report = EnergyReport(
            experiment_id=getattr(self, "experiment_id", "unknown"),
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_energy_kwh=total_energy,
            cpu_energy_kwh=cpu_energy,
            gpu_energy_kwh=gpu_energy,
            carbon_emissions_kg=carbon_emissions,
            carbon_intensity_gco2_kwh=carbon_intensity,
            avg_cpu_utilization=avg_cpu_util,
            avg_gpu_utilization=avg_gpu_util,
            peak_memory_usage_gb=peak_memory,
            peak_gpu_memory_usage_gb=peak_gpu_memory,
            avg_power_draw_watts=avg_power,
            peak_power_draw_watts=peak_power,
            estimated_cost_usd=estimated_cost,
            energy_price_per_kwh=energy_price,
            measurements=self.measurements.copy(),
        )

        return report

    def _save_report(self, report: EnergyReport):
        """Save energy report to file."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save detailed report
            report_file = output_dir / f"{report.experiment_id}_energy_report.json"
            with open(report_file, "w") as f:
                json.dump(report.to_dict(), f, indent=2)

            # Save measurements separately for analysis
            measurements_file = output_dir / f"{report.experiment_id}_measurements.json"
            measurements_data = [m.to_dict() for m in report.measurements]
            with open(measurements_file, "w") as f:
                json.dump(measurements_data, f, indent=2)

            logger.debug(f"Energy report saved to {report_file}")

        except Exception as e:
            logger.error(f"Failed to save energy report: {e}")

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current energy consumption statistics."""
        if self.monitor is None:
            return {}

        try:
            current_measurement = self.monitor.get_current_measurement()
            return {
                "current_power_watts": current_measurement.power_draw_watts,
                "total_energy_kwh": current_measurement.total_energy_kwh,
                "cpu_utilization": current_measurement.cpu_utilization,
                "gpu_utilization": current_measurement.gpu_utilization,
                "memory_usage_gb": current_measurement.memory_usage_gb,
                "gpu_memory_usage_gb": current_measurement.gpu_memory_usage_gb,
                "measurements_count": len(self.measurements),
            }
        except Exception as e:
            logger.error(f"Failed to get current stats: {e}")
            return {}


# Utility functions for easy integration


def track_energy(experiment_id: str = None, config: Optional[EnergyConfig] = None):
    """
    Context manager for energy tracking.

    Args:
        experiment_id: Unique identifier for the experiment
        config: Energy tracking configuration

    Returns:
        Context manager that yields EnergyTracker instance
    """
    tracker = EnergyTracker(config)
    return tracker.track(experiment_id)


def measure_training_energy(
    training_func,
    *args,
    experiment_id: str = None,
    config: Optional[EnergyConfig] = None,
    **kwargs,
):
    """
    Measure energy consumption of a training function.

    Args:
        training_func: Function to measure
        *args: Arguments for training function
        experiment_id: Unique identifier for the experiment
        config: Energy tracking configuration
        **kwargs: Keyword arguments for training function

    Returns:
        Tuple of (training_result, energy_report)
    """
    with track_energy(experiment_id, config) as tracker:
        result = training_func(*args, **kwargs)
        report = tracker.stop_tracking()
        return result, report


def measure_inference_energy(
    model, data_loader, experiment_id: str = None, config: Optional[EnergyConfig] = None
):
    """
    Measure energy consumption during model inference.

    Args:
        model: PyTorch model
        data_loader: Data loader for inference
        experiment_id: Unique identifier for the experiment
        config: Energy tracking configuration

    Returns:
        Tuple of (predictions, energy_report)
    """
    predictions = []

    with track_energy(experiment_id, config) as tracker:
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())

        report = tracker.stop_tracking()

    return np.concatenate(predictions), report

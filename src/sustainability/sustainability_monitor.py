"""
Comprehensive Sustainability Monitoring System.

This module integrates energy tracking, carbon calculation, ESG metrics collection,
and real-time monitoring with alerting and optimization recommendations.
"""

import json
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import warnings

try:
    from ..core.logging import get_logger, get_audit_logger
    from .energy_tracker import EnergyTracker, EnergyReport, EnergyConfig
    from .carbon_calculator import CarbonCalculator, CarbonFootprint, CarbonFootprintConfig
    from .esg_metrics import ESGMetricsCollector, ESGMetric, ESGScore, ESGReport, ESGCategory
    from .esg_dashboard import ESGDashboard, create_esg_dashboard
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.logging import get_logger, get_audit_logger
    from sustainability.energy_tracker import EnergyTracker, EnergyReport, EnergyConfig
    from sustainability.carbon_calculator import CarbonCalculator, CarbonFootprint, CarbonFootprintConfig
    from sustainability.esg_metrics import ESGMetricsCollector, ESGMetric, ESGScore, ESGReport, ESGCategory
    from sustainability.esg_dashboard import ESGDashboard, create_esg_dashboard

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SustainabilityAlert:
    """Container for sustainability alerts."""
    alert_id: str
    level: AlertLevel
    category: str
    message: str
    timestamp: datetime
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "category": self.category,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "recommendations": self.recommendations
        }


@dataclass
class SustainabilityConfig:
    """Configuration for sustainability monitoring."""
    # Energy tracking
    energy_config: Optional[EnergyConfig] = None
    
    # Carbon calculation
    carbon_config: Optional[CarbonFootprintConfig] = None
    
    # Monitoring settings
    monitoring_interval: int = 60  # seconds
    enable_real_time_monitoring: bool = True
    enable_dashboard: bool = True
    dashboard_port: int = 8050
    
    # Alert thresholds
    carbon_budget_warning_threshold: float = 0.8  # 80%
    carbon_budget_critical_threshold: float = 0.95  # 95%
    energy_efficiency_threshold: float = 0.7  # 70%
    
    # Optimization settings
    enable_optimization_recommendations: bool = True
    carbon_aware_scheduling: bool = True
    
    # Reporting
    output_dir: str = "sustainability_reports"
    save_reports: bool = True
    report_frequency: str = "daily"  # daily, weekly, monthly


class SustainabilityMonitor:
    """Comprehensive sustainability monitoring system."""
    
    def __init__(self, config: Optional[SustainabilityConfig] = None):
        self.config = config or SustainabilityConfig()
        
        # Initialize components
        self.energy_tracker = EnergyTracker(self.config.energy_config)
        self.carbon_calculator = CarbonCalculator(self.config.carbon_config)
        self.esg_collector = ESGMetricsCollector(self.carbon_calculator)
        
        # Initialize dashboard if enabled
        self.dashboard = None
        if self.config.enable_dashboard:
            try:
                self.dashboard = create_esg_dashboard(
                    esg_collector=self.esg_collector,
                    port=self.config.dashboard_port
                )
                logger.info(f"ESG Dashboard initialized on port {self.config.dashboard_port}")
            except ImportError:
                logger.warning("Dashboard not available - Dash/Plotly not installed")
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.active_experiments = {}
        self.alerts = []
        self.reports_history = []
        
        # Alert callbacks
        self.alert_callbacks = []
        
        logger.info("Sustainability monitor initialized")
    
    def start_monitoring(self):
        """Start real-time sustainability monitoring."""
        
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        
        if self.config.enable_real_time_monitoring:
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Real-time sustainability monitoring started")
        
        # Start dashboard if enabled
        if self.dashboard and self.config.enable_dashboard:
            dashboard_thread = threading.Thread(
                target=self._start_dashboard,
                daemon=True
            )
            dashboard_thread.start()
            logger.info("ESG Dashboard started")
        
        audit_logger.log_model_operation(
            user_id="system",
            model_id="sustainability_monitor",
            operation="start_monitoring",
            success=True,
            details={"monitoring_interval": self.config.monitoring_interval}
        )
    
    def stop_monitoring(self):
        """Stop sustainability monitoring."""
        
        self.is_monitoring = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Sustainability monitoring stopped")
        
        audit_logger.log_model_operation(
            user_id="system",
            model_id="sustainability_monitor",
            operation="stop_monitoring",
            success=True,
            details={}
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        
        while self.is_monitoring:
            try:
                # Collect current metrics
                self._collect_and_analyze_metrics()
                
                # Check for alerts
                self._check_alert_conditions()
                
                # Generate optimization recommendations
                if self.config.enable_optimization_recommendations:
                    self._generate_optimization_recommendations()
                
                # Sleep until next monitoring cycle
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _start_dashboard(self):
        """Start the ESG dashboard in a separate thread."""
        try:
            self.dashboard.run(host='0.0.0.0')
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
    
    def start_experiment_tracking(self, experiment_id: str, 
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking sustainability metrics for an experiment."""
        
        # Start energy tracking
        energy_experiment_id = self.energy_tracker.start_tracking(experiment_id)
        
        # Store experiment metadata
        self.active_experiments[experiment_id] = {
            "start_time": datetime.now(),
            "energy_experiment_id": energy_experiment_id,
            "metadata": metadata or {}
        }
        
        logger.info(f"Started sustainability tracking for experiment: {experiment_id}")
        
        audit_logger.log_model_operation(
            user_id="system",
            model_id="sustainability_monitor",
            operation="start_experiment_tracking",
            success=True,
            details={"experiment_id": experiment_id}
        )
        
        return experiment_id
    
    def stop_experiment_tracking(self, experiment_id: str) -> Dict[str, Any]:
        """Stop tracking and generate sustainability report for an experiment."""
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found in active experiments")
        
        experiment_data = self.active_experiments[experiment_id]
        
        # Stop energy tracking
        energy_report = self.energy_tracker.stop_tracking()
        
        # Calculate carbon footprint
        carbon_footprint = self.carbon_calculator.calculate_carbon_footprint(
            energy_report, 
            region=self.config.carbon_config.default_region if self.config.carbon_config else "US"
        )
        
        # Collect ESG metrics
        esg_metrics = self.esg_collector.collect_all_metrics(
            energy_reports=[energy_report],
            carbon_footprints=[carbon_footprint]
        )
        
        # Calculate ESG score
        esg_score = self.esg_collector.calculate_esg_score(esg_metrics)
        
        # Generate recommendations
        recommendations = self.esg_collector.generate_recommendations(esg_metrics)
        
        # Create experiment report
        experiment_report = {
            "experiment_id": experiment_id,
            "start_time": experiment_data["start_time"].isoformat(),
            "end_time": datetime.now().isoformat(),
            "metadata": experiment_data["metadata"],
            "energy_report": energy_report.to_dict(),
            "carbon_footprint": carbon_footprint.to_dict(),
            "esg_metrics": [metric.to_dict() for metric in esg_metrics],
            "esg_score": esg_score.to_dict(),
            "recommendations": recommendations
        }
        
        # Save report if configured
        if self.config.save_reports:
            self._save_experiment_report(experiment_report)
        
        # Remove from active experiments
        del self.active_experiments[experiment_id]
        
        logger.info(f"Completed sustainability tracking for experiment: {experiment_id}")
        
        audit_logger.log_model_operation(
            user_id="system",
            model_id="sustainability_monitor",
            operation="stop_experiment_tracking",
            success=True,
            details={
                "experiment_id": experiment_id,
                "total_emissions_kg": carbon_footprint.total_emissions_kg,
                "energy_kwh": energy_report.total_energy_kwh,
                "esg_score": esg_score.overall_score
            }
        )
        
        return experiment_report
    
    def _collect_and_analyze_metrics(self):
        """Collect and analyze current sustainability metrics."""
        
        # Get recent carbon footprints and energy reports
        recent_footprints = self.carbon_calculator.carbon_history[-10:]  # Last 10
        
        # Create mock energy reports for recent footprints (in real implementation, these would be stored)
        recent_energy_reports = []
        for footprint in recent_footprints:
            # Create mock energy report from footprint data
            from .energy_tracker import EnergyReport
            energy_report = EnergyReport(
                experiment_id=footprint.experiment_id,
                start_time=footprint.timestamp - timedelta(hours=1),
                end_time=footprint.timestamp,
                duration_seconds=3600,
                total_energy_kwh=footprint.energy_kwh,
                cpu_energy_kwh=footprint.energy_kwh * 0.7,
                gpu_energy_kwh=footprint.energy_kwh * 0.3
            )
            recent_energy_reports.append(energy_report)
        
        if recent_energy_reports and recent_footprints:
            # Collect ESG metrics
            current_metrics = self.esg_collector.collect_all_metrics(
                energy_reports=recent_energy_reports,
                carbon_footprints=recent_footprints
            )
            
            # Calculate current ESG score
            current_score = self.esg_collector.calculate_esg_score(current_metrics)
            
            logger.debug(f"Current ESG score: {current_score.overall_score:.2f}")
    
    def _check_alert_conditions(self):
        """Check for alert conditions and generate alerts."""
        
        # Check carbon budget alerts
        if self.config.carbon_config and self.config.carbon_config.enable_budget_monitoring:
            try:
                # Check daily budget
                if self.config.carbon_config.daily_carbon_budget_kg:
                    daily_status = self.carbon_calculator.track_carbon_budget("daily")
                    self._check_budget_alert(daily_status, "daily")
                
                # Check monthly budget
                if self.config.carbon_config.monthly_carbon_budget_kg:
                    monthly_status = self.carbon_calculator.track_carbon_budget("monthly")
                    self._check_budget_alert(monthly_status, "monthly")
                    
            except Exception as e:
                logger.warning(f"Error checking carbon budget alerts: {e}")
        
        # Check energy efficiency alerts
        self._check_energy_efficiency_alerts()
    
    def _check_budget_alert(self, budget_status, period: str):
        """Check and generate budget alerts."""
        
        usage_percentage = budget_status.usage_percentage / 100
        
        if usage_percentage >= self.config.carbon_budget_critical_threshold:
            alert = SustainabilityAlert(
                alert_id=f"carbon_budget_critical_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.CRITICAL,
                category="carbon_budget",
                message=f"CRITICAL: {period.title()} carbon budget usage at {usage_percentage*100:.1f}% "
                       f"({budget_status.current_usage_kg:.3f} kg CO2e / {budget_status.budget_limit_kg:.3f} kg CO2e)",
                timestamp=datetime.now(),
                metric_value=usage_percentage,
                threshold_value=self.config.carbon_budget_critical_threshold,
                recommendations=[
                    "Immediately reduce model training frequency",
                    "Switch to regions with cleaner energy grids",
                    "Implement aggressive model compression techniques"
                ]
            )
            self._add_alert(alert)
            
        elif usage_percentage >= self.config.carbon_budget_warning_threshold:
            alert = SustainabilityAlert(
                alert_id=f"carbon_budget_warning_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                category="carbon_budget",
                message=f"WARNING: {period.title()} carbon budget usage at {usage_percentage*100:.1f}% "
                       f"({budget_status.current_usage_kg:.3f} kg CO2e / {budget_status.budget_limit_kg:.3f} kg CO2e)",
                timestamp=datetime.now(),
                metric_value=usage_percentage,
                threshold_value=self.config.carbon_budget_warning_threshold,
                recommendations=[
                    "Consider optimizing model architectures",
                    "Schedule training during low-carbon grid times",
                    "Review and optimize hyperparameter search strategies"
                ]
            )
            self._add_alert(alert)
    
    def _check_energy_efficiency_alerts(self):
        """Check energy efficiency and generate alerts if needed."""
        
        # Get recent energy reports from active experiments
        if not self.active_experiments:
            return
        
        # Calculate average energy efficiency (placeholder logic)
        # In real implementation, this would use actual efficiency metrics
        current_efficiency = 0.75  # Placeholder
        
        if current_efficiency < self.config.energy_efficiency_threshold:
            alert = SustainabilityAlert(
                alert_id=f"energy_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                category="energy_efficiency",
                message=f"Energy efficiency below threshold: {current_efficiency:.2f} < {self.config.energy_efficiency_threshold:.2f}",
                timestamp=datetime.now(),
                metric_value=current_efficiency,
                threshold_value=self.config.energy_efficiency_threshold,
                recommendations=[
                    "Implement model pruning to reduce computational requirements",
                    "Use mixed precision training to improve efficiency",
                    "Optimize batch sizes and learning rates"
                ]
            )
            self._add_alert(alert)
    
    def _add_alert(self, alert: SustainabilityAlert):
        """Add alert and notify callbacks."""
        
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Log alert
        logger.warning(f"Sustainability Alert [{alert.level.value.upper()}]: {alert.message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Audit log
        audit_logger.log_model_operation(
            user_id="system",
            model_id="sustainability_monitor",
            operation="generate_alert",
            success=True,
            details={
                "alert_level": alert.level.value,
                "alert_category": alert.category,
                "metric_value": alert.metric_value,
                "threshold_value": alert.threshold_value
            }
        )
    
    def _generate_optimization_recommendations(self):
        """Generate optimization recommendations based on current metrics."""
        
        # Get recent metrics
        recent_metrics = self.esg_collector.get_metrics_history(days=1)
        
        if recent_metrics:
            recommendations = self.esg_collector.generate_recommendations(recent_metrics)
            
            if recommendations:
                logger.info(f"Generated {len(recommendations)} optimization recommendations")
                
                # Create info alert with recommendations
                alert = SustainabilityAlert(
                    alert_id=f"optimization_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.INFO,
                    category="optimization",
                    message="New optimization recommendations available",
                    timestamp=datetime.now(),
                    recommendations=recommendations
                )
                self._add_alert(alert)
    
    def _save_experiment_report(self, report: Dict[str, Any]):
        """Save experiment sustainability report."""
        
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = output_dir / f"{report['experiment_id']}_sustainability_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.debug(f"Sustainability report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Error saving sustainability report: {e}")
    
    def add_alert_callback(self, callback: Callable[[SustainabilityAlert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current sustainability monitoring status."""
        
        recent_alerts = [alert for alert in self.alerts 
                        if alert.timestamp >= datetime.now() - timedelta(hours=24)]
        
        return {
            "monitoring_active": self.is_monitoring,
            "active_experiments": len(self.active_experiments),
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL]),
            "warning_alerts": len([a for a in recent_alerts if a.level == AlertLevel.WARNING]),
            "dashboard_available": self.dashboard is not None,
            "dashboard_port": self.config.dashboard_port if self.dashboard else None
        }
    
    def generate_sustainability_report(self, period_days: int = 30) -> ESGReport:
        """Generate comprehensive sustainability report for a period."""
        
        period_start = datetime.now() - timedelta(days=period_days)
        period_end = datetime.now()
        
        # Get metrics for the period
        period_metrics = self.esg_collector.get_metrics_history(days=period_days)
        
        if not period_metrics:
            logger.warning("No metrics available for sustainability report")
            # Return empty report
            return ESGReport(
                report_id=f"sustainability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                period_start=period_start,
                period_end=period_end,
                generated_at=datetime.now(),
                current_score=ESGScore(
                    environmental_score=0,
                    social_score=0,
                    governance_score=0,
                    overall_score=0,
                    timestamp=datetime.now()
                )
            )
        
        # Calculate current ESG score
        current_score = self.esg_collector.calculate_esg_score(period_metrics)
        
        # Generate recommendations and alerts
        recommendations = self.esg_collector.generate_recommendations(period_metrics)
        alerts = self.esg_collector.generate_alerts(period_metrics)
        
        # Create report
        report = ESGReport(
            report_id=f"sustainability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            period_start=period_start,
            period_end=period_end,
            generated_at=datetime.now(),
            current_score=current_score,
            metrics=period_metrics,
            recommendations=recommendations,
            alerts=alerts
        )
        
        # Save report if configured
        if self.config.save_reports:
            self._save_sustainability_report(report)
        
        # Store in history
        self.reports_history.append(report)
        
        logger.info(f"Generated sustainability report for {period_days} days")
        
        return report
    
    def _save_sustainability_report(self, report: ESGReport):
        """Save sustainability report to file."""
        
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = output_dir / f"{report.report_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            
            logger.debug(f"Sustainability report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Error saving sustainability report: {e}")


# Context manager for easy experiment tracking

class SustainabilityTracker:
    """Context manager for sustainability tracking."""
    
    def __init__(self, monitor: SustainabilityMonitor, experiment_id: str,
                 metadata: Optional[Dict[str, Any]] = None):
        self.monitor = monitor
        self.experiment_id = experiment_id
        self.metadata = metadata
        self.report = None
    
    def __enter__(self):
        self.monitor.start_experiment_tracking(self.experiment_id, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.report = self.monitor.stop_experiment_tracking(self.experiment_id)
        return False
    
    def get_report(self) -> Optional[Dict[str, Any]]:
        """Get the sustainability report for this experiment."""
        return self.report


# Utility functions

def create_sustainability_monitor(config: Optional[SustainabilityConfig] = None) -> SustainabilityMonitor:
    """Create and configure sustainability monitor."""
    return SustainabilityMonitor(config)


def track_sustainability(monitor: SustainabilityMonitor, experiment_id: str,
                        metadata: Optional[Dict[str, Any]] = None) -> SustainabilityTracker:
    """Create sustainability tracking context manager."""
    return SustainabilityTracker(monitor, experiment_id, metadata)
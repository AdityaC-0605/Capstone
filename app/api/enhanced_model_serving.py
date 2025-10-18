"""
Enhanced Model Serving Infrastructure Integration.

This module provides enhanced integration between the model serving infrastructure
and the inference service, with additional features for production readiness.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

try:
    from fastapi import HTTPException
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    warnings.warn("FastAPI not available. Install with: pip install fastapi")

try:
    from .model_serving import (
        ModelServingManager, ModelServingConfig, ModelStatus,
        RoutingStrategy, ModelMetadata
    )
    from .inference_service import CreditApplication, PredictionResponse, PredictionStatus, RiskLevel
    from ..core.logging import get_logger, get_audit_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.logging import get_logger, get_audit_logger
    
    # Mock classes for testing
    class MockModelServingManager:
        def __init__(self, config=None):
            self.config = config
            self.models = {}
        
        async def predict(self, input_data, model_id=None, version=None, user_id=None):
            return {"prediction": 0.5, "confidence": 0.8, "model_used": f"{model_id}:{version}"}
        
        def get_health_status(self):
            return {"status": "healthy", "models": {}}
        
        def load_model(self, model_id, version, file_path=None):
            return True
    
    ModelServingManager = MockModelServingManager
    
    class MockModelServingConfig:
        def __init__(self):
            self.max_concurrent_requests = 100
    
    ModelServingConfig = MockModelServingConfig
    
    class ModelStatus:
        READY = "ready"
        LOADING = "loading"
        ERROR = "error"
    
    class RoutingStrategy:
        WEIGHTED = "weighted"
        A_B_TEST = "a_b_test"
        CHAMPION_CHALLENGER = "champion_challenger"
    
    class MockCreditApplication:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MockPredictionResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class PredictionStatus:
        SUCCESS = "success"
        ERROR = "error"
    
    class RiskLevel:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        VERY_HIGH = "very_high"
    
    CreditApplication = MockCreditApplication
    PredictionResponse = MockPredictionResponse

logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class ModelDeploymentConfig:
    """Configuration for model deployment."""
    model_id: str
    version: str
    file_path: Optional[str] = None
    traffic_percentage: float = 0.0
    is_champion: bool = False
    is_challenger: bool = False
    auto_promote: bool = False
    performance_threshold: float = 0.85
    max_error_rate: float = 0.05


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    enabled: bool = True
    interval_seconds: int = 30
    timeout_seconds: int = 5
    failure_threshold: int = 3
    success_threshold: int = 2
    readiness_check_enabled: bool = True
    liveness_check_enabled: bool = True


@dataclass
class GracefulUpdateConfig:
    """Configuration for graceful model updates."""
    enabled: bool = True
    drain_timeout_seconds: int = 60
    health_check_interval: int = 5
    rollback_on_failure: bool = True
    canary_percentage: float = 10.0
    canary_duration_minutes: int = 15


class EnhancedModelServingManager:
    """Enhanced model serving manager with production features."""
    
    def __init__(self, config: Optional[ModelServingConfig] = None):
        self.config = config or ModelServingConfig()
        self.base_manager = ModelServingManager(self.config)
        
        # Enhanced features
        self.health_config = HealthCheckConfig()
        self.update_config = GracefulUpdateConfig()
        self.deployment_configs: Dict[str, ModelDeploymentConfig] = {}
        
        # Health monitoring
        self.health_status = {"status": "healthy", "last_check": datetime.now()}
        self.readiness_status = {"ready": True, "last_check": datetime.now()}
        
        # Update management
        self.active_updates: Dict[str, Dict[str, Any]] = {}
        self.model_versions: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Enhanced model serving manager initialized")
    
    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        if self.health_config.enabled:
            asyncio.create_task(self._health_check_loop())
        
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._update_management_loop())
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_config.interval_seconds)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.health_config.interval_seconds)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check."""
        start_time = time.time()
        
        try:
            # Check base manager health
            base_health = self.base_manager.get_health_status()
            
            # Check model responsiveness
            model_health = await self._check_model_responsiveness()
            
            # Check resource usage
            resource_health = self._check_resource_usage()
            
            # Aggregate health status
            overall_healthy = (
                base_health.get("ready_models", 0) > 0 and
                model_health["responsive_models"] > 0 and
                resource_health["memory_ok"] and
                resource_health["cpu_ok"]
            )
            
            self.health_status = {
                "status": "healthy" if overall_healthy else "unhealthy",
                "last_check": datetime.now(),
                "check_duration_ms": (time.time() - start_time) * 1000,
                "base_manager": base_health,
                "model_responsiveness": model_health,
                "resource_usage": resource_health
            }
            
            # Update readiness status
            self.readiness_status = {
                "ready": overall_healthy,
                "last_check": datetime.now(),
                "details": {
                    "models_ready": base_health.get("ready_models", 0) > 0,
                    "models_responsive": model_health["responsive_models"] > 0,
                    "resources_ok": resource_health["memory_ok"] and resource_health["cpu_ok"]
                }
            }
            
            if not overall_healthy:
                logger.warning(f"Health check failed: {self.health_status}")
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            self.health_status = {
                "status": "error",
                "last_check": datetime.now(),
                "error": str(e)
            }
            self.readiness_status = {
                "ready": False,
                "last_check": datetime.now(),
                "error": str(e)
            }
    
    async def _check_model_responsiveness(self) -> Dict[str, Any]:
        """Check if models are responsive."""
        responsive_models = 0
        total_models = 0
        model_latencies = {}
        
        try:
            # Get loaded models
            models = self.base_manager.model_loader.list_models()
            total_models = len(models)
            
            # Test each model with a simple prediction
            test_input = {
                "age": 30,
                "income": 50000,
                "credit_score": 700,
                "debt_to_income_ratio": 0.3,
                "loan_amount": 10000
            }
            
            for model_metadata in models:
                if model_metadata.status == ModelStatus.READY:
                    try:
                        start_time = time.time()
                        
                        # Make test prediction
                        result = await self.base_manager.predict(
                            test_input,
                            model_id=model_metadata.model_id,
                            version=model_metadata.version
                        )
                        
                        latency = (time.time() - start_time) * 1000
                        model_latencies[f"{model_metadata.model_id}:{model_metadata.version}"] = latency
                        
                        if latency < 1000:  # Less than 1 second
                            responsive_models += 1
                        
                    except Exception as e:
                        logger.warning(f"Model {model_metadata.model_id}:{model_metadata.version} not responsive: {e}")
            
        except Exception as e:
            logger.error(f"Model responsiveness check error: {e}")
        
        return {
            "responsive_models": responsive_models,
            "total_models": total_models,
            "responsiveness_rate": responsive_models / max(total_models, 1),
            "model_latencies": model_latencies
        }
    
    def _check_resource_usage(self) -> Dict[str, Any]:
        """Check resource usage."""
        try:
            import psutil
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_ok = memory.percent < 90
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_ok = cpu_percent < 90
            
            return {
                "memory_ok": memory_ok,
                "memory_percent": memory.percent,
                "cpu_ok": cpu_ok,
                "cpu_percent": cpu_percent
            }
            
        except ImportError:
            # psutil not available, assume OK
            return {
                "memory_ok": True,
                "memory_percent": 0,
                "cpu_ok": True,
                "cpu_percent": 0
            }
        except Exception as e:
            logger.error(f"Resource check error: {e}")
            return {
                "memory_ok": False,
                "cpu_ok": False,
                "error": str(e)
            }
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring loop."""
        while True:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _update_performance_metrics(self):
        """Update performance metrics for all models."""
        try:
            models = self.base_manager.model_loader.list_models()
            
            for model_metadata in models:
                model_key = f"{model_metadata.model_id}:{model_metadata.version}"
                
                if model_key not in self.performance_metrics:
                    self.performance_metrics[model_key] = {
                        "requests": 0,
                        "errors": 0,
                        "total_latency": 0,
                        "last_updated": datetime.now()
                    }
                
                # Update from model metadata
                metrics = self.performance_metrics[model_key]
                metrics["requests"] = model_metadata.request_count
                metrics["errors"] = model_metadata.error_count
                metrics["error_rate"] = model_metadata.error_count / max(model_metadata.request_count, 1)
                metrics["last_updated"] = datetime.now()
                
                # Check for auto-promotion/demotion
                await self._check_auto_promotion(model_key, metrics)
                
        except Exception as e:
            logger.error(f"Performance metrics update error: {e}")
    
    async def _check_auto_promotion(self, model_key: str, metrics: Dict[str, Any]):
        """Check if model should be auto-promoted or demoted."""
        if model_key not in self.deployment_configs:
            return
        
        config = self.deployment_configs[model_key]
        
        if not config.auto_promote:
            return
        
        try:
            # Check if challenger should be promoted
            if config.is_challenger and metrics["requests"] > 100:
                error_rate = metrics["error_rate"]
                
                if error_rate <= config.max_error_rate:
                    logger.info(f"Auto-promoting challenger {model_key} to champion")
                    await self._promote_challenger_to_champion(model_key)
                elif error_rate > config.max_error_rate * 2:
                    logger.warning(f"Auto-demoting challenger {model_key} due to high error rate")
                    await self._demote_challenger(model_key)
            
        except Exception as e:
            logger.error(f"Auto-promotion check error for {model_key}: {e}")
    
    async def _update_management_loop(self):
        """Background update management loop."""
        while True:
            try:
                await self._process_active_updates()
                await asyncio.sleep(self.update_config.health_check_interval)
            except Exception as e:
                logger.error(f"Update management error: {e}")
                await asyncio.sleep(self.update_config.health_check_interval)
    
    async def _process_active_updates(self):
        """Process active model updates."""
        for update_id, update_info in list(self.active_updates.items()):
            try:
                await self._process_single_update(update_id, update_info)
            except Exception as e:
                logger.error(f"Update processing error for {update_id}: {e}")
    
    async def _process_single_update(self, update_id: str, update_info: Dict[str, Any]):
        """Process a single model update."""
        stage = update_info.get("stage", "unknown")
        
        if stage == "canary":
            await self._process_canary_stage(update_id, update_info)
        elif stage == "rollout":
            await self._process_rollout_stage(update_id, update_info)
        elif stage == "complete":
            await self._complete_update(update_id, update_info)
    
    async def _process_canary_stage(self, update_id: str, update_info: Dict[str, Any]):
        """Process canary deployment stage."""
        start_time = update_info.get("canary_start_time")
        if not start_time:
            return
        
        elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
        
        if elapsed_minutes >= self.update_config.canary_duration_minutes:
            # Check canary performance
            model_key = update_info["model_key"]
            metrics = self.performance_metrics.get(model_key, {})
            
            error_rate = metrics.get("error_rate", 1.0)
            
            if error_rate <= 0.05:  # 5% error rate threshold
                logger.info(f"Canary deployment successful for {update_id}, proceeding to rollout")
                update_info["stage"] = "rollout"
                update_info["rollout_start_time"] = datetime.now()
            else:
                logger.warning(f"Canary deployment failed for {update_id}, rolling back")
                await self._rollback_update(update_id, update_info)
    
    async def _process_rollout_stage(self, update_id: str, update_info: Dict[str, Any]):
        """Process full rollout stage."""
        # Gradually increase traffic to new model
        model_key = update_info["model_key"]
        current_percentage = update_info.get("current_percentage", 10.0)
        
        if current_percentage < 100.0:
            new_percentage = min(current_percentage + 20.0, 100.0)
            update_info["current_percentage"] = new_percentage
            
            # Update model weights
            await self._update_model_traffic(model_key, new_percentage)
            
            logger.info(f"Updated traffic for {model_key} to {new_percentage}%")
            
            if new_percentage >= 100.0:
                update_info["stage"] = "complete"
    
    async def _complete_update(self, update_id: str, update_info: Dict[str, Any]):
        """Complete model update."""
        logger.info(f"Model update {update_id} completed successfully")
        del self.active_updates[update_id]
    
    async def _rollback_update(self, update_id: str, update_info: Dict[str, Any]):
        """Rollback failed update."""
        model_key = update_info["model_key"]
        
        # Restore previous model weights
        previous_weights = update_info.get("previous_weights", {})
        if previous_weights:
            self.base_manager.model_router.set_model_weights(previous_weights)
        
        logger.warning(f"Rolled back update {update_id} for {model_key}")
        del self.active_updates[update_id]
    
    # Public API methods
    
    async def deploy_model(self, config: ModelDeploymentConfig) -> bool:
        """Deploy a new model with enhanced features."""
        try:
            # Load the model
            success = self.base_manager.model_loader.load_model(
                config.model_id,
                config.version,
                config.file_path
            )
            
            if not success:
                return False
            
            # Store deployment config
            model_key = f"{config.model_id}:{config.version}"
            self.deployment_configs[model_key] = config
            
            # Track model versions
            if config.model_id not in self.model_versions:
                self.model_versions[config.model_id] = []
            
            if config.version not in self.model_versions[config.model_id]:
                self.model_versions[config.model_id].append(config.version)
            
            # Configure routing if needed
            if config.traffic_percentage > 0:
                await self._configure_model_routing(config)
            
            logger.info(f"Model {model_key} deployed successfully")
            
            # Audit log
            audit_logger.log_model_operation(
                user_id="system",
                model_id=model_key,
                operation="deploy",
                success=True,
                details={
                    "traffic_percentage": config.traffic_percentage,
                    "is_champion": config.is_champion,
                    "is_challenger": config.is_challenger
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return False
    
    async def _configure_model_routing(self, config: ModelDeploymentConfig):
        """Configure routing for deployed model."""
        model_key = f"{config.model_id}:{config.version}"
        
        if config.is_champion and config.is_challenger:
            # Champion-challenger setup
            self.base_manager.model_router.set_routing_strategy(RoutingStrategy.CHAMPION_CHALLENGER)
        elif config.traffic_percentage < 100:
            # A/B testing or canary
            if config.traffic_percentage <= 20:
                self.base_manager.model_router.set_routing_strategy(RoutingStrategy.A_B_TEST)
            else:
                self.base_manager.model_router.set_routing_strategy(RoutingStrategy.WEIGHTED)
        
        # Set model weights
        current_weights = self.base_manager.model_router.model_weights.copy()
        current_weights[model_key] = config.traffic_percentage / 100.0
        self.base_manager.model_router.set_model_weights(current_weights)
    
    async def graceful_update_model(self, model_id: str, new_version: str, 
                                  file_path: Optional[str] = None) -> str:
        """Perform graceful model update with canary deployment."""
        if not self.update_config.enabled:
            raise ValueError("Graceful updates are disabled")
        
        update_id = f"update_{model_id}_{new_version}_{int(time.time())}"
        model_key = f"{model_id}:{new_version}"
        
        try:
            # Store current state
            current_weights = self.base_manager.model_router.model_weights.copy()
            
            # Load new model
            success = self.base_manager.model_loader.load_model(
                model_id, new_version, file_path
            )
            
            if not success:
                raise Exception(f"Failed to load model {model_key}")
            
            # Start canary deployment
            canary_config = ModelDeploymentConfig(
                model_id=model_id,
                version=new_version,
                file_path=file_path,
                traffic_percentage=self.update_config.canary_percentage,
                is_challenger=True
            )
            
            await self.deploy_model(canary_config)
            
            # Track update
            self.active_updates[update_id] = {
                "model_key": model_key,
                "stage": "canary",
                "canary_start_time": datetime.now(),
                "previous_weights": current_weights,
                "current_percentage": self.update_config.canary_percentage
            }
            
            logger.info(f"Started graceful update {update_id} for {model_key}")
            
            return update_id
            
        except Exception as e:
            logger.error(f"Graceful update failed: {e}")
            raise e
    
    async def _promote_challenger_to_champion(self, model_key: str):
        """Promote challenger model to champion."""
        if model_key in self.deployment_configs:
            config = self.deployment_configs[model_key]
            config.is_champion = True
            config.is_challenger = False
            config.traffic_percentage = 80.0
            
            await self._configure_model_routing(config)
            
            logger.info(f"Promoted {model_key} to champion")
    
    async def _demote_challenger(self, model_key: str):
        """Demote challenger model."""
        if model_key in self.deployment_configs:
            config = self.deployment_configs[model_key]
            config.traffic_percentage = 0.0
            
            await self._configure_model_routing(config)
            
            logger.warning(f"Demoted challenger {model_key}")
    
    async def _update_model_traffic(self, model_key: str, percentage: float):
        """Update traffic percentage for a model."""
        current_weights = self.base_manager.model_router.model_weights.copy()
        current_weights[model_key] = percentage / 100.0
        self.base_manager.model_router.set_model_weights(current_weights)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return self.health_status
    
    def get_readiness_status(self) -> Dict[str, Any]:
        """Get readiness status."""
        return self.readiness_status
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get deployment status for all models."""
        return {
            "deployed_models": len(self.deployment_configs),
            "active_updates": len(self.active_updates),
            "model_versions": self.model_versions,
            "deployment_configs": {
                k: {
                    "model_id": v.model_id,
                    "version": v.version,
                    "traffic_percentage": v.traffic_percentage,
                    "is_champion": v.is_champion,
                    "is_challenger": v.is_challenger,
                    "auto_promote": v.auto_promote
                }
                for k, v in self.deployment_configs.items()
            },
            "active_updates": self.active_updates,
            "performance_metrics": self.performance_metrics
        }
    
    async def predict(self, input_data: Dict[str, Any], model_id: Optional[str] = None,
                     version: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Make prediction through enhanced serving infrastructure."""
        return await self.base_manager.predict(input_data, model_id, version, user_id)


# Utility functions for integration

def create_enhanced_model_serving_manager(config: Optional[ModelServingConfig] = None) -> EnhancedModelServingManager:
    """Create enhanced model serving manager with default configuration."""
    if config is None:
        config = ModelServingConfig()
        # Enhanced defaults
        config.max_concurrent_requests = 200
        config.enable_ab_testing = True
        config.default_routing_strategy = RoutingStrategy.WEIGHTED
    
    return EnhancedModelServingManager(config)


async def serve_prediction_enhanced(
    application: CreditApplication,
    manager: EnhancedModelServingManager,
    model_id: Optional[str] = None,
    version: Optional[str] = None,
    user_id: Optional[str] = None
) -> PredictionResponse:
    """Serve prediction through enhanced infrastructure."""
    
    try:
        # Prepare input data
        input_data = {
            "age": application.age,
            "income": application.income,
            "employment_length": application.employment_length,
            "debt_to_income_ratio": application.debt_to_income_ratio,
            "credit_score": application.credit_score,
            "loan_amount": application.loan_amount,
            "loan_purpose": application.loan_purpose,
            "home_ownership": application.home_ownership,
            "verification_status": application.verification_status
        }
        
        # Make prediction
        result = await manager.predict(input_data, model_id, version, user_id)
        
        # Convert to response format
        risk_score = float(result.get("prediction", 0.5))
        confidence = float(result.get("confidence", 0.8))
        
        # Determine risk level
        if risk_score < 0.25:
            risk_level = RiskLevel.LOW
        elif risk_score < 0.5:
            risk_level = RiskLevel.MEDIUM
        elif risk_score < 0.75:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.VERY_HIGH
        
        return PredictionResponse(
            prediction_id=f"pred_{int(time.time())}",
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            model_version=result.get("model_used", "unknown"),
            prediction_timestamp=datetime.now(),
            processing_time_ms=result.get("processing_time_ms", 0),
            status=PredictionStatus.SUCCESS,
            message="Prediction completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Enhanced prediction serving failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create enhanced manager
        manager = create_enhanced_model_serving_manager()
        
        # Deploy a model
        config = ModelDeploymentConfig(
            model_id="credit_risk",
            version="1.0.0",
            traffic_percentage=100.0,
            is_champion=True
        )
        
        success = await manager.deploy_model(config)
        print(f"Model deployment: {'success' if success else 'failed'}")
        
        # Check health
        health = manager.get_health_status()
        print(f"Health status: {health['status']}")
        
        # Check deployment status
        deployment = manager.get_deployment_status()
        print(f"Deployed models: {deployment['deployed_models']}")
    
    # Run example
    asyncio.run(main())
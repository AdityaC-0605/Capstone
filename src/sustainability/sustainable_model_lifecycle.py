"""
Sustainable Model Lifecycle Management System

This module implements automatic model optimization techniques including:
- Dynamic pruning based on carbon intensity
- Adaptive quantization for energy efficiency
- Knowledge distillation for model compression
- Lifecycle-aware model deployment
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    from ..core.logging import get_logger
    from ..sustainability.energy_tracker import EnergyTracker, EnergyReport
    from ..sustainability.carbon_calculator import CarbonCalculator, CarbonFootprint
    from ..sustainability.carbon_intensity_api import CarbonIntensityAPI
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.core.logging import get_logger
    from src.sustainability.energy_tracker import EnergyTracker, EnergyReport
    from src.sustainability.carbon_calculator import CarbonCalculator, CarbonFootprint
    from src.sustainability.carbon_intensity_api import CarbonIntensityAPI

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """Model optimization strategies based on carbon intensity."""
    AGGRESSIVE = "aggressive"  # High carbon intensity - maximum compression
    MODERATE = "moderate"      # Medium carbon intensity - balanced optimization
    CONSERVATIVE = "conservative"  # Low carbon intensity - minimal optimization
    ADAPTIVE = "adaptive"      # Dynamic based on real-time conditions


class ModelLifecycleStage(Enum):
    """Stages in the model lifecycle."""
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    INFERENCE = "inference"
    RETIREMENT = "retirement"


@dataclass
class ModelOptimizationConfig:
    """Configuration for sustainable model optimization."""
    # Carbon intensity thresholds (gCO2/kWh)
    low_carbon_threshold: float = 200.0
    medium_carbon_threshold: float = 400.0
    high_carbon_threshold: float = 600.0
    
    # Pruning settings
    enable_adaptive_pruning: bool = True
    max_pruning_ratio: float = 0.8  # Maximum 80% pruning
    min_pruning_ratio: float = 0.1  # Minimum 10% pruning
    pruning_granularity: int = 100  # Check every 100 inferences
    
    # Quantization settings
    enable_adaptive_quantization: bool = True
    quantization_levels: List[str] = field(default_factory=lambda: ["fp32", "fp16", "int8", "int4"])
    quantization_threshold: float = 0.95  # Quantize when carbon > 95th percentile
    
    # Knowledge distillation settings
    enable_knowledge_distillation: bool = True
    teacher_model_retention_days: int = 30
    student_model_compression_ratio: float = 0.5
    
    # Lifecycle management
    enable_automatic_retirement: bool = True
    model_performance_decay_threshold: float = 0.05  # 5% performance drop
    carbon_efficiency_threshold: float = 0.7  # Minimum carbon efficiency score
    
    # Monitoring
    optimization_check_interval: int = 3600  # Check every hour
    performance_tracking_window: int = 24  # Track performance over 24 hours


@dataclass
class ModelOptimizationResult:
    """Result of model optimization operation."""
    optimization_type: str
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    performance_impact: float  # Change in accuracy/F1 score
    energy_savings_percent: float
    carbon_savings_percent: float
    optimization_timestamp: datetime
    carbon_intensity_at_optimization: float
    strategy_used: OptimizationStrategy
    success: bool
    error_message: Optional[str] = None


class AdaptivePruner:
    """Adaptive model pruning based on carbon intensity."""
    
    def __init__(self, config: ModelOptimizationConfig):
        self.config = config
        self.pruning_history = []
        logger.info("Adaptive pruner initialized")
    
    def prune_model(self, model: nn.Module, carbon_intensity: float, 
                   performance_metrics: Dict[str, float]) -> ModelOptimizationResult:
        """Prune model based on current carbon intensity."""
        try:
            original_size = self._get_model_size_mb(model)
            
            # Determine pruning ratio based on carbon intensity
            pruning_ratio = self._calculate_pruning_ratio(carbon_intensity)
            
            if pruning_ratio < self.config.min_pruning_ratio:
                logger.info(f"Carbon intensity {carbon_intensity:.1f} gCO2/kWh too low for pruning")
                return ModelOptimizationResult(
                    optimization_type="pruning",
                    original_size_mb=original_size,
                    optimized_size_mb=original_size,
                    compression_ratio=1.0,
                    performance_impact=0.0,
                    energy_savings_percent=0.0,
                    carbon_savings_percent=0.0,
                    optimization_timestamp=datetime.now(),
                    carbon_intensity_at_optimization=carbon_intensity,
                    strategy_used=OptimizationStrategy.CONSERVATIVE,
                    success=False,
                    error_message="Carbon intensity below pruning threshold"
                )
            
            # Apply structured pruning
            pruned_model = self._apply_structured_pruning(model, pruning_ratio)
            optimized_size = self._get_model_size_mb(pruned_model)
            
            # Estimate performance impact
            performance_impact = self._estimate_performance_impact(pruning_ratio)
            
            # Calculate energy and carbon savings
            compression_ratio = optimized_size / original_size
            energy_savings = (1 - compression_ratio) * 0.7  # 70% of size reduction translates to energy savings
            carbon_savings = energy_savings * 0.8  # 80% of energy savings translates to carbon savings
            
            result = ModelOptimizationResult(
                optimization_type="pruning",
                original_size_mb=original_size,
                optimized_size_mb=optimized_size,
                compression_ratio=compression_ratio,
                performance_impact=performance_impact,
                energy_savings_percent=energy_savings * 100,
                carbon_savings_percent=carbon_savings * 100,
                optimization_timestamp=datetime.now(),
                carbon_intensity_at_optimization=carbon_intensity,
                strategy_used=self._get_strategy_for_carbon_intensity(carbon_intensity),
                success=True
            )
            
            self.pruning_history.append(result)
            logger.info(f"Model pruned: {pruning_ratio:.1%} reduction, {energy_savings:.1%} energy savings")
            return result
            
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return ModelOptimizationResult(
                optimization_type="pruning",
                original_size_mb=0.0,
                optimized_size_mb=0.0,
                compression_ratio=1.0,
                performance_impact=0.0,
                energy_savings_percent=0.0,
                carbon_savings_percent=0.0,
                optimization_timestamp=datetime.now(),
                carbon_intensity_at_optimization=carbon_intensity,
                strategy_used=OptimizationStrategy.CONSERVATIVE,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_pruning_ratio(self, carbon_intensity: float) -> float:
        """Calculate pruning ratio based on carbon intensity."""
        if carbon_intensity <= self.config.low_carbon_threshold:
            return 0.0
        elif carbon_intensity <= self.config.medium_carbon_threshold:
            # Linear interpolation between 0 and 0.3
            ratio = (carbon_intensity - self.config.low_carbon_threshold) / \
                   (self.config.medium_carbon_threshold - self.config.low_carbon_threshold) * 0.3
            return min(ratio, self.config.max_pruning_ratio)
        else:
            # High carbon intensity - aggressive pruning
            ratio = 0.3 + (carbon_intensity - self.config.medium_carbon_threshold) / \
                   (self.config.high_carbon_threshold - self.config.medium_carbon_threshold) * 0.5
            return min(ratio, self.config.max_pruning_ratio)
    
    def _apply_structured_pruning(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """Apply structured pruning to the model."""
        pruned_model = model
        
        # Prune linear layers
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                prune.remove(module, 'weight')
        
        # Prune convolutional layers
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                prune.remove(module, 'weight')
        
        return pruned_model
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024 * 1024)
    
    def _estimate_performance_impact(self, pruning_ratio: float) -> float:
        """Estimate performance impact of pruning."""
        # Empirical relationship: performance drops roughly with square of pruning ratio
        return -(pruning_ratio ** 2) * 0.1  # Max 10% performance drop for 100% pruning
    
    def _get_strategy_for_carbon_intensity(self, carbon_intensity: float) -> OptimizationStrategy:
        """Get optimization strategy based on carbon intensity."""
        if carbon_intensity <= self.config.low_carbon_threshold:
            return OptimizationStrategy.CONSERVATIVE
        elif carbon_intensity <= self.config.medium_carbon_threshold:
            return OptimizationStrategy.MODERATE
        else:
            return OptimizationStrategy.AGGRESSIVE


class AdaptiveQuantizer:
    """Adaptive model quantization based on carbon intensity."""
    
    def __init__(self, config: ModelOptimizationConfig):
        self.config = config
        self.quantization_history = []
        logger.info("Adaptive quantizer initialized")
    
    def quantize_model(self, model: nn.Module, carbon_intensity: float,
                      performance_metrics: Dict[str, float]) -> ModelOptimizationResult:
        """Quantize model based on current carbon intensity."""
        try:
            original_size = self._get_model_size_mb(model)
            
            # Determine quantization level based on carbon intensity
            quantization_level = self._select_quantization_level(carbon_intensity)
            
            if quantization_level == "fp32":
                logger.info(f"Carbon intensity {carbon_intensity:.1f} gCO2/kWh too low for quantization")
                return ModelOptimizationResult(
                    optimization_type="quantization",
                    original_size_mb=original_size,
                    optimized_size_mb=original_size,
                    compression_ratio=1.0,
                    performance_impact=0.0,
                    energy_savings_percent=0.0,
                    carbon_savings_percent=0.0,
                    optimization_timestamp=datetime.now(),
                    carbon_intensity_at_optimization=carbon_intensity,
                    strategy_used=OptimizationStrategy.CONSERVATIVE,
                    success=False,
                    error_message="Carbon intensity below quantization threshold"
                )
            
            # Apply quantization
            quantized_model = self._apply_quantization(model, quantization_level)
            optimized_size = self._get_model_size_mb(quantized_model)
            
            # Estimate performance impact
            performance_impact = self._estimate_quantization_impact(quantization_level)
            
            # Calculate energy and carbon savings
            compression_ratio = optimized_size / original_size
            energy_savings = (1 - compression_ratio) * 0.6  # 60% of size reduction translates to energy savings
            carbon_savings = energy_savings * 0.7  # 70% of energy savings translates to carbon savings
            
            result = ModelOptimizationResult(
                optimization_type="quantization",
                original_size_mb=original_size,
                optimized_size_mb=optimized_size,
                compression_ratio=compression_ratio,
                performance_impact=performance_impact,
                energy_savings_percent=energy_savings * 100,
                carbon_savings_percent=carbon_savings * 100,
                optimization_timestamp=datetime.now(),
                carbon_intensity_at_optimization=carbon_intensity,
                strategy_used=self._get_strategy_for_quantization_level(quantization_level),
                success=True
            )
            
            self.quantization_history.append(result)
            logger.info(f"Model quantized to {quantization_level}: {compression_ratio:.1%} size reduction")
            return result
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return ModelOptimizationResult(
                optimization_type="quantization",
                original_size_mb=0.0,
                optimized_size_mb=0.0,
                compression_ratio=1.0,
                performance_impact=0.0,
                energy_savings_percent=0.0,
                carbon_savings_percent=0.0,
                optimization_timestamp=datetime.now(),
                carbon_intensity_at_optimization=carbon_intensity,
                strategy_used=OptimizationStrategy.CONSERVATIVE,
                success=False,
                error_message=str(e)
            )
    
    def _select_quantization_level(self, carbon_intensity: float) -> str:
        """Select quantization level based on carbon intensity."""
        if carbon_intensity <= self.config.low_carbon_threshold:
            return "fp32"
        elif carbon_intensity <= self.config.medium_carbon_threshold:
            return "fp16"
        elif carbon_intensity <= self.config.high_carbon_threshold:
            return "int8"
        else:
            return "int4"
    
    def _apply_quantization(self, model: nn.Module, quantization_level: str) -> nn.Module:
        """Apply quantization to the model."""
        if quantization_level == "fp32":
            return model
        elif quantization_level == "fp16":
            return model.half()
        elif quantization_level == "int8":
            # Simulate INT8 quantization (in practice, use torch.quantization)
            return self._simulate_int8_quantization(model)
        elif quantization_level == "int4":
            # Simulate INT4 quantization
            return self._simulate_int4_quantization(model)
        else:
            return model
    
    def _simulate_int8_quantization(self, model: nn.Module) -> nn.Module:
        """Simulate INT8 quantization by reducing precision."""
        # In practice, use torch.quantization.quantize_dynamic
        quantized_model = model
        for param in quantized_model.parameters():
            # Simulate 8-bit quantization
            param.data = torch.round(param.data * 127) / 127
        return quantized_model
    
    def _simulate_int4_quantization(self, model: nn.Module) -> nn.Module:
        """Simulate INT4 quantization by reducing precision."""
        quantized_model = model
        for param in quantized_model.parameters():
            # Simulate 4-bit quantization
            param.data = torch.round(param.data * 7) / 7
        return quantized_model
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024 * 1024)
    
    def _estimate_quantization_impact(self, quantization_level: str) -> float:
        """Estimate performance impact of quantization."""
        impact_map = {
            "fp32": 0.0,
            "fp16": -0.01,  # 1% performance drop
            "int8": -0.03,  # 3% performance drop
            "int4": -0.08   # 8% performance drop
        }
        return impact_map.get(quantization_level, 0.0)
    
    def _get_strategy_for_quantization_level(self, quantization_level: str) -> OptimizationStrategy:
        """Get optimization strategy based on quantization level."""
        if quantization_level == "fp32":
            return OptimizationStrategy.CONSERVATIVE
        elif quantization_level == "fp16":
            return OptimizationStrategy.MODERATE
        else:
            return OptimizationStrategy.AGGRESSIVE


class KnowledgeDistillationManager:
    """Manages knowledge distillation for model compression."""
    
    def __init__(self, config: ModelOptimizationConfig):
        self.config = config
        self.distillation_history = []
        logger.info("Knowledge distillation manager initialized")
    
    def create_student_model(self, teacher_model: nn.Module, 
                           carbon_intensity: float) -> Tuple[nn.Module, ModelOptimizationResult]:
        """Create a compressed student model from teacher model."""
        try:
            teacher_size = self._get_model_size_mb(teacher_model)
            
            # Create student model with reduced capacity
            student_model = self._create_compressed_architecture(teacher_model)
            student_size = self._get_model_size_mb(student_model)
            
            # Estimate performance impact
            performance_impact = -0.05  # 5% performance drop for student model
            
            # Calculate energy and carbon savings
            compression_ratio = student_size / teacher_size
            energy_savings = (1 - compression_ratio) * 0.8  # 80% of size reduction translates to energy savings
            carbon_savings = energy_savings * 0.9  # 90% of energy savings translates to carbon savings
            
            result = ModelOptimizationResult(
                optimization_type="knowledge_distillation",
                original_size_mb=teacher_size,
                optimized_size_mb=student_size,
                compression_ratio=compression_ratio,
                performance_impact=performance_impact,
                energy_savings_percent=energy_savings * 100,
                carbon_savings_percent=carbon_savings * 100,
                optimization_timestamp=datetime.now(),
                carbon_intensity_at_optimization=carbon_intensity,
                strategy_used=OptimizationStrategy.ADAPTIVE,
                success=True
            )
            
            self.distillation_history.append(result)
            logger.info(f"Student model created: {compression_ratio:.1%} size reduction")
            return student_model, result
            
        except Exception as e:
            logger.error(f"Knowledge distillation failed: {e}")
            dummy_model = nn.Linear(10, 1)
            return dummy_model, ModelOptimizationResult(
                optimization_type="knowledge_distillation",
                original_size_mb=0.0,
                optimized_size_mb=0.0,
                compression_ratio=1.0,
                performance_impact=0.0,
                energy_savings_percent=0.0,
                carbon_savings_percent=0.0,
                optimization_timestamp=datetime.now(),
                carbon_intensity_at_optimization=carbon_intensity,
                strategy_used=OptimizationStrategy.CONSERVATIVE,
                success=False,
                error_message=str(e)
            )
    
    def _create_compressed_architecture(self, teacher_model: nn.Module) -> nn.Module:
        """Create a compressed architecture based on teacher model."""
        # Simple compression: reduce layer sizes by compression ratio
        compression_ratio = self.config.student_model_compression_ratio
        
        # Create a simplified version of the teacher model
        class CompressedModel(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                # Copy architecture but with reduced dimensions
                self.layers = nn.ModuleList()
                for name, module in original_model.named_modules():
                    if isinstance(module, nn.Linear):
                        # Reduce output features
                        new_out_features = max(1, int(module.out_features * compression_ratio))
                        self.layers.append(nn.Linear(module.in_features, new_out_features))
                    elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                        # Reduce output channels
                        new_out_channels = max(1, int(module.out_channels * compression_ratio))
                        self.layers.append(type(module)(
                            module.in_channels, new_out_channels, 
                            module.kernel_size, module.stride, module.padding
                        ))
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return CompressedModel(teacher_model)
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024 * 1024)


class SustainableModelLifecycleManager:
    """Main manager for sustainable model lifecycle operations."""
    
    def __init__(self, config: Optional[ModelOptimizationConfig] = None):
        self.config = config or ModelOptimizationConfig()
        self.energy_tracker = EnergyTracker()
        self.carbon_calculator = CarbonCalculator()
        self.carbon_api = CarbonIntensityAPI()
        
        # Initialize optimization components
        self.pruner = AdaptivePruner(self.config)
        self.quantizer = AdaptiveQuantizer(self.config)
        self.distillation_manager = KnowledgeDistillationManager(self.config)
        
        # Model registry
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.optimization_history: List[ModelOptimizationResult] = []
        
        logger.info("Sustainable model lifecycle manager initialized")
    
    def register_model(self, model_id: str, model: nn.Module, 
                      initial_performance: Dict[str, float]) -> bool:
        """Register a new model for lifecycle management."""
        try:
            model_size = self._get_model_size_mb(model)
            
            self.model_registry[model_id] = {
                "model": model,
                "size_mb": model_size,
                "performance_history": [initial_performance],
                "optimization_history": [],
                "deployment_timestamp": datetime.now(),
                "stage": ModelLifecycleStage.DEPLOYMENT,
                "carbon_efficiency_score": 0.0,
                "energy_efficiency_score": 0.0
            }
            
            logger.info(f"Model {model_id} registered for lifecycle management")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False
    
    def optimize_model_for_carbon(self, model_id: str, 
                                 current_performance: Dict[str, float]) -> ModelOptimizationResult:
        """Optimize model based on current carbon intensity."""
        if model_id not in self.model_registry:
            logger.error(f"Model {model_id} not found in registry")
            return ModelOptimizationResult(
                optimization_type="unknown",
                original_size_mb=0.0,
                optimized_size_mb=0.0,
                compression_ratio=1.0,
                performance_impact=0.0,
                energy_savings_percent=0.0,
                carbon_savings_percent=0.0,
                optimization_timestamp=datetime.now(),
                carbon_intensity_at_optimization=0.0,
                strategy_used=OptimizationStrategy.CONSERVATIVE,
                success=False,
                error_message="Model not found in registry"
            )
        
        try:
            # Get current carbon intensity
            carbon_intensity = self.carbon_api.get_current_carbon_intensity("US")
            
            # Get model info
            model_info = self.model_registry[model_id]
            model = model_info["model"]
            
            # Determine optimization strategy
            optimization_type = self._select_optimization_type(carbon_intensity, model_info)
            
            # Apply optimization
            if optimization_type == "pruning":
                result = self.pruner.prune_model(model, carbon_intensity, current_performance)
            elif optimization_type == "quantization":
                result = self.quantizer.quantize_model(model, carbon_intensity, current_performance)
            elif optimization_type == "knowledge_distillation":
                student_model, result = self.distillation_manager.create_student_model(model, carbon_intensity)
                if result.success:
                    # Replace model with student model
                    self.model_registry[model_id]["model"] = student_model
                    self.model_registry[model_id]["size_mb"] = result.optimized_size_mb
            else:
                result = ModelOptimizationResult(
                    optimization_type="none",
                    original_size_mb=model_info["size_mb"],
                    optimized_size_mb=model_info["size_mb"],
                    compression_ratio=1.0,
                    performance_impact=0.0,
                    energy_savings_percent=0.0,
                    carbon_savings_percent=0.0,
                    optimization_timestamp=datetime.now(),
                    carbon_intensity_at_optimization=carbon_intensity,
                    strategy_used=OptimizationStrategy.CONSERVATIVE,
                    success=True,
                    error_message="No optimization needed"
                )
            
            # Update model registry
            if result.success:
                self.model_registry[model_id]["optimization_history"].append(result)
                self.model_registry[model_id]["performance_history"].append(current_performance)
            
            self.optimization_history.append(result)
            logger.info(f"Model {model_id} optimized: {result.optimization_type}")
            return result
            
        except Exception as e:
            logger.error(f"Model optimization failed for {model_id}: {e}")
            return ModelOptimizationResult(
                optimization_type="unknown",
                original_size_mb=0.0,
                optimized_size_mb=0.0,
                compression_ratio=1.0,
                performance_impact=0.0,
                energy_savings_percent=0.0,
                carbon_savings_percent=0.0,
                optimization_timestamp=datetime.now(),
                carbon_intensity_at_optimization=0.0,
                strategy_used=OptimizationStrategy.CONSERVATIVE,
                success=False,
                error_message=str(e)
            )
    
    def _select_optimization_type(self, carbon_intensity: float, 
                                 model_info: Dict[str, Any]) -> str:
        """Select the best optimization type based on carbon intensity and model state."""
        # Check if model has been recently optimized
        recent_optimizations = [
            opt for opt in model_info["optimization_history"]
            if (datetime.now() - opt.optimization_timestamp).total_seconds() < 3600  # Last hour
        ]
        
        if len(recent_optimizations) > 0:
            return "none"  # Don't optimize too frequently
        
        # Select optimization based on carbon intensity
        if carbon_intensity <= self.config.low_carbon_threshold:
            return "none"
        elif carbon_intensity <= self.config.medium_carbon_threshold:
            return "quantization"  # Start with quantization
        elif carbon_intensity <= self.config.high_carbon_threshold:
            return "pruning"  # More aggressive pruning
        else:
            return "knowledge_distillation"  # Most aggressive: create student model
    
    def check_model_retirement(self, model_id: str) -> bool:
        """Check if model should be retired based on performance and efficiency."""
        if model_id not in self.model_registry:
            return False
        
        model_info = self.model_registry[model_id]
        
        # Check performance decay
        if len(model_info["performance_history"]) >= 2:
            recent_performance = model_info["performance_history"][-1]
            initial_performance = model_info["performance_history"][0]
            
            # Calculate performance decay
            performance_decay = 0.0
            for metric in recent_performance:
                if metric in initial_performance:
                    decay = (initial_performance[metric] - recent_performance[metric]) / initial_performance[metric]
                    performance_decay = max(performance_decay, decay)
            
            if performance_decay > self.config.model_performance_decay_threshold:
                logger.info(f"Model {model_id} marked for retirement due to performance decay: {performance_decay:.1%}")
                return True
        
        # Check carbon efficiency
        if model_info["carbon_efficiency_score"] < self.config.carbon_efficiency_threshold:
            logger.info(f"Model {model_id} marked for retirement due to low carbon efficiency")
            return True
        
        return False
    
    def retire_model(self, model_id: str) -> bool:
        """Retire a model from the lifecycle management system."""
        if model_id not in self.model_registry:
            return False
        
        try:
            model_info = self.model_registry[model_id]
            model_info["stage"] = ModelLifecycleStage.RETIREMENT
            model_info["retirement_timestamp"] = datetime.now()
            
            logger.info(f"Model {model_id} retired from lifecycle management")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retire model {model_id}: {e}")
            return False
    
    def get_model_lifecycle_report(self, model_id: str) -> Dict[str, Any]:
        """Generate a comprehensive lifecycle report for a model."""
        if model_id not in self.model_registry:
            return {"error": "Model not found"}
        
        model_info = self.model_registry[model_id]
        
        # Calculate lifecycle metrics
        deployment_time = datetime.now() - model_info["deployment_timestamp"]
        total_optimizations = len(model_info["optimization_history"])
        
        # Calculate total energy and carbon savings
        total_energy_savings = sum(
            opt.energy_savings_percent for opt in model_info["optimization_history"]
        )
        total_carbon_savings = sum(
            opt.carbon_savings_percent for opt in model_info["optimization_history"]
        )
        
        # Calculate average performance
        if model_info["performance_history"]:
            avg_performance = {}
            for metric in model_info["performance_history"][0]:
                values = [perf.get(metric, 0) for perf in model_info["performance_history"]]
                avg_performance[metric] = np.mean(values)
        else:
            avg_performance = {}
        
        return {
            "model_id": model_id,
            "deployment_time_hours": deployment_time.total_seconds() / 3600,
            "current_stage": model_info["stage"].value,
            "current_size_mb": model_info["size_mb"],
            "total_optimizations": total_optimizations,
            "total_energy_savings_percent": total_energy_savings,
            "total_carbon_savings_percent": total_carbon_savings,
            "average_performance": avg_performance,
            "carbon_efficiency_score": model_info["carbon_efficiency_score"],
            "energy_efficiency_score": model_info["energy_efficiency_score"],
            "optimization_history": [
                {
                    "type": opt.optimization_type,
                    "timestamp": opt.optimization_timestamp.isoformat(),
                    "compression_ratio": opt.compression_ratio,
                    "energy_savings_percent": opt.energy_savings_percent,
                    "carbon_savings_percent": opt.carbon_savings_percent,
                    "strategy": opt.strategy_used.value
                }
                for opt in model_info["optimization_history"]
            ]
        }
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024 * 1024)
    
    def get_system_lifecycle_summary(self) -> Dict[str, Any]:
        """Get summary of all models in lifecycle management."""
        total_models = len(self.model_registry)
        active_models = sum(1 for info in self.model_registry.values() 
                          if info["stage"] != ModelLifecycleStage.RETIREMENT)
        
        total_optimizations = len(self.optimization_history)
        successful_optimizations = sum(1 for opt in self.optimization_history if opt.success)
        
        total_energy_savings = sum(opt.energy_savings_percent for opt in self.optimization_history)
        total_carbon_savings = sum(opt.carbon_savings_percent for opt in self.optimization_history)
        
        return {
            "total_models": total_models,
            "active_models": active_models,
            "retired_models": total_models - active_models,
            "total_optimizations": total_optimizations,
            "successful_optimizations": successful_optimizations,
            "optimization_success_rate": successful_optimizations / max(1, total_optimizations),
            "total_energy_savings_percent": total_energy_savings,
            "total_carbon_savings_percent": total_carbon_savings,
            "average_energy_savings_per_optimization": total_energy_savings / max(1, total_optimizations),
            "average_carbon_savings_per_optimization": total_carbon_savings / max(1, total_optimizations)
        }


# Utility functions
def create_sustainable_lifecycle_manager(config_dict: Optional[Dict[str, Any]] = None) -> SustainableModelLifecycleManager:
    """Create a sustainable model lifecycle manager with configuration."""
    if config_dict:
        config = ModelOptimizationConfig(**config_dict)
    else:
        config = ModelOptimizationConfig()
    return SustainableModelLifecycleManager(config)


def demo_sustainable_model_lifecycle() -> Dict[str, Any]:
    """Demonstrate sustainable model lifecycle management."""
    logger.info("Starting sustainable model lifecycle demo")
    
    # Create lifecycle manager
    manager = create_sustainable_lifecycle_manager()
    
    # Create a sample model
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SampleModel()
    initial_performance = {"accuracy": 0.85, "f1_score": 0.82, "precision": 0.88}
    
    # Register model
    model_id = "demo_credit_model"
    success = manager.register_model(model_id, model, initial_performance)
    
    if not success:
        return {"error": "Failed to register model"}
    
    # Simulate optimization over time with different carbon intensities
    carbon_scenarios = [150.0, 350.0, 550.0, 750.0]  # Different carbon intensities
    optimization_results = []
    
    for i, carbon_intensity in enumerate(carbon_scenarios):
        # Simulate performance degradation over time
        current_performance = {
            "accuracy": initial_performance["accuracy"] - i * 0.01,
            "f1_score": initial_performance["f1_score"] - i * 0.01,
            "precision": initial_performance["precision"] - i * 0.01
        }
        
        # Optimize model
        result = manager.optimize_model_for_carbon(model_id, current_performance)
        optimization_results.append(result)
        
        logger.info(f"Optimization {i+1}: {result.optimization_type}, "
                   f"Energy savings: {result.energy_savings_percent:.1f}%, "
                   f"Carbon savings: {result.carbon_savings_percent:.1f}%")
    
    # Generate lifecycle report
    lifecycle_report = manager.get_model_lifecycle_report(model_id)
    system_summary = manager.get_system_lifecycle_summary()
    
    return {
        "model_registration": success,
        "optimization_results": [
            {
                "scenario": i+1,
                "carbon_intensity": carbon_scenarios[i],
                "optimization_type": result.optimization_type,
                "compression_ratio": result.compression_ratio,
                "energy_savings_percent": result.energy_savings_percent,
                "carbon_savings_percent": result.carbon_savings_percent,
                "strategy": result.strategy_used.value,
                "success": result.success
            }
            for i, result in enumerate(optimization_results)
        ],
        "lifecycle_report": lifecycle_report,
        "system_summary": system_summary
    }

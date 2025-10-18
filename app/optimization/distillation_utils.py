"""
Utility functions for knowledge distillation.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import copy

from .knowledge_distillation import (
    KnowledgeDistiller, DistillationConfig, DistillationResult
)

# Utility functions
def distill_knowledge(teacher_model: nn.Module, student_model: nn.Module,
                     X: pd.DataFrame, y: pd.Series, 
                     config: Optional[DistillationConfig] = None) -> DistillationResult:
    """
    Convenience function to perform knowledge distillation.
    
    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        X: Training features
        y: Training targets
        config: Distillation configuration
    
    Returns:
        DistillationResult with distillation results
    """
    distiller = KnowledgeDistiller(config)
    return distiller.distill_knowledge(teacher_model, student_model, X, y)


def get_default_distillation_config() -> DistillationConfig:
    """Get default distillation configuration."""
    return DistillationConfig()


def get_high_temperature_config(temperature: float = 8.0) -> DistillationConfig:
    """Get high temperature distillation configuration for better knowledge transfer."""
    return DistillationConfig(
        temperature=temperature,
        distillation_loss_weight=0.8,
        student_loss_weight=0.2,
        epochs=50,
        learning_rate=0.001,
        use_scheduler=True
    )


def get_progressive_distillation_config() -> DistillationConfig:
    """Get progressive distillation configuration."""
    return DistillationConfig(
        temperature=6.0,
        use_progressive_distillation=True,
        progressive_stages=[8.0, 6.0, 4.0, 2.0],
        epochs=80,
        distillation_loss_weight=0.7,
        student_loss_weight=0.3
    )


def get_feature_matching_config() -> DistillationConfig:
    """Get distillation configuration with feature matching."""
    return DistillationConfig(
        temperature=4.0,
        use_feature_matching=True,
        feature_matching_weight=0.1,
        distillation_loss_weight=0.6,
        student_loss_weight=0.3,
        epochs=60
    )


def create_student_model(teacher_model: nn.Module, compression_ratio: float = 0.5) -> nn.Module:
    """
    Create a smaller student model based on teacher architecture.
    
    Args:
        teacher_model: Teacher model to base student on
        compression_ratio: Ratio of student to teacher parameters
    
    Returns:
        Student model with reduced capacity
    """
    # This is a simplified implementation for sequential models
    if isinstance(teacher_model, nn.Sequential):
        student_layers = []
        
        for layer in teacher_model:
            if isinstance(layer, nn.Linear):
                # Reduce linear layer size
                in_features = layer.in_features
                out_features = max(1, int(layer.out_features * compression_ratio))
                student_layers.append(nn.Linear(in_features, out_features))
                
                # Update input size for next layer
                if student_layers and isinstance(student_layers[-2], nn.Linear):
                    student_layers[-2] = nn.Linear(
                        student_layers[-2].in_features,
                        out_features
                    )
            else:
                # Keep other layers as is (activations, dropout, etc.)
                student_layers.append(copy.deepcopy(layer))
        
        return nn.Sequential(*student_layers)
    
    else:
        # For custom models, return a simplified version
        # This would need to be customized based on the specific model architecture
        logger.warning("Custom student model creation not implemented for this architecture")
        return copy.deepcopy(teacher_model)


def analyze_distillation_impact(teacher_model: nn.Module, student_model: nn.Module,
                               X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Analyze the impact of knowledge distillation.
    
    Args:
        teacher_model: Teacher model
        student_model: Distilled student model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary with analysis results
    """
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Evaluate both models
    teacher_perf = _evaluate_model_simple(teacher_model, X_test_tensor, y_test_tensor)
    student_perf = _evaluate_model_simple(student_model, X_test_tensor, y_test_tensor)
    
    # Calculate model statistics
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    teacher_size = sum(p.numel() * p.element_size() for p in teacher_model.parameters()) / (1024 * 1024)
    student_size = sum(p.numel() * p.element_size() for p in student_model.parameters()) / (1024 * 1024)
    
    # Compression metrics
    compression_ratio = teacher_params / student_params if student_params > 0 else 1.0
    parameter_reduction = 1.0 - (student_params / teacher_params) if teacher_params > 0 else 0.0
    size_reduction = 1.0 - (student_size / teacher_size) if teacher_size > 0 else 0.0
    
    # Performance impact
    performance_drop = {
        key: teacher_perf[key] - student_perf.get(key, 0.0)
        for key in teacher_perf.keys()
    }
    
    # Knowledge transfer efficiency
    knowledge_retention = {
        key: (student_perf.get(key, 0.0) / teacher_perf[key]) if teacher_perf[key] > 0 else 0.0
        for key in teacher_perf.keys()
    }
    
    analysis = {
        'compression_metrics': {
            'compression_ratio': compression_ratio,
            'parameter_reduction': parameter_reduction,
            'size_reduction': size_reduction,
            'teacher_params': teacher_params,
            'student_params': student_params,
            'teacher_size_mb': teacher_size,
            'student_size_mb': student_size
        },
        'performance_metrics': {
            'teacher_performance': teacher_perf,
            'student_performance': student_perf,
            'performance_drop': performance_drop,
            'knowledge_retention': knowledge_retention
        },
        'distillation_efficiency': {
            'auc_retention': knowledge_retention.get('roc_auc', 0.0),
            'f1_retention': knowledge_retention.get('f1_score', 0.0),
            'compression_vs_performance': compression_ratio / (1.0 + performance_drop.get('roc_auc', 0.0))
        }
    }
    
    return analysis


def compare_distillation_temperatures(teacher_model: nn.Module, student_model: nn.Module,
                                    X: pd.DataFrame, y: pd.Series,
                                    temperatures: List[float] = [2.0, 4.0, 6.0, 8.0]) -> Dict[float, DistillationResult]:
    """
    Compare distillation performance across different temperatures.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model architecture
        X: Training features
        y: Training targets
        temperatures: List of temperatures to test
    
    Returns:
        Dictionary mapping temperatures to results
    """
    results = {}
    
    for temp in temperatures:
        logger.info(f"Testing temperature: {temp}")
        
        config = DistillationConfig(
            temperature=temp,
            epochs=20,  # Shorter for comparison
            early_stopping_patience=5
        )
        
        # Use a fresh copy of student model for each experiment
        student_copy = copy.deepcopy(student_model)
        result = distill_knowledge(teacher_model, student_copy, X, y, config)
        results[temp] = result
    
    return results


def validate_distilled_model(teacher_model: nn.Module, student_model: nn.Module,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           threshold: float = 0.05) -> Dict[str, Any]:
    """
    Validate that distilled model maintains acceptable performance.
    
    Args:
        teacher_model: Teacher model
        student_model: Distilled student model
        X_test: Test features
        y_test: Test targets
        threshold: Maximum acceptable performance drop
    
    Returns:
        Validation results
    """
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Evaluate both models
    teacher_perf = _evaluate_model_simple(teacher_model, X_test_tensor, y_test_tensor)
    student_perf = _evaluate_model_simple(student_model, X_test_tensor, y_test_tensor)
    
    # Calculate drops
    accuracy_drop = teacher_perf['accuracy'] - student_perf['accuracy']
    auc_drop = teacher_perf['roc_auc'] - student_perf['roc_auc']
    f1_drop = teacher_perf['f1_score'] - student_perf['f1_score']
    
    # Validation
    accuracy_valid = accuracy_drop <= threshold
    auc_valid = auc_drop <= threshold
    f1_valid = f1_drop <= threshold
    
    overall_valid = accuracy_valid and auc_valid and f1_valid
    
    return {
        'validation_passed': overall_valid,
        'accuracy_drop': accuracy_drop,
        'auc_drop': auc_drop,
        'f1_drop': f1_drop,
        'accuracy_valid': accuracy_valid,
        'auc_valid': auc_valid,
        'f1_valid': f1_valid,
        'threshold': threshold,
        'teacher_performance': teacher_perf,
        'student_performance': student_perf
    }


def load_distilled_model(model_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a distilled model from disk.
    
    Args:
        model_path: Path to saved distilled model
    
    Returns:
        Tuple of (loaded_model, metadata)
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    metadata = {
        'config': checkpoint.get('config'),
        'distillation_method': checkpoint.get('distillation_method'),
        'temperature': checkpoint.get('temperature'),
        'saved_at': checkpoint.get('saved_at')
    }
    
    print(f"Loaded distilled model with temperature: {metadata['temperature']}")
    
    # Note: This is a placeholder - actual implementation would need to reconstruct
    # the student model architecture based on the saved configuration
    return None, metadata


def create_ensemble_student(teacher_models: List[nn.Module], student_architecture: nn.Module,
                          X: pd.DataFrame, y: pd.Series,
                          config: Optional[DistillationConfig] = None) -> DistillationResult:
    """
    Distill knowledge from multiple teacher models into a single student.
    
    Args:
        teacher_models: List of teacher models
        student_architecture: Student model architecture
        X: Training features
        y: Training targets
        config: Distillation configuration
    
    Returns:
        DistillationResult with ensemble distillation results
    """
    # This is a simplified implementation
    # In practice, you'd need to handle ensemble predictions properly
    
    if not teacher_models:
        raise ValueError("At least one teacher model is required")
    
    # For simplicity, use the first teacher model
    # In a full implementation, you'd ensemble the teacher predictions
    primary_teacher = teacher_models[0]
    
    logger.info(f"Ensemble distillation with {len(teacher_models)} teachers (simplified to use first teacher)")
    
    return distill_knowledge(primary_teacher, student_architecture, X, y, config)


# Helper functions
def _evaluate_model_simple(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """Simple model evaluation."""
    try:
        model.eval()
        
        with torch.no_grad():
            outputs = model(X)
            probs = torch.sigmoid(outputs.squeeze()).numpy()
            preds = (probs > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        
        return {
            'accuracy': accuracy_score(y.numpy(), preds),
            'roc_auc': roc_auc_score(y.numpy(), probs),
            'f1_score': f1_score(y.numpy(), preds, average='weighted')
        }
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {'accuracy': 0.0, 'roc_auc': 0.0, 'f1_score': 0.0}


# Import logger for utility functions
try:
    from ..core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
"""Optimization package for hyperparameter tuning and model optimization."""

from .hyperparameter_tuning import (
    HyperparameterOptimizer,
    OptimizationConfig,
    OptimizationResult,
    optimize_all_models,
    get_fast_optimization_config,
    get_production_optimization_config
)

from .model_pruning import (
    ModelPruner,
    PruningConfig,
    PruningResult,
    MagnitudePruner,
    StructuredPruner,
    GradualPruner,
    prune_model,
    get_default_pruning_config,
    get_magnitude_pruning_config,
    get_structured_pruning_config,
    get_gradual_pruning_config,
    analyze_pruning_impact,
    compare_pruning_methods
)

from .model_quantization import (
    ModelQuantizer,
    QuantizationConfig,
    QuantizationResult,
    QATQuantizer,
    PostTrainingStaticQuantizer,
    PostTrainingDynamicQuantizer,
    QuantizableModel
)

from .quantization_utils import (
    quantize_model,
    get_default_quantization_config,
    get_qat_config,
    get_static_quantization_config,
    get_dynamic_quantization_config,
    get_mobile_quantization_config,
    analyze_quantization_impact,
    compare_quantization_methods,
    benchmark_quantized_model,
    validate_quantized_model_accuracy
)

from .knowledge_distillation import (
    KnowledgeDistiller,
    DistillationConfig,
    DistillationResult,
    DistillationLoss,
    FeatureMatchingLoss,
    AttentionTransferLoss
)

from .distillation_utils import (
    distill_knowledge,
    get_default_distillation_config,
    get_high_temperature_config,
    get_progressive_distillation_config,
    get_feature_matching_config,
    create_student_model,
    analyze_distillation_impact,
    compare_distillation_temperatures,
    validate_distilled_model,
    create_ensemble_student
)

__all__ = [
    'HyperparameterOptimizer',
    'OptimizationConfig', 
    'OptimizationResult',
    'optimize_all_models',
    'get_fast_optimization_config',
    'get_production_optimization_config',
    'ModelPruner',
    'PruningConfig',
    'PruningResult',
    'MagnitudePruner',
    'StructuredPruner',
    'GradualPruner',
    'prune_model',
    'get_default_pruning_config',
    'get_magnitude_pruning_config',
    'get_structured_pruning_config',
    'get_gradual_pruning_config',
    'analyze_pruning_impact',
    'compare_pruning_methods',
    'ModelQuantizer',
    'QuantizationConfig',
    'QuantizationResult',
    'QATQuantizer',
    'PostTrainingStaticQuantizer',
    'PostTrainingDynamicQuantizer',
    'QuantizableModel',
    'quantize_model',
    'get_default_quantization_config',
    'get_qat_config',
    'get_static_quantization_config',
    'get_dynamic_quantization_config',
    'get_mobile_quantization_config',
    'analyze_quantization_impact',
    'compare_quantization_methods',
    'benchmark_quantized_model',
    'validate_quantized_model_accuracy',
    'KnowledgeDistiller',
    'DistillationConfig',
    'DistillationResult',
    'DistillationLoss',
    'FeatureMatchingLoss',
    'AttentionTransferLoss',
    'distill_knowledge',
    'get_default_distillation_config',
    'get_high_temperature_config',
    'get_progressive_distillation_config',
    'get_feature_matching_config',
    'create_student_model',
    'analyze_distillation_impact',
    'compare_distillation_temperatures',
    'validate_distilled_model',
    'create_ensemble_student'
]
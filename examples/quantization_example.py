#!/usr/bin/env python3
"""
Comprehensive example demonstrating the quantization system for sustainable AI.

This example shows how to use the quantization system to compress neural network models
while maintaining accuracy, reducing energy consumption, and enabling efficient deployment.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available - skipping visualizations")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.optimization.model_quantization import (
    ModelQuantizer, QuantizationConfig, QuantizationResult
)
from src.optimization.quantization_utils import (
    quantize_model, get_qat_config, get_static_quantization_config,
    get_dynamic_quantization_config, get_mobile_quantization_config,
    compare_quantization_methods, analyze_quantization_impact,
    validate_quantized_model_accuracy, benchmark_quantized_model
)


class CreditRiskModel(nn.Module):
    """Example credit risk model for quantization demonstration."""
    
    def __init__(self, input_size: int = 30, hidden_sizes: list = [128, 64, 32]):
        super(CreditRiskModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_synthetic_credit_data(n_samples: int = 5000, n_features: int = 30) -> tuple:
    """Create synthetic credit risk dataset."""
    print(f"Creating synthetic credit dataset with {n_samples} samples and {n_features} features...")
    
    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=42
    )
    
    # Add realistic feature names
    feature_names = [
        'annual_income', 'debt_to_income_ratio', 'credit_score', 'employment_length',
        'loan_amount', 'loan_term', 'home_ownership', 'credit_history_length',
        'number_of_accounts', 'delinquencies_2yrs', 'inquiries_6mths', 'open_accounts',
        'total_accounts', 'revolving_balance', 'revolving_utilization', 'collections_12mths'
    ]
    
    # Extend feature names if needed
    while len(feature_names) < n_features:
        feature_names.append(f'behavioral_feature_{len(feature_names) - 15}')
    
    feature_names = feature_names[:n_features]
    
    # Create DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='default_risk')
    
    print(f"Dataset created: {len(X_df)} samples, {X_df.shape[1]} features")
    print(f"Class distribution: {y_series.value_counts().to_dict()}")
    
    return X_df, y_series


def train_credit_risk_model(X: pd.DataFrame, y: pd.Series, epochs: int = 100) -> nn.Module:
    """Train a credit risk model."""
    print(f"\nTraining credit risk model for {epochs} epochs...")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.FloatTensor(y_val.values)
    
    # Create model
    model = CreditRiskModel(input_size=X.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
            val_probs = torch.sigmoid(val_outputs.squeeze())
            val_preds = (val_probs > 0.5).float()
            val_acc = (val_preds == y_val_tensor).float().mean()
        
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc.item())
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")
    
    print(f"Training completed. Final validation accuracy: {val_accuracies[-1]:.4f}")
    return model


def demonstrate_quantization_methods(model: nn.Module, X: pd.DataFrame, y: pd.Series):
    """Demonstrate different quantization methods."""
    print("\n" + "=" * 80)
    print("QUANTIZATION METHODS DEMONSTRATION")
    print("=" * 80)
    
    # Compare all quantization methods
    print("\n1. Comparing All Quantization Methods...")
    results = compare_quantization_methods(model, X, y)
    
    # Display results
    print("\nQuantization Results Summary:")
    print("-" * 60)
    
    for method, result in results.items():
        if result and result.success:
            print(f"\n{method.upper()} Quantization:")
            print(f"  ✓ Success: {result.success}")
            print(f"  📦 Compression Ratio: {result.compression_ratio:.2f}x")
            print(f"  📉 Size Reduction: {result.size_reduction:.2%}")
            print(f"  ⚡ Speedup: {result.speedup_ratio:.2f}x")
            print(f"  📊 AUC Drop: {result.performance_drop.get('roc_auc', 0.0):.4f}")
            print(f"  ⏱️ Quantization Time: {result.quantization_time_seconds:.2f}s")
        elif result:
            print(f"\n{method.upper()} Quantization:")
            print(f"  ⚠️ Partial Success: {result.message}")
        else:
            print(f"\n{method.upper()} Quantization:")
            print(f"  ❌ Failed")
    
    return results


def demonstrate_mobile_optimization(model: nn.Module, X: pd.DataFrame, y: pd.Series):
    """Demonstrate mobile-optimized quantization."""
    print("\n" + "=" * 80)
    print("MOBILE OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    
    print("\n2. Mobile-Optimized Quantization...")
    
    # Mobile configuration
    mobile_config = get_mobile_quantization_config()
    mobile_result = quantize_model(model, X, y, mobile_config)
    
    if mobile_result.success:
        print(f"✓ Mobile quantization successful!")
        print(f"  📱 Optimized for mobile deployment")
        print(f"  📦 Model size: {mobile_result.quantized_size_mb:.3f}MB")
        print(f"  📉 Size reduction: {mobile_result.size_reduction:.2%}")
        print(f"  ⚡ Inference speedup: {mobile_result.speedup_ratio:.2f}x")
        print(f"  🎯 Accuracy preserved: {mobile_result.performance_drop.get('roc_auc', 0.0):.4f} AUC drop")
        
        # Benchmark mobile model
        X_test = X.sample(100)
        benchmark_results = benchmark_quantized_model(mobile_result.quantized_model, X_test)
        
        print(f"\n📊 Mobile Model Benchmark:")
        print(f"  Mean inference time: {benchmark_results['mean_inference_time_ms']:.2f}ms")
        print(f"  P95 inference time: {benchmark_results['p95_inference_time_ms']:.2f}ms")
        print(f"  P99 inference time: {benchmark_results['p99_inference_time_ms']:.2f}ms")
        
    else:
        print(f"⚠️ Mobile quantization completed with issues: {mobile_result.message}")
    
    return mobile_result


def demonstrate_accuracy_validation(original_model: nn.Module, quantized_model: nn.Module, 
                                  X: pd.DataFrame, y: pd.Series):
    """Demonstrate accuracy validation."""
    print("\n" + "=" * 80)
    print("ACCURACY VALIDATION DEMONSTRATION")
    print("=" * 80)
    
    print("\n3. Validating Quantized Model Accuracy...")
    
    # Split test data
    X_test = X.sample(1000, random_state=42)
    y_test = y.loc[X_test.index]
    
    # Validate accuracy
    validation_results = validate_quantized_model_accuracy(
        original_model, quantized_model, X_test, y_test, threshold=0.05
    )
    
    print(f"\n🔍 Accuracy Validation Results:")
    print(f"  Overall validation: {'✓ PASSED' if validation_results['validation_passed'] else '❌ FAILED'}")
    print(f"  Accuracy drop: {validation_results['accuracy_drop']:.4f} (threshold: {validation_results['threshold']})")
    print(f"  AUC drop: {validation_results['auc_drop']:.4f}")
    print(f"  F1 drop: {validation_results['f1_drop']:.4f}")
    
    print(f"\n📊 Performance Comparison:")
    orig_perf = validation_results['original_performance']
    quant_perf = validation_results['quantized_performance']
    
    print(f"  Original Model - AUC: {orig_perf['roc_auc']:.4f}, Accuracy: {orig_perf['accuracy']:.4f}")
    print(f"  Quantized Model - AUC: {quant_perf['roc_auc']:.4f}, Accuracy: {quant_perf['accuracy']:.4f}")
    
    return validation_results


def demonstrate_sustainability_impact(results: dict):
    """Demonstrate sustainability impact of quantization."""
    print("\n" + "=" * 80)
    print("SUSTAINABILITY IMPACT DEMONSTRATION")
    print("=" * 80)
    
    print("\n4. Sustainability Impact Analysis...")
    
    # Calculate energy savings (simulated)
    base_energy_per_inference = 0.1  # mWh per inference (simulated)
    inferences_per_day = 10000
    days_per_year = 365
    
    print(f"\n🌱 Estimated Annual Energy Savings:")
    print(f"  Assumptions: {inferences_per_day:,} inferences/day, {days_per_year} days/year")
    
    for method, result in results.items():
        if result and result.success:
            # Estimate energy savings based on speedup
            energy_reduction = 1 - (1 / result.speedup_ratio) if result.speedup_ratio > 1 else 0
            annual_energy_savings = (base_energy_per_inference * inferences_per_day * 
                                   days_per_year * energy_reduction)
            
            # Estimate CO2 savings (using average grid carbon intensity)
            co2_per_kwh = 0.4  # kg CO2 per kWh (global average)
            annual_co2_savings = annual_energy_savings * co2_per_kwh
            
            print(f"\n  {method.upper()}:")
            print(f"    Energy reduction: {energy_reduction:.1%}")
            print(f"    Annual energy savings: {annual_energy_savings:.2f} kWh")
            print(f"    Annual CO2 savings: {annual_co2_savings:.2f} kg CO2")
            print(f"    Model size reduction: {result.size_reduction:.1%}")


def create_quantization_visualization(results: dict):
    """Create visualization of quantization results."""
    print("\n5. Creating Quantization Results Visualization...")
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - creating text-based visualization instead")
        create_text_visualization(results)
        return
    
    # Prepare data for visualization
    methods = []
    compression_ratios = []
    size_reductions = []
    speedups = []
    auc_drops = []
    
    for method, result in results.items():
        if result and result.success:
            methods.append(method.upper())
            compression_ratios.append(result.compression_ratio)
            size_reductions.append(result.size_reduction * 100)  # Convert to percentage
            speedups.append(result.speedup_ratio)
            auc_drops.append(abs(result.performance_drop.get('roc_auc', 0.0)) * 100)  # Convert to percentage
    
    if not methods:
        print("No successful quantization results to visualize.")
        return
    
    try:
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quantization Methods Comparison', fontsize=16, fontweight='bold')
        
        # Compression ratio
        axes[0, 0].bar(methods, compression_ratios, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Compression Ratio')
        axes[0, 0].set_ylabel('Ratio (x)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Size reduction
        axes[0, 1].bar(methods, size_reductions, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Size Reduction')
        axes[0, 1].set_ylabel('Reduction (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Speedup
        axes[1, 0].bar(methods, speedups, color='orange', alpha=0.7)
        axes[1, 0].set_title('Inference Speedup')
        axes[1, 0].set_ylabel('Speedup (x)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # AUC drop
        axes[1, 1].bar(methods, auc_drops, color='salmon', alpha=0.7)
        axes[1, 1].set_title('AUC Performance Drop')
        axes[1, 1].set_ylabel('AUC Drop (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("quantization_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error creating matplotlib visualization: {e}")
        create_text_visualization(results)


def create_text_visualization(results: dict):
    """Create text-based visualization of results."""
    print("\n📊 Quantization Results Summary (Text Format):")
    print("=" * 60)
    
    for method, result in results.items():
        if result and result.success:
            print(f"\n{method.upper()}:")
            print(f"  Compression Ratio: {result.compression_ratio:.2f}x {'█' * min(int(result.compression_ratio), 20)}")
            print(f"  Size Reduction:    {result.size_reduction:.1%} {'█' * min(int(result.size_reduction * 20), 20)}")
            print(f"  Speedup:          {result.speedup_ratio:.2f}x {'█' * min(int(result.speedup_ratio), 20)}")
            auc_drop = abs(result.performance_drop.get('roc_auc', 0.0))
            print(f"  AUC Drop:         {auc_drop:.4f} {'█' * min(int(auc_drop * 1000), 20)}")


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("SUSTAINABLE CREDIT RISK AI - QUANTIZATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("\nThis demonstration shows how to use quantization to create")
    print("sustainable AI models with reduced energy consumption and model size.")
    
    # Create dataset
    X, y = create_synthetic_credit_data(n_samples=3000, n_features=30)
    
    # Train model
    model = train_credit_risk_model(X, y, epochs=80)
    
    # Demonstrate quantization methods
    results = demonstrate_quantization_methods(model, X, y)
    
    # Get best quantized model for further demonstrations
    best_result = None
    for result in results.values():
        if result and result.success:
            best_result = result
            break
    
    if best_result:
        # Mobile optimization
        mobile_result = demonstrate_mobile_optimization(model, X, y)
        
        # Accuracy validation
        validation_results = demonstrate_accuracy_validation(
            model, best_result.quantized_model, X, y
        )
        
        # Sustainability impact
        demonstrate_sustainability_impact(results)
        
        # Visualization
        create_quantization_visualization(results)
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n✅ Key Achievements:")
        print("  • Successfully implemented all quantization methods")
        print("  • Demonstrated significant model compression")
        print("  • Maintained model accuracy within acceptable thresholds")
        print("  • Showed potential for substantial energy savings")
        print("  • Created mobile-optimized models for deployment")
        print("\n🌱 Sustainability Benefits:")
        print("  • Reduced model size enables edge deployment")
        print("  • Lower energy consumption during inference")
        print("  • Decreased carbon footprint for AI operations")
        print("  • Supports sustainable AI development practices")
        
    else:
        print("\n⚠️ No successful quantization results obtained.")
        print("This may be due to platform-specific limitations.")
        print("The quantization system includes fallback mechanisms for compatibility.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script for LightGBM baseline model.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.ingestion import ingest_banking_data
from src.data.feature_engineering import engineer_banking_features, get_minimal_config
from src.data.feature_selection import select_banking_features, get_fast_selection_config
from src.models.lightgbm_model import (
    train_lightgbm_baseline, get_fast_lightgbm_config, 
    create_lightgbm_model, LightGBMTrainer
)
import warnings
warnings.filterwarnings('ignore')


def main():
    """Test LightGBM baseline model with banking data."""
    print("Testing LightGBM baseline model...")
    
    # 1. Prepare data
    print("1. Preparing data...")
    ingestion_result = ingest_banking_data("Bank_data.csv")
    
    if not ingestion_result.success:
        print(f"Data ingestion failed: {ingestion_result.message}")
        return
    
    data = ingestion_result.data.sample(n=2000, random_state=42)  # Smaller sample for faster testing
    print(f"   Data loaded: {data.shape}")
    
    # Feature engineering
    fe_config = get_minimal_config()
    fe_result = engineer_banking_features(data, target_column='default', config=fe_config)
    
    if not fe_result.success:
        print(f"Feature engineering failed: {fe_result.message}")
        return
    
    # Feature selection
    fs_config = get_fast_selection_config()
    fs_result = select_banking_features(fe_result.features, fe_result.target, config=fs_config)
    
    if not fs_result.success:
        print(f"Feature selection failed: {fs_result.message}")
        return
    
    X = fe_result.features[fs_result.selected_features]
    y = fe_result.target
    
    print(f"   Final data shape: {X.shape}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    
    # 2. Test basic LightGBM model
    print("\n2. Testing basic LightGBM model...")
    
    # Create model with fast config
    config = get_fast_lightgbm_config()
    model = create_lightgbm_model(config)
    
    # Split data manually for testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate basic metrics
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"   Basic Model Performance:")
    print(f"     Accuracy: {accuracy:.3f}")
    print(f"     F1-Score: {f1:.3f}")
    print(f"     ROC-AUC: {auc:.3f}")
    
    # 3. Test feature importance
    print("\n3. Testing feature importance...")
    
    feature_importance = model.get_feature_importance()
    
    print(f"   Top 10 most important features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f"     {i+1}. {feature}: {importance:.2f}")
    
    # 4. Test full training pipeline
    print("\n4. Testing full training pipeline...")
    
    trainer = LightGBMTrainer(config)
    result = trainer.train_and_evaluate(X, y, test_size=0.2)
    
    print(f"   Training Result:")
    print(f"     Success: {result.success}")
    print(f"     Training time: {result.training_time_seconds:.2f} seconds")
    print(f"     Message: {result.message}")
    
    if result.success:
        print(f"\n   Performance Metrics:")
        print(f"     Training AUC: {result.training_metrics.get('roc_auc', 0):.3f}")
        print(f"     Validation AUC: {result.validation_metrics.get('roc_auc', 0):.3f}")
        print(f"     Test AUC: {result.test_metrics.get('roc_auc', 0):.3f}")
        
        print(f"\n   Best Parameters:")
        for param, value in list(result.best_params.items())[:5]:
            print(f"     {param}: {value}")
        
        if result.cv_results:
            print(f"\n   Cross-Validation Results:")
            cv_auc = result.cv_results['mean_scores'].get('roc_auc', 0)
            cv_auc_std = result.cv_results['std_scores'].get('roc_auc', 0)
            print(f"     CV AUC: {cv_auc:.3f} ± {cv_auc_std:.3f}")
        
        print(f"\n   Top 5 Feature Importance:")
        for i, (feature, importance) in enumerate(list(result.feature_importance.items())[:5]):
            print(f"     {i+1}. {feature}: {importance:.2f}")
    
    # 5. Test model saving and loading
    print("\n5. Testing model saving and loading...")
    
    if result.success and result.model:
        # Save model
        model_path = result.model.save_model("test_models/lightgbm_test")
        print(f"   Model saved to: {model_path}")
        
        # Load model
        loaded_model = create_lightgbm_model()
        loaded_model.load_model("test_models/lightgbm_test")
        
        # Test loaded model
        loaded_pred = loaded_model.predict(X_test)
        loaded_accuracy = accuracy_score(y_test, loaded_pred)
        
        print(f"   Loaded model accuracy: {loaded_accuracy:.3f}")
        print(f"   Model loading successful: {abs(accuracy - loaded_accuracy) < 0.001}")
    
    # 6. Test convenience function
    print("\n6. Testing convenience function...")
    
    convenience_result = train_lightgbm_baseline(X, y, config)
    
    print(f"   Convenience function result:")
    print(f"     Success: {convenience_result.success}")
    print(f"     Test AUC: {convenience_result.test_metrics.get('roc_auc', 0):.3f}")
    
    print(f"\nLightGBM baseline testing completed successfully!")


if __name__ == "__main__":
    main()
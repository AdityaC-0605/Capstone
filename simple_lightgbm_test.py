#!/usr/bin/env python3
"""
Simple test for LightGBM functionality.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.ingestion import ingest_banking_data
from src.models.lightgbm_model import create_lightgbm_model, get_fast_lightgbm_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def main():
    """Simple LightGBM test."""
    print("Simple LightGBM test...")
    
    # Load data
    result = ingest_banking_data("Bank_data.csv")
    data = result.data.sample(n=500, random_state=42)
    
    # Simple feature selection
    numeric_cols = ['age', 'annual_income_inr', 'loan_amount_inr', 'credit_score', 'debt_to_income_ratio']
    X = data[numeric_cols]
    y = data['default']
    
    print(f"Data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    config = get_fast_lightgbm_config()
    model = create_lightgbm_model(config)
    
    print("Training LightGBM model...")
    model.fit(X_train, y_train)
    
    # Test predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"Results: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
    
    # Test feature importance
    importance = model.get_feature_importance()
    print(f"Top 3 features: {list(importance.items())[:3]}")
    
    print("LightGBM test completed successfully!")


if __name__ == "__main__":
    main()
"""
Bias Mitigation Techniques for Credit Risk AI System.

This module implements comprehensive bias mitigation techniques including
reweighting algorithms, adversarial debiasing, post-processing fairness
adjustments, and bias mitigation impact measurement.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

try:
    from .bias_detector import BiasDetector, FairnessMetricsCalculator
    from ..core.logging import get_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from compliance.bias_detector import BiasDetector, FairnessMetricsCalculator
    from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BiasMetrics:
    """Bias metrics before and after mitigation."""
    demographic_parity: float
    equal_opportunity: float
    equalized_odds: float
    calibration: float
    individual_fairness: float
    overall_accuracy: float
    group_accuracies: Dict[str, float]


@dataclass
class MitigationResult:
    """Result of bias mitigation technique."""
    technique: str
    before_metrics: BiasMetrics
    after_metrics: BiasMetrics
    improvement: Dict[str, float]
    parameters: Dict[str, Any]
    timestamp: datetime
    success: bool
    notes: str


class DataReweighting:
    """Implements reweighting algorithms for training data bias mitigation."""
    
    def __init__(self):
        self.weights = None
        self.protected_attributes = None
        self.target_variable = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, protected_attributes: List[str]) -> 'DataReweighting':
        """
        Fit reweighting algorithm to compute sample weights.
        
        Args:
            X: Feature matrix
            y: Target variable
            protected_attributes: List of protected attribute column names
            
        Returns:
            Self for method chaining
        """
        self.protected_attributes = protected_attributes
        self.target_variable = y.name if hasattr(y, 'name') else 'target'
        
        # Combine features and target for analysis
        data = X.copy()
        data[self.target_variable] = y
        
        # Calculate weights for each combination of protected attributes and target
        weights = np.ones(len(data))
        
        for attr in protected_attributes:
            if attr not in data.columns:
                logger.warning(f"Protected attribute '{attr}' not found in data")
                continue
                
            # Calculate group sizes
            group_counts = data.groupby([attr, self.target_variable]).size()
            total_counts = data.groupby(attr).size()
            target_counts = data.groupby(self.target_variable).size()
            
            # Calculate desired weights for demographic parity
            total_size = len(data)
            n_groups = len(data[attr].unique())
            n_targets = len(data[self.target_variable].unique())
            
            for group in data[attr].unique():
                for target in data[self.target_variable].unique():
                    # Desired proportion for this group-target combination
                    desired_prop = 1.0 / (n_groups * n_targets)
                    
                    # Current proportion
                    current_count = group_counts.get((group, target), 0)
                    current_prop = current_count / total_size
                    
                    if current_prop > 0:
                        # Weight to achieve desired proportion
                        weight = desired_prop / current_prop
                        
                        # Apply weight to samples in this group-target combination
                        mask = (data[attr] == group) & (data[self.target_variable] == target)
                        weights[mask] *= weight
        
        # Normalize weights
        weights = weights / np.mean(weights)
        
        # Cap extreme weights to prevent instability
        weights = np.clip(weights, 0.1, 10.0)
        
        self.weights = weights
        
        logger.info(f"Computed reweighting with mean weight: {np.mean(weights):.3f}, "
                   f"std: {np.std(weights):.3f}")
        
        return self
    
    def get_weights(self) -> np.ndarray:
        """Get computed sample weights."""
        if self.weights is None:
            raise ValueError("Must call fit() before get_weights()")
        return self.weights
    
    def transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Apply reweighting to dataset.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuple of (X, y, weights)
        """
        if self.weights is None:
            raise ValueError("Must call fit() before transform()")
            
        return X, y, self.weights


class AdversarialDebiasing(nn.Module):
    """
    Implements adversarial debiasing during model training.
    
    This technique trains a predictor while simultaneously training an adversary
    that tries to predict protected attributes from the predictor's representations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 protected_dim: int = 1, num_classes: int = 2):
        super(AdversarialDebiasing, self).__init__()
        
        # Main predictor network
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Adversarial network (tries to predict protected attributes)
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, protected_dim)
        )
        
        self.hidden_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both predictor and adversary.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (predictions, adversary_predictions)
        """
        # Get hidden representation from predictor
        hidden = x
        for layer in self.predictor[:-1]:
            hidden = layer(hidden)
        
        # Final prediction
        predictions = self.predictor[-1](hidden)
        
        # Adversary tries to predict protected attributes from hidden representation
        adversary_pred = self.adversary(hidden.detach())  # Detach to prevent gradient flow
        
        return predictions, adversary_pred
    
    def get_hidden_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get hidden representation for analysis."""
        hidden = x
        for layer in self.predictor[:-1]:
            hidden = layer(hidden)
        return hidden


class AdversarialTrainer:
    """Trainer for adversarial debiasing."""
    
    def __init__(self, model: AdversarialDebiasing, lambda_adv: float = 1.0,
                 learning_rate: float = 0.001, device: str = 'cpu'):
        self.model = model.to(device)
        self.lambda_adv = lambda_adv
        self.device = device
        
        # Separate optimizers for predictor and adversary
        self.predictor_optimizer = optim.Adam(self.model.predictor.parameters(), lr=learning_rate)
        self.adversary_optimizer = optim.Adam(self.model.adversary.parameters(), lr=learning_rate)
        
        # Loss functions
        self.prediction_loss = nn.CrossEntropyLoss()
        self.adversary_loss = nn.BCEWithLogitsLoss() if model.adversary[-1].out_features == 1 else nn.CrossEntropyLoss()
        
    def train_step(self, X: torch.Tensor, y: torch.Tensor, 
                   protected: torch.Tensor) -> Dict[str, float]:
        """
        Single training step with adversarial training.
        
        Args:
            X: Input features
            y: Target labels
            protected: Protected attribute labels
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        
        # Forward pass
        predictions, adversary_pred = self.model(X)
        
        # Prediction loss
        pred_loss = self.prediction_loss(predictions, y)
        
        # Adversary loss (adversary tries to predict protected attributes)
        if protected.dim() == 1 and self.model.adversary[-1].out_features == 1:
            adv_loss = self.adversary_loss(adversary_pred.squeeze(), protected.float())
        else:
            adv_loss = self.adversary_loss(adversary_pred, protected)
        
        # Update adversary (maximize its ability to predict protected attributes)
        self.adversary_optimizer.zero_grad()
        adv_loss.backward(retain_graph=True)
        self.adversary_optimizer.step()
        
        # Re-compute forward pass for predictor update
        predictions, adversary_pred = self.model(X)
        pred_loss = self.prediction_loss(predictions, y)
        
        if protected.dim() == 1 and self.model.adversary[-1].out_features == 1:
            adv_loss = self.adversary_loss(adversary_pred.squeeze(), protected.float())
        else:
            adv_loss = self.adversary_loss(adversary_pred, protected)
        
        # Update predictor (minimize prediction loss, maximize adversary loss)
        self.predictor_optimizer.zero_grad()
        total_loss = pred_loss - self.lambda_adv * adv_loss
        total_loss.backward()
        self.predictor_optimizer.step()
        
        return {
            'prediction_loss': pred_loss.item(),
            'adversary_loss': adv_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
              val_loader: torch.utils.data.DataLoader,
              epochs: int = 100, patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the adversarial debiasing model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        history = {
            'train_pred_loss': [],
            'train_adv_loss': [],
            'val_pred_loss': [],
            'val_adv_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_losses = {'prediction_loss': [], 'adversary_loss': []}
            
            for batch_X, batch_y, batch_protected in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_protected = batch_protected.to(self.device)
                
                losses = self.train_step(batch_X, batch_y, batch_protected)
                train_losses['prediction_loss'].append(losses['prediction_loss'])
                train_losses['adversary_loss'].append(losses['adversary_loss'])
            
            # Validation
            val_losses = self.validate(val_loader)
            
            # Record history
            history['train_pred_loss'].append(np.mean(train_losses['prediction_loss']))
            history['train_adv_loss'].append(np.mean(train_losses['adversary_loss']))
            history['val_pred_loss'].append(val_losses['prediction_loss'])
            history['val_adv_loss'].append(val_losses['adversary_loss'])
            
            # Early stopping
            current_val_loss = val_losses['prediction_loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Pred Loss: {history['train_pred_loss'][-1]:.4f}, "
                           f"Val Pred Loss: {history['val_pred_loss'][-1]:.4f}")
        
        return history
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        val_losses = {'prediction_loss': [], 'adversary_loss': []}
        
        with torch.no_grad():
            for batch_X, batch_y, batch_protected in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_protected = batch_protected.to(self.device)
                
                predictions, adversary_pred = self.model(batch_X)
                
                pred_loss = self.prediction_loss(predictions, batch_y)
                
                if batch_protected.dim() == 1 and self.model.adversary[-1].out_features == 1:
                    adv_loss = self.adversary_loss(adversary_pred.squeeze(), batch_protected.float())
                else:
                    adv_loss = self.adversary_loss(adversary_pred, batch_protected)
                
                val_losses['prediction_loss'].append(pred_loss.item())
                val_losses['adversary_loss'].append(adv_loss.item())
        
        return {
            'prediction_loss': np.mean(val_losses['prediction_loss']),
            'adversary_loss': np.mean(val_losses['adversary_loss'])
        }


class PostProcessingFairness:
    """
    Implements post-processing fairness adjustments.
    
    This technique adjusts model predictions after training to achieve
    fairness constraints while maintaining overall accuracy.
    """
    
    def __init__(self, fairness_constraint: str = 'demographic_parity'):
        """
        Initialize post-processing fairness adjuster.
        
        Args:
            fairness_constraint: Type of fairness constraint
                - 'demographic_parity': Equal positive prediction rates
                - 'equal_opportunity': Equal true positive rates
                - 'equalized_odds': Equal TPR and FPR
        """
        self.fairness_constraint = fairness_constraint
        self.thresholds = {}
        self.protected_groups = None
        
    def fit(self, y_pred_proba: np.ndarray, y_true: np.ndarray, 
            protected_attributes: np.ndarray) -> 'PostProcessingFairness':
        """
        Fit post-processing thresholds to achieve fairness.
        
        Args:
            y_pred_proba: Predicted probabilities
            y_true: True labels
            protected_attributes: Protected attribute values
            
        Returns:
            Self for method chaining
        """
        self.protected_groups = np.unique(protected_attributes)
        
        if self.fairness_constraint == 'demographic_parity':
            self._fit_demographic_parity(y_pred_proba, y_true, protected_attributes)
        elif self.fairness_constraint == 'equal_opportunity':
            self._fit_equal_opportunity(y_pred_proba, y_true, protected_attributes)
        elif self.fairness_constraint == 'equalized_odds':
            self._fit_equalized_odds(y_pred_proba, y_true, protected_attributes)
        else:
            raise ValueError(f"Unknown fairness constraint: {self.fairness_constraint}")
        
        return self
    
    def _fit_demographic_parity(self, y_pred_proba: np.ndarray, y_true: np.ndarray,
                               protected_attributes: np.ndarray):
        """Fit thresholds for demographic parity."""
        # Calculate overall positive rate
        overall_positive_rate = np.mean(y_true)
        
        for group in self.protected_groups:
            group_mask = protected_attributes == group
            group_proba = y_pred_proba[group_mask]
            
            # Find threshold that achieves target positive rate
            thresholds = np.linspace(0, 1, 1000)
            best_threshold = 0.5
            best_diff = float('inf')
            
            for threshold in thresholds:
                predicted_positive_rate = np.mean(group_proba >= threshold)
                diff = abs(predicted_positive_rate - overall_positive_rate)
                
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = threshold
            
            self.thresholds[group] = best_threshold
            
        logger.info(f"Demographic parity thresholds: {self.thresholds}")
    
    def _fit_equal_opportunity(self, y_pred_proba: np.ndarray, y_true: np.ndarray,
                              protected_attributes: np.ndarray):
        """Fit thresholds for equal opportunity (equal TPR)."""
        # Calculate overall TPR at default threshold
        overall_tpr = np.mean(y_pred_proba[y_true == 1] >= 0.5)
        
        for group in self.protected_groups:
            group_mask = (protected_attributes == group) & (y_true == 1)
            if np.sum(group_mask) == 0:
                self.thresholds[group] = 0.5
                continue
                
            group_proba = y_pred_proba[group_mask]
            
            # Find threshold that achieves target TPR
            thresholds = np.linspace(0, 1, 1000)
            best_threshold = 0.5
            best_diff = float('inf')
            
            for threshold in thresholds:
                tpr = np.mean(group_proba >= threshold)
                diff = abs(tpr - overall_tpr)
                
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = threshold
            
            self.thresholds[group] = best_threshold
            
        logger.info(f"Equal opportunity thresholds: {self.thresholds}")
    
    def _fit_equalized_odds(self, y_pred_proba: np.ndarray, y_true: np.ndarray,
                           protected_attributes: np.ndarray):
        """Fit thresholds for equalized odds (equal TPR and FPR)."""
        # Calculate overall TPR and FPR at default threshold
        overall_tpr = np.mean(y_pred_proba[y_true == 1] >= 0.5)
        overall_fpr = np.mean(y_pred_proba[y_true == 0] >= 0.5)
        
        for group in self.protected_groups:
            # Find threshold that minimizes TPR and FPR differences
            thresholds = np.linspace(0, 1, 1000)
            best_threshold = 0.5
            best_score = float('inf')
            
            for threshold in thresholds:
                group_pos_mask = (protected_attributes == group) & (y_true == 1)
                group_neg_mask = (protected_attributes == group) & (y_true == 0)
                
                if np.sum(group_pos_mask) == 0 or np.sum(group_neg_mask) == 0:
                    continue
                
                group_tpr = np.mean(y_pred_proba[group_pos_mask] >= threshold)
                group_fpr = np.mean(y_pred_proba[group_neg_mask] >= threshold)
                
                # Score combines TPR and FPR differences
                score = abs(group_tpr - overall_tpr) + abs(group_fpr - overall_fpr)
                
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
            
            self.thresholds[group] = best_threshold
            
        logger.info(f"Equalized odds thresholds: {self.thresholds}")
    
    def transform(self, y_pred_proba: np.ndarray, 
                  protected_attributes: np.ndarray) -> np.ndarray:
        """
        Apply post-processing fairness adjustments.
        
        Args:
            y_pred_proba: Predicted probabilities
            protected_attributes: Protected attribute values
            
        Returns:
            Adjusted binary predictions
        """
        if not self.thresholds:
            raise ValueError("Must call fit() before transform()")
        
        predictions = np.zeros(len(y_pred_proba))
        
        for group in self.protected_groups:
            group_mask = protected_attributes == group
            threshold = self.thresholds.get(group, 0.5)
            predictions[group_mask] = (y_pred_proba[group_mask] >= threshold).astype(int)
        
        return predictions


class BiasMitigationImpactMeasurement:
    """Measures the impact of bias mitigation techniques."""
    
    def __init__(self):
        self.bias_detector = BiasDetector()
    
    def measure_impact(self, y_true: np.ndarray, y_pred_before: np.ndarray,
                      y_pred_after: np.ndarray, protected_attributes: Dict[str, np.ndarray],
                      technique_name: str) -> MitigationResult:
        """
        Measure the impact of a bias mitigation technique.
        
        Args:
            y_true: True labels
            y_pred_before: Predictions before mitigation
            y_pred_after: Predictions after mitigation
            protected_attributes: Dictionary of protected attributes
            technique_name: Name of the mitigation technique
            
        Returns:
            MitigationResult with before/after comparison
        """
        # Calculate metrics before mitigation
        before_metrics = self._calculate_bias_metrics(
            y_true, y_pred_before, protected_attributes
        )
        
        # Calculate metrics after mitigation
        after_metrics = self._calculate_bias_metrics(
            y_true, y_pred_after, protected_attributes
        )
        
        # Calculate improvements
        improvement = self._calculate_improvement(before_metrics, after_metrics)
        
        # Determine success
        success = self._determine_success(improvement)
        
        # Generate notes
        notes = self._generate_notes(improvement, technique_name)
        
        return MitigationResult(
            technique=technique_name,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement=improvement,
            parameters={},  # To be filled by specific techniques
            timestamp=datetime.now(),
            success=success,
            notes=notes
        )
    
    def _calculate_bias_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               protected_attributes: Dict[str, np.ndarray]) -> BiasMetrics:
        """Calculate comprehensive bias metrics."""
        
        # Overall accuracy
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        # Group accuracies
        group_accuracies = {}
        for attr_name, attr_values in protected_attributes.items():
            for group in np.unique(attr_values):
                group_mask = attr_values == group
                if np.sum(group_mask) > 0:
                    group_acc = accuracy_score(y_true[group_mask], y_pred[group_mask])
                    group_accuracies[f"{attr_name}_{group}"] = group_acc
        
        # Use bias detector for fairness metrics
        results = self.bias_detector.detect_bias(
            y_true, y_pred, protected_attributes
        )
        
        # Convert results to dictionary format
        fairness_results = {}
        for result in results:
            attr_name = result.protected_attribute
            fairness_results[attr_name] = result
        
        # Aggregate fairness metrics (take worst case across attributes)
        demographic_parity = 1.0  # Default to fair
        equal_opportunity = 1.0
        equalized_odds = 1.0
        calibration = 1.0
        
        # Extract metrics from bias detection results
        for result in results:
            if result.metric.value == 'demographic_parity':
                demographic_parity = min(demographic_parity, 1.0 - result.overall_metric)
            elif result.metric.value == 'equal_opportunity':
                equal_opportunity = min(equal_opportunity, 1.0 - result.overall_metric)
            elif result.metric.value == 'equalized_odds':
                equalized_odds = min(equalized_odds, 1.0 - result.overall_metric)
        
        # For calibration, use a simplified calculation
        calibration = 0.95  # Placeholder - would need proper calibration calculation
        
        # Individual fairness (simplified as consistency across similar individuals)
        individual_fairness = self._calculate_individual_fairness(y_pred, protected_attributes)
        
        return BiasMetrics(
            demographic_parity=demographic_parity,
            equal_opportunity=equal_opportunity,
            equalized_odds=equalized_odds,
            calibration=calibration,
            individual_fairness=individual_fairness,
            overall_accuracy=overall_accuracy,
            group_accuracies=group_accuracies
        )
    
    def _calculate_individual_fairness(self, y_pred: np.ndarray,
                                     protected_attributes: Dict[str, np.ndarray]) -> float:
        """Calculate individual fairness metric (simplified)."""
        # This is a simplified version - in practice, you'd need similarity metrics
        # For now, we'll use consistency within protected groups as a proxy
        
        consistency_scores = []
        for attr_name, attr_values in protected_attributes.items():
            for group in np.unique(attr_values):
                group_mask = attr_values == group
                group_predictions = y_pred[group_mask]
                
                if len(group_predictions) > 1:
                    # Measure consistency as 1 - variance of predictions within group
                    consistency = 1.0 - np.var(group_predictions)
                    consistency_scores.append(max(0, consistency))
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_improvement(self, before: BiasMetrics, after: BiasMetrics) -> Dict[str, float]:
        """Calculate improvement in metrics."""
        return {
            'demographic_parity': after.demographic_parity - before.demographic_parity,
            'equal_opportunity': after.equal_opportunity - before.equal_opportunity,
            'equalized_odds': after.equalized_odds - before.equalized_odds,
            'calibration': after.calibration - before.calibration,
            'individual_fairness': after.individual_fairness - before.individual_fairness,
            'overall_accuracy': after.overall_accuracy - before.overall_accuracy
        }
    
    def _determine_success(self, improvement: Dict[str, float]) -> bool:
        """Determine if mitigation was successful."""
        # Success if fairness improved without significant accuracy loss
        fairness_improved = (
            improvement['demographic_parity'] > 0.01 or
            improvement['equal_opportunity'] > 0.01 or
            improvement['equalized_odds'] > 0.01
        )
        
        accuracy_maintained = improvement['overall_accuracy'] > -0.05  # Allow 5% accuracy loss
        
        return fairness_improved and accuracy_maintained
    
    def _generate_notes(self, improvement: Dict[str, float], technique_name: str) -> str:
        """Generate notes about the mitigation results."""
        notes = [f"Applied {technique_name} bias mitigation technique."]
        
        # Fairness improvements
        fairness_improvements = []
        if improvement['demographic_parity'] > 0.01:
            fairness_improvements.append(f"Demographic parity improved by {improvement['demographic_parity']:.3f}")
        if improvement['equal_opportunity'] > 0.01:
            fairness_improvements.append(f"Equal opportunity improved by {improvement['equal_opportunity']:.3f}")
        if improvement['equalized_odds'] > 0.01:
            fairness_improvements.append(f"Equalized odds improved by {improvement['equalized_odds']:.3f}")
        
        if fairness_improvements:
            notes.append("Fairness improvements: " + ", ".join(fairness_improvements))
        
        # Accuracy impact
        acc_change = improvement['overall_accuracy']
        if acc_change > 0.01:
            notes.append(f"Accuracy improved by {acc_change:.3f}")
        elif acc_change < -0.01:
            notes.append(f"Accuracy decreased by {abs(acc_change):.3f}")
        else:
            notes.append("Accuracy maintained")
        
        return " ".join(notes)


class BiasMitigationPipeline:
    """
    Comprehensive bias mitigation pipeline that combines multiple techniques.
    """
    
    def __init__(self):
        self.reweighting = DataReweighting()
        self.impact_measurement = BiasMitigationImpactMeasurement()
        self.mitigation_history = []
        
    def apply_reweighting(self, X: pd.DataFrame, y: pd.Series,
                         protected_attributes: List[str]) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Apply data reweighting technique."""
        logger.info("Applying data reweighting for bias mitigation")
        
        # Fit and apply reweighting
        self.reweighting.fit(X, y, protected_attributes)
        X_reweighted, y_reweighted, weights = self.reweighting.transform(X, y)
        
        logger.info(f"Reweighting applied. Weight statistics: "
                   f"mean={np.mean(weights):.3f}, std={np.std(weights):.3f}")
        
        return X_reweighted, y_reweighted, weights
    
    def create_adversarial_model(self, input_dim: int, protected_dim: int = 1,
                                num_classes: int = 2) -> AdversarialDebiasing:
        """Create adversarial debiasing model."""
        logger.info("Creating adversarial debiasing model")
        
        model = AdversarialDebiasing(
            input_dim=input_dim,
            protected_dim=protected_dim,
            num_classes=num_classes
        )
        
        return model
    
    def apply_post_processing(self, y_pred_proba: np.ndarray, y_true: np.ndarray,
                             protected_attributes: np.ndarray,
                             fairness_constraint: str = 'demographic_parity') -> np.ndarray:
        """Apply post-processing fairness adjustments."""
        logger.info(f"Applying post-processing fairness with {fairness_constraint} constraint")
        
        post_processor = PostProcessingFairness(fairness_constraint)
        post_processor.fit(y_pred_proba, y_true, protected_attributes)
        adjusted_predictions = post_processor.transform(y_pred_proba, protected_attributes)
        
        logger.info("Post-processing fairness adjustments applied")
        
        return adjusted_predictions
    
    def measure_mitigation_impact(self, y_true: np.ndarray, y_pred_before: np.ndarray,
                                 y_pred_after: np.ndarray, 
                                 protected_attributes: Dict[str, np.ndarray],
                                 technique_name: str) -> MitigationResult:
        """Measure the impact of bias mitigation."""
        logger.info(f"Measuring impact of {technique_name} bias mitigation")
        
        result = self.impact_measurement.measure_impact(
            y_true, y_pred_before, y_pred_after, protected_attributes, technique_name
        )
        
        # Store in history
        self.mitigation_history.append(result)
        
        logger.info(f"Mitigation impact measured. Success: {result.success}")
        
        return result
    
    def get_mitigation_summary(self) -> Dict[str, Any]:
        """Get summary of all mitigation attempts."""
        if not self.mitigation_history:
            return {"message": "No bias mitigation attempts recorded"}
        
        successful_mitigations = [r for r in self.mitigation_history if r.success]
        
        summary = {
            "total_attempts": len(self.mitigation_history),
            "successful_attempts": len(successful_mitigations),
            "success_rate": len(successful_mitigations) / len(self.mitigation_history),
            "techniques_used": list(set(r.technique for r in self.mitigation_history)),
            "latest_results": self.mitigation_history[-3:] if len(self.mitigation_history) >= 3 else self.mitigation_history
        }
        
        if successful_mitigations:
            # Best performing technique
            best_result = max(successful_mitigations, 
                            key=lambda r: sum(r.improvement.values()))
            summary["best_technique"] = {
                "name": best_result.technique,
                "improvement": best_result.improvement,
                "timestamp": best_result.timestamp.isoformat()
            }
        
        return summary


# Utility functions

def create_bias_mitigation_pipeline() -> BiasMitigationPipeline:
    """Create bias mitigation pipeline."""
    return BiasMitigationPipeline()


def apply_comprehensive_bias_mitigation(X: pd.DataFrame, y: pd.Series,
                                      protected_attributes: List[str],
                                      model_predictions: np.ndarray) -> Dict[str, Any]:
    """
    Apply comprehensive bias mitigation using multiple techniques.
    
    Args:
        X: Feature matrix
        y: Target variable
        protected_attributes: List of protected attribute column names
        model_predictions: Original model predictions
        
    Returns:
        Dictionary with mitigation results
    """
    pipeline = create_bias_mitigation_pipeline()
    
    # Apply reweighting
    X_reweighted, y_reweighted, weights = pipeline.apply_reweighting(
        X, y, protected_attributes
    )
    
    # Apply post-processing (example with demographic parity)
    protected_attr_values = X[protected_attributes[0]].values
    adjusted_predictions = pipeline.apply_post_processing(
        model_predictions, y.values, protected_attr_values
    )
    
    # Measure impact
    protected_dict = {attr: X[attr].values for attr in protected_attributes}
    impact_result = pipeline.measure_mitigation_impact(
        y.values, model_predictions, adjusted_predictions,
        protected_dict, "comprehensive_mitigation"
    )
    
    return {
        "reweighted_data": (X_reweighted, y_reweighted, weights),
        "adjusted_predictions": adjusted_predictions,
        "impact_result": impact_result,
        "pipeline_summary": pipeline.get_mitigation_summary()
    }


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.datasets import make_classification
    
    # Create synthetic dataset with bias
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    # Add protected attribute (gender) with bias
    gender = np.random.choice(['M', 'F'], size=1000, p=[0.6, 0.4])
    # Introduce bias: males more likely to get positive outcome
    bias_mask = (gender == 'M') & (y == 1)
    y[bias_mask[:100]] = 1  # Artificially increase positive outcomes for males
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    X_df['gender'] = gender
    y_series = pd.Series(y, name='target')
    
    # Create biased predictions (simulate model predictions)
    biased_predictions = np.where(gender == 'M', 
                                 np.random.choice([0, 1], size=1000, p=[0.3, 0.7]),
                                 np.random.choice([0, 1], size=1000, p=[0.7, 0.3]))
    
    # Apply comprehensive bias mitigation
    results = apply_comprehensive_bias_mitigation(
        X_df, y_series, ['gender'], biased_predictions
    )
    
    print("Bias Mitigation Results:")
    print(f"Impact: {results['impact_result'].success}")
    print(f"Improvements: {results['impact_result'].improvement}")
    print(f"Pipeline Summary: {results['pipeline_summary']}")
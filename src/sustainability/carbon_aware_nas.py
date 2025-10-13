"""
Carbon-Aware Neural Architecture Search (NAS) for Sustainable AI.

This module implements advanced NAS techniques that prioritize energy efficiency
and carbon footprint reduction as primary objectives, not just secondary metrics.
It includes multi-objective optimization, real-time carbon intensity integration,
and automated architecture discovery for sustainable credit risk models.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time
import requests
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from abc import ABC, abstractmethod

try:
    from ..core.logging import get_logger, get_audit_logger
    from .carbon_calculator import CarbonCalculator, CarbonFootprintConfig
    from .energy_tracker import EnergyTracker, EnergyConfig
    from ..nas.neural_architecture_search import ArchitectureSpec, NASConfig
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.logging import get_logger, get_audit_logger
    from sustainability.carbon_calculator import CarbonCalculator, CarbonFootprintConfig
    from sustainability.energy_tracker import EnergyTracker, EnergyConfig
    from nas.neural_architecture_search import ArchitectureSpec, NASConfig

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class CarbonOptimizationObjective(Enum):
    """Carbon optimization objectives for NAS."""
    MINIMIZE_CARBON_FOOTPRINT = "minimize_carbon_footprint"
    MINIMIZE_ENERGY_CONSUMPTION = "minimize_energy_consumption"
    MAXIMIZE_CARBON_EFFICIENCY = "maximize_carbon_efficiency"  # Performance per CO2
    MINIMIZE_TRAINING_TIME = "minimize_training_time"
    BALANCED_SUSTAINABILITY = "balanced_sustainability"


@dataclass
class CarbonAwareNASConfig:
    """Configuration for carbon-aware neural architecture search."""
    
    # Primary objectives
    primary_objective: CarbonOptimizationObjective = CarbonOptimizationObjective.BALANCED_SUSTAINABILITY
    carbon_weight: float = 0.4  # Weight for carbon footprint in multi-objective optimization
    performance_weight: float = 0.4  # Weight for model performance
    efficiency_weight: float = 0.2  # Weight for energy efficiency
    
    # Carbon constraints
    max_carbon_budget_kg: float = 0.1  # Maximum CO2 emissions per architecture evaluation
    max_energy_budget_kwh: float = 0.05  # Maximum energy consumption per evaluation
    carbon_intensity_threshold: float = 400.0  # gCO2/kWh - avoid training above this
    
    # Real-time carbon integration
    enable_real_time_carbon: bool = True
    carbon_api_url: str = "https://api.electricitymap.org/v3/carbon-intensity/latest"
    carbon_api_key: Optional[str] = None
    carbon_check_interval: int = 300  # seconds
    
    # Architecture search parameters
    max_architectures: int = 50
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Model constraints
    max_parameters: int = 1000000  # 1M parameters max
    max_layers: int = 10
    min_accuracy_threshold: float = 0.85  # Minimum acceptable accuracy
    
    # Energy tracking
    energy_config: Optional[EnergyConfig] = None
    carbon_config: Optional[CarbonFootprintConfig] = None
    
    # Output settings
    save_architectures: bool = True
    output_dir: str = "carbon_aware_nas_results"
    generate_visualizations: bool = True


@dataclass
class CarbonAwareArchitecture:
    """Enhanced architecture specification with carbon metrics."""
    
    # Basic architecture info
    architecture_id: str
    layers: List[Dict[str, Any]]
    total_parameters: int
    estimated_latency_ms: float
    
    # Carbon and energy metrics
    estimated_energy_mj: float = 0.0
    estimated_carbon_kg: float = 0.0
    carbon_efficiency_score: float = 0.0  # Performance per CO2
    energy_efficiency_score: float = 0.0  # Performance per kWh
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    is_evaluated: bool = False
    
    # Carbon context
    carbon_intensity_at_training: float = 0.0  # gCO2/kWh when trained
    training_region: str = "unknown"
    training_timestamp: Optional[datetime] = None
    
    # Fitness score for optimization
    fitness_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        def convert_numpy(value):
            """Convert numpy types to Python native types."""
            if hasattr(value, 'item'):
                return value.item()
            elif isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            else:
                return value
        
        return {
            "architecture_id": str(self.architecture_id),
            "layers": self.layers,
            "total_parameters": convert_numpy(self.total_parameters),
            "estimated_latency_ms": convert_numpy(self.estimated_latency_ms),
            "estimated_energy_mj": convert_numpy(self.estimated_energy_mj),
            "estimated_carbon_kg": convert_numpy(self.estimated_carbon_kg),
            "carbon_efficiency_score": convert_numpy(self.carbon_efficiency_score),
            "energy_efficiency_score": convert_numpy(self.energy_efficiency_score),
            "performance_metrics": {k: convert_numpy(v) for k, v in self.performance_metrics.items()},
            "training_time": convert_numpy(self.training_time),
            "is_evaluated": bool(self.is_evaluated),
            "carbon_intensity_at_training": convert_numpy(self.carbon_intensity_at_training),
            "training_region": str(self.training_region),
            "training_timestamp": self.training_timestamp.isoformat() if self.training_timestamp else None,
            "fitness_score": convert_numpy(self.fitness_score)
        }


class CarbonIntensityAPI:
    """Real-time carbon intensity data provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.last_check = None
        self.cached_intensity = None
        self.cache_duration = timedelta(minutes=5)
        
    def get_current_carbon_intensity(self, region: str = "US") -> float:
        """Get current carbon intensity for a region."""
        
        # Check cache first
        if (self.cached_intensity is not None and 
            self.last_check and 
            datetime.now() - self.last_check < self.cache_duration):
            return self.cached_intensity
        
        try:
            # For demo purposes, simulate realistic carbon intensity
            # In production, integrate with real APIs like ElectricityMap
            hour = datetime.now().hour
            
            # Simulate daily carbon intensity pattern
            if 2 <= hour <= 6:  # Night - cleaner energy
                base_intensity = 200
            elif 10 <= hour <= 16:  # Peak - dirtier energy
                base_intensity = 450
            else:  # Normal hours
                base_intensity = 350
            
            # Add some randomness
            intensity = base_intensity + np.random.normal(0, 50)
            intensity = max(100, min(600, intensity))  # Clamp to realistic range
            
            self.cached_intensity = intensity
            self.last_check = datetime.now()
            
            logger.debug(f"Carbon intensity for {region}: {intensity:.1f} gCO2/kWh")
            return intensity
            
        except Exception as e:
            logger.warning(f"Failed to get carbon intensity: {e}")
            return 400.0  # Default fallback
    
    def is_optimal_for_training(self, region: str = "US", threshold: float = 300.0) -> bool:
        """Check if current conditions are optimal for training."""
        intensity = self.get_current_carbon_intensity(region)
        return intensity <= threshold


class CarbonAwareArchitectureEvaluator:
    """Evaluates architectures with carbon footprint as primary metric."""
    
    def __init__(self, config: CarbonAwareNASConfig):
        self.config = config
        self.carbon_calculator = CarbonCalculator(config.carbon_config)
        self.energy_tracker = EnergyTracker(config.energy_config)
        self.carbon_api = CarbonIntensityAPI(config.carbon_api_key)
        
        # Performance tracking
        self.evaluation_history = []
        self.best_architectures = []
        
    def evaluate_architecture(self, architecture: CarbonAwareArchitecture, 
                            X: pd.DataFrame, y: pd.Series) -> CarbonAwareArchitecture:
        """Evaluate architecture with carbon-aware metrics."""
        
        start_time = time.time()
        
        try:
            # Check carbon intensity before training
            current_intensity = self.carbon_api.get_current_carbon_intensity()
            architecture.carbon_intensity_at_training = current_intensity
            architecture.training_timestamp = datetime.now()
            
            # Skip training if carbon intensity is too high
            if (self.config.enable_real_time_carbon and 
                current_intensity > self.config.carbon_intensity_threshold):
                logger.warning(f"Skipping training due to high carbon intensity: {current_intensity:.1f} gCO2/kWh")
                architecture.performance_metrics = {
                    'accuracy': 0.0,
                    'roc_auc': 0.0,
                    'f1_score': 0.0,
                    'carbon_skipped': True
                }
                architecture.training_time = 0.0
                architecture.is_evaluated = True
                return architecture
            
            # Start energy tracking
            energy_experiment_id = self.energy_tracker.start_tracking(architecture.architecture_id)
            
            # Create and train model
            model = self._create_model_from_architecture(architecture, X.shape[1])
            performance_metrics = self._train_and_evaluate_model(model, X, y)
            
            # Stop energy tracking
            energy_report = self.energy_tracker.stop_tracking()
            
            # Calculate carbon footprint
            carbon_footprint = self.carbon_calculator.calculate_carbon_footprint(
                energy_report, 
                region="US"  # Could be made configurable
            )
            
            # Update architecture with metrics
            architecture.estimated_energy_mj = energy_report.total_energy_kwh * 3.6  # Convert to MJ
            architecture.estimated_carbon_kg = carbon_footprint.total_emissions_kg
            architecture.performance_metrics = performance_metrics
            architecture.training_time = time.time() - start_time
            architecture.is_evaluated = True
            
            # Calculate efficiency scores
            if architecture.estimated_carbon_kg > 0:
                architecture.carbon_efficiency_score = (
                    performance_metrics.get('roc_auc', 0.0) / architecture.estimated_carbon_kg
                )
            
            if architecture.estimated_energy_mj > 0:
                architecture.energy_efficiency_score = (
                    performance_metrics.get('roc_auc', 0.0) / architecture.estimated_energy_mj
                )
            
            # Check carbon budget
            if architecture.estimated_carbon_kg > self.config.max_carbon_budget_kg:
                logger.warning(f"Architecture {architecture.architecture_id} exceeded carbon budget: "
                             f"{architecture.estimated_carbon_kg:.4f} kg > {self.config.max_carbon_budget_kg:.4f} kg")
            
            # Store evaluation
            self.evaluation_history.append(architecture)
            
            logger.info(f"Evaluated architecture {architecture.architecture_id}: "
                       f"AUC={performance_metrics.get('roc_auc', 0.0):.4f}, "
                       f"Carbon={architecture.estimated_carbon_kg:.4f} kg, "
                       f"Energy={architecture.estimated_energy_mj:.4f} MJ")
            
        except Exception as e:
            logger.error(f"Error evaluating architecture {architecture.architecture_id}: {e}")
            architecture.performance_metrics = {
                'accuracy': 0.0,
                'roc_auc': 0.0,
                'f1_score': 0.0,
                'error': str(e)
            }
            architecture.training_time = time.time() - start_time
            architecture.is_evaluated = True
        
        return architecture
    
    def _create_model_from_architecture(self, architecture: CarbonAwareArchitecture, 
                                      input_dim: int) -> nn.Module:
        """Create PyTorch model from architecture specification."""
        layers = []
        current_dim = input_dim
        
        for layer_spec in architecture.layers:
            if layer_spec['type'] == 'linear':
                layers.append(nn.Linear(current_dim, layer_spec['hidden_size']))
                current_dim = layer_spec['hidden_size']
                
                # Add normalization if specified
                if layer_spec.get('use_batch_norm', False):
                    layers.append(nn.BatchNorm1d(current_dim))
                elif layer_spec.get('use_layer_norm', False):
                    layers.append(nn.LayerNorm(current_dim))
                
                # Add activation
                activation = layer_spec.get('activation', 'relu')
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                
                # Add dropout if specified
                if layer_spec.get('dropout', 0) > 0:
                    layers.append(nn.Dropout(layer_spec['dropout']))
        
        # Add final output layer
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def _train_and_evaluate_model(self, model: nn.Module, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train and evaluate a model quickly."""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)
        
        # Quick training (reduced epochs for NAS)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        model.train()
        for epoch in range(10):  # Reduced epochs for speed
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).numpy().flatten()
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        # Convert probabilities to binary predictions
        binary_preds = (predictions > 0.5).astype(int)
        
        try:
            roc_auc = roc_auc_score(y, predictions)
        except:
            roc_auc = 0.5
        
        accuracy = accuracy_score(y, binary_preds)
        f1 = f1_score(y, binary_preds, average='weighted')
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1_score': f1
        }


class CarbonAwareNAS:
    """Main carbon-aware neural architecture search system."""
    
    def __init__(self, config: Optional[CarbonAwareNASConfig] = None):
        self.config = config or CarbonAwareNASConfig()
        self.evaluator = CarbonAwareArchitectureEvaluator(self.config)
        
        # Search state
        self.population = []
        self.generation = 0
        self.best_architecture = None
        self.search_history = []
        
        # Results storage
        self.results_dir = Path(self.config.output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Carbon-aware NAS initialized")
    
    def search(self, X: pd.DataFrame, y: pd.Series) -> CarbonAwareArchitecture:
        """Perform carbon-aware neural architecture search."""
        
        logger.info("Starting carbon-aware neural architecture search")
        
        # Initialize population
        self._initialize_population(X.shape[1])
        
        # Evolution loop
        for generation in range(self.config.generations):
            self.generation = generation
            logger.info(f"Generation {generation + 1}/{self.config.generations}")
            
            # Evaluate population
            self._evaluate_population(X, y)
            
            # Select best architectures
            self._selection()
            
            # Generate new population
            self._crossover_and_mutation(X.shape[1])
            
            # Update best architecture
            self._update_best_architecture()
            
            # Log progress
            self._log_generation_progress()
        
        # Final evaluation and ranking
        self._final_evaluation(X, y)
        
        # Save results
        self._save_results()
        
        logger.info(f"Carbon-aware NAS completed. Best architecture: {self.best_architecture.architecture_id}")
        
        return self.best_architecture
    
    def _initialize_population(self, input_dim: int):
        """Initialize random population of architectures."""
        
        self.population = []
        
        for i in range(self.config.population_size):
            architecture = self._generate_random_architecture(input_dim, f"arch_{i}")
            self.population.append(architecture)
        
        logger.info(f"Initialized population of {len(self.population)} architectures")
    
    def _generate_random_architecture(self, input_dim: int, architecture_id: str) -> CarbonAwareArchitecture:
        """Generate a random architecture."""
        
        num_layers = np.random.randint(2, self.config.max_layers + 1)
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            # Random layer size
            hidden_size = np.random.choice([32, 64, 128, 256, 512])
            hidden_size = min(hidden_size, current_dim * 2)  # Don't expand too much
            
            layer_spec = {
                'type': 'linear',
                'hidden_size': hidden_size,
                'activation': np.random.choice(['relu', 'tanh', 'sigmoid']),
                'use_batch_norm': np.random.random() < 0.3,
                'use_layer_norm': np.random.random() < 0.2,
                'dropout': np.random.uniform(0, 0.5) if np.random.random() < 0.4 else 0
            }
            
            layers.append(layer_spec)
            current_dim = hidden_size
        
        # Calculate total parameters
        total_params = self._calculate_parameters(layers, input_dim)
        
        # Estimate latency (rough approximation)
        estimated_latency = total_params * 0.001  # 1ms per 1000 parameters
        
        return CarbonAwareArchitecture(
            architecture_id=architecture_id,
            layers=layers,
            total_parameters=total_params,
            estimated_latency_ms=estimated_latency
        )
    
    def _calculate_parameters(self, layers: List[Dict], input_dim: int) -> int:
        """Calculate total number of parameters in architecture."""
        
        total_params = 0
        current_dim = input_dim
        
        for layer in layers:
            if layer['type'] == 'linear':
                # Linear layer parameters
                total_params += current_dim * layer['hidden_size'] + layer['hidden_size']
                current_dim = layer['hidden_size']
                
                # Batch norm parameters
                if layer.get('use_batch_norm', False):
                    total_params += current_dim * 2  # gamma and beta
        
        # Final output layer
        total_params += current_dim * 1 + 1
        
        return total_params
    
    def _evaluate_population(self, X: pd.DataFrame, y: pd.Series):
        """Evaluate all architectures in the population."""
        
        logger.info(f"Evaluating {len(self.population)} architectures")
        
        # Use parallel evaluation for speed
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for architecture in self.population:
                if not architecture.is_evaluated:
                    future = executor.submit(self.evaluator.evaluate_architecture, architecture, X, y)
                    futures.append(future)
            
            # Wait for all evaluations to complete
            for future in futures:
                try:
                    future.result(timeout=300)  # 5 minute timeout per architecture
                except Exception as e:
                    logger.error(f"Architecture evaluation failed: {e}")
    
    def _selection(self):
        """Select best architectures based on carbon-aware fitness."""
        
        # Calculate fitness scores
        for architecture in self.population:
            if architecture.is_evaluated:
                architecture.fitness_score = self._calculate_fitness(architecture)
        
        # Sort by fitness
        evaluated_architectures = [a for a in self.population if a.is_evaluated]
        evaluated_architectures.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Keep top 50% for next generation
        num_keep = max(2, len(evaluated_architectures) // 2)
        self.population = evaluated_architectures[:num_keep]
        
        logger.info(f"Selected {len(self.population)} best architectures")
    
    def _calculate_fitness(self, architecture: CarbonAwareArchitecture) -> float:
        """Calculate fitness score based on carbon-aware objectives."""
        
        if not architecture.is_evaluated:
            return 0.0
        
        performance = architecture.performance_metrics.get('roc_auc', 0.0)
        
        if self.config.primary_objective == CarbonOptimizationObjective.MINIMIZE_CARBON_FOOTPRINT:
            # Maximize performance while minimizing carbon
            carbon_penalty = max(0, architecture.estimated_carbon_kg - self.config.max_carbon_budget_kg * 0.5)
            return performance - self.config.carbon_weight * carbon_penalty * 10
        
        elif self.config.primary_objective == CarbonOptimizationObjective.MAXIMIZE_CARBON_EFFICIENCY:
            # Maximize performance per CO2
            if architecture.estimated_carbon_kg > 0:
                return performance / architecture.estimated_carbon_kg
            else:
                return performance
        
        elif self.config.primary_objective == CarbonOptimizationObjective.BALANCED_SUSTAINABILITY:
            # Balanced multi-objective optimization
            carbon_score = max(0, 1 - architecture.estimated_carbon_kg / self.config.max_carbon_budget_kg)
            energy_score = max(0, 1 - architecture.estimated_energy_mj / (self.config.max_energy_budget_kwh * 3.6))
            
            return (self.config.performance_weight * performance +
                   self.config.carbon_weight * carbon_score +
                   self.config.efficiency_weight * energy_score)
        
        else:
            return performance
    
    def _crossover_and_mutation(self, input_dim: int):
        """Generate new architectures through crossover and mutation."""
        
        new_population = list(self.population)  # Keep current population
        
        while len(new_population) < self.config.population_size:
            # Select parents
            parent1 = np.random.choice(self.population)
            parent2 = np.random.choice(self.population)
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child = self._crossover(parent1, parent2, input_dim)
            else:
                child = self._generate_random_architecture(input_dim, f"arch_{len(new_population)}")
            
            # Mutation
            if np.random.random() < self.config.mutation_rate:
                child = self._mutate(child, input_dim)
            
            new_population.append(child)
        
        self.population = new_population
    
    def _crossover(self, parent1: CarbonAwareArchitecture, parent2: CarbonAwareArchitecture, 
                  input_dim: int) -> CarbonAwareArchitecture:
        """Create child architecture through crossover."""
        
        # Simple crossover: take layers from both parents
        child_layers = []
        
        max_layers = max(len(parent1.layers), len(parent2.layers))
        
        for i in range(max_layers):
            if i < len(parent1.layers) and i < len(parent2.layers):
                # Choose layer from either parent
                if np.random.random() < 0.5:
                    child_layers.append(parent1.layers[i].copy())
                else:
                    child_layers.append(parent2.layers[i].copy())
            elif i < len(parent1.layers):
                child_layers.append(parent1.layers[i].copy())
            else:
                child_layers.append(parent2.layers[i].copy())
        
        # Calculate parameters and latency
        total_params = self._calculate_parameters(child_layers, input_dim)
        estimated_latency = total_params * 0.001
        
        return CarbonAwareArchitecture(
            architecture_id=f"crossover_{len(self.population)}",
            layers=child_layers,
            total_parameters=total_params,
            estimated_latency_ms=estimated_latency
        )
    
    def _mutate(self, architecture: CarbonAwareArchitecture, input_dim: int) -> CarbonAwareArchitecture:
        """Apply mutations to architecture."""
        
        mutated_layers = []
        
        for layer in architecture.layers:
            mutated_layer = layer.copy()
            
            # Random mutations
            if np.random.random() < 0.1:  # 10% chance to change layer size
                mutated_layer['hidden_size'] = np.random.choice([32, 64, 128, 256, 512])
            
            if np.random.random() < 0.1:  # 10% chance to change activation
                mutated_layer['activation'] = np.random.choice(['relu', 'tanh', 'sigmoid'])
            
            if np.random.random() < 0.1:  # 10% chance to toggle batch norm
                mutated_layer['use_batch_norm'] = not mutated_layer.get('use_batch_norm', False)
            
            if np.random.random() < 0.1:  # 10% chance to change dropout
                mutated_layer['dropout'] = np.random.uniform(0, 0.5)
            
            mutated_layers.append(mutated_layer)
        
        # Calculate new parameters
        total_params = self._calculate_parameters(mutated_layers, input_dim)
        estimated_latency = total_params * 0.001
        
        return CarbonAwareArchitecture(
            architecture_id=f"mutated_{architecture.architecture_id}",
            layers=mutated_layers,
            total_parameters=total_params,
            estimated_latency_ms=estimated_latency
        )
    
    def _update_best_architecture(self):
        """Update the best architecture found so far."""
        
        evaluated_architectures = [a for a in self.population if a.is_evaluated]
        if not evaluated_architectures:
            return
        
        # Find best by fitness
        best_in_generation = max(evaluated_architectures, key=lambda x: x.fitness_score)
        
        if (self.best_architecture is None or 
            best_in_generation.fitness_score > self.best_architecture.fitness_score):
            self.best_architecture = best_in_generation
            logger.info(f"New best architecture: {self.best_architecture.architecture_id} "
                       f"(fitness: {self.best_architecture.fitness_score:.4f})")
    
    def _log_generation_progress(self):
        """Log progress for current generation."""
        
        evaluated_architectures = [a for a in self.population if a.is_evaluated]
        
        if evaluated_architectures:
            avg_performance = np.mean([a.performance_metrics.get('roc_auc', 0.0) for a in evaluated_architectures])
            avg_carbon = np.mean([a.estimated_carbon_kg for a in evaluated_architectures])
            avg_energy = np.mean([a.estimated_energy_mj for a in evaluated_architectures])
            
            logger.info(f"Generation {self.generation + 1} - "
                       f"Avg Performance: {avg_performance:.4f}, "
                       f"Avg Carbon: {avg_carbon:.4f} kg, "
                       f"Avg Energy: {avg_energy:.4f} MJ")
    
    def _final_evaluation(self, X: pd.DataFrame, y: pd.Series):
        """Final evaluation and ranking of all architectures."""
        
        # Evaluate any remaining unevaluated architectures
        self._evaluate_population(X, y)
        
        # Rank all evaluated architectures
        evaluated_architectures = [a for a in self.population if a.is_evaluated]
        evaluated_architectures.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Update best architecture
        if evaluated_architectures:
            self.best_architecture = evaluated_architectures[0]
        
        logger.info(f"Final evaluation complete. Best architecture: {self.best_architecture.architecture_id}")
    
    def _save_results(self):
        """Save search results and visualizations."""
        
        if not self.config.save_architectures:
            return
        
        # Save best architecture
        if self.best_architecture:
            best_arch_file = self.results_dir / "best_architecture.json"
            with open(best_arch_file, 'w') as f:
                json.dump(self.best_architecture.to_dict(), f, indent=2)
        
        # Save all evaluated architectures
        evaluated_architectures = [a for a in self.population if a.is_evaluated]
        all_architectures_file = self.results_dir / "all_architectures.json"
        with open(all_architectures_file, 'w') as f:
            json.dump([a.to_dict() for a in evaluated_architectures], f, indent=2)
        
        # Save search summary
        summary = {
            "search_config": {
                "primary_objective": self.config.primary_objective.value,
                "carbon_weight": self.config.carbon_weight,
                "performance_weight": self.config.performance_weight,
                "efficiency_weight": self.config.efficiency_weight,
                "max_carbon_budget_kg": self.config.max_carbon_budget_kg,
                "max_energy_budget_kwh": self.config.max_energy_budget_kwh
            },
            "search_results": {
                "total_architectures_evaluated": len(evaluated_architectures),
                "generations": self.config.generations,
                "best_architecture_id": self.best_architecture.architecture_id if self.best_architecture else None,
                "best_fitness_score": self.best_architecture.fitness_score if self.best_architecture else 0.0,
                "best_performance": self.best_architecture.performance_metrics.get('roc_auc', 0.0) if self.best_architecture else 0.0,
                "best_carbon_kg": self.best_architecture.estimated_carbon_kg if self.best_architecture else 0.0,
                "best_energy_mj": self.best_architecture.estimated_energy_mj if self.best_architecture else 0.0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = self.results_dir / "search_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def get_carbon_efficiency_ranking(self) -> List[Tuple[CarbonAwareArchitecture, float]]:
        """Get architectures ranked by carbon efficiency (performance per CO2)."""
        
        evaluated_architectures = [a for a in self.population if a.is_evaluated]
        
        carbon_efficiency_ranking = []
        for arch in evaluated_architectures:
            if arch.estimated_carbon_kg > 0:
                efficiency = arch.performance_metrics.get('roc_auc', 0.0) / arch.estimated_carbon_kg
                carbon_efficiency_ranking.append((arch, efficiency))
        
        carbon_efficiency_ranking.sort(key=lambda x: x[1], reverse=True)
        return carbon_efficiency_ranking
    
    def get_energy_efficiency_ranking(self) -> List[Tuple[CarbonAwareArchitecture, float]]:
        """Get architectures ranked by energy efficiency (performance per kWh)."""
        
        evaluated_architectures = [a for a in self.population if a.is_evaluated]
        
        energy_efficiency_ranking = []
        for arch in evaluated_architectures:
            if arch.estimated_energy_mj > 0:
                efficiency = arch.performance_metrics.get('roc_auc', 0.0) / arch.estimated_energy_mj
                energy_efficiency_ranking.append((arch, efficiency))
        
        energy_efficiency_ranking.sort(key=lambda x: x[1], reverse=True)
        return energy_efficiency_ranking


# Utility functions

def create_carbon_aware_nas(config: Optional[CarbonAwareNASConfig] = None) -> CarbonAwareNAS:
    """Create and configure carbon-aware NAS system."""
    return CarbonAwareNAS(config)


def run_carbon_aware_search(X: pd.DataFrame, y: pd.Series, 
                           config: Optional[CarbonAwareNASConfig] = None) -> CarbonAwareArchitecture:
    """Run carbon-aware neural architecture search."""
    
    nas = create_carbon_aware_nas(config)
    return nas.search(X, y)


def compare_carbon_efficiency(architectures: List[CarbonAwareArchitecture]) -> pd.DataFrame:
    """Compare architectures by carbon efficiency metrics."""
    
    comparison_data = []
    
    for arch in architectures:
        if arch.is_evaluated:
            comparison_data.append({
                'architecture_id': arch.architecture_id,
                'performance_auc': arch.performance_metrics.get('roc_auc', 0.0),
                'carbon_kg': arch.estimated_carbon_kg,
                'energy_mj': arch.estimated_energy_mj,
                'carbon_efficiency': arch.carbon_efficiency_score,
                'energy_efficiency': arch.energy_efficiency_score,
                'total_parameters': arch.total_parameters,
                'training_time': arch.training_time
            })
    
    return pd.DataFrame(comparison_data)

"""
Neural Architecture Search (NAS) framework for automated architecture discovery.
Implements multi-objective NAS optimizing for accuracy, latency, and energy efficiency.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time
import random
from abc import ABC, abstractmethod
import copy

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

try:
    from ..models.dnn_model import DNNModel, DNNTrainer, DNNConfig
    from ..core.interfaces import BaseModel, TrainingMetrics
    from ..core.logging import get_logger, get_audit_logger
    from ..optimization.hyperparameter_tuning import EnergyTracker
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from models.dnn_model import DNNModel, DNNTrainer, DNNConfig
    from core.interfaces import BaseModel, TrainingMetrics
    from core.logging import get_logger, get_audit_logger
    from optimization.hyperparameter_tuning import EnergyTracker

    # Create minimal implementations for testing
    class MockAuditLogger:
        def log_model_operation(self, **kwargs):
            pass

    def get_audit_logger():
        return MockAuditLogger()


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class ArchitectureSpec:
    """Specification for a neural architecture."""

    architecture_id: str
    layers: List[Dict[str, Any]]
    total_params: int = 0
    flops: int = 0
    memory_mb: float = 0.0
    estimated_latency_ms: float = 0.0
    estimated_energy_mj: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    is_evaluated: bool = False


@dataclass
class SearchSpace:
    """Definition of the neural architecture search space."""

    # Layer types
    layer_types: List[str] = field(
        default_factory=lambda: ["linear", "conv1d", "lstm", "attention"]
    )

    # Linear layer options
    linear_hidden_sizes: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512, 1024]
    )
    max_linear_layers: int = 6
    min_linear_layers: int = 2

    # Convolutional layer options
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    max_conv_layers: int = 4

    # LSTM layer options
    lstm_hidden_sizes: List[int] = field(default_factory=lambda: [64, 128, 256])
    lstm_num_layers: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Attention layer options
    attention_heads: List[int] = field(default_factory=lambda: [4, 8, 16])
    attention_dims: List[int] = field(default_factory=lambda: [64, 128, 256])

    # Activation functions
    activations: List[str] = field(
        default_factory=lambda: ["relu", "gelu", "swish", "leaky_relu"]
    )

    # Regularization
    dropout_rates: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    use_batch_norm: List[bool] = field(default_factory=lambda: [True, False])
    use_layer_norm: List[bool] = field(default_factory=lambda: [True, False])

    # Skip connections
    use_skip_connections: List[bool] = field(default_factory=lambda: [True, False])
    skip_connection_types: List[str] = field(
        default_factory=lambda: ["residual", "dense"]
    )

    # Architecture constraints
    max_total_params: int = 10_000_000  # 10M parameters
    max_memory_mb: float = 1000.0  # 1GB memory
    max_latency_ms: float = 100.0  # 100ms inference time


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search."""

    # Search strategy
    search_strategy: str = (
        "evolutionary"  # 'random', 'evolutionary', 'bayesian', 'progressive'
    )

    # Multi-objective optimization
    objectives: List[str] = field(
        default_factory=lambda: ["accuracy", "latency", "energy"]
    )
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {"accuracy": 0.6, "latency": 0.2, "energy": 0.2}
    )

    # Search parameters
    population_size: int = 20
    num_generations: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7

    # Progressive search
    progressive_stages: List[int] = field(
        default_factory=lambda: [5, 10, 20]
    )  # Epochs per stage
    progressive_population_growth: float = 1.5

    # Early stopping
    early_stopping_patience: int = 3
    min_improvement: float = 0.01

    # Evaluation
    max_epochs_per_architecture: int = 20
    validation_split: float = 0.2
    use_weight_sharing: bool = False  # For efficiency

    # Resource constraints
    max_search_time_hours: float = 24.0
    max_architectures_to_evaluate: int = 100

    # Pareto optimization
    pareto_front_size: int = 10
    diversity_threshold: float = 0.1

    # Results storage
    save_results: bool = True
    results_path: str = "nas_results"


@dataclass
class NASResult:
    """Result of Neural Architecture Search."""

    success: bool
    best_architectures: List[ArchitectureSpec]
    pareto_front: List[ArchitectureSpec]
    search_history: List[ArchitectureSpec]
    search_time_seconds: float
    total_architectures_evaluated: int
    best_single_objective_scores: Dict[str, float]
    convergence_history: List[Dict[str, float]]
    search_space: SearchSpace
    config: NASConfig
    message: str


class ArchitectureGenerator(ABC):
    """Abstract base class for architecture generators."""

    @abstractmethod
    def generate_architecture(self, search_space: SearchSpace) -> ArchitectureSpec:
        """Generate a new architecture."""
        pass

    @abstractmethod
    def mutate_architecture(
        self, architecture: ArchitectureSpec, search_space: SearchSpace
    ) -> ArchitectureSpec:
        """Mutate an existing architecture."""
        pass

    @abstractmethod
    def crossover_architectures(
        self,
        parent1: ArchitectureSpec,
        parent2: ArchitectureSpec,
        search_space: SearchSpace,
    ) -> ArchitectureSpec:
        """Create offspring from two parent architectures."""
        pass


class RandomArchitectureGenerator(ArchitectureGenerator):
    """Random architecture generator."""

    def generate_architecture(self, search_space: SearchSpace) -> ArchitectureSpec:
        """Generate a random architecture."""
        layers = []
        architecture_id = f"random_{int(time.time() * 1000000) % 1000000}"

        # Determine number of layers
        num_layers = random.randint(
            search_space.min_linear_layers, search_space.max_linear_layers
        )

        # Generate layers
        for i in range(num_layers):
            layer_type = random.choice(
                ["linear"]
            )  # Start with linear layers for simplicity

            if layer_type == "linear":
                layer_spec = {
                    "type": "linear",
                    "hidden_size": random.choice(search_space.linear_hidden_sizes),
                    "activation": random.choice(search_space.activations),
                    "dropout": random.choice(search_space.dropout_rates),
                    "use_batch_norm": random.choice(search_space.use_batch_norm),
                    "use_layer_norm": random.choice(search_space.use_layer_norm),
                }

            layers.append(layer_spec)

        return ArchitectureSpec(architecture_id=architecture_id, layers=layers)

    def mutate_architecture(
        self, architecture: ArchitectureSpec, search_space: SearchSpace
    ) -> ArchitectureSpec:
        """Mutate an architecture by changing random parameters."""
        mutated = copy.deepcopy(architecture)
        mutated.architecture_id = f"mutated_{int(time.time() * 1000000) % 1000000}"
        mutated.is_evaluated = False
        mutated.performance_metrics = {}

        # Choose random layer to mutate
        if mutated.layers:
            layer_idx = random.randint(0, len(mutated.layers) - 1)
            layer = mutated.layers[layer_idx]

            # Mutate random parameter
            mutation_type = random.choice(
                ["hidden_size", "activation", "dropout", "normalization"]
            )

            if mutation_type == "hidden_size" and layer["type"] == "linear":
                layer["hidden_size"] = random.choice(search_space.linear_hidden_sizes)
            elif mutation_type == "activation":
                layer["activation"] = random.choice(search_space.activations)
            elif mutation_type == "dropout":
                layer["dropout"] = random.choice(search_space.dropout_rates)
            elif mutation_type == "normalization":
                layer["use_batch_norm"] = random.choice(search_space.use_batch_norm)
                layer["use_layer_norm"] = random.choice(search_space.use_layer_norm)

        return mutated

    def crossover_architectures(
        self,
        parent1: ArchitectureSpec,
        parent2: ArchitectureSpec,
        search_space: SearchSpace,
    ) -> ArchitectureSpec:
        """Create offspring by combining two parent architectures."""
        offspring_id = f"crossover_{int(time.time() * 1000000) % 1000000}"

        # Simple crossover: take layers from both parents
        min_layers = min(len(parent1.layers), len(parent2.layers))
        max_layers = max(len(parent1.layers), len(parent2.layers))

        # Choose crossover point
        crossover_point = random.randint(1, min_layers - 1) if min_layers > 1 else 1

        # Create offspring layers
        offspring_layers = []
        offspring_layers.extend(parent1.layers[:crossover_point])
        offspring_layers.extend(parent2.layers[crossover_point:])

        # Ensure we don't exceed max layers
        if len(offspring_layers) > search_space.max_linear_layers:
            offspring_layers = offspring_layers[: search_space.max_linear_layers]

        return ArchitectureSpec(architecture_id=offspring_id, layers=offspring_layers)


class ArchitectureEvaluator:
    """Evaluates neural architectures."""

    def __init__(self, config: NASConfig):
        self.config = config
        self.energy_tracker = EnergyTracker()

    def evaluate_architecture(
        self, architecture: ArchitectureSpec, X: pd.DataFrame, y: pd.Series
    ) -> ArchitectureSpec:
        """
        Evaluate a neural architecture.

        Args:
            architecture: Architecture specification to evaluate
            X: Training features
            y: Training targets

        Returns:
            Updated architecture with performance metrics
        """
        start_time = time.time()

        try:
            # Create model from architecture
            model = self._create_model_from_architecture(architecture, X.shape[1])

            # Calculate model statistics
            architecture.total_params = sum(p.numel() for p in model.parameters())
            architecture.memory_mb = self._estimate_memory_usage(model)
            architecture.estimated_latency_ms = self._estimate_latency(
                model, X.shape[1]
            )

            # Check constraints
            if not self._check_constraints(architecture):
                architecture.performance_metrics = {
                    "accuracy": 0.0,
                    "roc_auc": 0.0,
                    "f1_score": 0.0,
                    "constraint_violation": True,
                }
                architecture.is_evaluated = True
                return architecture

            # Train and evaluate model
            performance_metrics = self._train_and_evaluate_model(model, X, y)

            # Calculate energy consumption
            architecture.estimated_energy_mj = self._estimate_energy_consumption(
                architecture.total_params, self.config.max_epochs_per_architecture
            )

            # Store results
            architecture.performance_metrics = performance_metrics
            architecture.training_time = time.time() - start_time
            architecture.is_evaluated = True

            logger.info(
                f"Evaluated architecture {architecture.architecture_id}: "
                f"AUC={performance_metrics.get('roc_auc', 0.0):.4f}, "
                f"Params={architecture.total_params}, "
                f"Latency={architecture.estimated_latency_ms:.2f}ms"
            )

        except Exception as e:
            logger.error(
                f"Error evaluating architecture {architecture.architecture_id}: {e}"
            )
            architecture.performance_metrics = {
                "accuracy": 0.0,
                "roc_auc": 0.0,
                "f1_score": 0.0,
                "error": str(e),
            }
            architecture.training_time = time.time() - start_time
            architecture.is_evaluated = True

        return architecture

    def _create_model_from_architecture(
        self, architecture: ArchitectureSpec, input_dim: int
    ) -> nn.Module:
        """Create PyTorch model from architecture specification."""
        layers = []
        current_dim = input_dim

        for layer_spec in architecture.layers:
            if layer_spec["type"] == "linear":
                # Linear layer
                layers.append(nn.Linear(current_dim, layer_spec["hidden_size"]))
                current_dim = layer_spec["hidden_size"]

                # Normalization
                if layer_spec.get("use_batch_norm", False):
                    layers.append(nn.BatchNorm1d(current_dim))
                elif layer_spec.get("use_layer_norm", False):
                    layers.append(nn.LayerNorm(current_dim))

                # Activation
                activation = layer_spec.get("activation", "relu")
                if activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "swish":
                    layers.append(nn.SiLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(0.01, inplace=True))

                # Dropout
                dropout_rate = layer_spec.get("dropout", 0.0)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(current_dim, 1))

        return nn.Sequential(*layers)

    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

        # Estimate activation memory (rough approximation)
        activation_memory = param_size * 2  # Rough estimate

        total_bytes = param_size + buffer_size + activation_memory
        return total_bytes / (1024 * 1024)  # Convert to MB

    def _estimate_latency(self, model: nn.Module, input_dim: int) -> float:
        """Estimate inference latency in milliseconds."""
        # Simple estimation based on parameter count and operations
        total_params = sum(p.numel() for p in model.parameters())

        # Rough estimation: 1M parameters ≈ 1ms on modern hardware
        estimated_latency = total_params / 1_000_000

        return max(0.1, estimated_latency)  # Minimum 0.1ms

    def _estimate_energy_consumption(self, num_params: int, epochs: int) -> float:
        """Estimate energy consumption in millijoules."""
        # Simple estimation based on parameter count and training epochs
        # Rough approximation: 1M parameters * 1 epoch ≈ 1 mJ
        return (num_params / 1_000_000) * epochs

    def _check_constraints(self, architecture: ArchitectureSpec) -> bool:
        """Check if architecture satisfies constraints."""
        search_space = SearchSpace()  # Use default constraints

        if architecture.total_params > search_space.max_total_params:
            return False
        if architecture.memory_mb > search_space.max_memory_mb:
            return False
        if architecture.estimated_latency_ms > search_space.max_latency_ms:
            return False

        return True

    def _train_and_evaluate_model(
        self, model: nn.Module, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Train and evaluate the model."""
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.config.validation_split,
                random_state=42,
                stratify=y,
            )

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.FloatTensor(y_train.values)
            X_val_tensor = torch.FloatTensor(X_val.values)
            y_val_tensor = torch.FloatTensor(y_val.values)

            # Setup training
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Training loop
            model.train()
            for epoch in range(self.config.max_epochs_per_architecture):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_probs = torch.sigmoid(val_outputs.squeeze()).numpy()
                val_preds = (val_probs > 0.5).astype(int)

            # Calculate metrics
            metrics = {
                "accuracy": np.mean(val_preds == y_val.values),
                "roc_auc": roc_auc_score(y_val.values, val_probs),
                "f1_score": f1_score(y_val.values, val_preds, average="weighted"),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {"accuracy": 0.0, "roc_auc": 0.0, "f1_score": 0.0}


class NeuralArchitectureSearch:
    """Main Neural Architecture Search class."""

    def __init__(
        self,
        config: Optional[NASConfig] = None,
        search_space: Optional[SearchSpace] = None,
    ):
        self.config = config or NASConfig()
        self.search_space = search_space or SearchSpace()
        self.generator = RandomArchitectureGenerator()
        self.evaluator = ArchitectureEvaluator(self.config)

        # Search state
        self.population: List[ArchitectureSpec] = []
        self.search_history: List[ArchitectureSpec] = []
        self.pareto_front: List[ArchitectureSpec] = []
        self.generation = 0

    def search(self, X: pd.DataFrame, y: pd.Series) -> NASResult:
        """
        Run Neural Architecture Search.

        Args:
            X: Training features
            y: Training targets

        Returns:
            NASResult with search results
        """
        start_time = datetime.now()

        try:
            logger.info("Starting Neural Architecture Search")
            logger.info(f"Search strategy: {self.config.search_strategy}")
            logger.info(f"Population size: {self.config.population_size}")
            logger.info(f"Generations: {self.config.num_generations}")

            # Initialize population
            self._initialize_population()

            # Evaluate initial population
            self._evaluate_population(X, y)

            # Evolution loop
            convergence_history = []

            for generation in range(self.config.num_generations):
                self.generation = generation

                logger.info(
                    f"Generation {generation + 1}/{self.config.num_generations}"
                )

                # Create next generation
                if self.config.search_strategy == "evolutionary":
                    self._evolutionary_step(X, y)
                elif self.config.search_strategy == "random":
                    self._random_step(X, y)
                else:
                    self._evolutionary_step(X, y)  # Default to evolutionary

                # Update Pareto front
                self._update_pareto_front()

                # Record convergence
                best_scores = self._get_best_scores()
                convergence_history.append(best_scores)

                logger.info(
                    f"Best scores - Accuracy: {best_scores.get('accuracy', 0.0):.4f}, "
                    f"Latency: {best_scores.get('latency', 0.0):.2f}ms"
                )

                # Check early stopping
                if self._should_stop_early(convergence_history):
                    logger.info(f"Early stopping at generation {generation + 1}")
                    break

            # Get final results
            best_architectures = self._get_best_architectures()

            search_time = (datetime.now() - start_time).total_seconds()

            # Log completion
            audit_logger.log_model_operation(
                user_id="system",
                model_id="nas_search",
                operation="nas_completed",
                success=True,
                details={
                    "search_time_seconds": search_time,
                    "total_architectures": len(self.search_history),
                    "generations": self.generation + 1,
                    "pareto_front_size": len(self.pareto_front),
                    "best_accuracy": best_scores.get("accuracy", 0.0),
                },
            )

            logger.info(f"NAS completed in {search_time:.2f} seconds")
            logger.info(f"Evaluated {len(self.search_history)} architectures")

            return NASResult(
                success=True,
                best_architectures=best_architectures,
                pareto_front=self.pareto_front,
                search_history=self.search_history,
                search_time_seconds=search_time,
                total_architectures_evaluated=len(self.search_history),
                best_single_objective_scores=best_scores,
                convergence_history=convergence_history,
                search_space=self.search_space,
                config=self.config,
                message="NAS completed successfully",
            )

        except Exception as e:
            search_time = (datetime.now() - start_time).total_seconds()
            error_message = f"NAS failed: {str(e)}"
            logger.error(error_message)

            return NASResult(
                success=False,
                best_architectures=[],
                pareto_front=[],
                search_history=self.search_history,
                search_time_seconds=search_time,
                total_architectures_evaluated=len(self.search_history),
                best_single_objective_scores={},
                convergence_history=[],
                search_space=self.search_space,
                config=self.config,
                message=error_message,
            )

    def _initialize_population(self):
        """Initialize the population with random architectures."""
        self.population = []
        for _ in range(self.config.population_size):
            architecture = self.generator.generate_architecture(self.search_space)
            self.population.append(architecture)

    def _evaluate_population(self, X: pd.DataFrame, y: pd.Series):
        """Evaluate all architectures in the population."""
        for architecture in self.population:
            if not architecture.is_evaluated:
                evaluated_arch = self.evaluator.evaluate_architecture(
                    architecture, X, y
                )
                self.search_history.append(evaluated_arch)

    def _evolutionary_step(self, X: pd.DataFrame, y: pd.Series):
        """Perform one evolutionary step."""
        # Selection
        parents = self._tournament_selection()

        # Create offspring
        offspring = []

        while len(offspring) < self.config.population_size:
            # Crossover
            if random.random() < self.config.crossover_rate and len(parents) >= 2:
                parent1, parent2 = random.sample(parents, 2)
                child = self.generator.crossover_architectures(
                    parent1, parent2, self.search_space
                )
                offspring.append(child)

            # Mutation
            if random.random() < self.config.mutation_rate and parents:
                parent = random.choice(parents)
                mutated = self.generator.mutate_architecture(parent, self.search_space)
                offspring.append(mutated)

            # Random generation
            if len(offspring) < self.config.population_size:
                random_arch = self.generator.generate_architecture(self.search_space)
                offspring.append(random_arch)

        # Evaluate offspring
        for architecture in offspring:
            if not architecture.is_evaluated:
                evaluated_arch = self.evaluator.evaluate_architecture(
                    architecture, X, y
                )
                self.search_history.append(evaluated_arch)

        # Replace population
        self.population = offspring[: self.config.population_size]

    def _random_step(self, X: pd.DataFrame, y: pd.Series):
        """Perform random search step."""
        new_population = []

        for _ in range(self.config.population_size):
            architecture = self.generator.generate_architecture(self.search_space)
            evaluated_arch = self.evaluator.evaluate_architecture(architecture, X, y)
            new_population.append(evaluated_arch)
            self.search_history.append(evaluated_arch)

        self.population = new_population

    def _tournament_selection(self, tournament_size: int = 3) -> List[ArchitectureSpec]:
        """Select parents using tournament selection."""
        parents = []

        for _ in range(self.config.population_size // 2):
            tournament = random.sample(
                self.population, min(tournament_size, len(self.population))
            )
            winner = max(tournament, key=lambda x: self._calculate_fitness(x))
            parents.append(winner)

        return parents

    def _calculate_fitness(self, architecture: ArchitectureSpec) -> float:
        """Calculate fitness score for an architecture."""
        if not architecture.is_evaluated or not architecture.performance_metrics:
            return 0.0

        # Multi-objective fitness calculation
        accuracy = architecture.performance_metrics.get("roc_auc", 0.0)
        latency = architecture.estimated_latency_ms
        energy = architecture.estimated_energy_mj

        # Normalize objectives (lower is better for latency and energy)
        normalized_accuracy = accuracy  # Already 0-1
        normalized_latency = 1.0 / (1.0 + latency / 100.0)  # Normalize around 100ms
        normalized_energy = 1.0 / (1.0 + energy / 10.0)  # Normalize around 10mJ

        # Weighted combination
        fitness = (
            self.config.objective_weights.get("accuracy", 0.6) * normalized_accuracy
            + self.config.objective_weights.get("latency", 0.2) * normalized_latency
            + self.config.objective_weights.get("energy", 0.2) * normalized_energy
        )

        return fitness

    def _update_pareto_front(self):
        """Update the Pareto front with non-dominated solutions."""
        all_evaluated = [arch for arch in self.search_history if arch.is_evaluated]

        if not all_evaluated:
            return

        # Find Pareto front
        pareto_front = []

        for candidate in all_evaluated:
            is_dominated = False

            for other in all_evaluated:
                if self._dominates(other, candidate):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(candidate)

        # Limit size and ensure diversity
        if len(pareto_front) > self.config.pareto_front_size:
            pareto_front = self._select_diverse_solutions(
                pareto_front, self.config.pareto_front_size
            )

        self.pareto_front = pareto_front

    def _dominates(self, arch1: ArchitectureSpec, arch2: ArchitectureSpec) -> bool:
        """Check if arch1 dominates arch2 in multi-objective sense."""
        if not (arch1.is_evaluated and arch2.is_evaluated):
            return False

        # Get objectives (accuracy: higher is better, latency/energy: lower is better)
        acc1 = arch1.performance_metrics.get("roc_auc", 0.0)
        lat1 = arch1.estimated_latency_ms
        eng1 = arch1.estimated_energy_mj

        acc2 = arch2.performance_metrics.get("roc_auc", 0.0)
        lat2 = arch2.estimated_latency_ms
        eng2 = arch2.estimated_energy_mj

        # arch1 dominates arch2 if it's better in all objectives
        better_accuracy = acc1 >= acc2
        better_latency = lat1 <= lat2
        better_energy = eng1 <= eng2

        # At least one objective must be strictly better
        strictly_better = acc1 > acc2 or lat1 < lat2 or eng1 < eng2

        return better_accuracy and better_latency and better_energy and strictly_better

    def _select_diverse_solutions(
        self, solutions: List[ArchitectureSpec], target_size: int
    ) -> List[ArchitectureSpec]:
        """Select diverse solutions from the Pareto front."""
        if len(solutions) <= target_size:
            return solutions

        # Simple diversity selection based on objective space distance
        selected = [solutions[0]]  # Start with first solution
        remaining = solutions[1:]

        while len(selected) < target_size and remaining:
            # Find solution with maximum minimum distance to selected solutions
            best_candidate = None
            best_min_distance = -1

            for candidate in remaining:
                min_distance = float("inf")

                for selected_sol in selected:
                    distance = self._calculate_objective_distance(
                        candidate, selected_sol
                    )
                    min_distance = min(min_distance, distance)

                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def _calculate_objective_distance(
        self, arch1: ArchitectureSpec, arch2: ArchitectureSpec
    ) -> float:
        """Calculate distance between two architectures in objective space."""
        acc1 = arch1.performance_metrics.get("roc_auc", 0.0)
        lat1 = arch1.estimated_latency_ms / 100.0  # Normalize
        eng1 = arch1.estimated_energy_mj / 10.0  # Normalize

        acc2 = arch2.performance_metrics.get("roc_auc", 0.0)
        lat2 = arch2.estimated_latency_ms / 100.0
        eng2 = arch2.estimated_energy_mj / 10.0

        # Euclidean distance in normalized objective space
        distance = np.sqrt((acc1 - acc2) ** 2 + (lat1 - lat2) ** 2 + (eng1 - eng2) ** 2)
        return distance

    def _get_best_scores(self) -> Dict[str, float]:
        """Get best scores for each objective."""
        evaluated_archs = [arch for arch in self.search_history if arch.is_evaluated]

        if not evaluated_archs:
            return {"accuracy": 0.0, "latency": float("inf"), "energy": float("inf")}

        best_accuracy = max(
            arch.performance_metrics.get("roc_auc", 0.0) for arch in evaluated_archs
        )
        best_latency = min(arch.estimated_latency_ms for arch in evaluated_archs)
        best_energy = min(arch.estimated_energy_mj for arch in evaluated_archs)

        return {
            "accuracy": best_accuracy,
            "latency": best_latency,
            "energy": best_energy,
        }

    def _get_best_architectures(self) -> List[ArchitectureSpec]:
        """Get the best architectures from search."""
        if self.pareto_front:
            return self.pareto_front

        # Fallback: return top architectures by fitness
        evaluated_archs = [arch for arch in self.search_history if arch.is_evaluated]
        evaluated_archs.sort(key=self._calculate_fitness, reverse=True)

        return evaluated_archs[: min(5, len(evaluated_archs))]

    def _should_stop_early(self, convergence_history: List[Dict[str, float]]) -> bool:
        """Check if search should stop early."""
        if len(convergence_history) < self.config.early_stopping_patience:
            return False

        # Check if accuracy hasn't improved significantly
        recent_scores = [
            h.get("accuracy", 0.0)
            for h in convergence_history[-self.config.early_stopping_patience :]
        ]

        if len(recent_scores) < 2:
            return False

        max_recent = max(recent_scores)
        min_recent = min(recent_scores)

        improvement = max_recent - min_recent

        return improvement < self.config.min_improvement


# Utility functions
def run_nas_search(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[NASConfig] = None,
    search_space: Optional[SearchSpace] = None,
) -> NASResult:
    """
    Convenience function to run NAS search.

    Args:
        X: Training features
        y: Training targets
        config: NAS configuration
        search_space: Architecture search space

    Returns:
        NASResult with search results
    """
    nas = NeuralArchitectureSearch(config, search_space)
    return nas.search(X, y)


def get_default_nas_config() -> NASConfig:
    """Get default NAS configuration."""
    return NASConfig()


def get_fast_nas_config() -> NASConfig:
    """Get fast NAS configuration for testing."""
    return NASConfig(
        population_size=5,
        num_generations=3,
        max_epochs_per_architecture=5,
        max_architectures_to_evaluate=15,
        early_stopping_patience=2,
    )


def get_comprehensive_nas_config() -> NASConfig:
    """Get comprehensive NAS configuration for production."""
    return NASConfig(
        population_size=50,
        num_generations=20,
        max_epochs_per_architecture=50,
        max_architectures_to_evaluate=500,
        max_search_time_hours=48.0,
        pareto_front_size=20,
    )


def get_sustainable_search_space() -> SearchSpace:
    """Get search space optimized for sustainable architectures."""
    return SearchSpace(
        linear_hidden_sizes=[32, 64, 128, 256],  # Smaller sizes for efficiency
        max_linear_layers=4,  # Fewer layers
        dropout_rates=[0.1, 0.2, 0.3],  # Regularization for smaller models
        activations=["relu", "gelu"],  # Efficient activations
        max_total_params=1_000_000,  # 1M parameter limit
        max_memory_mb=100.0,  # 100MB memory limit
        max_latency_ms=10.0,  # 10ms latency limit
    )


def create_architecture_from_config(config: Dict[str, Any]) -> ArchitectureSpec:
    """
    Create architecture specification from configuration dictionary.

    Args:
        config: Configuration dictionary with layer specifications

    Returns:
        ArchitectureSpec object
    """
    layers = []

    # Parse configuration
    hidden_layers = config.get("hidden_layers", [64, 32])
    activation = config.get("activation", "relu")
    dropout_rate = config.get("dropout_rate", 0.2)
    use_batch_norm = config.get("use_batch_norm", True)

    # Create layer specifications
    for hidden_size in hidden_layers:
        layer_spec = {
            "type": "linear",
            "hidden_size": hidden_size,
            "activation": activation,
            "dropout": dropout_rate,
            "use_batch_norm": use_batch_norm,
            "use_layer_norm": False,
        }
        layers.append(layer_spec)

    return ArchitectureSpec(
        architecture_id=f"config_{hash(str(config)) % 1000000}", layers=layers
    )


def export_architecture_to_config(architecture: ArchitectureSpec) -> Dict[str, Any]:
    """
    Export architecture specification to configuration dictionary.

    Args:
        architecture: Architecture specification

    Returns:
        Configuration dictionary
    """
    config = {
        "architecture_id": architecture.architecture_id,
        "hidden_layers": [],
        "activations": [],
        "dropout_rates": [],
        "use_batch_norm": [],
        "total_params": architecture.total_params,
        "performance_metrics": architecture.performance_metrics,
    }

    for layer in architecture.layers:
        if layer["type"] == "linear":
            config["hidden_layers"].append(layer["hidden_size"])
            config["activations"].append(layer.get("activation", "relu"))
            config["dropout_rates"].append(layer.get("dropout", 0.0))
            config["use_batch_norm"].append(layer.get("use_batch_norm", False))

    return config


def save_nas_results(result: NASResult, path: Optional[str] = None) -> str:
    """
    Save NAS results to disk.

    Args:
        result: NAS result to save
        path: Save path (optional)

    Returns:
        Path where results were saved
    """
    save_path = path or result.config.results_path
    results_dir = Path(save_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save main results
    results_data = {
        "success": result.success,
        "search_time_seconds": result.search_time_seconds,
        "total_architectures_evaluated": result.total_architectures_evaluated,
        "best_single_objective_scores": result.best_single_objective_scores,
        "convergence_history": result.convergence_history,
        "config": result.config.__dict__,
        "search_space": result.search_space.__dict__,
        "message": result.message,
    }

    results_file = results_dir / "nas_results.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    # Save best architectures
    best_archs = []
    for arch in result.best_architectures:
        arch_data = {
            "architecture_id": arch.architecture_id,
            "layers": arch.layers,
            "total_params": arch.total_params,
            "estimated_latency_ms": arch.estimated_latency_ms,
            "estimated_energy_mj": arch.estimated_energy_mj,
            "performance_metrics": arch.performance_metrics,
            "training_time": arch.training_time,
        }
        best_archs.append(arch_data)

    best_archs_file = results_dir / "best_architectures.json"
    with open(best_archs_file, "w") as f:
        json.dump(best_archs, f, indent=2, default=str)

    # Save Pareto front
    pareto_data = []
    for arch in result.pareto_front:
        arch_data = {
            "architecture_id": arch.architecture_id,
            "layers": arch.layers,
            "total_params": arch.total_params,
            "estimated_latency_ms": arch.estimated_latency_ms,
            "estimated_energy_mj": arch.estimated_energy_mj,
            "performance_metrics": arch.performance_metrics,
        }
        pareto_data.append(arch_data)

    pareto_file = results_dir / "pareto_front.json"
    with open(pareto_file, "w") as f:
        json.dump(pareto_data, f, indent=2, default=str)

    logger.info(f"NAS results saved to {results_dir}")
    return str(results_dir)


def load_nas_results(path: str) -> Dict[str, Any]:
    """
    Load NAS results from disk.

    Args:
        path: Path to results directory

    Returns:
        Dictionary with loaded results
    """
    results_dir = Path(path)

    # Load main results
    results_file = results_dir / "nas_results.json"
    with open(results_file, "r") as f:
        results_data = json.load(f)

    # Load best architectures
    best_archs_file = results_dir / "best_architectures.json"
    if best_archs_file.exists():
        with open(best_archs_file, "r") as f:
            results_data["best_architectures"] = json.load(f)

    # Load Pareto front
    pareto_file = results_dir / "pareto_front.json"
    if pareto_file.exists():
        with open(pareto_file, "r") as f:
            results_data["pareto_front"] = json.load(f)

    return results_data

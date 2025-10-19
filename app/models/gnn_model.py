"""
Graph Neural Network (GNN) implementation for credit risk prediction.
Uses PyTorch Geometric for graph operations and includes graph construction,
convolution layers, attention mechanisms, and mixed precision training.
"""

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GraphConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import from_networkx, to_networkx

try:
    from ..core.interfaces import BaseModel, TrainingMetrics
    from ..core.logging import get_audit_logger, get_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.interfaces import BaseModel, TrainingMetrics
    from core.logging import get_audit_logger, get_logger

    # Create minimal implementations for testing
    class MockAuditLogger:
        def log_model_operation(self, **kwargs):
            pass

    def get_audit_logger():
        return MockAuditLogger()


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class GNNConfig:
    """Configuration for Graph Neural Network model."""

    # Graph construction parameters
    k_neighbors: int = 5  # Number of neighbors for k-NN graph
    graph_construction_method: str = "knn"  # 'knn', 'threshold', 'correlation'
    similarity_threshold: float = 0.7
    correlation_threshold: float = 0.5

    # Architecture parameters
    input_dim: int = 14
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    conv_type: str = "gat"  # 'gcn', 'gat', 'graph_conv'
    num_heads: int = 4  # For GAT layers
    dropout_rate: float = 0.3

    # Graph pooling parameters
    pooling_method: str = "attention"  # 'mean', 'max', 'add', 'attention'
    attention_dim: int = 64

    # Output layers
    output_hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    use_batch_norm: bool = True
    use_layer_norm: bool = False

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    min_delta: float = 1e-4

    # Optimization parameters
    optimizer: str = "adam"  # 'adam', 'adamw', 'sgd'
    weight_decay: float = 1e-4
    gradient_clip_value: float = 1.0

    # Loss function parameters
    loss_function: str = "focal"  # 'bce', 'focal', 'weighted_bce'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # 'onecycle', 'cosine', 'step', 'plateau'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Mixed precision training
    use_mixed_precision: bool = True

    # Regularization
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0

    # Model saving
    save_model: bool = True
    model_path: str = "models/gnn"
    save_best_only: bool = True

    # Device configuration
    device: str = "auto"  # 'auto', 'cpu', 'cuda', 'mps'


@dataclass
class GNNResult:
    """Result of GNN training and evaluation."""

    success: bool
    model: Optional["GNNModel"]
    config: GNNConfig
    training_metrics: List[TrainingMetrics]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    graph_statistics: Dict[str, Any]
    training_time_seconds: float
    model_path: Optional[str]
    best_epoch: int
    message: str


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class AttentionPooling(nn.Module):
    """Attention-based graph pooling layer."""

    def __init__(self, input_dim: int, attention_dim: int):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, input_dim)
            batch: Batch assignment for each node (num_nodes,)

        Returns:
            pooled: Graph-level representations (batch_size, input_dim)
        """
        # Calculate attention scores
        attention_scores = self.attention(x)  # (num_nodes, 1)
        attention_weights = torch.softmax(attention_scores, dim=0)  # (num_nodes, 1)

        # Apply attention weights and pool
        weighted_features = x * attention_weights  # (num_nodes, input_dim)

        # Sum over nodes in each graph
        pooled = torch_geometric.utils.scatter(
            weighted_features, batch, dim=0, reduce="sum"
        )

        return pooled


class GraphConstructor:
    """Constructs graphs from tabular data using various methods."""

    def __init__(self, method: str = "knn", k: int = 5, threshold: float = 0.7):
        self.method = method
        self.k = k
        self.threshold = threshold

    def construct_graph(self, X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct graph from feature matrix.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_feature_dim)
        """
        if self.method == "knn":
            return self._construct_knn_graph(X)
        elif self.method == "threshold":
            return self._construct_threshold_graph(X)
        elif self.method == "correlation":
            return self._construct_correlation_graph(X)
        else:
            raise ValueError(f"Unknown graph construction method: {self.method}")

    def _construct_knn_graph(self, X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct k-nearest neighbors graph."""
        # Build k-NN graph
        knn_graph = kneighbors_graph(
            X, n_neighbors=self.k, mode="connectivity", include_self=False
        )

        # Convert to edge list
        edge_list = []
        edge_weights = []

        # Get non-zero entries (edges)
        rows, cols = knn_graph.nonzero()

        for i, j in zip(rows, cols):
            # Calculate edge weight as similarity (inverse of distance)
            distance = np.linalg.norm(X[i] - X[j])
            weight = 1.0 / (1.0 + distance)

            edge_list.append([i, j])
            edge_weights.append(weight)

        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        return edge_index, edge_attr

    def _construct_threshold_graph(
        self, X: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct graph based on similarity threshold."""
        n_samples = X.shape[0]
        edge_list = []
        edge_weights = []

        # Calculate pairwise similarities
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Cosine similarity
                similarity = np.dot(X[i], X[j]) / (
                    np.linalg.norm(X[i]) * np.linalg.norm(X[j])
                )

                if similarity > self.threshold:
                    edge_list.extend([[i, j], [j, i]])  # Undirected graph
                    edge_weights.extend([similarity, similarity])

        if not edge_list:
            # If no edges, create a minimal connected graph
            for i in range(min(5, n_samples - 1)):
                edge_list.extend([[i, i + 1], [i + 1, i]])
                edge_weights.extend([0.5, 0.5])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        return edge_index, edge_attr

    def _construct_correlation_graph(
        self, X: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct graph based on feature correlations."""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X)

        edge_list = []
        edge_weights = []

        n_samples = X.shape[0]
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                correlation = abs(corr_matrix[i, j])

                if correlation > self.threshold and not np.isnan(correlation):
                    edge_list.extend([[i, j], [j, i]])
                    edge_weights.extend([correlation, correlation])

        if not edge_list:
            # Fallback to k-NN if no correlations found
            return self._construct_knn_graph(X)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        return edge_index, edge_attr


class GNNModel(BaseModel):
    """Graph Neural Network model implementation."""

    def __init__(self, config: Optional[GNNConfig] = None):
        super(GNNModel, self).__init__()
        self.config = config or GNNConfig()
        self.device = self._get_device()

        # Build network architecture
        self._build_network()

        # Initialize weights
        self._initialize_weights()

        # Move to device
        self.to(self.device)

        # Training state
        self.is_trained = False
        self.feature_names = None
        self.scaler = StandardScaler()
        self.graph_constructor = GraphConstructor(
            method=self.config.graph_construction_method,
            k=self.config.k_neighbors,
            threshold=self.config.similarity_threshold,
        )
        self.best_state_dict = None
        self.training_history = []

    def _get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)

    def _build_network(self):
        """Build the GNN architecture."""
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()

        input_dim = self.config.input_dim
        for hidden_dim in self.config.hidden_dims:
            if self.config.conv_type == "gcn":
                conv = GCNConv(input_dim, hidden_dim)
            elif self.config.conv_type == "gat":
                conv = GATConv(
                    input_dim,
                    hidden_dim // self.config.num_heads,
                    heads=self.config.num_heads,
                    dropout=self.config.dropout_rate,
                )
            elif self.config.conv_type == "graph_conv":
                conv = GraphConv(input_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown convolution type: {self.config.conv_type}")

            self.conv_layers.append(conv)
            input_dim = hidden_dim

        # Graph pooling
        if self.config.pooling_method == "attention":
            self.pooling = AttentionPooling(input_dim, self.config.attention_dim)
        else:
            self.pooling = None

        # Output layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.config.output_hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.config.dropout_rate))

            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.output_layers = nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the GNN.

        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features (num_nodes, input_dim)
                - edge_index: Edge indices (2, num_edges)
                - edge_attr: Edge attributes (num_edges, edge_feature_dim)
                - batch: Batch assignment for each node (num_nodes,)

        Returns:
            output: Graph-level predictions (batch_size,)
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Apply graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            if self.config.conv_type == "gat":
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_attr)

            x = F.relu(x)
            x = F.dropout(x, p=self.config.dropout_rate, training=self.training)

        # Graph pooling
        if self.config.pooling_method == "attention":
            x = self.pooling(x, batch)
        elif self.config.pooling_method == "mean":
            x = global_mean_pool(x, batch)
        elif self.config.pooling_method == "max":
            x = global_max_pool(x, batch)
        elif self.config.pooling_method == "add":
            x = global_add_pool(x, batch)

        # Apply output layers
        x = self.output_layers(x)

        return x.squeeze(-1)

    def predict_proba(self, data_list: List[Data]) -> torch.Tensor:
        """Get prediction probabilities."""
        self.eval()
        with torch.no_grad():
            # Create batch from data list
            batch = Batch.from_data_list(data_list).to(self.device)

            logits = self.forward(batch)
            probabilities = torch.sigmoid(logits)

            # Return as 2D array for binary classification
            neg_probs = 1 - probabilities
            return torch.stack([neg_probs, probabilities], dim=1)

    def predict(self, data_list: List[Data], threshold: float = 0.5) -> torch.Tensor:
        """Make binary predictions."""
        probs = self.predict_proba(data_list)
        return (probs[:, 1] > threshold).long()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance using gradient-based method."""
        if not self.is_trained:
            return {}

        try:
            self.eval()

            # Create dummy graph data
            dummy_x = torch.randn(
                10, self.config.input_dim, requires_grad=True, device=self.device
            )
            dummy_edge_index = torch.tensor(
                [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], device=self.device
            )
            dummy_edge_attr = torch.ones(5, 1, device=self.device)
            dummy_batch = torch.zeros(10, dtype=torch.long, device=self.device)

            dummy_data = Data(
                x=dummy_x,
                edge_index=dummy_edge_index,
                edge_attr=dummy_edge_attr,
                batch=dummy_batch,
            )

            output = self.forward(dummy_data)

            # Compute gradients
            gradients = torch.autograd.grad(output.sum(), dummy_x, create_graph=False)[
                0
            ]
            importance_scores = torch.abs(gradients).mean(dim=0).cpu().numpy()

            # Create feature importance dictionary
            feature_importance = {}
            for i, score in enumerate(importance_scores):
                feature_name = (
                    self.feature_names[i] if self.feature_names else f"feature_{i}"
                )
                feature_importance[feature_name] = float(score)

            # Sort by importance
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            return feature_importance

        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}")
            # Fallback: uniform importance
            if self.feature_names:
                return {
                    name: 1.0 / len(self.feature_names) for name in self.feature_names
                }
            else:
                return {
                    f"feature_{i}": 1.0 / self.config.input_dim
                    for i in range(self.config.input_dim)
                }

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        # Create save path
        save_path = path or self.config.model_path
        model_dir = Path(save_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_file = model_dir / "gnn_model.pth"
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "best_state_dict": self.best_state_dict,
                "config": self.config,
                "feature_names": self.feature_names,
                "scaler_mean": (
                    self.scaler.mean_.tolist()
                    if hasattr(self.scaler, "mean_")
                    else None
                ),
                "scaler_scale": (
                    self.scaler.scale_.tolist()
                    if hasattr(self.scaler, "scale_")
                    else None
                ),
                "training_history": self.training_history,
                "device": str(self.device),
            },
            model_file,
        )

        # Save metadata
        metadata = {
            "model_type": "gnn",
            "input_dim": self.config.input_dim,
            "hidden_dims": self.config.hidden_dims,
            "conv_type": self.config.conv_type,
            "pooling_method": self.config.pooling_method,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "config": self.config.__dict__,
            "saved_at": datetime.now().isoformat(),
        }

        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {model_file}")
        return str(model_file)

    def load_model(self, path: str) -> "GNNModel":
        """Load a trained model."""
        model_path = Path(path)

        if model_path.is_file():
            model_file = model_path
        else:
            model_file = model_path / "gnn_model.pth"

        # Load model
        checkpoint = torch.load(
            model_file, map_location=self.device, weights_only=False
        )

        # Restore configuration and architecture
        self.config = checkpoint["config"]
        self.feature_names = checkpoint.get("feature_names")
        self.training_history = checkpoint.get("training_history", [])

        # Rebuild network with loaded config
        self._build_network()

        # Load state dict
        self.load_state_dict(checkpoint["model_state_dict"])
        self.best_state_dict = checkpoint.get("best_state_dict")

        # Restore scaler
        if "scaler_mean" in checkpoint and "scaler_scale" in checkpoint:
            if checkpoint["scaler_mean"] is not None:
                self.scaler.mean_ = np.array(checkpoint["scaler_mean"])
            if checkpoint["scaler_scale"] is not None:
                self.scaler.scale_ = np.array(checkpoint["scaler_scale"])

        self.is_trained = True
        self.to(self.device)

        logger.info(f"Model loaded from {model_file}")
        return self


class GraphDataset(Dataset):
    """Dataset for graph data."""

    def __init__(self, data_list: List[Data]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_graphs(batch):
    """Collate function for graph data."""
    return Batch.from_data_list(batch)


class GNNTrainer:
    """Trainer for Graph Neural Network models."""

    def __init__(self, config: Optional[GNNConfig] = None):
        self.config = config or GNNConfig()

    def train_and_evaluate(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> GNNResult:
        """Train and evaluate GNN model."""
        start_time = datetime.now()

        try:
            logger.info("Starting GNN training and evaluation")

            # Prepare data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

            logger.info(
                f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
            )

            # Update config with actual input dimension
            self.config.input_dim = X_train.shape[1]

            # Create and train model
            model = GNNModel(self.config)
            model.feature_names = list(X_train.columns)

            # Convert data to graphs
            train_graphs, train_targets = self._create_graph_data(
                model, X_train, y_train
            )
            val_graphs, val_targets = self._create_graph_data(model, X_val, y_val)
            test_graphs, test_targets = self._create_graph_data(model, X_test, y_test)

            # Calculate graph statistics
            graph_stats = self._calculate_graph_statistics(train_graphs)
            logger.info(f"Graph statistics: {graph_stats}")

            # Train model
            training_metrics = self._train_model(
                model, train_graphs, train_targets, val_graphs, val_targets
            )

            # Evaluate model
            validation_metrics = self._evaluate_model(
                model, val_graphs, val_targets, "Validation"
            )
            test_metrics = self._evaluate_model(
                model, test_graphs, test_targets, "Test"
            )

            # Get feature importance
            feature_importance = model.get_feature_importance()

            # Save model if requested
            model_path = None
            if self.config.save_model:
                model_path = model.save_model()

            training_time = (datetime.now() - start_time).total_seconds()

            # Find best epoch
            best_epoch = 0
            if training_metrics:
                try:
                    best_auc = max(training_metrics, key=lambda x: x.auc_roc)
                    best_epoch = best_auc.epoch
                except (ValueError, AttributeError):
                    best_epoch = len(training_metrics) - 1

            # Log training completion
            audit_logger.log_model_operation(
                user_id="system",
                model_id="gnn_baseline",
                operation="training_completed",
                success=True,
                details={
                    "training_time_seconds": training_time,
                    "test_auc": test_metrics.get("roc_auc", 0.0),
                    "best_epoch": best_epoch,
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "graph_statistics": graph_stats,
                },
            )

            logger.info(f"GNN training completed in {training_time:.2f} seconds")

            return GNNResult(
                success=True,
                model=model,
                config=self.config,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                feature_importance=feature_importance,
                graph_statistics=graph_stats,
                training_time_seconds=training_time,
                model_path=model_path,
                best_epoch=best_epoch,
                message="GNN training completed successfully",
            )

        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            error_message = f"GNN training failed: {str(e)}"
            logger.error(error_message)

            return GNNResult(
                success=False,
                model=None,
                config=self.config,
                training_metrics=[],
                validation_metrics={},
                test_metrics={},
                feature_importance={},
                graph_statistics={},
                training_time_seconds=training_time,
                model_path=None,
                best_epoch=0,
                message=error_message,
            )

    def _create_graph_data(
        self, model: GNNModel, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[List[Data], torch.Tensor]:
        """Convert tabular data to graph format."""
        # Scale features
        X_scaled = (
            model.scaler.fit_transform(X)
            if not model.is_trained
            else model.scaler.transform(X)
        )

        # Construct graph
        edge_index, edge_attr = model.graph_constructor.construct_graph(X_scaled)

        # Create node features (each sample becomes a node)
        node_features = torch.FloatTensor(X_scaled)
        targets = torch.FloatTensor(y.values)

        # Create individual graphs for each sample (ego graphs)
        graphs = []
        for i in range(len(X_scaled)):
            # Create ego graph centered on node i
            ego_nodes = [i]

            # Find neighbors of node i
            neighbors = []
            for edge_idx in range(edge_index.shape[1]):
                if edge_index[0, edge_idx] == i:
                    neighbors.append(edge_index[1, edge_idx].item())
                elif edge_index[1, edge_idx] == i:
                    neighbors.append(edge_index[0, edge_idx].item())

            # Limit number of neighbors
            neighbors = list(set(neighbors))[: min(10, len(neighbors))]
            ego_nodes.extend(neighbors)

            # Create subgraph
            ego_nodes = list(set(ego_nodes))
            node_mapping = {
                old_idx: new_idx for new_idx, old_idx in enumerate(ego_nodes)
            }

            # Extract subgraph edges
            sub_edge_index = []
            sub_edge_attr = []

            for edge_idx in range(edge_index.shape[1]):
                src, dst = (
                    edge_index[0, edge_idx].item(),
                    edge_index[1, edge_idx].item(),
                )
                if src in node_mapping and dst in node_mapping:
                    sub_edge_index.append([node_mapping[src], node_mapping[dst]])
                    sub_edge_attr.append(edge_attr[edge_idx].item())

            if not sub_edge_index:
                # Create self-loop if no edges
                sub_edge_index = [[0, 0]]
                sub_edge_attr = [1.0]

            # Create Data object
            sub_x = node_features[ego_nodes]
            sub_edge_index = (
                torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()
            )
            sub_edge_attr = torch.tensor(sub_edge_attr, dtype=torch.float).unsqueeze(1)

            graph_data = Data(
                x=sub_x,
                edge_index=sub_edge_index,
                edge_attr=sub_edge_attr,
                y=targets[i],
            )

            graphs.append(graph_data)

        return graphs, targets

    def _calculate_graph_statistics(self, graphs: List[Data]) -> Dict[str, Any]:
        """Calculate statistics about the constructed graphs."""
        num_nodes = [graph.x.shape[0] for graph in graphs]
        num_edges = [graph.edge_index.shape[1] for graph in graphs]

        stats = {
            "num_graphs": len(graphs),
            "avg_nodes_per_graph": np.mean(num_nodes),
            "avg_edges_per_graph": np.mean(num_edges),
            "min_nodes": np.min(num_nodes),
            "max_nodes": np.max(num_nodes),
            "min_edges": np.min(num_edges),
            "max_edges": np.max(num_edges),
            "total_nodes": np.sum(num_nodes),
            "total_edges": np.sum(num_edges),
        }

        return stats

    def _train_model(
        self,
        model: GNNModel,
        train_graphs: List[Data],
        train_targets: torch.Tensor,
        val_graphs: List[Data],
        val_targets: torch.Tensor,
    ) -> List[TrainingMetrics]:
        """Train the GNN model."""

        # Create data loaders
        train_dataset = GraphDataset(train_graphs)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_graphs,
        )

        val_dataset = GraphDataset(val_graphs)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
        )

        # Setup loss function
        if self.config.loss_function == "focal":
            criterion = FocalLoss(
                alpha=self.config.focal_alpha, gamma=self.config.focal_gamma
            )
        elif self.config.loss_function == "weighted_bce":
            pos_weight = torch.tensor(
                [len(train_targets) / (2 * sum(train_targets))]
            ).to(model.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Setup optimizer
        if self.config.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )

        # Setup scheduler
        scheduler = None
        if self.config.use_scheduler:
            if self.config.scheduler_type == "onecycle":
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.config.learning_rate,
                    steps_per_epoch=len(train_loader),
                    epochs=self.config.epochs,
                )
            elif self.config.scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.config.epochs
                )
            elif self.config.scheduler_type == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=30, gamma=self.config.scheduler_factor
                )
            elif self.config.scheduler_type == "plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=self.config.scheduler_patience,
                    factor=self.config.scheduler_factor,
                )

        # Mixed precision scaler
        scaler = GradScaler() if self.config.use_mixed_precision else None

        # Training loop
        training_metrics = []
        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                batch = batch.to(model.device)
                optimizer.zero_grad()

                if self.config.use_mixed_precision and scaler is not None:
                    with autocast():
                        outputs = model(batch)
                        loss = criterion(outputs, batch.y)

                        # Add L1/L2 regularization
                        if self.config.l1_lambda > 0:
                            l1_reg = sum(p.abs().sum() for p in model.parameters())
                            loss += self.config.l1_lambda * l1_reg

                        if self.config.l2_lambda > 0:
                            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
                            loss += self.config.l2_lambda * l2_reg

                    scaler.scale(loss).backward()

                    # Gradient clipping
                    if self.config.gradient_clip_value > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.gradient_clip_value
                        )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch)
                    loss = criterion(outputs, batch.y)

                    # Add L1/L2 regularization
                    if self.config.l1_lambda > 0:
                        l1_reg = sum(p.abs().sum() for p in model.parameters())
                        loss += self.config.l1_lambda * l1_reg

                    if self.config.l2_lambda > 0:
                        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
                        loss += self.config.l2_lambda * l2_reg

                    loss.backward()

                    # Gradient clipping
                    if self.config.gradient_clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.gradient_clip_value
                        )

                    optimizer.step()

                train_loss += loss.item()

                # Update scheduler (for OneCycleLR)
                if scheduler and self.config.scheduler_type == "onecycle":
                    scheduler.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets_list = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(model.device)
                    outputs = model(batch)
                    loss = criterion(outputs, batch.y)
                    val_loss += loss.item()

                    probs = torch.sigmoid(outputs).cpu().numpy()
                    val_predictions.extend(probs)
                    val_targets_list.extend(batch.y.cpu().numpy())

            # Calculate metrics
            val_predictions = np.array(val_predictions)
            val_targets_array = np.array(val_targets_list)
            val_pred_binary = (val_predictions > 0.5).astype(int)

            val_auc = roc_auc_score(val_targets_array, val_predictions)
            val_f1 = f1_score(val_targets_array, val_pred_binary, average="weighted")
            val_precision = precision_score(
                val_targets_array, val_pred_binary, average="weighted", zero_division=0
            )
            val_recall = recall_score(
                val_targets_array, val_pred_binary, average="weighted", zero_division=0
            )

            # Create training metrics
            metrics = TrainingMetrics(
                experiment_id=f"gnn_epoch_{epoch}",
                model_type="gnn",
                epoch=epoch,
                train_loss=train_loss / len(train_loader),
                val_loss=val_loss / len(val_loader),
                auc_roc=val_auc,
                f1_score=val_f1,
                precision=val_precision,
                recall=val_recall,
                energy_consumed_kwh=0.0,  # TODO: Integrate with sustainability monitor
                carbon_emissions_kg=0.0,
                training_time_seconds=0.0,
            )
            training_metrics.append(metrics)

            # Update scheduler (except OneCycleLR)
            if scheduler and self.config.scheduler_type != "onecycle":
                if self.config.scheduler_type == "plateau":
                    scheduler.step(val_loss / len(val_loader))
                else:
                    scheduler.step()

            # Early stopping and best model saving
            if val_auc > best_val_auc + self.config.min_delta:
                best_val_auc = val_auc
                patience_counter = 0
                if self.config.save_best_only:
                    model.best_state_dict = model.state_dict().copy()
            else:
                patience_counter += 1

            # Log progress
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                logger.info(
                    f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                    f"Val Loss: {val_loss/len(val_loader):.4f}, Val AUC: {val_auc:.4f}"
                )

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model if saved
        if model.best_state_dict is not None:
            model.load_state_dict(model.best_state_dict)

        model.is_trained = True
        model.training_history = training_metrics

        return training_metrics

    def _evaluate_model(
        self,
        model: GNNModel,
        graphs: List[Data],
        targets: torch.Tensor,
        dataset_name: str,
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        try:
            # Make predictions
            model.eval()
            probs = model.predict_proba(graphs)
            predictions = model.predict(graphs)

            # Convert to numpy
            probs_np = probs.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(targets_np, predictions_np),
                "precision": precision_score(
                    targets_np, predictions_np, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    targets_np, predictions_np, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    targets_np, predictions_np, average="weighted", zero_division=0
                ),
                "roc_auc": roc_auc_score(targets_np, probs_np[:, 1]),
            }

            logger.info(
                f"{dataset_name} Metrics - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed for {dataset_name}: {e}")
            return {}


# Factory functions and utilities
def create_gnn_model(config: Optional[GNNConfig] = None) -> GNNModel:
    """Create a GNN model instance."""
    return GNNModel(config)


def train_gnn_baseline(
    X: pd.DataFrame, y: pd.Series, config: Optional[GNNConfig] = None
) -> GNNResult:
    """Convenience function to train GNN baseline."""
    trainer = GNNTrainer(config)
    return trainer.train_and_evaluate(X, y)


def get_default_gnn_config() -> GNNConfig:
    """Get default GNN configuration."""
    return GNNConfig()


def get_fast_gnn_config() -> GNNConfig:
    """Get fast GNN configuration for testing."""
    return GNNConfig(
        hidden_dims=[64, 32],
        epochs=20,
        early_stopping_patience=5,
        batch_size=16,
        use_mixed_precision=False,
        k_neighbors=3,
    )


def get_optimized_gnn_config() -> GNNConfig:
    """Get optimized GNN configuration for production."""
    return GNNConfig(
        hidden_dims=[256, 128, 64, 32],
        epochs=200,
        early_stopping_patience=25,
        batch_size=64,
        learning_rate=0.0005,
        use_mixed_precision=True,
        use_scheduler=True,
        scheduler_type="cosine",
        dropout_rate=0.4,
        weight_decay=1e-3,
        k_neighbors=8,
        conv_type="gat",
        num_heads=8,
        pooling_method="attention",
    )

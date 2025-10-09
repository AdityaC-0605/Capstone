"""
Experiment tracking and model registry system with MLflow integration.
Provides reproducibility framework, model lineage tracking, and experiment comparison.
"""

import os
import json
import pickle
import hashlib
import platform
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# MLflow imports (with fallback)
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from ..core.interfaces import DataProcessor
from ..core.config import get_config
from ..core.logging import get_logger, get_audit_logger


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    # MLflow settings
    tracking_uri: Optional[str] = None
    experiment_name: str = "credit_risk_experiments"
    artifact_location: Optional[str] = None
    
    # Reproducibility settings
    enable_seed_management: bool = True
    global_seed: int = 42
    
    # Tracking settings
    track_environment: bool = True
    track_dependencies: bool = True
    track_git_info: bool = True
    track_system_info: bool = True
    
    # Model registry settings
    enable_model_registry: bool = True
    model_registry_stage: str = "None"  # None, Staging, Production, Archived
    
    # Artifact settings
    save_model_artifacts: bool = True
    save_data_artifacts: bool = False
    save_plots: bool = True
    
    # Comparison settings
    enable_experiment_comparison: bool = True
    comparison_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'f1', 'roc_auc'])


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment run."""
    experiment_id: str
    run_id: str
    run_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    user: str
    tags: Dict[str, str]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    model_info: Optional[Dict[str, Any]]


@dataclass
class ModelLineage:
    """Model lineage information for compliance."""
    model_id: str
    model_name: str
    model_version: str
    parent_run_id: str
    creation_time: datetime
    creator: str
    data_sources: List[str]
    feature_pipeline: Dict[str, Any]
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    compliance_info: Dict[str, Any]


class ReproducibilityManager:
    """Manages reproducibility across experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.seed_state = {}
    
    def set_global_seed(self, seed: Optional[int] = None):
        """Set global seed for reproducibility."""
        if not self.config.enable_seed_management:
            return
        
        seed = seed or self.config.global_seed
        
        # Set Python random seed
        import random
        random.seed(seed)
        
        # Set NumPy seed
        np.random.seed(seed)
        
        # Set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        
        # Set scikit-learn seed (for algorithms that support it)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        self.seed_state = {
            'global_seed': seed,
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Global seed set to {seed}")
    
    def get_seed_state(self) -> Dict[str, Any]:
        """Get current seed state."""
        return self.seed_state.copy()
    
    def restore_seed_state(self, seed_state: Dict[str, Any]):
        """Restore seed state."""
        if not self.config.enable_seed_management:
            return
        
        try:
            import random
            if 'python_random_state' in seed_state:
                random.setstate(seed_state['python_random_state'])
            
            if 'numpy_random_state' in seed_state:
                np.random.set_state(seed_state['numpy_random_state'])
            
            if 'global_seed' in seed_state:
                self.set_global_seed(seed_state['global_seed'])
            
            logger.info("Seed state restored")
            
        except Exception as e:
            logger.warning(f"Failed to restore seed state: {e}")


class EnvironmentTracker:
    """Tracks environment and system information."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        env_info = {}
        
        if self.config.track_system_info:
            env_info.update(self._get_system_info())
        
        if self.config.track_dependencies:
            env_info.update(self._get_dependency_info())
        
        if self.config.track_git_info:
            env_info.update(self._get_git_info())
        
        return env_info
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return {
                'system': {
                    'platform': platform.platform(),
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'python_version': platform.python_version(),
                    'python_implementation': platform.python_implementation()
                }
            }
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            return {'system': {}}
    
    def _get_dependency_info(self) -> Dict[str, Any]:
        """Get dependency information."""
        try:
            # Get installed packages
            result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
            if result.returncode == 0:
                packages = {}
                for line in result.stdout.strip().split('\n'):
                    if '==' in line:
                        name, version = line.split('==', 1)
                        packages[name] = version
                
                return {'dependencies': packages}
            else:
                return {'dependencies': {}}
        except Exception as e:
            logger.warning(f"Failed to get dependency info: {e}")
            return {'dependencies': {}}
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get Git repository information."""
        try:
            git_info = {}
            
            # Get current commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True)
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Get repository URL
            result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], capture_output=True, text=True)
            if result.returncode == 0:
                git_info['repository_url'] = result.stdout.strip()
            
            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
            if result.returncode == 0:
                git_info['has_uncommitted_changes'] = bool(result.stdout.strip())
            
            return {'git': git_info}
            
        except Exception as e:
            logger.warning(f"Failed to get git info: {e}")
            return {'git': {}}


class MLflowTracker:
    """MLflow-based experiment tracker."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client = None
        self.current_run = None
        self.experiment_id = None
        
        if MLFLOW_AVAILABLE:
            self._initialize_mlflow()
        else:
            logger.warning("MLflow not available. Using fallback tracking.")
    
    def _initialize_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            # Set tracking URI
            if self.config.tracking_uri:
                mlflow.set_tracking_uri(self.config.tracking_uri)
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
                if experiment is None:
                    self.experiment_id = mlflow.create_experiment(
                        self.config.experiment_name,
                        artifact_location=self.config.artifact_location
                    )
                else:
                    self.experiment_id = experiment.experiment_id
            except Exception as e:
                logger.warning(f"Failed to create/get experiment: {e}")
                self.experiment_id = "0"  # Default experiment
            
            self.client = MlflowClient()
            logger.info(f"MLflow initialized with experiment: {self.config.experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.client = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new MLflow run."""
        if not MLFLOW_AVAILABLE or not self.client:
            return self._start_fallback_run(run_name, tags)
        
        try:
            # Generate run name if not provided
            if not run_name:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Start MLflow run
            self.current_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags
            )
            
            run_id = self.current_run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return self._start_fallback_run(run_name, tags)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if not MLFLOW_AVAILABLE or not self.current_run:
            return
        
        try:
            # Convert complex objects to strings
            processed_params = {}
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    processed_params[key] = json.dumps(value)
                else:
                    processed_params[key] = str(value)
            
            mlflow.log_params(processed_params)
            
        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if not MLFLOW_AVAILABLE or not self.current_run:
            return
        
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    mlflow.log_metric(key, value, step=step)
                    
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact to MLflow."""
        if not MLFLOW_AVAILABLE or not self.current_run:
            return
        
        try:
            if os.path.isfile(artifact_path):
                mlflow.log_artifact(artifact_path, artifact_name)
            elif os.path.isdir(artifact_path):
                mlflow.log_artifacts(artifact_path, artifact_name)
                
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")
    
    def log_model(self, model, model_name: str, **kwargs):
        """Log model to MLflow."""
        if not MLFLOW_AVAILABLE or not self.current_run:
            return
        
        try:
            # Determine model type and log accordingly
            if hasattr(model, 'fit') and hasattr(model, 'predict'):
                # Scikit-learn style model
                mlflow.sklearn.log_model(model, model_name, **kwargs)
            else:
                # Generic model - save as pickle
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    pickle.dump(model, f)
                    temp_path = f.name
                
                mlflow.log_artifact(temp_path, f"{model_name}.pkl")
                os.unlink(temp_path)
                
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
    
    def end_run(self, status: str = "FINISHED"):
        """End current MLflow run."""
        if not MLFLOW_AVAILABLE or not self.current_run:
            return
        
        try:
            mlflow.end_run(status=status)
            run_id = self.current_run.info.run_id
            self.current_run = None
            logger.info(f"Ended MLflow run: {run_id}")
            
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")
    
    def _start_fallback_run(self, run_name: Optional[str], tags: Optional[Dict[str, str]]) -> str:
        """Start fallback run when MLflow is not available."""
        run_id = f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Started fallback run: {run_id}")
        return run_id


class ModelRegistry:
    """Model registry for versioning and lifecycle management."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.registry_path = Path("model_registry")
        self.registry_path.mkdir(exist_ok=True)
        
        # Initialize registry metadata
        self.registry_file = self.registry_path / "registry.json"
        self.models = self._load_registry()
    
    def register_model(self, model, model_name: str, run_id: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a model in the registry."""
        try:
            # Generate model version
            existing_versions = [
                m['version'] for m in self.models.get(model_name, [])
            ]
            
            if existing_versions:
                latest_version = max(int(v) for v in existing_versions)
                version = str(latest_version + 1)
            else:
                version = "1"
            
            # Create model directory
            model_dir = self.registry_path / model_name / f"version_{version}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Create model metadata
            model_info = {
                'model_name': model_name,
                'version': version,
                'run_id': run_id,
                'creation_time': datetime.now().isoformat(),
                'model_path': str(model_path),
                'stage': self.config.model_registry_stage,
                'metadata': metadata or {}
            }
            
            # Update registry
            if model_name not in self.models:
                self.models[model_name] = []
            
            self.models[model_name].append(model_info)
            self._save_registry()
            
            logger.info(f"Registered model {model_name} version {version}")
            
            return version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return ""
    
    def get_model(self, model_name: str, version: Optional[str] = None, 
                  stage: Optional[str] = None):
        """Get model from registry."""
        try:
            if model_name not in self.models:
                return None
            
            model_versions = self.models[model_name]
            
            # Filter by stage if specified
            if stage:
                model_versions = [m for m in model_versions if m['stage'] == stage]
            
            # Get specific version or latest
            if version:
                model_info = next((m for m in model_versions if m['version'] == version), None)
            else:
                # Get latest version
                model_info = max(model_versions, key=lambda x: int(x['version']))
            
            if not model_info:
                return None
            
            # Load model
            with open(model_info['model_path'], 'rb') as f:
                model = pickle.load(f)
            
            return model, model_info
            
        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            return None
    
    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all registered models."""
        return self.models.copy()
    
    def transition_model_stage(self, model_name: str, version: str, stage: str) -> bool:
        """Transition model to different stage."""
        try:
            if model_name not in self.models:
                return False
            
            for model_info in self.models[model_name]:
                if model_info['version'] == version:
                    model_info['stage'] = stage
                    model_info['stage_transition_time'] = datetime.now().isoformat()
                    self._save_registry()
                    
                    logger.info(f"Transitioned {model_name} v{version} to {stage}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            return False
    
    def _load_registry(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load registry from file."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")
            return {}
    
    def _save_registry(self):
        """Save registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.models, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")


class ExperimentTracker(DataProcessor):
    """Main experiment tracking system."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.reproducibility_manager = ReproducibilityManager(self.config)
        self.environment_tracker = EnvironmentTracker(self.config)
        self.mlflow_tracker = MLflowTracker(self.config)
        self.model_registry = ModelRegistry(self.config)
        
        # Current experiment state
        self.current_run_id = None
        self.experiment_metadata = {}
    
    def start_experiment(self, experiment_name: Optional[str] = None, 
                        tags: Optional[Dict[str, str]] = None,
                        seed: Optional[int] = None) -> str:
        """Start a new experiment."""
        try:
            # Set reproducibility
            self.reproducibility_manager.set_global_seed(seed)
            
            # Start MLflow run
            run_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_run_id = self.mlflow_tracker.start_run(run_name, tags)
            
            # Log environment information
            env_info = self.environment_tracker.get_environment_info()
            self.mlflow_tracker.log_params(env_info)
            
            # Log seed state
            seed_state = self.reproducibility_manager.get_seed_state()
            self.mlflow_tracker.log_params({'seed_state': seed_state})
            
            # Initialize experiment metadata
            self.experiment_metadata = {
                'run_id': self.current_run_id,
                'start_time': datetime.now(),
                'environment': env_info,
                'seed_state': seed_state,
                'tags': tags or {}
            }
            
            logger.info(f"Started experiment: {run_name}")
            
            # Log experiment start
            audit_logger.log_data_access(
                user_id="system",
                resource="experiment_tracker",
                action="experiment_start",
                success=True,
                details={
                    "run_id": self.current_run_id,
                    "experiment_name": run_name
                }
            )
            
            return self.current_run_id
            
        except Exception as e:
            error_message = f"Failed to start experiment: {str(e)}"
            logger.error(error_message)
            return ""
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log experiment parameters."""
        if not self.current_run_id:
            logger.warning("No active experiment. Start experiment first.")
            return
        
        try:
            self.mlflow_tracker.log_params(params)
            self.experiment_metadata.setdefault('parameters', {}).update(params)
            
        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log experiment metrics."""
        if not self.current_run_id:
            logger.warning("No active experiment. Start experiment first.")
            return
        
        try:
            self.mlflow_tracker.log_metrics(metrics, step)
            self.experiment_metadata.setdefault('metrics', {}).update(metrics)
            
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def log_model(self, model, model_name: str, register: bool = True, 
                  metadata: Optional[Dict[str, Any]] = None):
        """Log and optionally register model."""
        if not self.current_run_id:
            logger.warning("No active experiment. Start experiment first.")
            return
        
        try:
            # Log model to MLflow
            self.mlflow_tracker.log_model(model, model_name)
            
            # Register model if requested
            if register and self.config.enable_model_registry:
                version = self.model_registry.register_model(
                    model, model_name, self.current_run_id, metadata
                )
                
                if version:
                    self.experiment_metadata.setdefault('registered_models', []).append({
                        'name': model_name,
                        'version': version
                    })
            
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact."""
        if not self.current_run_id:
            logger.warning("No active experiment. Start experiment first.")
            return
        
        try:
            self.mlflow_tracker.log_artifact(artifact_path, artifact_name)
            
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")
    
    def end_experiment(self, status: str = "FINISHED"):
        """End current experiment."""
        if not self.current_run_id:
            logger.warning("No active experiment to end.")
            return
        
        try:
            # End MLflow run
            self.mlflow_tracker.end_run(status)
            
            # Update metadata
            self.experiment_metadata['end_time'] = datetime.now()
            self.experiment_metadata['status'] = status
            
            logger.info(f"Ended experiment: {self.current_run_id}")
            
            # Log experiment end
            audit_logger.log_data_access(
                user_id="system",
                resource="experiment_tracker",
                action="experiment_end",
                success=True,
                details={
                    "run_id": self.current_run_id,
                    "status": status
                }
            )
            
            # Reset state
            run_id = self.current_run_id
            self.current_run_id = None
            
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to end experiment: {e}")
            return None
    
    def process(self, data: pd.DataFrame) -> bool:
        """Process method for DataProcessor interface."""
        # This is a tracking system, so we just validate the data
        return self.validate(data)
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate data for experiment tracking."""
        try:
            if data.empty:
                logger.error("Data is empty")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Experiment tracking validation failed: {e}")
            return False


class ExperimentComparator:
    """Compare and analyze experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client = MlflowClient() if MLFLOW_AVAILABLE else None
    
    def compare_experiments(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiment runs."""
        if not MLFLOW_AVAILABLE or not self.client:
            return self._fallback_comparison(run_ids)
        
        try:
            comparison = {
                'run_ids': run_ids,
                'comparison_time': datetime.now().isoformat(),
                'metrics_comparison': {},
                'parameters_comparison': {},
                'best_run': None,
                'recommendations': []
            }
            
            # Get run information
            runs = []
            for run_id in run_ids:
                try:
                    run = self.client.get_run(run_id)
                    runs.append(run)
                except Exception as e:
                    logger.warning(f"Failed to get run {run_id}: {e}")
            
            if not runs:
                return comparison
            
            # Compare metrics
            all_metrics = set()
            for run in runs:
                all_metrics.update(run.data.metrics.keys())
            
            for metric in all_metrics:
                metric_values = []
                for run in runs:
                    value = run.data.metrics.get(metric)
                    metric_values.append(value if value is not None else 0.0)
                
                comparison['metrics_comparison'][metric] = {
                    'values': dict(zip(run_ids, metric_values)),
                    'best_run_id': run_ids[np.argmax(metric_values)] if metric_values else None,
                    'worst_run_id': run_ids[np.argmin(metric_values)] if metric_values else None,
                    'mean': np.mean(metric_values) if metric_values else 0.0,
                    'std': np.std(metric_values) if metric_values else 0.0
                }
            
            # Compare parameters
            all_params = set()
            for run in runs:
                all_params.update(run.data.params.keys())
            
            for param in all_params:
                param_values = []
                for run in runs:
                    value = run.data.params.get(param, 'N/A')
                    param_values.append(value)
                
                comparison['parameters_comparison'][param] = {
                    'values': dict(zip(run_ids, param_values)),
                    'unique_values': len(set(param_values)),
                    'most_common': max(set(param_values), key=param_values.count) if param_values else None
                }
            
            # Determine best run based on primary metric
            primary_metric = self.config.comparison_metrics[0] if self.config.comparison_metrics else 'accuracy'
            if primary_metric in comparison['metrics_comparison']:
                best_run_id = comparison['metrics_comparison'][primary_metric]['best_run_id']
                comparison['best_run'] = best_run_id
            
            # Generate recommendations
            recommendations = self._generate_comparison_recommendations(comparison)
            comparison['recommendations'] = recommendations
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare experiments: {e}")
            return {'error': str(e)}
    
    def get_experiment_leaderboard(self, experiment_name: Optional[str] = None, 
                                  metric: str = 'accuracy', top_k: int = 10) -> List[Dict[str, Any]]:
        """Get leaderboard of best experiments."""
        if not MLFLOW_AVAILABLE or not self.client:
            return []
        
        try:
            # Get experiment
            if experiment_name:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id if experiment else "0"
            else:
                experiment_id = "0"  # Default experiment
            
            # Search runs
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="",
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=1000,
                order_by=[f"metrics.{metric} DESC"]
            )
            
            # Build leaderboard
            leaderboard = []
            for i, run in enumerate(runs[:top_k]):
                entry = {
                    'rank': i + 1,
                    'run_id': run.info.run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                    'start_time': run.info.start_time,
                    'status': run.info.status,
                    'metrics': dict(run.data.metrics),
                    'primary_metric': run.data.metrics.get(metric, 0.0)
                }
                leaderboard.append(entry)
            
            return leaderboard
            
        except Exception as e:
            logger.error(f"Failed to get leaderboard: {e}")
            return []
    
    def _fallback_comparison(self, run_ids: List[str]) -> Dict[str, Any]:
        """Fallback comparison when MLflow is not available."""
        return {
            'run_ids': run_ids,
            'comparison_time': datetime.now().isoformat(),
            'error': 'MLflow not available for comparison',
            'recommendations': ['Install MLflow for full experiment comparison capabilities']
        }
    
    def _generate_comparison_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []
        
        # Check metric performance
        metrics_comp = comparison.get('metrics_comparison', {})
        for metric, data in metrics_comp.items():
            if metric in self.config.comparison_metrics:
                values = list(data['values'].values())
                if values:
                    cv = data['std'] / data['mean'] if data['mean'] != 0 else float('inf')
                    if cv > 0.1:
                        recommendations.append(f"High variance in {metric} across runs - consider parameter tuning")
        
        # Check parameter consistency
        params_comp = comparison.get('parameters_comparison', {})
        inconsistent_params = [
            param for param, data in params_comp.items()
            if data['unique_values'] > 1
        ]
        
        if inconsistent_params:
            recommendations.append(f"Parameters varied across runs: {inconsistent_params[:3]}")
        
        # Best run recommendation
        if comparison.get('best_run'):
            recommendations.append(f"Best performing run: {comparison['best_run']}")
        
        return recommendations


# Factory functions and utilities
def create_experiment_tracker(config: Optional[ExperimentConfig] = None) -> ExperimentTracker:
    """Create an experiment tracker instance."""
    return ExperimentTracker(config)


def get_default_experiment_config() -> ExperimentConfig:
    """Get default experiment tracking configuration."""
    return ExperimentConfig()


def get_mlflow_config(tracking_uri: str, experiment_name: str) -> ExperimentConfig:
    """Get MLflow-specific configuration."""
    return ExperimentConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        enable_model_registry=True,
        track_environment=True,
        track_dependencies=True,
        track_git_info=True
    )


def get_local_config(experiment_name: str = "local_experiments") -> ExperimentConfig:
    """Get configuration for local experiment tracking."""
    return ExperimentConfig(
        tracking_uri="file:./mlruns",
        experiment_name=experiment_name,
        enable_model_registry=True,
        save_model_artifacts=True,
        save_data_artifacts=False
    )


class ExperimentContext:
    """Context manager for experiments."""
    
    def __init__(self, tracker: ExperimentTracker, experiment_name: str, 
                 tags: Optional[Dict[str, str]] = None, seed: Optional[int] = None):
        self.tracker = tracker
        self.experiment_name = experiment_name
        self.tags = tags
        self.seed = seed
        self.run_id = None
    
    def __enter__(self):
        """Start experiment."""
        self.run_id = self.tracker.start_experiment(self.experiment_name, self.tags, self.seed)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End experiment."""
        status = "FAILED" if exc_type else "FINISHED"
        self.tracker.end_experiment(status)


def experiment_context(tracker: ExperimentTracker, experiment_name: str,
                      tags: Optional[Dict[str, str]] = None, 
                      seed: Optional[int] = None) -> ExperimentContext:
    """Create experiment context manager."""
    return ExperimentContext(tracker, experiment_name, tags, seed)


def track_model_training(tracker: ExperimentTracker, model, model_name: str,
                        training_params: Dict[str, Any], 
                        metrics: Dict[str, float],
                        artifacts: Optional[List[str]] = None) -> str:
    """Convenience function to track model training."""
    try:
        # Log parameters
        tracker.log_parameters(training_params)
        
        # Log metrics
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(model, model_name)
        
        # Log artifacts
        if artifacts:
            for artifact in artifacts:
                tracker.log_artifact(artifact)
        
        return tracker.current_run_id or ""
        
    except Exception as e:
        logger.error(f"Failed to track model training: {e}")
        return ""


def export_experiment_comparison(comparison: Dict[str, Any], file_path: str) -> bool:
    """Export experiment comparison to file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        logger.info(f"Experiment comparison exported to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export comparison: {e}")
        return False


def create_model_lineage(model_info: Dict[str, Any], run_id: str, 
                        data_sources: List[str], feature_pipeline: Dict[str, Any],
                        training_config: Dict[str, Any]) -> ModelLineage:
    """Create model lineage record."""
    return ModelLineage(
        model_id=hashlib.md5(f"{model_info['name']}_{run_id}".encode()).hexdigest(),
        model_name=model_info['name'],
        model_version=model_info.get('version', '1'),
        parent_run_id=run_id,
        creation_time=datetime.now(),
        creator=os.getenv('USER', 'unknown'),
        data_sources=data_sources,
        feature_pipeline=feature_pipeline,
        training_config=training_config,
        performance_metrics=model_info.get('metrics', {}),
        compliance_info={
            'data_lineage_tracked': True,
            'reproducible': True,
            'audit_logged': True
        }
    )


# Visualization utilities (basic implementations)
def plot_experiment_metrics(comparison: Dict[str, Any], save_path: Optional[str] = None) -> bool:
    """Plot experiment metrics comparison."""
    try:
        import matplotlib.pyplot as plt
        
        metrics_comp = comparison.get('metrics_comparison', {})
        if not metrics_comp:
            return False
        
        # Create subplots for each metric
        n_metrics = len(metrics_comp)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, (metric, data) in enumerate(metrics_comp.items()):
            ax = axes[i]
            
            run_ids = list(data['values'].keys())
            values = list(data['values'].values())
            
            ax.bar(range(len(run_ids)), values)
            ax.set_title(f'{metric.title()} Comparison')
            ax.set_xlabel('Run ID')
            ax.set_ylabel(metric.title())
            ax.set_xticks(range(len(run_ids)))
            ax.set_xticklabels([rid[:8] for rid in run_ids], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Experiment metrics plot saved to {save_path}")
        
        plt.close()
        return True
        
    except ImportError:
        logger.warning("Matplotlib not available for plotting")
        return False
    except Exception as e:
        logger.error(f"Failed to plot metrics: {e}")
        return False


def generate_experiment_report(tracker: ExperimentTracker, run_id: str, 
                             output_path: str) -> bool:
    """Generate comprehensive experiment report."""
    try:
        if not MLFLOW_AVAILABLE or not tracker.mlflow_tracker.client:
            return False
        
        # Get run information
        run = tracker.mlflow_tracker.client.get_run(run_id)
        
        # Create report
        report = {
            'experiment_report': {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'status': run.info.status,
                'duration_seconds': (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None
            },
            'parameters': dict(run.data.params),
            'metrics': dict(run.data.metrics),
            'tags': dict(run.data.tags),
            'artifacts': [artifact.path for artifact in tracker.mlflow_tracker.client.list_artifacts(run_id)],
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Experiment report generated: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate experiment report: {e}")
        return False
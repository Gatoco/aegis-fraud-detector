# MLflow Configuration for Aegis Fraud Detection System
"""
MLflow client configuration to connect to the containerized tracking server.
This module provides utilities for experiment tracking, model versioning,
and artifact management in both development and production environments.
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowConfig:
    """MLflow configuration manager for fraud detection experiments."""
    
    def __init__(self):
        """Initialize MLflow configuration."""
        self.setup_tracking_uri()
        self.setup_experiment()
    
    def setup_tracking_uri(self):
        """Configure MLflow tracking URI based on environment."""
        # Check if running in Docker container
        if os.getenv('MLFLOW_TRACKING_URI'):
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        elif self._is_docker_environment():
            tracking_uri = "http://mlflow:5000"
        else:
            # Local development
            tracking_uri = "http://localhost:5000"
        
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        
        # Verify connection
        try:
            mlflow.get_tracking_uri()
            logger.info("Successfully connected to MLflow tracking server")
        except Exception as e:
            logger.warning(f"Could not connect to MLflow server: {e}")
            logger.info("Falling back to local file storage")
            mlflow.set_tracking_uri("file:///workspace/mlruns")
    
    def setup_experiment(self, experiment_name: str = "aegis-fraud-detection"):
        """Setup or get existing MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=self._get_artifact_location()
                )
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(experiment_name)
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to setup experiment: {e}")
            return None
    
    def _is_docker_environment(self) -> bool:
        """Check if running inside Docker container."""
        return (
            os.path.exists('/.dockerenv') or 
            os.getenv('DOCKER_CONTAINER') == 'true'
        )
    
    def _get_artifact_location(self) -> str:
        """Get appropriate artifact storage location."""
        if self._is_docker_environment():
            return "/workspace/artifacts"
        else:
            return str(Path.cwd() / "artifacts")

class ExperimentTracker:
    """Enhanced experiment tracking utilities for fraud detection."""
    
    def __init__(self, config: MLflowConfig):
        """Initialize experiment tracker."""
        self.config = config
    
    def log_dataset_info(self, 
                        train_shape: tuple, 
                        test_shape: tuple, 
                        fraud_rate: float,
                        **kwargs):
        """Log dataset information to MLflow."""
        mlflow.log_param("train_samples", train_shape[0])
        mlflow.log_param("train_features", train_shape[1])
        mlflow.log_param("test_samples", test_shape[0])
        mlflow.log_param("fraud_rate", fraud_rate)
        
        for key, value in kwargs.items():
            mlflow.log_param(f"data_{key}", value)
    
    def log_preprocessing_steps(self, steps: Dict[str, Any]):
        """Log preprocessing pipeline steps."""
        for step_name, step_info in steps.items():
            mlflow.log_param(f"preprocessing_{step_name}", step_info)
    
    def log_model_with_signature(self, 
                                model, 
                                model_name: str,
                                signature=None,
                                input_example=None):
        """Log model with proper signature and metadata."""
        if hasattr(model, 'get_params'):
            # Log hyperparameters
            params = model.get_params()
            for param, value in params.items():
                mlflow.log_param(f"model_{param}", value)
        
        # Log model based on type
        if model_name.lower().startswith('xgb'):
            mlflow.xgboost.log_model(
                model, 
                "model", 
                signature=signature,
                input_example=input_example
            )
        elif model_name.lower().startswith('lgb'):
            mlflow.lightgbm.log_model(
                model, 
                "model",
                signature=signature,
                input_example=input_example
            )
        else:
            mlflow.sklearn.log_model(
                model, 
                "model",
                signature=signature,
                input_example=input_example
            )
    
    def log_fraud_detection_metrics(self, 
                                  y_true, 
                                  y_pred, 
                                  y_pred_proba=None):
        """Log fraud detection specific metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        # Basic classification metrics
        mlflow.log_metric("accuracy", accuracy_score(y_true, y_pred))
        mlflow.log_metric("precision", precision_score(y_true, y_pred))
        mlflow.log_metric("recall", recall_score(y_true, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_true, y_pred))
        
        # Fraud detection specific metrics
        if y_pred_proba is not None:
            mlflow.log_metric("roc_auc", roc_auc_score(y_true, y_pred_proba))
            mlflow.log_metric("avg_precision", average_precision_score(y_true, y_pred_proba))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        mlflow.log_metric("true_negatives", tn)
        mlflow.log_metric("false_positives", fp)
        mlflow.log_metric("false_negatives", fn)
        mlflow.log_metric("true_positives", tp)
        
        # Business metrics
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        mlflow.log_metric("false_positive_rate", false_positive_rate)
        mlflow.log_metric("false_negative_rate", false_negative_rate)

# Global configuration instance
mlflow_config = MLflowConfig()
experiment_tracker = ExperimentTracker(mlflow_config)

# Convenience functions
def start_run(run_name: Optional[str] = None, **kwargs):
    """Start MLflow run with proper configuration."""
    return mlflow.start_run(run_name=run_name, **kwargs)

def log_artifact(local_path: str, artifact_path: Optional[str] = None):
    """Log artifact to MLflow."""
    mlflow.log_artifact(local_path, artifact_path)

def log_figure(figure, filename: str):
    """Log matplotlib figure to MLflow."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        figure.savefig(tmp.name, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(tmp.name, f"figures/{filename}")

# Example usage in training scripts:
"""
from src.mlflow_config import start_run, experiment_tracker, log_figure

with start_run(run_name="baseline_logistic_regression"):
    # Log dataset info
    experiment_tracker.log_dataset_info(
        train_shape=X_train.shape,
        test_shape=X_test.shape,
        fraud_rate=y_train.mean()
    )
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Log model and metrics
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    experiment_tracker.log_model_with_signature(model, "logistic_regression")
    experiment_tracker.log_fraud_detection_metrics(y_test, y_pred, y_pred_proba)
"""

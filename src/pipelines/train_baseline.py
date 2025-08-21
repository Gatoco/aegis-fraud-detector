"""
Aegis Fraud Detector - Baseline Training Pipeline

This module implements a baseline training pipeline for fraud detection using
Logistic Regression with basic preprocessing and MLflow tracking.

Sprint 1.2: Pipeline de Entrenamiento Baseline
- Simple preprocessing: imputation + StandardScaler
- Logistic Regression with class_weight='balanced'
- MLflow integration for experiment tracking
- Comprehensive metrics evaluation

Author: Aegis Development Team
Date: August 2025
Sprint: 1.2 - Baseline Training Pipeline
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, auc, average_precision_score,
    precision_score, recall_score, f1_score, matthews_corrcoef
)
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaselineTrainingPipeline:
    """
    Baseline training pipeline for fraud detection.
    
    This class implements a complete training pipeline with:
    - Data loading and basic preprocessing
    - Feature engineering and selection
    - Model training with Logistic Regression
    - MLflow experiment tracking
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, 
                 data_path: str = "data/01_raw",
                 experiment_name: str = "fraud-detection-baseline",
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the baseline training pipeline.
        
        Args:
            data_path: Path to the raw data directory
            experiment_name: MLflow experiment name
            test_size: Fraction for test split
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.experiment_name = experiment_name
        self.test_size = test_size
        self.random_state = random_state
        
        # Data containers
        self.train_transaction = None
        self.train_identity = None
        self.merged_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Pipeline components
        self.preprocessor = None
        self.model = None
        self.pipeline = None
        
        # Results storage
        self.training_results = {}
        self.evaluation_metrics = {}
        
        # MLflow setup
        self.mlflow_client = None
        self.experiment_id = None
        self.run_id = None
        
    def setup_mlflow(self) -> None:
        """Initialize MLflow experiment and tracking."""
        try:
            # Set tracking URI (should be configured in environment)
            mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
            mlflow.set_tracking_uri(mlflow_uri)
            
            # Create or get experiment
            try:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"✅ Created new MLflow experiment: {self.experiment_name}")
            except Exception:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                self.experiment_id = experiment.experiment_id
                logger.info(f"✅ Using existing MLflow experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_client = MlflowClient()
            
            logger.info(f"🔗 MLflow tracking URI: {mlflow_uri}")
            logger.info(f"📊 Experiment ID: {self.experiment_id}")
            
        except Exception as e:
            logger.error(f"❌ MLflow setup failed: {str(e)}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and merge transaction and identity datasets.
        
        Returns:
            Merged DataFrame with transaction and identity data
        """
        logger.info("📂 Loading datasets...")
        
        try:
            # Load datasets
            self.train_transaction = pd.read_csv(self.data_path / "train_transaction.csv")
            self.train_identity = pd.read_csv(self.data_path / "train_identity.csv")
            
            logger.info(f"  📊 Transaction data: {self.train_transaction.shape}")
            logger.info(f"  📊 Identity data: {self.train_identity.shape}")
            
            # Merge datasets
            self.merged_data = self.train_transaction.merge(
                self.train_identity, 
                on='TransactionID', 
                how='left'
            )
            
            logger.info(f"  📊 Merged data: {self.merged_data.shape}")
            logger.info(f"  🎯 Fraud rate: {self.merged_data['isFraud'].mean():.4f}")
            
            return self.merged_data
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {str(e)}")
            raise
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply basic preprocessing to the dataset.
        
        Returns:
            Tuple of (features, target) after preprocessing
        """
        logger.info("🔧 Preprocessing data...")
        
        if self.merged_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Separate features and target
        target_col = 'isFraud'
        feature_cols = [col for col in self.merged_data.columns 
                       if col not in [target_col, 'TransactionID']]
        
        X = self.merged_data[feature_cols].copy()
        y = self.merged_data[target_col].copy()
        
        # Basic feature selection - remove features with >95% missing values
        missing_threshold = 0.95
        missing_percentages = X.isnull().mean()
        features_to_keep = missing_percentages[missing_percentages < missing_threshold].index
        
        logger.info(f"  🔍 Original features: {len(X.columns)}")
        logger.info(f"  🔍 Features after missing value filter: {len(features_to_keep)}")
        
        X = X[features_to_keep]
        
        # Separate numerical and categorical features
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"  🔢 Numerical features: {len(numerical_features)}")
        logger.info(f"  🏷️ Categorical features: {len(categorical_features)}")
        
        # For baseline, we'll use only numerical features to simplify
        X_numerical = X[numerical_features]
        
        # Store feature information
        self.feature_info = {
            'total_features': len(X.columns),
            'numerical_features': len(numerical_features),
            'categorical_features': len(categorical_features),
            'selected_features': len(X_numerical.columns),
            'feature_names': X_numerical.columns.tolist()
        }
        
        logger.info(f"  ✅ Using {len(X_numerical.columns)} numerical features for baseline")
        
        return X_numerical, y
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create scikit-learn preprocessing pipeline.
        
        Returns:
            Configured ColumnTransformer for preprocessing
        """
        logger.info("🛠️ Creating preprocessing pipeline...")
        
        # Numerical preprocessing: imputation + scaling
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Create preprocessor (for now, only numerical features)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, slice(None))  # All features are numerical
            ]
        )
        
        self.preprocessor = preprocessor
        logger.info("✅ Preprocessing pipeline created")
        
        return preprocessor
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.info("✂️ Splitting data...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"  📊 Training set: {self.X_train.shape}")
        logger.info(f"  📊 Test set: {self.X_test.shape}")
        logger.info(f"  🎯 Train fraud rate: {self.y_train.mean():.4f}")
        logger.info(f"  🎯 Test fraud rate: {self.y_test.mean():.4f}")
    
    def create_model(self) -> LogisticRegression:
        """
        Create and configure the baseline model.
        
        Returns:
            Configured LogisticRegression model
        """
        logger.info("🤖 Creating baseline model...")
        
        # Logistic Regression with balanced class weights
        model = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000,
            solver='liblinear'  # Good for small datasets
        )
        
        self.model = model
        logger.info("✅ Baseline model (Logistic Regression) created")
        
        return model
    
    def create_training_pipeline(self) -> Pipeline:
        """
        Create complete training pipeline.
        
        Returns:
            Complete scikit-learn Pipeline
        """
        logger.info("🔧 Creating complete training pipeline...")
        
        if self.preprocessor is None:
            self.create_preprocessing_pipeline()
        
        if self.model is None:
            self.create_model()
        
        # Create complete pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        
        self.pipeline = pipeline
        logger.info("✅ Complete training pipeline created")
        
        return pipeline
    
    def train_model(self) -> Dict[str, Any]:
        """
        Train the model and return training results.
        
        Returns:
            Dictionary containing training results
        """
        logger.info("🚀 Training baseline model...")
        
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_training_pipeline() first.")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"baseline_logistic_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            self.run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("preprocessing", "SimpleImputer+StandardScaler")
            mlflow.log_param("feature_count", len(self.X_train.columns))
            mlflow.log_param("training_samples", len(self.X_train))
            mlflow.log_param("test_samples", len(self.X_test))
            mlflow.log_param("fraud_rate_train", float(self.y_train.mean()))
            mlflow.log_param("fraud_rate_test", float(self.y_test.mean()))
            
            # Train the model
            start_time = datetime.now()
            self.pipeline.fit(self.X_train, self.y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Make predictions
            y_pred = self.pipeline.predict(self.X_test)
            y_pred_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(
                self.pipeline,
                "model",
                registered_model_name="fraud_detection_baseline"
            )
            
            # Store results
            self.training_results = {
                'model': self.pipeline,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'metrics': metrics,
                'training_time': training_time,
                'run_id': self.run_id
            }
            
            logger.info(f"✅ Model training completed in {training_time:.2f} seconds")
            logger.info(f"📊 MLflow run ID: {self.run_id}")
            
            return self.training_results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of calculated metrics
        """
        logger.info("📊 Calculating evaluation metrics...")
        
        # Basic classification metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # ROC AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall AUC (AUPRC)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        auprc = auc(recall_curve, precision_curve)
        
        # Average Precision Score (another way to calculate AUPRC)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'auprc': auprc,
            'avg_precision': avg_precision,
            'matthews_corr_coef': mcc,
            'specificity': specificity,
            'negative_predictive_value': npv,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        self.evaluation_metrics = metrics
        
        logger.info("✅ Metrics calculated successfully")
        return metrics
    
    def evaluate_model(self) -> None:
        """Print comprehensive model evaluation results."""
        if not self.training_results:
            raise ValueError("Model not trained. Call train_model() first.")
        
        metrics = self.training_results['metrics']
        
        print("\n" + "="*60)
        print("📊 BASELINE MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n🎯 PRIMARY METRICS:")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  F1-Score:     {metrics['f1_score']:.4f}")
        print(f"  AUPRC:        {metrics['auprc']:.4f}")
        print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
        
        print(f"\n📈 ADDITIONAL METRICS:")
        print(f"  MCC:          {metrics['matthews_corr_coef']:.4f}")
        print(f"  Specificity:  {metrics['specificity']:.4f}")
        print(f"  NPV:          {metrics['negative_predictive_value']:.4f}")
        
        print(f"\n🔢 CONFUSION MATRIX:")
        print(f"  True Positives:   {metrics['true_positives']:,}")
        print(f"  False Positives:  {metrics['false_positives']:,}")
        print(f"  True Negatives:   {metrics['true_negatives']:,}")
        print(f"  False Negatives:  {metrics['false_negatives']:,}")
        
        print(f"\n⏱️ TRAINING INFO:")
        print(f"  Training Time:    {self.training_results['training_time']:.2f} seconds")
        print(f"  MLflow Run ID:    {self.training_results['run_id']}")
        
        print("\n" + "="*60)
    
    def cross_validate_model(self, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation for more robust evaluation.
        
        Args:
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        logger.info(f"🔄 Performing {cv_folds}-fold cross-validation...")
        
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_training_pipeline() first.")
        
        # Combine training and test data for CV
        X_combined = pd.concat([self.X_train, self.X_test], axis=0)
        y_combined = pd.concat([self.y_train, self.y_test], axis=0)
        
        # Define scoring metrics
        scoring = ['precision', 'recall', 'f1', 'roc_auc']
        
        # Perform cross-validation
        cv_results = cross_validate(
            self.pipeline, 
            X_combined, 
            y_combined,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calculate statistics
        cv_stats = {}
        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            cv_stats[f'{metric}_test_mean'] = np.mean(test_scores)
            cv_stats[f'{metric}_test_std'] = np.std(test_scores)
            cv_stats[f'{metric}_train_mean'] = np.mean(train_scores)
            cv_stats[f'{metric}_train_std'] = np.std(train_scores)
        
        logger.info("✅ Cross-validation completed")
        return cv_stats
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete baseline training pipeline.
        
        Returns:
            Complete results dictionary
        """
        logger.info("🚀 Running complete baseline training pipeline...")
        
        try:
            # Setup MLflow
            self.setup_mlflow()
            
            # Load and preprocess data
            self.load_data()
            X, y = self.preprocess_data()
            
            # Split data
            self.split_data(X, y)
            
            # Create and train pipeline
            self.create_training_pipeline()
            training_results = self.train_model()
            
            # Evaluate model
            self.evaluate_model()
            
            # Cross-validation (optional, for additional validation)
            cv_results = self.cross_validate_model()
            
            # Compile complete results
            complete_results = {
                'training_results': training_results,
                'cv_results': cv_results,
                'feature_info': self.feature_info,
                'data_info': {
                    'total_samples': len(self.merged_data),
                    'train_samples': len(self.X_train),
                    'test_samples': len(self.X_test),
                    'fraud_rate': float(y.mean())
                },
                'mlflow_info': {
                    'experiment_name': self.experiment_name,
                    'experiment_id': self.experiment_id,
                    'run_id': self.run_id
                }
            }
            
            logger.info("✅ Complete baseline pipeline executed successfully!")
            return complete_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}")
            raise


def main():
    """Main execution function."""
    logger.info("🚀 Starting Aegis Fraud Detection - Baseline Training Pipeline")
    
    try:
        # Initialize pipeline
        pipeline = BaselineTrainingPipeline(
            data_path="data/01_raw",
            experiment_name="fraud-detection-baseline-sprint-1-2",
            test_size=0.2,
            random_state=42
        )
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        logger.info("🎉 Baseline training pipeline completed successfully!")
        logger.info(f"📊 Check MLflow UI at: http://localhost:5001")
        logger.info(f"🔬 Experiment: {results['mlflow_info']['experiment_name']}")
        logger.info(f"🆔 Run ID: {results['mlflow_info']['run_id']}")
        
        return results
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

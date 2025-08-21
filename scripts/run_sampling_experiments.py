"""
Sampling experiment runner for Sprint 2.2.

This script runs comprehensive experiments comparing different sampling strategies
on the fraud detection dataset with MLflow tracking.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report, confusion_matrix
)
import joblib

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from sampling.sampling_strategies import (
    create_sampling_pipeline, SAMPLING_CONFIGURATIONS,
    SamplingExperimentTracker
)

warnings.filterwarnings('ignore')


class SamplingExperimentRunner:
    """Comprehensive sampling experiment runner with MLflow integration."""
    
    def __init__(self, 
                 experiment_name: str = "fraud-detection-sampling-sprint-2-2",
                 random_state: int = 42):
        """
        Initialize experiment runner.
        
        Args:
            experiment_name: MLflow experiment name
            random_state: Random state for reproducibility
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.tracker = SamplingExperimentTracker()
        
        # Setup MLflow
        self._setup_mlflow()
        
        # Load preprocessing pipeline from Sprint 2.1
        self.preprocessing_pipeline = self._load_preprocessing_pipeline()
        
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5001'
        mlflow.set_tracking_uri('http://localhost:5001')
        
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                print(f"Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                print(f"Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
                
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            print(f"Error setting up MLflow: {e}")
            print("Continuing without MLflow tracking...")
    
    def _load_preprocessing_pipeline(self):
        """Load preprocessing pipeline from Sprint 2.1."""
        pipeline_path = Path("models/feature_pipeline_v1.0.pkl")
        
        if pipeline_path.exists():
            print(f"Loading preprocessing pipeline from {pipeline_path}")
            return joblib.load(pipeline_path)
        else:
            print("Preprocessing pipeline not found. Creating simple preprocessing...")
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import StandardScaler, SimpleImputer
            from sklearn.pipeline import Pipeline
            
            # Simple preprocessing for demonstration
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            return ColumnTransformer([
                ('numeric', numeric_transformer, slice(None))
            ])
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training data for sampling experiments.
        
        Returns:
            Tuple of (features, target)
        """
        # Try to load processed data from Sprint 2.1
        train_path = Path("data/02_processed/fraud_train_v1.0.parquet")
        
        if train_path.exists():
            print(f"Loading training data from {train_path}")
            df = pd.read_parquet(train_path)
            X = df.drop('isFraud', axis=1)
            y = df['isFraud']
            
        else:
            print("Processed data not found. Loading raw data...")
            # Load raw data and apply basic preprocessing
            transaction_path = Path("data/01_raw/train_transaction.csv")
            identity_path = Path("data/01_raw/train_identity.csv")
            
            if not transaction_path.exists():
                raise FileNotFoundError(f"Training data not found at {transaction_path}")
            
            # Load sample for experimentation
            print("Loading sample of raw data for experiments...")
            train_transaction = pd.read_csv(transaction_path, nrows=50000)
            
            if identity_path.exists():
                train_identity = pd.read_csv(identity_path)
                df = train_transaction.merge(train_identity, on='TransactionID', how='left')
            else:
                df = train_transaction.copy()
            
            # Basic feature selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['TransactionID', 'isFraud']]
            
            # Select features with less than 50% missing values
            completeness = df[numeric_cols].isnull().mean()
            good_features = completeness[completeness < 0.5].index.tolist()
            
            X = df[good_features]
            y = df['isFraud']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution: {Counter(y)}")
        print(f"Fraud rate: {y.mean():.4f}")
        
        return X, y
    
    def run_single_experiment(self, 
                            X: pd.DataFrame, 
                            y: pd.Series,
                            sampling_config: str,
                            cv_folds: int = 3) -> Dict[str, Any]:
        """
        Run a single sampling experiment with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            sampling_config: Sampling configuration name
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*60}")
        print(f"Running experiment: {sampling_config}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create model
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
        )
        
        # Create sampling pipeline
        if sampling_config == 'no_sampling':
            # Use regular sklearn pipeline for no sampling
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([
                ('preprocessing', self.preprocessing_pipeline),
                ('classifier', model)
            ])
        else:
            pipeline = create_sampling_pipeline(
                preprocessing_pipeline=self.preprocessing_pipeline,
                sampling_config=sampling_config,
                model=model,
                random_state=self.random_state
            )
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Scoring metrics
        scoring = {
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision'
        }
        
        # Run cross-validation
        print("Running cross-validation...")
        cv_results = cross_validate(
            pipeline, X, y, 
            cv=cv, 
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )
        
        training_time = time.time() - start_time
        
        # Calculate mean and std of metrics
        results = {
            'sampling_strategy': sampling_config,
            'cv_folds': cv_folds,
            'training_time': training_time,
        }
        
        for metric in scoring.keys():
            scores = cv_results[f'test_{metric}']
            results[f'{metric}_mean'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)
            results[f'{metric}_scores'] = scores.tolist()
        
        # Get sampling configuration details
        config_params = SAMPLING_CONFIGURATIONS.get(sampling_config, {})
        results['sampling_parameters'] = config_params
        
        # Original class distribution
        original_dist = Counter(y)
        results['original_class_distribution'] = dict(original_dist)
        results['original_fraud_rate'] = y.mean()
        
        # Print results
        print(f"Results for {sampling_config}:")
        print(f"  Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
        print(f"  Recall:    {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
        print(f"  F1-Score:  {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
        print(f"  ROC-AUC:   {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
        print(f"  AUPRC:     {results['average_precision_mean']:.4f} ± {results['average_precision_std']:.4f}")
        print(f"  Training time: {training_time:.2f} seconds")
        
        return results
    
    def log_experiment_to_mlflow(self, results: Dict[str, Any]):
        """Log experiment results to MLflow."""
        try:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("sampling_strategy", results['sampling_strategy'])
                mlflow.log_param("cv_folds", results['cv_folds'])
                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_param("class_weight", "balanced")
                mlflow.log_param("solver", "liblinear")
                mlflow.log_param("max_iter", 1000)
                
                # Log sampling parameters
                for key, value in results['sampling_parameters'].items():
                    mlflow.log_param(f"sampling_{key}", value)
                
                # Log class distribution
                mlflow.log_param("original_fraud_rate", results['original_fraud_rate'])
                for class_label, count in results['original_class_distribution'].items():
                    mlflow.log_param(f"original_class_{class_label}_count", count)
                
                # Log metrics (mean values)
                mlflow.log_metric("precision", results['precision_mean'])
                mlflow.log_metric("recall", results['recall_mean'])
                mlflow.log_metric("f1_score", results['f1_mean'])
                mlflow.log_metric("roc_auc", results['roc_auc_mean'])
                mlflow.log_metric("auprc", results['average_precision_mean'])
                mlflow.log_metric("training_time", results['training_time'])
                
                # Log metrics (std values)
                mlflow.log_metric("precision_std", results['precision_std'])
                mlflow.log_metric("recall_std", results['recall_std'])
                mlflow.log_metric("f1_score_std", results['f1_std'])
                mlflow.log_metric("roc_auc_std", results['roc_auc_std'])
                mlflow.log_metric("auprc_std", results['average_precision_std'])
                
                # Log cross-validation scores as artifacts
                cv_scores_df = pd.DataFrame({
                    'fold': range(len(results['precision_scores'])),
                    'precision': results['precision_scores'],
                    'recall': results['recall_scores'],
                    'f1_score': results['f1_scores'],
                    'roc_auc': results['roc_auc_scores'],
                    'auprc': results['average_precision_scores']
                })
                
                cv_scores_path = "cv_scores.csv"
                cv_scores_df.to_csv(cv_scores_path, index=False)
                mlflow.log_artifact(cv_scores_path)
                os.remove(cv_scores_path)
                
                run_id = mlflow.active_run().info.run_id
                print(f"Logged to MLflow run: {run_id}")
                
        except Exception as e:
            print(f"Error logging to MLflow: {e}")
    
    def run_all_experiments(self, sampling_strategies: List[str] = None) -> pd.DataFrame:
        """
        Run experiments for all sampling strategies.
        
        Args:
            sampling_strategies: List of sampling strategy names to test
            
        Returns:
            DataFrame with all experiment results
        """
        if sampling_strategies is None:
            sampling_strategies = list(SAMPLING_CONFIGURATIONS.keys())
        
        print(f"Running {len(sampling_strategies)} sampling experiments...")
        print(f"Strategies: {sampling_strategies}")
        
        # Load data
        X, y = self.load_data()
        
        all_results = []
        
        for strategy in sampling_strategies:
            try:
                # Run experiment
                results = self.run_single_experiment(X, y, strategy)
                all_results.append(results)
                
                # Log to MLflow
                self.log_experiment_to_mlflow(results)
                
                # Track in local tracker
                self.tracker.track_experiment(
                    strategy=strategy,
                    original_distribution=results['original_class_distribution'],
                    sampled_distribution=results['original_class_distribution'],  # Will be updated
                    performance_metrics={
                        'precision': results['precision_mean'],
                        'recall': results['recall_mean'],
                        'f1_score': results['f1_mean'],
                        'roc_auc': results['roc_auc_mean'],
                        'auprc': results['average_precision_mean']
                    },
                    parameters=results['sampling_parameters']
                )
                
            except Exception as e:
                print(f"Error running experiment {strategy}: {e}")
                continue
        
        # Create summary DataFrame
        summary_df = self._create_summary_dataframe(all_results)
        
        # Save results
        self._save_results(summary_df, all_results)
        
        return summary_df
    
    def _create_summary_dataframe(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create summary DataFrame from all results."""
        summary_data = []
        
        for result in all_results:
            row = {
                'sampling_strategy': result['sampling_strategy'],
                'precision_mean': result['precision_mean'],
                'precision_std': result['precision_std'],
                'recall_mean': result['recall_mean'],
                'recall_std': result['recall_std'],
                'f1_mean': result['f1_mean'],
                'f1_std': result['f1_std'],
                'roc_auc_mean': result['roc_auc_mean'],
                'roc_auc_std': result['roc_auc_std'],
                'auprc_mean': result['average_precision_mean'],
                'auprc_std': result['average_precision_std'],
                'training_time': result['training_time']
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Sort by F1-Score (descending)
        df = df.sort_values('f1_mean', ascending=False).reset_index(drop=True)
        
        return df
    
    def _save_results(self, summary_df: pd.DataFrame, all_results: List[Dict[str, Any]]):
        """Save experiment results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_path = f"docs/sprints/Sprint_2_2_Sampling_Results_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary to: {summary_path}")
        
        # Save detailed results
        detailed_path = f"docs/sprints/Sprint_2_2_Detailed_Results_{timestamp}.pkl"
        joblib.dump(all_results, detailed_path)
        print(f"Saved detailed results to: {detailed_path}")
    
    def analyze_results(self, summary_df: pd.DataFrame):
        """Analyze and display experiment results."""
        print(f"\n{'='*80}")
        print("SAMPLING EXPERIMENT ANALYSIS - SPRINT 2.2")
        print(f"{'='*80}")
        
        print(f"\nTOP 5 STRATEGIES BY F1-SCORE:")
        print("-" * 50)
        top_5 = summary_df.head(5)
        for idx, row in top_5.iterrows():
            print(f"{idx+1}. {row['sampling_strategy']:<20} | "
                  f"F1: {row['f1_mean']:.4f} ± {row['f1_std']:.4f} | "
                  f"P: {row['precision_mean']:.4f} | "
                  f"R: {row['recall_mean']:.4f}")
        
        print(f"\nBEST STRATEGY DETAILS:")
        print("-" * 30)
        best = summary_df.iloc[0]
        print(f"Strategy: {best['sampling_strategy']}")
        print(f"Precision: {best['precision_mean']:.4f} ± {best['precision_std']:.4f}")
        print(f"Recall:    {best['recall_mean']:.4f} ± {best['recall_std']:.4f}")
        print(f"F1-Score:  {best['f1_mean']:.4f} ± {best['f1_std']:.4f}")
        print(f"ROC-AUC:   {best['roc_auc_mean']:.4f} ± {best['roc_auc_std']:.4f}")
        print(f"AUPRC:     {best['auprc_mean']:.4f} ± {best['auprc_std']:.4f}")
        
        # Compare with no sampling
        no_sampling = summary_df[summary_df['sampling_strategy'] == 'no_sampling']
        if not no_sampling.empty:
            baseline = no_sampling.iloc[0]
            print(f"\nIMPROVEMENT OVER NO SAMPLING:")
            print("-" * 35)
            print(f"F1-Score improvement: {((best['f1_mean'] / baseline['f1_mean']) - 1) * 100:.1f}%")
            print(f"Precision improvement: {((best['precision_mean'] / baseline['precision_mean']) - 1) * 100:.1f}%")
            print(f"Recall change: {((best['recall_mean'] / baseline['recall_mean']) - 1) * 100:.1f}%")


def main():
    """Main execution function."""
    print("="*80)
    print("SPRINT 2.2: SAMPLING STRATEGIES EXPERIMENT")
    print("="*80)
    
    # Initialize experiment runner
    runner = SamplingExperimentRunner()
    
    # Define strategies to test
    strategies_to_test = [
        'no_sampling',           # Baseline
        'smote_default',         # Standard SMOTE
        'smote_conservative',    # Conservative SMOTE
        'smote_balanced',        # Fully balanced SMOTE
        'adasyn_default',        # ADASYN
        'adasyn_conservative',   # Conservative ADASYN
        'borderline_smote',      # Borderline SMOTE
        'random_under_balanced', # Random undersampling
        'random_under_conservative', # Conservative undersampling
        'smote_tomek',          # SMOTE + Tomek links
        'smote_enn'             # SMOTE + ENN
    ]
    
    # Run all experiments
    results_df = runner.run_all_experiments(strategies_to_test)
    
    # Analyze results
    runner.analyze_results(results_df)
    
    print(f"\n{'='*80}")
    print("SPRINT 2.2 COMPLETED SUCCESSFULLY!")
    print("Check MLflow UI at http://localhost:5001 for detailed tracking")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

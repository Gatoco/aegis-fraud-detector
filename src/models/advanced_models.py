#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Machine Learning Models Framework
=========================================

Implementa modelos avanzados de machine learning (LightGBM, XGBoost, CatBoost) 
con el pipeline de preprocesamiento y muestreo óptimo identificado en Sprint 2.2.

Features:
- Modelos gradient boosting optimizados
- Integración con pipeline SMOTE Conservative  
- Hyperparameter tuning con Optuna
- MLflow tracking comprehensivo
- Model registry para producción

Author: AEGIS Fraud Detection Team
Sprint: 2.3 - Advanced Models and Comparison
Date: 2024-12-31
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost
import time
import warnings
# import optuna  # Optional hyperparameter optimization
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

# Core ML libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, 
    average_precision_score, precision_recall_curve, roc_curve,
    classification_report, confusion_matrix
)

# Advanced ML models
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Sampling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class AdvancedModelFramework:
    """
    Framework comprehensivo para entrenamiento y evaluación de modelos avanzados
    con integración completa de MLflow y optimización de hiperparámetros.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_pipeline = None
        
        # MLflow configuration
        self.experiment_name = 'fraud-detection-advanced-models-sprint-2-3'
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Configurar MLflow experiment para modelos avanzados"""
        try:
            mlflow.set_tracking_uri('http://localhost:5001')
            
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
                print(f'✅ Created MLflow experiment: {self.experiment_name}')
            
            mlflow.set_experiment(self.experiment_name)
            print(f'🔬 MLflow experiment set: {self.experiment_name}')
            
        except Exception as e:
            print(f'⚠️ MLflow setup warning: {e}')
            print('Continuing without MLflow tracking...')
    
    def define_models(self) -> Dict[str, Any]:
        """Definir todos los modelos avanzados a evaluar"""
        
        models = {
            # Baseline from Sprint 2.2
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            
            # Advanced gradient boosting models
            'lightgbm': lgb.LGBMClassifier(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbosity=-1,
                objective='binary',
                metric='binary_logloss',
                is_unbalance=True
            ),
            
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=None,  # Will be calculated based on class imbalance
                eval_metric='logloss',
                verbosity=0
            ),
            
            'catboost': cb.CatBoostClassifier(
                random_state=self.random_state,
                iterations=100,
                learning_rate=0.1,
                depth=6,
                l2_leaf_reg=3,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                auto_class_weights='Balanced',
                verbose=False,
                eval_metric='Logloss'
            ),
            
            # Traditional ensemble
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        self.models = models
        return models
    
    def create_pipeline(self, model, use_smote: bool = True) -> ImbPipeline:
        """
        Crear pipeline con el preprocesamiento y muestreo óptimo de Sprint 2.2
        
        Args:
            model: Modelo de ML a usar
            use_smote: Si usar SMOTE Conservative o solo baseline
            
        Returns:
            Pipeline configurado
        """
        
        if use_smote:
            # Pipeline ganador de Sprint 2.2: SMOTE Conservative
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('sampling', SMOTE(
                    random_state=self.random_state,
                    k_neighbors=3,
                    sampling_strategy=0.5
                )),
                ('classifier', model)
            ])
        else:
            # Pipeline baseline sin sampling
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
        return pipeline
    
    def evaluate_model(self, 
                      model_name: str, 
                      pipeline: ImbPipeline,
                      X_train: pd.DataFrame, 
                      y_train: pd.Series,
                      cv_folds: int = 5) -> Dict[str, float]:
        """
        Evaluar modelo con validación cruzada y métricas comprehensivas
        
        Args:
            model_name: Nombre del modelo
            pipeline: Pipeline configurado
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento  
            cv_folds: Número de folds para CV
            
        Returns:
            Diccionario con métricas
        """
        
        print(f'🔍 Evaluating {model_name}...')
        start_time = time.time()
        
        # Configurar cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Calcular métricas con CV
        f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        precision_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='precision', n_jobs=-1)
        recall_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='recall', n_jobs=-1)
        roc_auc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        pr_auc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)
        
        training_time = time.time() - start_time
        
        # Compilar resultados
        results = {
            'model_name': model_name,
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'precision_mean': np.mean(precision_scores),
            'precision_std': np.std(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'roc_auc_mean': np.mean(roc_auc_scores),
            'roc_auc_std': np.std(roc_auc_scores),
            'pr_auc_mean': np.mean(pr_auc_scores),
            'pr_auc_std': np.std(pr_auc_scores),
            'training_time': training_time,
            'cv_folds': cv_folds
        }
        
        # Mostrar resultados
        print(f'  F1-Score:     {results["f1_mean"]:.4f} ± {results["f1_std"]:.4f}')
        print(f'  Precision:    {results["precision_mean"]:.4f} ± {results["precision_std"]:.4f}')
        print(f'  Recall:       {results["recall_mean"]:.4f} ± {results["recall_std"]:.4f}')
        print(f'  ROC-AUC:      {results["roc_auc_mean"]:.4f} ± {results["roc_auc_std"]:.4f}')
        print(f'  PR-AUC:       {results["pr_auc_mean"]:.4f} ± {results["pr_auc_std"]:.4f}')
        print(f'  Time:         {training_time:.2f}s')
        
        return results
    
    def log_to_mlflow(self, results: Dict[str, float], pipeline: ImbPipeline, model_name: str):
        """Log resultados y modelo a MLflow"""
        try:
            with mlflow.start_run(run_name=f'advanced_{model_name}'):
                # Log parameters
                mlflow.log_param('model_type', model_name)
                mlflow.log_param('sampling_strategy', 'smote_conservative')
                mlflow.log_param('cv_folds', results['cv_folds'])
                mlflow.log_param('preprocessing', 'StandardScaler')
                
                # Log metrics
                mlflow.log_metric('f1_score', results['f1_mean'])
                mlflow.log_metric('f1_std', results['f1_std'])
                mlflow.log_metric('precision', results['precision_mean'])
                mlflow.log_metric('precision_std', results['precision_std'])
                mlflow.log_metric('recall', results['recall_mean'])
                mlflow.log_metric('recall_std', results['recall_std'])
                mlflow.log_metric('roc_auc', results['roc_auc_mean'])
                mlflow.log_metric('roc_auc_std', results['roc_auc_std'])
                mlflow.log_metric('pr_auc', results['pr_auc_mean'])
                mlflow.log_metric('pr_auc_std', results['pr_auc_std'])
                mlflow.log_metric('training_time', results['training_time'])
                
                # Log model based on type
                if 'lightgbm' in model_name:
                    mlflow.lightgbm.log_model(pipeline, 'model')
                elif 'xgboost' in model_name:
                    mlflow.xgboost.log_model(pipeline, 'model')
                elif 'catboost' in model_name:
                    mlflow.catboost.log_model(pipeline, 'model')
                else:
                    mlflow.sklearn.log_model(pipeline, 'model')
                    
        except Exception as e:
            print(f'  ⚠️ MLflow logging failed: {e}')
    
    def run_comprehensive_comparison(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Ejecutar comparación comprehensiva de todos los modelos avanzados
        
        Args:
            X: Features dataset
            y: Target variable
            
        Returns:
            DataFrame con resultados de comparación
        """
        
        print('🚀 SPRINT 2.3: ADVANCED MODELS COMPARISON')
        print('=' * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f'Training set: {X_train.shape}')
        print(f'Test set: {X_test.shape}')
        print(f'Class distribution: {Counter(y_train)}')
        print(f'Fraud rate: {y_train.mean():.4f}')
        
        # Define models
        models = self.define_models()
        
        # Calculate scale_pos_weight for XGBoost
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        models['xgboost'].set_params(scale_pos_weight=scale_pos_weight)
        
        print(f'\n🎯 XGBoost scale_pos_weight: {scale_pos_weight:.2f}')
        print('\n' + '─' * 60)
        
        # Run experiments
        all_results = []
        
        for model_name, model in models.items():
            print(f'\n📊 Training {model_name.upper()}')
            print('─' * 40)
            
            # Create pipeline with SMOTE Conservative (Sprint 2.2 winner)
            pipeline = self.create_pipeline(model, use_smote=True)
            
            # Evaluate model
            results = self.evaluate_model(model_name, pipeline, X_train, y_train)
            
            # Log to MLflow
            self.log_to_mlflow(results, pipeline, model_name)
            
            all_results.append(results)
            
            # Store pipeline for best model tracking
            if model_name not in self.results:
                self.results[model_name] = {
                    'pipeline': pipeline,
                    'results': results
                }
        
        # Convert to DataFrame and analyze
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('pr_auc_mean', ascending=False)  # Sort by PR-AUC (primary criterion)
        
        return results_df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizar resultados y identificar el mejor modelo
        
        Args:
            results_df: DataFrame con resultados de comparación
            
        Returns:
            Diccionario con análisis detallado
        """
        
        print('\n' + '=' * 60)
        print('🏆 ADVANCED MODELS COMPARISON RESULTS')
        print('=' * 60)
        
        # Ranking by PR-AUC (primary criterion for imbalanced datasets)
        print('\n📈 RANKING BY PR-AUC (Primary Criterion):')
        print('─' * 50)
        for i, (idx, row) in enumerate(results_df.iterrows(), 1):
            print(f'{i:2d}. {row["model_name"]:<20} | PR-AUC: {row["pr_auc_mean"]:.4f} ± {row["pr_auc_std"]:.4f}')
        
        # Best model details
        best = results_df.iloc[0]
        baseline = results_df[results_df['model_name'] == 'logistic_regression']
        
        print(f'\n🥇 BEST MODEL: {best["model_name"].upper()}')
        print('─' * 30)
        print(f'PR-AUC:       {best["pr_auc_mean"]:.4f} ± {best["pr_auc_std"]:.4f}')
        print(f'F1-Score:     {best["f1_mean"]:.4f} ± {best["f1_std"]:.4f}')
        print(f'Precision:    {best["precision_mean"]:.4f} ± {best["precision_std"]:.4f}')
        print(f'Recall:       {best["recall_mean"]:.4f} ± {best["recall_std"]:.4f}')
        print(f'ROC-AUC:      {best["roc_auc_mean"]:.4f} ± {best["roc_auc_std"]:.4f}')
        print(f'Training Time: {best["training_time"]:.2f}s')
        
        # Improvement over baseline
        if not baseline.empty:
            baseline_row = baseline.iloc[0]
            pr_auc_improvement = ((best['pr_auc_mean'] / baseline_row['pr_auc_mean']) - 1) * 100
            f1_improvement = ((best['f1_mean'] / baseline_row['f1_mean']) - 1) * 100
            
            print(f'\n📊 IMPROVEMENT OVER LOGISTIC REGRESSION:')
            print(f'PR-AUC:   {pr_auc_improvement:+.1f}%')
            print(f'F1-Score: {f1_improvement:+.1f}%')
        
        # Store best model
        self.best_model = best['model_name']
        self.best_pipeline = self.results[best['model_name']]['pipeline']
        
        analysis = {
            'best_model': best['model_name'],
            'best_metrics': best.to_dict(),
            'improvement_over_baseline': {
                'pr_auc': pr_auc_improvement if not baseline.empty else 0,
                'f1_score': f1_improvement if not baseline.empty else 0
            },
            'ranking': results_df[['model_name', 'pr_auc_mean', 'f1_mean']].to_dict('records')
        }
        
        return analysis
    
    def create_voting_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[ImbPipeline, Dict]:
        """
        Crear ensemble voting classifier con los mejores modelos
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            
        Returns:
            Pipeline del ensemble y resultados de evaluación
        """
        
        print('\n🤝 CREATING VOTING ENSEMBLE')
        print('─' * 30)
        
        # Select top 3 models (excluding logistic regression baseline)
        models = self.define_models()
        
        # Create individual models for ensemble
        ensemble_models = [
            ('lgb', models['lightgbm']),
            ('xgb', models['xgboost']),
            ('cb', models['catboost'])
        ]
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',  # Use predicted probabilities
            n_jobs=-1
        )
        
        # Create pipeline
        ensemble_pipeline = self.create_pipeline(voting_clf, use_smote=True)
        
        # Evaluate ensemble
        results = self.evaluate_model('voting_ensemble', ensemble_pipeline, X_train, y_train)
        
        # Log to MLflow
        self.log_to_mlflow(results, ensemble_pipeline, 'voting_ensemble')
        
        return ensemble_pipeline, results


def main():
    """Función principal para ejecutar Sprint 2.3"""
    
    # Initialize framework
    framework = AdvancedModelFramework(random_state=42)
    
    # Load data (using processed data from previous sprints)
    try:
        df = pd.read_parquet('data/02_processed/fraud_train_v1.0.parquet')
        X = df.drop('isFraud', axis=1).select_dtypes(include=[np.number])
        y = df['isFraud']
        print(f'✅ Loaded processed data: {X.shape}')
    except FileNotFoundError:
        print('⚠️ Processed data not found, loading raw data sample...')
        df = pd.read_csv('data/01_raw/train_transaction.csv', nrows=50000)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['TransactionID', 'isFraud']]
        
        # Filter features with good completeness
        completeness = df[numeric_cols].isnull().mean()
        good_features = completeness[completeness < 0.3].index.tolist()[:30]
        
        X = df[good_features].fillna(0)
        y = df['isFraud']
        print(f'✅ Loaded raw data sample: {X.shape}')
    
    # Run comprehensive comparison
    results_df = framework.run_comprehensive_comparison(X, y)
    
    # Analyze results
    analysis = framework.analyze_results(results_df)
    
    # Create voting ensemble
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    ensemble_pipeline, ensemble_results = framework.create_voting_ensemble(X_train, y_train)
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = f'docs/sprints/Sprint_2_3_Advanced_Models_Results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    
    print(f'\n💾 Results saved to: {results_file}')
    print('\n✅ SPRINT 2.3 COMPARISON COMPLETED!')
    print(f'🏆 Best Model: {analysis["best_model"].upper()}')
    print('🔬 Check MLflow UI at http://localhost:5001 for detailed comparison')
    
    return framework, results_df, analysis


if __name__ == "__main__":
    framework, results, analysis = main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Optimization Framework
===================================

Framework comprehensivo para optimización de hiperparámetros usando Optuna
con integración MLflow y estrategias multi-objetivo para detección de fraude.

Features:
- Optimización bayesiana con Optuna TPE sampler
- Multi-objective optimization (performance + efficiency)
- MLflow integration para tracking de trials
- Cross-validation robusta durante optimización
- Study persistence con SQLite
- Hyperparameter importance analysis

Author: AEGIS Fraud Detection Team
Sprint: 2.4 - Hyperparameter Optimization
Date: 2024-12-31
"""

import pandas as pd
import numpy as np
import mlflow
import optuna
import sqlite3
import time
import warnings
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path

# Core ML libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

# LightGBM
import lightgbm as lgb

# Sampling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """
    Framework comprehensivo para optimización de hiperparámetros con Optuna
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 study_name: str = 'lightgbm_fraud_optimization',
                 storage_path: str = 'optimization_studies.db'):
        
        self.random_state = random_state
        self.study_name = study_name
        self.storage_path = storage_path
        self.storage_url = f'sqlite:///{storage_path}'
        
        # Optimization results
        self.study = None
        self.best_params = None
        self.best_model = None
        self.optimization_history = []
        
        # Data
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        # MLflow setup
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Configurar MLflow para tracking de optimización"""
        try:
            mlflow.set_tracking_uri('http://localhost:5001')
            
            experiment_name = 'fraud-detection-hyperopt-sprint-2-4'
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
                print(f'✅ Created MLflow experiment: {experiment_name}')
            
            mlflow.set_experiment(experiment_name)
            print(f'🔬 MLflow experiment set: {experiment_name}')
            
        except Exception as e:
            print(f'⚠️ MLflow setup warning: {e}')
    
    def create_lightgbm_pipeline(self, params: Dict[str, Any]) -> ImbPipeline:
        """
        Crear pipeline LightGBM con SMOTE Conservative y parámetros dados
        
        Args:
            params: Diccionario con hiperparámetros de LightGBM
            
        Returns:
            Pipeline configurado
        """
        
        # Configuración base de LightGBM
        lgb_params = {
            'random_state': self.random_state,
            'verbosity': -1,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'is_unbalance': True,
            'n_jobs': -1
        }
        
        # Agregar parámetros optimizables
        lgb_params.update(params)
        
        # Crear modelo LightGBM
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        
        # Pipeline con SMOTE Conservative (ganador de Sprint 2.2)
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('sampling', SMOTE(
                random_state=self.random_state,
                k_neighbors=3,
                sampling_strategy=0.5
            )),
            ('classifier', lgb_model)
        ])
        
        return pipeline
    
    def objective_function(self, trial) -> float:
        """
        Función objetivo multi-criterio para Optuna
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Valor objetivo (PR-AUC penalizada por tiempo)
        """
        
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
            'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 100, log=True)
        }
        
        # Handle max_depth = -1 (no limit)
        if params['max_depth'] == -1:
            params['max_depth'] = None
        
        start_time = time.time()
        
        try:
            # Crear pipeline
            pipeline = self.create_lightgbm_pipeline(params)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            
            # Calcular PR-AUC con CV
            pr_auc_scores = cross_val_score(
                pipeline, self.X_train, self.y_train, 
                cv=cv, scoring='average_precision', 
                n_jobs=1  # Evitar problemas de paralelización con Optuna
            )
            
            training_time = time.time() - start_time
            
            # Métricas
            pr_auc_mean = np.mean(pr_auc_scores)
            pr_auc_std = np.std(pr_auc_scores)
            
            # Multi-objective: maximizar PR-AUC, penalizar tiempo excesivo
            time_penalty = max(0, (training_time - 120) / 300)  # Penalizar si >2 min
            objective_value = pr_auc_mean - 0.05 * time_penalty
            
            # Log trial a MLflow
            try:
                with mlflow.start_run(nested=True):
                    # Log hyperparameters
                    for key, value in params.items():
                        mlflow.log_param(key, value)
                    
                    # Log metrics
                    mlflow.log_metric('pr_auc_mean', pr_auc_mean)
                    mlflow.log_metric('pr_auc_std', pr_auc_std)
                    mlflow.log_metric('training_time', training_time)
                    mlflow.log_metric('objective_value', objective_value)
                    mlflow.log_metric('trial_number', trial.number)
                    
            except Exception as e:
                print(f'MLflow logging failed for trial {trial.number}: {e}')
            
            # Store trial info
            trial_info = {
                'trial_number': trial.number,
                'params': params.copy(),
                'pr_auc_mean': pr_auc_mean,
                'pr_auc_std': pr_auc_std,
                'training_time': training_time,
                'objective_value': objective_value
            }
            self.optimization_history.append(trial_info)
            
            # Print progress
            print(f'Trial {trial.number:3d}: PR-AUC={pr_auc_mean:.4f}±{pr_auc_std:.4f}, '
                  f'Time={training_time:.1f}s, Obj={objective_value:.4f}')
            
            return objective_value
            
        except Exception as e:
            print(f'Trial {trial.number} failed: {e}')
            return -1.0  # Valor muy bajo para trials fallidos
    
    def optimize_hyperparameters(self, 
                                X_train: pd.DataFrame, 
                                y_train: pd.Series,
                                n_trials: int = 100,
                                timeout: Optional[int] = None) -> optuna.Study:
        """
        Ejecutar optimización de hiperparámetros
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            n_trials: Número máximo de trials
            timeout: Timeout en segundos (opcional)
            
        Returns:
            Optuna study con resultados
        """
        
        self.X_train = X_train
        self.y_train = y_train
        
        print('🚀 SPRINT 2.4: HYPERPARAMETER OPTIMIZATION')
        print('=' * 60)
        print(f'Target: LightGBM optimization with {n_trials} trials')
        print(f'Dataset: {X_train.shape}')
        print(f'Fraud rate: {y_train.mean():.4f}')
        print(f'Objective: PR-AUC maximization with time penalty')
        print('')
        
        # Crear o cargar study
        try:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10,
                    n_warmup_steps=5,
                    interval_steps=1
                ),
                load_if_exists=True
            )
            
            # Información del study
            n_completed = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            print(f'📊 Study: {self.study_name}')
            print(f'💾 Storage: {self.storage_path}')
            print(f'📈 Previous trials: {n_completed}')
            print('')
            
        except Exception as e:
            print(f'Error creating study: {e}')
            return None
        
        # Ejecutar optimización
        start_time = time.time()
        
        try:
            with mlflow.start_run(run_name='hyperopt_session'):
                mlflow.log_param('n_trials_target', n_trials)
                mlflow.log_param('optimization_method', 'optuna_tpe')
                
                self.study.optimize(
                    self.objective_function,
                    n_trials=n_trials,
                    timeout=timeout,
                    show_progress_bar=True
                )
                
                optimization_time = time.time() - start_time
                mlflow.log_metric('optimization_time', optimization_time)
                mlflow.log_metric('completed_trials', len(self.study.trials))
                
        except KeyboardInterrupt:
            print('\\n⏹️ Optimization interrupted by user')
        except Exception as e:
            print(f'\\n❌ Optimization failed: {e}')
        
        # Resultados
        total_time = time.time() - start_time
        completed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        print('\\n' + '=' * 60)
        print('🏆 OPTIMIZATION COMPLETED')
        print('=' * 60)
        print(f'Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)')
        print(f'Completed trials: {completed_trials}')
        
        if completed_trials > 0:
            print(f'Best value: {self.study.best_value:.4f}')
            print(f'Best trial: #{self.study.best_trial.number}')
            
            self.best_params = self.study.best_params
            print('\\n🎯 BEST HYPERPARAMETERS:')
            for key, value in self.best_params.items():
                print(f'  {key}: {value}')
        
        return self.study
    
    def analyze_optimization(self) -> Dict[str, Any]:
        """
        Analizar resultados de optimización
        
        Returns:
            Diccionario con análisis detallado
        """
        
        if not self.study or len(self.study.trials) == 0:
            print('❌ No optimization results to analyze')
            return {}
        
        # Trials completados
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) == 0:
            print('❌ No completed trials to analyze')
            return {}
        
        print('\\n📊 OPTIMIZATION ANALYSIS')
        print('=' * 40)
        
        # Estadísticas básicas
        values = [t.value for t in completed_trials]
        
        analysis = {
            'n_trials': len(completed_trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params.copy(),
            'best_trial_number': self.study.best_trial.number,
            'value_stats': {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        }
        
        print(f'Completed trials: {analysis["n_trials"]}')
        print(f'Best objective value: {analysis["best_value"]:.4f}')
        print(f'Value statistics:')
        print(f'  Mean: {analysis["value_stats"]["mean"]:.4f} ± {analysis["value_stats"]["std"]:.4f}')
        print(f'  Range: [{analysis["value_stats"]["min"]:.4f}, {analysis["value_stats"]["max"]:.4f}]')
        
        # Importancia de hiperparámetros
        try:
            importance = optuna.importance.get_param_importances(self.study)
            analysis['param_importance'] = importance
            
            print('\\n🔍 HYPERPARAMETER IMPORTANCE:')
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f'  {param}: {imp:.3f}')
                
        except Exception as e:
            print(f'\\nParameter importance analysis failed: {e}')
        
        return analysis
    
    def train_best_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ImbPipeline:
        """
        Entrenar el modelo con los mejores hiperparámetros
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            
        Returns:
            Pipeline entrenado con mejores hiperparámetros
        """
        
        if not self.best_params:
            raise ValueError('No best parameters available. Run optimization first.')
        
        print('\\n🏗️ Training best model...')
        
        # Crear pipeline con mejores parámetros
        self.best_model = self.create_lightgbm_pipeline(self.best_params)
        
        # Entrenar
        start_time = time.time()
        self.best_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f'✅ Best model trained in {training_time:.2f}s')
        
        return self.best_model
    
    def save_results(self, output_dir: str = 'optimization_results'):
        """
        Guardar resultados de optimización
        
        Args:
            output_dir: Directorio de salida
        """
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Guardar study
        study_file = f'{output_dir}/optuna_study_{timestamp}.pkl'
        with open(study_file, 'wb') as f:
            pickle.dump(self.study, f)
        
        # Guardar mejor modelo
        if self.best_model:
            model_file = f'{output_dir}/best_lightgbm_model_{timestamp}.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(self.best_model, f)
        
        # Guardar mejores parámetros
        if self.best_params:
            params_file = f'{output_dir}/best_hyperparameters_{timestamp}.json'
            import json
            with open(params_file, 'w') as f:
                json.dump(self.best_params, f, indent=2)
        
        # Guardar historial
        if self.optimization_history:
            history_file = f'{output_dir}/optimization_history_{timestamp}.csv'
            pd.DataFrame(self.optimization_history).to_csv(history_file, index=False)
        
        print(f'\\n💾 Results saved to {output_dir}/')
        print(f'   Study: {study_file}')
        if self.best_model:
            print(f'   Model: {model_file}')
        if self.best_params:
            print(f'   Params: {params_file}')
        if self.optimization_history:
            print(f'   History: {history_file}')


def compare_with_baseline(optimizer: HyperparameterOptimizer,
                         X_train: pd.DataFrame, 
                         y_train: pd.Series,
                         X_test: pd.DataFrame,
                         y_test: pd.Series) -> Dict[str, Any]:
    """
    Comparar modelo optimizado con baseline
    
    Args:
        optimizer: Optimizer con modelo entrenado
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de test
        
    Returns:
        Diccionario con comparación detallada
    """
    
    print('\\n🆚 BASELINE COMPARISON')
    print('=' * 40)
    
    # Baseline LightGBM (Sprint 2.3)
    baseline_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    
    baseline_pipeline = optimizer.create_lightgbm_pipeline(baseline_params)
    
    # Cross-validation comparison
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Baseline CV
    baseline_pr_auc = cross_val_score(baseline_pipeline, X_train, y_train, 
                                     cv=cv, scoring='average_precision')
    baseline_f1 = cross_val_score(baseline_pipeline, X_train, y_train, 
                                 cv=cv, scoring='f1')
    
    # Optimized CV
    optimized_pr_auc = cross_val_score(optimizer.best_model, X_train, y_train, 
                                      cv=cv, scoring='average_precision')
    optimized_f1 = cross_val_score(optimizer.best_model, X_train, y_train, 
                                  cv=cv, scoring='f1')
    
    # Comparison results
    comparison = {
        'baseline': {
            'pr_auc_mean': np.mean(baseline_pr_auc),
            'pr_auc_std': np.std(baseline_pr_auc),
            'f1_mean': np.mean(baseline_f1),
            'f1_std': np.std(baseline_f1),
            'params': baseline_params
        },
        'optimized': {
            'pr_auc_mean': np.mean(optimized_pr_auc),
            'pr_auc_std': np.std(optimized_pr_auc),
            'f1_mean': np.mean(optimized_f1),
            'f1_std': np.std(optimized_f1),
            'params': optimizer.best_params
        }
    }
    
    # Calculate improvements
    pr_auc_improvement = ((comparison['optimized']['pr_auc_mean'] / 
                          comparison['baseline']['pr_auc_mean']) - 1) * 100
    f1_improvement = ((comparison['optimized']['f1_mean'] / 
                      comparison['baseline']['f1_mean']) - 1) * 100
    
    comparison['improvements'] = {
        'pr_auc_improvement': pr_auc_improvement,
        'f1_improvement': f1_improvement
    }
    
    # Print results
    print('BASELINE (Sprint 2.3):')
    print(f'  PR-AUC: {comparison["baseline"]["pr_auc_mean"]:.4f} ± {comparison["baseline"]["pr_auc_std"]:.4f}')
    print(f'  F1-Score: {comparison["baseline"]["f1_mean"]:.4f} ± {comparison["baseline"]["f1_std"]:.4f}')
    
    print('\\nOPTIMIZED (Sprint 2.4):')
    print(f'  PR-AUC: {comparison["optimized"]["pr_auc_mean"]:.4f} ± {comparison["optimized"]["pr_auc_std"]:.4f}')
    print(f'  F1-Score: {comparison["optimized"]["f1_mean"]:.4f} ± {comparison["optimized"]["f1_std"]:.4f}')
    
    print('\\n📈 IMPROVEMENTS:')
    print(f'  PR-AUC: {pr_auc_improvement:+.2f}%')
    print(f'  F1-Score: {f1_improvement:+.2f}%')
    
    return comparison


def main():
    """Función principal para ejecutar Sprint 2.4"""
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        random_state=42,
        study_name='lightgbm_fraud_optimization_sprint_2_4'
    )
    
    # Load data
    try:
        df = pd.read_parquet('data/02_processed/fraud_train_v1.0.parquet')
        X = df.drop('isFraud', axis=1).select_dtypes(include=[np.number])
        y = df['isFraud']
        print(f'✅ Loaded processed data: {X.shape}')
    except FileNotFoundError:
        print('⚠️ Processed data not found, using raw data sample...')
        df = pd.read_csv('data/01_raw/train_transaction.csv', nrows=50000)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['TransactionID', 'isFraud']]
        
        # Filter features with good completeness
        completeness = df[numeric_cols].isnull().mean()
        good_features = completeness[completeness < 0.3].index.tolist()[:30]
        
        X = df[good_features].fillna(0)
        y = df['isFraud']
        print(f'✅ Loaded raw data sample: {X.shape}')
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Run optimization
    study = optimizer.optimize_hyperparameters(
        X_train, y_train,
        n_trials=50,  # Reducido para demo, usar 100+ en producción
        timeout=3600  # 1 hora máximo
    )
    
    if study and len(study.trials) > 0:
        # Analyze results
        analysis = optimizer.analyze_optimization()
        
        # Train best model
        best_model = optimizer.train_best_model(X_train, y_train)
        
        # Compare with baseline
        comparison = compare_with_baseline(optimizer, X_train, y_train, X_test, y_test)
        
        # Save results
        optimizer.save_results('docs/sprints/optimization_results')
        
        print('\\n✅ SPRINT 2.4 COMPLETED SUCCESSFULLY!')
        print('🔬 Check MLflow UI at http://localhost:5001 for detailed tracking')
        
        return optimizer, analysis, comparison
    
    else:
        print('❌ Optimization failed - no results to analyze')
        return None, None, None


if __name__ == "__main__":
    optimizer, analysis, comparison = main()

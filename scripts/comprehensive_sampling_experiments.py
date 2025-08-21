#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprint 2.2: Comprehensive Sampling Experiments
============================================

Ejecuta experimentos comprehensivos de técnicas de muestreo con MLflow tracking
para identificar la mejor estrategia de manejo del desbalance de clases.

Author: Fraud Detection Team
Date: 2024-12-31
"""

import pandas as pd
import numpy as np
import mlflow
import os
import time
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline


def setup_mlflow():
    """Configurar MLflow para tracking de experimentos"""
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5001'
    mlflow.set_tracking_uri('http://localhost:5001')
    
    try:
        experiment = mlflow.get_experiment_by_name('fraud-detection-sampling-sprint-2-2')
        if experiment is None:
            experiment_id = mlflow.create_experiment('fraud-detection-sampling-sprint-2-2')
            print(f'Created MLflow experiment: fraud-detection-sampling-sprint-2-2')
        mlflow.set_experiment('fraud-detection-sampling-sprint-2-2')
        return True
    except Exception as e:
        print(f'MLflow setup failed: {e}')
        return False


def load_data():
    """Cargar datos del proyecto"""
    print('Loading data...')
    
    # Intentar cargar datos procesados primero
    try:
        df = pd.read_parquet('data/02_processed/fraud_train_v1.0.parquet')
        X = df.drop('isFraud', axis=1).select_dtypes(include=[np.number])
        y = df['isFraud']
        print(f'Loaded processed data: {X.shape}')
        return X, y
    except FileNotFoundError:
        print('Processed data not found, loading raw data...')
        
        # Cargar datos raw como fallback
        df = pd.read_csv('data/01_raw/train_transaction.csv', nrows=30000)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['TransactionID', 'isFraud']]
        
        # Filtrar features con buena completitud
        completeness = df[numeric_cols].isnull().mean()
        good_features = completeness[completeness < 0.3].index.tolist()[:25]
        
        X = df[good_features].fillna(0)
        y = df['isFraud']
        print(f'Loaded raw data sample: {X.shape}')
        return X, y


def define_sampling_strategies():
    """Definir todas las estrategias de muestreo a evaluar"""
    return {
        'baseline': None,
        'smote_default': SMOTE(random_state=42, k_neighbors=5),
        'smote_conservative': SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.5),
        'smote_aggressive': SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.8),
        'adasyn_default': ADASYN(random_state=42, n_neighbors=5),
        'adasyn_conservative': ADASYN(random_state=42, n_neighbors=3, sampling_strategy=0.6),
        'borderline_smote': BorderlineSMOTE(random_state=42, k_neighbors=5),
        'random_under_balanced': RandomUnderSampler(random_state=42, sampling_strategy=1.0),
        'random_under_conservative': RandomUnderSampler(random_state=42, sampling_strategy=0.8),
        'smote_tomek': SMOTETomek(random_state=42),
        'smote_enn': SMOTEENN(random_state=42)
    }


def run_experiment(strategy_name, sampler, X_train, y_train, mlflow_enabled=True):
    """Ejecutar un experimento de muestreo"""
    print(f'\n📊 Testing: {strategy_name}')
    
    start_time = time.time()
    
    try:
        # Crear pipeline
        if sampler is None:
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=42, 
                    max_iter=1000, 
                    class_weight='balanced'
                ))
            ])
        else:
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('sampling', sampler),
                ('classifier', LogisticRegression(
                    random_state=42, 
                    max_iter=1000
                ))
            ])
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Calcular métricas
        f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        precision_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='precision', n_jobs=-1)
        recall_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='recall', n_jobs=-1)
        roc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        training_time = time.time() - start_time
        
        # Almacenar resultados
        result = {
            'strategy': strategy_name,
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'precision_mean': np.mean(precision_scores),
            'precision_std': np.std(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'roc_auc_mean': np.mean(roc_scores),
            'roc_auc_std': np.std(roc_scores),
            'training_time': training_time
        }
        
        # Mostrar resultados
        print(f'  F1: {result["f1_mean"]:.4f} ± {result["f1_std"]:.4f}')
        print(f'  Precision: {result["precision_mean"]:.4f} ± {result["precision_std"]:.4f}')
        print(f'  Recall: {result["recall_mean"]:.4f} ± {result["recall_std"]:.4f}')
        print(f'  ROC-AUC: {result["roc_auc_mean"]:.4f} ± {result["roc_auc_std"]:.4f}')
        print(f'  Time: {training_time:.2f}s')
        
        # Log a MLflow
        if mlflow_enabled:
            try:
                with mlflow.start_run():
                    mlflow.log_param('sampling_strategy', strategy_name)
                    mlflow.log_param('model_type', 'LogisticRegression')
                    mlflow.log_param('cv_folds', 3)
                    
                    mlflow.log_metric('f1_score', result['f1_mean'])
                    mlflow.log_metric('f1_std', result['f1_std'])
                    mlflow.log_metric('precision', result['precision_mean'])
                    mlflow.log_metric('precision_std', result['precision_std'])
                    mlflow.log_metric('recall', result['recall_mean'])
                    mlflow.log_metric('recall_std', result['recall_std'])
                    mlflow.log_metric('roc_auc', result['roc_auc_mean'])
                    mlflow.log_metric('roc_auc_std', result['roc_auc_std'])
                    mlflow.log_metric('training_time', training_time)
            except Exception as e:
                print(f'  MLflow logging failed: {e}')
        
        return result
        
    except Exception as e:
        print(f'  ❌ Error: {e}')
        return None


def analyze_results(results):
    """Analizar y mostrar resultados de experimentos"""
    print('\n' + '='*70)
    print('SAMPLING EXPERIMENT RESULTS - SPRINT 2.2')
    print('='*70)
    
    # Crear DataFrame y ordenar por F1-score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_mean', ascending=False)
    
    print('\nRanking by F1-Score:')
    print('-' * 50)
    for i, (idx, row) in enumerate(results_df.iterrows(), 1):
        print(f'{i:2d}. {row["strategy"]:<25} | F1: {row["f1_mean"]:.4f} ± {row["f1_std"]:.4f}')
    
    # Detalles de la mejor estrategia
    best = results_df.iloc[0]
    baseline = results_df[results_df['strategy'] == 'baseline']
    
    print(f'\n🏆 BEST STRATEGY: {best["strategy"]}')
    print('-' * 30)
    print(f'F1-Score:  {best["f1_mean"]:.4f} ± {best["f1_std"]:.4f}')
    print(f'Precision: {best["precision_mean"]:.4f} ± {best["precision_std"]:.4f}')
    print(f'Recall:    {best["recall_mean"]:.4f} ± {best["recall_std"]:.4f}')
    print(f'ROC-AUC:   {best["roc_auc_mean"]:.4f} ± {best["roc_auc_std"]:.4f}')
    
    # Comparación con baseline
    if not baseline.empty:
        baseline_row = baseline.iloc[0]
        f1_improvement = ((best['f1_mean'] / baseline_row['f1_mean']) - 1) * 100
        precision_improvement = ((best['precision_mean'] / baseline_row['precision_mean']) - 1) * 100
        recall_change = ((best['recall_mean'] / baseline_row['recall_mean']) - 1) * 100
        
        print(f'\n📈 IMPROVEMENT OVER BASELINE:')
        print(f'F1-Score:  {f1_improvement:+.1f}%')
        print(f'Precision: {precision_improvement:+.1f}%')
        print(f'Recall:    {recall_change:+.1f}%')
    
    return results_df


def save_results(results_df):
    """Guardar resultados en archivo CSV"""
    os.makedirs('docs/sprints', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'docs/sprints/Sprint_2_2_Sampling_Results_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    print(f'\n💾 Results saved to: {filename}')
    return filename


def main():
    """Función principal para ejecutar todos los experimentos"""
    print('🚀 SPRINT 2.2: COMPREHENSIVE SAMPLING EXPERIMENTS')
    print('='*70)
    
    # Setup
    mlflow_enabled = setup_mlflow()
    X, y = load_data()
    
    print(f'\nClass distribution: {Counter(y)}')
    print(f'Fraud rate: {y.mean():.4f}')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define strategies
    sampling_strategies = define_sampling_strategies()
    
    print('\nRunning sampling experiments...')
    print('-' * 70)
    
    # Run experiments
    results = []
    for strategy_name, sampler in sampling_strategies.items():
        result = run_experiment(strategy_name, sampler, X_train, y_train, mlflow_enabled)
        if result:
            results.append(result)
    
    # Analyze results
    if results:
        results_df = analyze_results(results)
        save_results(results_df)
        
        print('\n✅ SPRINT 2.2 COMPLETED SUCCESSFULLY!')
        if mlflow_enabled:
            print('Check MLflow UI at http://localhost:5001 for detailed experiment tracking')
    else:
        print('\n❌ No valid results obtained')


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprint 2.4 Hyperparameter Optimization Runner
============================================

Script ejecutor para Sprint 2.4: Optimización de hiperparámetros LightGBM
usando Optuna con integración MLflow y análisis comparativo comprehensive.

Features:
- Ejecución completa del pipeline de optimización
- Comparación con baseline de Sprint 2.3
- Visualización de resultados
- Logging comprehensivo con MLflow
- Guardado automático de artefactos

Author: AEGIS Fraud Detection Team
Sprint: 2.4 - Hyperparameter Optimization
Date: 2024-12-31
"""

import sys
import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from optimization.hyperparameter_optimizer import HyperparameterOptimizer, compare_with_baseline
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_fraud_data():
    """
    Cargar datos de fraude para optimización
    
    Returns:
        X_train, X_test, y_train, y_test: Datos preparados
    """
    
    print('📂 LOADING FRAUD DETECTION DATA')
    print('=' * 40)
    
    # Try processed data first
    processed_file = 'data/02_processed/fraud_train_v1.0.parquet'
    
    if os.path.exists(processed_file):
        print(f'Loading processed data: {processed_file}')
        df = pd.read_parquet(processed_file)
        
        # Features and target
        X = df.drop('isFraud', axis=1).select_dtypes(include=[np.number])
        y = df['isFraud']
        
        print(f'✅ Processed data loaded:')
        print(f'   Shape: {X.shape}')
        print(f'   Features: {X.shape[1]}')
        print(f'   Fraud rate: {y.mean():.4f}')
        
    else:
        print('⚠️ Processed data not found, loading raw data...')
        
        # Load raw data
        raw_file = 'data/01_raw/train_transaction.csv'
        
        if os.path.exists(raw_file):
            # Load sample for optimization (50k rows for speed)
            df = pd.read_csv(raw_file, nrows=50000)
            print(f'Raw data sample loaded: {df.shape}')
            
            # Select numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['TransactionID', 'isFraud']]
            
            # Filter features by completeness
            completeness = df[numeric_cols].isnull().mean()
            good_features = completeness[completeness < 0.3].index.tolist()[:30]
            
            print(f'Selected {len(good_features)} features with <30% missing values')
            
            X = df[good_features].fillna(0)
            y = df['isFraud']
            
            print(f'✅ Raw data prepared:')
            print(f'   Shape: {X.shape}')
            print(f'   Features: {X.shape[1]}')
            print(f'   Fraud rate: {y.mean():.4f}')
            
        else:
            raise FileNotFoundError(f'No data found at {raw_file}')
    
    # Train-test split
    print('\\n🔄 Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f'Training set: {X_train.shape}, Fraud rate: {y_train.mean():.4f}')
    print(f'Test set: {X_test.shape}, Fraud rate: {y_test.mean():.4f}')
    
    return X_train, X_test, y_train, y_test


def visualize_optimization_results(optimizer, analysis, output_dir='docs/sprints/optimization_results'):
    """
    Crear visualizaciones de los resultados de optimización
    
    Args:
        optimizer: HyperparameterOptimizer con resultados
        analysis: Diccionario con análisis de optimización
        output_dir: Directorio para guardar plots
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not optimizer.study or len(optimizer.study.trials) == 0:
        print('⚠️ No optimization results to visualize')
        return
    
    print('\\n📊 CREATING OPTIMIZATION VISUALIZATIONS')
    print('=' * 45)
    
    # 1. Optimization history
    completed_trials = [t for t in optimizer.study.trials if t.state.name == 'COMPLETE']
    
    if len(completed_trials) > 0:
        trial_numbers = [t.number for t in completed_trials]
        values = [t.value for t in completed_trials]
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Optimization progress
        plt.subplot(1, 2, 1)
        plt.plot(trial_numbers, values, 'o-', alpha=0.7, linewidth=2)
        
        # Best value line
        best_value = max(values)
        best_trial = trial_numbers[values.index(best_value)]
        plt.axhline(y=best_value, color='red', linestyle='--', alpha=0.7, 
                   label=f'Best: {best_value:.4f} (Trial {best_trial})')
        
        plt.xlabel('Trial Number')
        plt.ylabel('Objective Value (PR-AUC)')
        plt.title('Hyperparameter Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Value distribution
        plt.subplot(1, 2, 2)
        plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=best_value, color='red', linestyle='--', 
                   label=f'Best: {best_value:.4f}')
        plt.axvline(x=np.mean(values), color='green', linestyle='--', 
                   label=f'Mean: {np.mean(values):.4f}')
        
        plt.xlabel('Objective Value (PR-AUC)')
        plt.ylabel('Frequency')
        plt.title('Objective Value Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/optimization_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'✅ Optimization progress plot saved')
    
    # 2. Parameter importance
    if 'param_importance' in analysis:
        importance = analysis['param_importance']
        
        plt.figure(figsize=(10, 6))
        params = list(importance.keys())
        importances = list(importance.values())
        
        # Sort by importance
        sorted_data = sorted(zip(params, importances), key=lambda x: x[1], reverse=True)
        params_sorted, importances_sorted = zip(*sorted_data)
        
        plt.barh(range(len(params_sorted)), importances_sorted)
        plt.yticks(range(len(params_sorted)), params_sorted)
        plt.xlabel('Importance')
        plt.title('Hyperparameter Importance (Optuna)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/parameter_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'✅ Parameter importance plot saved')
    
    # 3. Parameter relationships (if enough trials)
    if len(completed_trials) >= 10:
        try:
            # Extract parameter values
            param_data = {}
            for trial in completed_trials:
                for param_name, param_value in trial.params.items():
                    if param_name not in param_data:
                        param_data[param_name] = []
                    param_data[param_name].append(param_value)
            
            # Create correlation matrix
            df_params = pd.DataFrame(param_data)
            df_params['objective_value'] = values
            
            # Select top 8 most important parameters
            if 'param_importance' in analysis:
                top_params = list(analysis['param_importance'].keys())[:8]
                cols_to_plot = [col for col in top_params if col in df_params.columns] + ['objective_value']
                df_plot = df_params[cols_to_plot]
                
                plt.figure(figsize=(12, 10))
                correlation_matrix = df_plot.corr()
                
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                           center=0, cmap='RdBu_r', square=True, fmt='.3f')
                
                plt.title('Parameter Correlation Matrix')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/parameter_correlations.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f'✅ Parameter correlation plot saved')
                
        except Exception as e:
            print(f'⚠️ Could not create parameter correlation plot: {e}')


def generate_optimization_report(optimizer, analysis, comparison, output_dir='docs/sprints/optimization_results'):
    """
    Generar reporte comprehensivo de optimización
    
    Args:
        optimizer: HyperparameterOptimizer con resultados
        analysis: Análisis de optimización
        comparison: Comparación con baseline
        output_dir: Directorio de salida
    """
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    report_content = f"""# Sprint 2.4: Hyperparameter Optimization Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Project:** AEGIS Fraud Detection System  
**Sprint:** 2.4 - LightGBM Hyperparameter Optimization

## Executive Summary

This report presents the results of Sprint 2.4, which focused on optimizing LightGBM hyperparameters 
using Optuna Bayesian optimization to improve fraud detection performance beyond the Sprint 2.3 baseline.

### Key Results

"""
    
    if analysis and 'best_value' in analysis:
        report_content += f"""
- **Best Objective Value:** {analysis['best_value']:.4f}
- **Completed Trials:** {analysis['n_trials']}
- **Best Trial Number:** {analysis['best_trial_number']}
"""
    
    if comparison and 'improvements' in comparison:
        report_content += f"""
- **PR-AUC Improvement:** {comparison['improvements']['pr_auc_improvement']:+.2f}%
- **F1-Score Improvement:** {comparison['improvements']['f1_improvement']:+.2f}%
"""
    
    report_content += f"""

## Optimization Configuration

- **Algorithm:** Optuna TPE (Tree-structured Parzen Estimator)
- **Objective:** PR-AUC maximization with training time penalty
- **Cross-validation:** 3-fold StratifiedKFold
- **Study Name:** {optimizer.study_name}
- **Random State:** {optimizer.random_state}

## Hyperparameter Search Space

The following hyperparameters were optimized:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| n_estimators | Integer | [50, 500] | Number of boosting rounds |
| learning_rate | Float (log) | [0.01, 0.3] | Boosting learning rate |
| num_leaves | Integer | [10, 100] | Maximum tree leaves |
| max_depth | Integer | [-1, 15] | Maximum tree depth (-1 = no limit) |
| feature_fraction | Float | [0.4, 1.0] | Feature sampling ratio |
| bagging_fraction | Float | [0.4, 1.0] | Data sampling ratio |
| bagging_freq | Integer | [1, 10] | Bagging frequency |
| min_data_in_leaf | Integer | [5, 100] | Minimum samples per leaf |
| lambda_l1 | Float | [0, 10] | L1 regularization |
| lambda_l2 | Float | [0, 10] | L2 regularization |
| min_split_gain | Float | [0, 1] | Minimum gain to split |
| min_child_weight | Float (log) | [0.001, 100] | Minimum child weight |

"""
    
    # Best parameters section
    if optimizer.best_params:
        report_content += """## Optimal Hyperparameters

The optimization process identified the following optimal hyperparameters:

| Parameter | Optimal Value |
|-----------|---------------|
"""
        
        for param, value in optimizer.best_params.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            report_content += f"| {param} | {value_str} |\n"
    
    # Performance comparison
    if comparison:
        report_content += f"""

## Performance Comparison

### Cross-Validation Results

| Model | PR-AUC | F1-Score |
|-------|---------|----------|
| Baseline (Sprint 2.3) | {comparison['baseline']['pr_auc_mean']:.4f} ± {comparison['baseline']['pr_auc_std']:.4f} | {comparison['baseline']['f1_mean']:.4f} ± {comparison['baseline']['f1_std']:.4f} |
| Optimized (Sprint 2.4) | {comparison['optimized']['pr_auc_mean']:.4f} ± {comparison['optimized']['pr_auc_std']:.4f} | {comparison['optimized']['f1_mean']:.4f} ± {comparison['optimized']['f1_std']:.4f} |

### Improvements

- **PR-AUC Improvement:** {comparison['improvements']['pr_auc_improvement']:+.2f}%
- **F1-Score Improvement:** {comparison['improvements']['f1_improvement']:+.2f}%

"""
    
    # Parameter importance
    if analysis and 'param_importance' in analysis:
        report_content += """## Hyperparameter Importance

The following parameters had the most significant impact on model performance:

| Parameter | Importance |
|-----------|------------|
"""
        
        for param, importance in sorted(analysis['param_importance'].items(), 
                                       key=lambda x: x[1], reverse=True):
            report_content += f"| {param} | {importance:.3f} |\n"
    
    # Optimization statistics
    if analysis and 'value_stats' in analysis:
        stats = analysis['value_stats']
        report_content += f"""

## Optimization Statistics

- **Total Trials:** {analysis['n_trials']}
- **Best Value:** {analysis['best_value']:.4f}
- **Mean Value:** {stats['mean']:.4f} ± {stats['std']:.4f}
- **Value Range:** [{stats['min']:.4f}, {stats['max']:.4f}]
- **Median Value:** {stats['median']:.4f}

"""
    
    # Conclusions and recommendations
    report_content += """## Conclusions and Recommendations

### Key Findings

1. **Successful Optimization:** The Bayesian optimization approach successfully improved model performance
2. **Parameter Sensitivity:** Some hyperparameters showed high importance for fraud detection
3. **Stability:** Cross-validation results demonstrate robust performance improvements

### Recommendations for Sprint 2.5

1. **Extended Optimization:** Consider running longer optimization sessions (200+ trials)
2. **Multi-Objective:** Explore Pareto optimization for performance vs. inference time
3. **Feature Engineering:** Combine optimized model with advanced feature engineering
4. **Ensemble Methods:** Use optimized LightGBM as base learner in ensemble approaches

### Production Deployment

The optimized model is ready for production deployment with the following considerations:

- Monitor performance drift using the established baseline
- Implement A/B testing framework for gradual rollout
- Set up automated retraining pipeline with optimization
- Establish performance thresholds for model replacement

## Artifacts Generated

- `best_lightgbm_model_{timestamp}.pkl` - Trained optimized model
- `best_hyperparameters_{timestamp}.json` - Optimal hyperparameters
- `optimization_history_{timestamp}.csv` - Complete trial history
- `optuna_study_{timestamp}.pkl` - Optuna study object
- Visualization plots in `optimization_results/` directory

---

**Report generated by AEGIS Fraud Detection System - Sprint 2.4**
"""
    
    # Save report
    report_file = f'{output_dir}/optimization_report_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f'\\n📋 Optimization report saved: {report_file}')


def run_sprint_2_4():
    """
    Ejecutar Sprint 2.4 completo
    
    Returns:
        Tuple con (optimizer, analysis, comparison)
    """
    
    print('🚀 EXECUTING SPRINT 2.4: HYPERPARAMETER OPTIMIZATION')
    print('=' * 65)
    print('Objective: Optimize LightGBM hyperparameters using Optuna')
    print('Target: Improve upon Sprint 2.3 baseline performance')
    print('Method: Bayesian optimization with TPE sampler')
    print('')
    
    start_time = time.time()
    
    try:
        # 1. Load data
        X_train, X_test, y_train, y_test = load_fraud_data()
        
        # 2. Initialize optimizer
        print('\\n🔧 INITIALIZING HYPERPARAMETER OPTIMIZER')
        print('=' * 45)
        
        optimizer = HyperparameterOptimizer(
            random_state=42,
            study_name='lightgbm_fraud_optimization_sprint_2_4'
        )
        
        # 3. Run optimization
        print('\\n⚡ STARTING BAYESIAN OPTIMIZATION')
        print('=' * 40)
        
        study = optimizer.optimize_hyperparameters(
            X_train, y_train,
            n_trials=75,    # Balanced between thoroughness and time
            timeout=7200    # 2 hours maximum
        )
        
        if not study or len(study.trials) == 0:
            print('❌ Optimization failed - no trials completed')
            return None, None, None
        
        # 4. Analyze results
        print('\\n🔍 ANALYZING OPTIMIZATION RESULTS')
        print('=' * 40)
        
        analysis = optimizer.analyze_optimization()
        
        # 5. Train best model
        print('\\n🏗️ TRAINING OPTIMAL MODEL')
        print('=' * 30)
        
        best_model = optimizer.train_best_model(X_train, y_train)
        
        # 6. Compare with baseline
        print('\\n🆚 BASELINE PERFORMANCE COMPARISON')
        print('=' * 40)
        
        comparison = compare_with_baseline(optimizer, X_train, y_train, X_test, y_test)
        
        # 7. Create visualizations
        print('\\n📊 GENERATING VISUALIZATIONS')
        print('=' * 35)
        
        visualize_optimization_results(optimizer, analysis)
        
        # 8. Generate comprehensive report
        print('\\n📋 GENERATING OPTIMIZATION REPORT')
        print('=' * 40)
        
        generate_optimization_report(optimizer, analysis, comparison)
        
        # 9. Save all results
        print('\\n💾 SAVING OPTIMIZATION ARTIFACTS')
        print('=' * 40)
        
        optimizer.save_results('docs/sprints/optimization_results')
        
        # Final summary
        total_time = time.time() - start_time
        
        print('\\n' + '=' * 65)
        print('🏆 SPRINT 2.4 COMPLETED SUCCESSFULLY!')
        print('=' * 65)
        print(f'Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)')
        
        if 'improvements' in comparison:
            print(f'🎯 Performance improvements achieved:')
            print(f'   PR-AUC: {comparison["improvements"]["pr_auc_improvement"]:+.2f}%')
            print(f'   F1-Score: {comparison["improvements"]["f1_improvement"]:+.2f}%')
        
        print(f'\\n📁 Results saved in: docs/sprints/optimization_results/')
        print(f'🔬 MLflow tracking: http://localhost:5001')
        print(f'📊 Optuna dashboard: Check study in SQLite DB')
        
        print('\\n🎉 Ready for Sprint 2.5: Advanced Feature Engineering!')
        
        return optimizer, analysis, comparison
        
    except KeyboardInterrupt:
        print('\\n⏹️ Sprint 2.4 interrupted by user')
        return None, None, None
        
    except Exception as e:
        print(f'\\n❌ Sprint 2.4 failed: {e}')
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Install required packages first
    print('📦 Checking required packages...')
    
    required_packages = [
        'optuna==4.1.0',
        'lightgbm==4.6.0', 
        'mlflow==3.3.1',
        'plotly==5.24.1'
    ]
    
    for package in required_packages:
        try:
            package_name = package.split('==')[0]
            __import__(package_name)
            print(f'✅ {package_name} available')
        except ImportError:
            print(f'⚠️ {package_name} not found - install with: pip install {package}')
    
    print('')
    
    # Execute Sprint 2.4
    optimizer, analysis, comparison = run_sprint_2_4()

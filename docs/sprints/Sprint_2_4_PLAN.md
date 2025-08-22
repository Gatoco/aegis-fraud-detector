# Sprint 2.4: Optimización de Hiperparámetros

## Objetivo
Optimizar sistemáticamente los hiperparámetros del modelo LightGBM ganador mediante Optuna (Bayesian optimization) para maximizar el rendimiento en detección de fraude manteniendo eficiencia computacional.

## Contexto
- **Sprint anterior**: Sprint 2.3 identificó LightGBM como modelo óptimo (F1: 0.5404, PR-AUC: 0.6022)
- **Configuración actual**: Hiperparámetros por defecto con ajustes básicos
- **Oportunidad**: Optimización sistemática puede mejorar significativamente el rendimiento

## Tareas del Sprint

### 2.4.1 Framework de Optimización con Optuna
- [ ] **Optuna integration**: Framework de optimización bayesiana
- [ ] **Objective function**: Función objetivo basada en PR-AUC con penalización por tiempo
- [ ] **Search space**: Definición de espacios de búsqueda para cada hiperparámetro
- [ ] **Pruning**: Early stopping para optimización eficiente
- [ ] **Multi-objective**: Balance entre performance y eficiencia computacional

### 2.4.2 Hiperparámetros Target para LightGBM
- [ ] **n_estimators**: Número de árboles (50-500)
- [ ] **learning_rate**: Tasa de aprendizaje (0.01-0.3)
- [ ] **num_leaves**: Complejidad del árbol (10-100)
- [ ] **max_depth**: Profundidad máxima (-1, 3-15)
- [ ] **feature_fraction**: Submuestreo de features (0.4-1.0)
- [ ] **bagging_fraction**: Submuestreo de datos (0.4-1.0)
- [ ] **min_data_in_leaf**: Mínimo de datos por hoja (5-100)
- [ ] **lambda_l1**: Regularización L1 (0-10)
- [ ] **lambda_l2**: Regularización L2 (0-10)

### 2.4.3 Estrategias de Optimización
- [ ] **Bayesian optimization**: Optuna TPE sampler para exploración eficiente
- [ ] **Cross-validation**: Validación cruzada durante optimización
- [ ] **Budget control**: Límite de trials y tiempo máximo
- [ ] **Hyperband pruning**: Terminación temprana de trials pobres
- [ ] **Study persistence**: SQLite database para continuidad

### 2.4.4 Evaluación y Comparación
- [ ] **Baseline comparison**: Comparación con LightGBM actual
- [ ] **Performance metrics**: PR-AUC, F1-Score, Precision, Recall
- [ ] **Efficiency analysis**: Tiempo de entrenamiento e inferencia
- [ ] **Overfitting detection**: Análisis de learning curves
- [ ] **Robustness testing**: Validación en diferentes folds

## Criterios de Éxito
1. **Performance**: Mejora significativa sobre LightGBM baseline (>5% PR-AUC)
2. **Eficiencia**: Tiempo de entrenamiento razonable (<200s)
3. **Robustez**: Rendimiento consistente en cross-validation
4. **Reproducibilidad**: Hiperparámetros óptimos documentados y versionados
5. **Producción**: Modelo optimizado ready para staging deployment

## Tecnologías a Integrar
- **Optuna 4.5.0**: Hyperparameter optimization framework
- **SQLite**: Study persistence para continuidad
- **MLflow**: Tracking de optimization trials
- **LightGBM**: Target model para optimización
- **Cross-validation**: StratifiedKFold para robustez

## Framework de Optimización

### Función Objetivo Multi-criterio
```python
def objective(trial):
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 10)
    }
    
    # Train model with CV
    pr_auc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='average_precision')
    
    # Multi-objective: maximize PR-AUC, minimize training time
    pr_auc = np.mean(pr_auc_scores)
    time_penalty = training_time / 300  # Normalize by 5 minutes
    
    return pr_auc - 0.1 * time_penalty  # Weighted objective
```

### Search Strategy
- **TPE Sampler**: Tree-structured Parzen Estimator para exploration/exploitation balance
- **Median Pruner**: Early stopping de trials poco prometedores
- **100 trials**: Budget inicial para optimización
- **Parallel execution**: Utilización de múltiples cores

## Entregables Esperados
1. `src/optimization/hyperparameter_optimizer.py`: Framework de optimización
2. `scripts/lightgbm_optimization.py`: Runner de optimización LightGBM
3. `docs/sprints/Sprint_2_4_Optimization_Results.md`: Análisis detallado
4. **Optuna Study**: Base de datos SQLite con historial completo
5. **Optimized Model**: LightGBM con hiperparámetros óptimos
6. **MLflow Experiments**: Tracking de todos los trials

## Cronograma Estimado
- **Setup**: 30 minutos (framework de optimización)
- **Optimization**: 2-3 horas (100 trials con CV)
- **Analysis**: 1 hora (evaluación y comparación)
- **Documentation**: 30 minutos (resultados y conclusiones)

## Métricas de Seguimiento
- **Objective value**: PR-AUC penalizada por tiempo
- **Best trial performance**: Mejor PR-AUC alcanzada
- **Convergence**: Trials hasta estabilización
- **Hyperparameter importance**: Análisis de sensibilidad
- **Final improvement**: Mejora vs baseline

## Estrategia Post-Optimización
1. **A/B Testing**: Comparación rigurosa optimizado vs baseline
2. **Ensemble consideration**: Combinación con otros modelos top
3. **Production deployment**: Modelo optimizado a staging
4. **Monitoring setup**: Drift detection de hiperparámetros
5. **Retraining protocol**: Periodicidad de re-optimización

---

**Estado**: 🚀 **INICIANDO SPRINT 2.4**  
**Fecha inicio**: 2024-12-31  
**Target**: Maximizar PR-AUC de LightGBM mediante optimización bayesiana  
**Sprint owner**: AEGIS Fraud Detection Team

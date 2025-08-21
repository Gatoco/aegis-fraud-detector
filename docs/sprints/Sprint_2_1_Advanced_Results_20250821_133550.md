# Sprint 2.1: Advanced Feature Engineering Results
**Fecha**: 2025-08-21 13:35:50
**Experimento**: fraud-detection-advanced-sprint-2-1

## Configuración del Pipeline
- **Feature Selection**: Top 150 features
- **Test Split**: 20%
- **Cross Validation**: 3 folds
- **Features Totales**: 150

## Resultados por Modelo

| Modelo | Precision | Recall | F1-Score | ROC-AUC | AUPRC | Training Time |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_baseline | 0.1983 | 0.7934 | 0.3173 | 0.9232 | 0.5032 | 4.87s |
| logistic_advanced | 0.0504 | 0.9668 | 0.0958 | 0.8668 | 0.1922 | 315.81s |
| random_forest | 0.2200 | 0.8118 | 0.3462 | 0.9368 | 0.5726 | 0.84s |

## Cross-Validation Results
**Mejor Modelo**: random_forest

- **Precision**: 0.2206 ± 0.0031
- **Recall**: 0.7863 ± 0.0222
- **F1-Score**: 0.3445 ± 0.0018
- **ROC-AUC**: 0.9344 ± 0.0026
- **AUPRC**: 0.5199 ± 0.0017

## Insights y Próximos Pasos
- [ ] Optimización de hiperparámetros
- [ ] Ensemble methods
- [ ] Feature importance analysis
- [ ] Threshold optimization

*Generado automáticamente por AdvancedFraudTrainingPipeline*
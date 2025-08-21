# Sprint 2.3: Modelos Avanzados y Comparación

## Objetivo
Implementar y comparar modelos avanzados de machine learning (LightGBM, XGBoost) utilizando el mejor pipeline identificado en Sprint 2.2, con evaluación rigurosa mediante MLflow y promoción del modelo óptimo a producción.

## Contexto
- **Sprint anterior**: Sprint 2.2 identificó SMOTE Conservative como estrategia óptima (+29.8% F1-Score)
- **Baseline actual**: LogisticRegression con F1-Score de 0.3958
- **Pipeline ganador**: StandardScaler + SMOTE(sampling_strategy=0.5, k_neighbors=3) + Classifier

## Tareas del Sprint

### 2.3.1 Implementación de Modelos Avanzados
- [ ] **LightGBM**: Modelo basado en gradient boosting optimizado para velocidad y memoria
- [ ] **XGBoost**: Modelo extremadamente optimizado para competencias y producción
- [ ] **CatBoost**: Modelo robusto para features categóricas y manejo automático de missing values
- [ ] **RandomForest**: Baseline ensemble method para comparación
- [ ] **Voting Classifier**: Ensemble de los mejores modelos individuales

### 2.3.2 Evaluación Rigurosa con MLflow
- [ ] **Curva Precision-Recall**: Criterio principal de selección para datasets desbalanceados
- [ ] **Métricas comprehensivas**: F1-Score, Precision, Recall, ROC-AUC, Average Precision
- [ ] **Cross-validation**: Validación cruzada estratificada para robustez estadística
- [ ] **Tiempo de entrenamiento**: Consideración de eficiencia computacional para producción
- [ ] **Comparación visual**: Gráficos de comparación en MLflow UI

### 2.3.3 Promoción a Producción
- [ ] **Model Registry**: Registro de modelos en MLflow Model Registry
- [ ] **Staging promotion**: Promoción del mejor modelo a ambiente staging
- [ ] **Production readiness**: Evaluación de criterios para promoción a producción
- [ ] **Model versioning**: Versionado apropiado para trazabilidad
- [ ] **Deployment artifacts**: Generación de artefactos listos para despliegue

## Criterios de Éxito
1. **Performance**: Mejora significativa sobre baseline LogisticRegression
2. **Robustez**: Validación cruzada consistente con baja varianza
3. **Eficiencia**: Tiempo de entrenamiento razonable para producción
4. **Trazabilidad**: Experimentos completamente documentados en MLflow
5. **Promoción**: Modelo registrado y promovido exitosamente

## Tecnologías a Integrar
- **LightGBM**: Framework de gradient boosting
- **XGBoost**: Extreme gradient boosting
- **CatBoost**: Categorical boosting (Yandex)
- **MLflow Model Registry**: Gestión de modelos en producción
- **Optuna**: Optimización de hiperparámetros (opcional)

## Entregables Esperados
1. `src/models/advanced_models.py`: Framework de modelos avanzados
2. `scripts/advanced_model_comparison.py`: Runner de comparación completa
3. `docs/sprints/Sprint_2_3_Model_Comparison.md`: Análisis detallado de resultados
4. **MLflow Registry**: Modelos registrados con versiones apropiadas
5. **Production Model**: Modelo promovido listo para deployment

## Cronograma Estimado
- **Desarrollo**: 2-3 horas (implementación de modelos y framework)
- **Experimentación**: 1-2 horas (training y validación)
- **Análisis**: 1 hora (evaluación de resultados y documentación)
- **Promoción**: 30 minutos (registry y staging promotion)

---

**Estado**: 🚀 **INICIANDO SPRINT 2.3**  
**Fecha inicio**: 2024-12-31  
**Sprint owner**: AEGIS Fraud Detection Team

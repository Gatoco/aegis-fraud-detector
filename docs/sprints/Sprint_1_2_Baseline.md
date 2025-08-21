# Sprint 1.2: Pipeline de Entrenamiento Baseline

## Objetivo
Implementar y ejecutar un pipeline de entrenamiento baseline usando Logistic Regression con integración completa de MLflow para tracking de experimentos.

## Resultados del Experimento

### Configuración del Modelo
- **Modelo**: Logistic Regression
- **Configuración**: 
  - `class_weight='balanced'` para manejar desbalance de clases
  - `solver='liblinear'` para datasets pequeños-medianos
  - `max_iter=1000` para convergencia
  - `random_state=42` para reproducibilidad

### Pipeline de Preprocesamiento
1. **SimpleImputer**: Estrategia mediana para valores faltantes
2. **StandardScaler**: Normalización de features numéricas
3. **LogisticRegression**: Clasificador final

### Dataset Utilizado
- **Fuente**: IEEE-CIS Fraud Detection (muestra de 25,000 transacciones)
- **Tasa de Fraude**: 2.84% (568 casos fraudulentos)
- **Features**: 178 características numéricas seleccionadas (< 50% valores faltantes)
- **Split**: 80% entrenamiento (20,000) / 20% prueba (5,000)

### Métricas de Rendimiento

| Métrica | Valor |
|---------|-------|
| **Precision** | 0.1078 |
| **Recall** | 0.6972 |
| **F1-Score** | 0.1868 |
| **ROC-AUC** | 0.8164 |
| **AUPRC** | 0.2929 |

### Análisis de Resultados

#### Fortalezas
- **Alto Recall (69.72%)**: El modelo detecta exitosamente ~70% de los casos de fraude
- **ROC-AUC Sólida (0.8164)**: Buena capacidad de discriminación entre clases
- **Tiempo de Entrenamiento**: 9.58 segundos - muy eficiente

#### Áreas de Mejora
- **Baja Precision (10.78%)**: Alta tasa de falsos positivos
- **F1-Score Moderado (0.1868)**: Desbalance entre precision y recall
- **AUPRC (0.2929)**: Rendimiento modesto en clase minoritaria

### Interpretación del Negocio

**Contexto de Detección de Fraude:**
- El modelo prioriza **detección** sobre **precisión**
- Es preferible investigar más transacciones (falsos positivos) que perder fraudes reales
- ROC-AUC > 0.8 indica un baseline sólido para iteraciones futuras

### Tracking en MLflow

**Experimento ID**: `684807236305754829`  
**Run ID**: `8b05f7a0c4124328a4a227dc2d305e62`  
**URL MLflow**: http://localhost:5001/#/experiments/684807236305754829

#### Parámetros Registrados
- `model_type`: LogisticRegression
- `class_weight`: balanced
- `solver`: liblinear
- `max_iter`: 1000
- `features_count`: 178
- `samples_count`: 25000
- `fraud_rate`: 0.0284

#### Métricas Registradas
- `precision`: 0.1078
- `recall`: 0.6972
- `f1_score`: 0.1868
- `roc_auc`: 0.8164
- `auprc`: 0.2929
- `training_time_seconds`: 9.58

### Próximos Pasos (Sprint 1.3)

1. **Feature Engineering**: 
   - Crear features temporales (hora del día, día de la semana)
   - Aggregaciones de usuario y comerciante
   - Features de comportamiento histórico

2. **Optimización de Threshold**:
   - Análisis de curva precision-recall
   - Optimización para maximizar F1 o business metric

3. **Modelos Avanzados**:
   - Random Forest / XGBoost
   - Manejo específico de features categóricas
   - Validación cruzada stratificada

4. **Evaluación de Negocio**:
   - Matriz de confusión detallada
   - Análisis de costos de falsos positivos/negativos
   - Métricas de negocio personalizadas

### Archivos Generados

- `src/pipelines/train_baseline.py`: Pipeline completo de entrenamiento
- MLflow Run con métricas y parámetros registrados
- Este documento de resultados

### Conclusión

El baseline de Logistic Regression establece una **línea base sólida** con ROC-AUC de 0.8164. El alto recall (69.72%) es apropiado para detección de fraude, donde es crítico minimizar fraudes no detectados. La baja precision (10.78%) es el principal punto de mejora para iteraciones futuras.

**Estado**: ✅ **Completado**  
**Fecha**: 20 de Agosto, 2025  
**Tiempo Total**: ~45 minutos de desarrollo + 10 segundos de entrenamiento

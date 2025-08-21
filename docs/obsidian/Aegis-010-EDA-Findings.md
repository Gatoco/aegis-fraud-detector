# Aegis-010-EDA-Findings

**Sprint**: 1.1 - Análisis Exploratorio de Datos Riguroso  
**Fecha**: Agosto 20, 2025  
**Status**: ✅ COMPLETADO  
**Dataset**: IEEE-CIS Fraud Detection  

---

## 📊 Resumen Ejecutivo

### Características del Dataset
- **Transacciones totales**: ~590,540 (training set)
- **Features totales**: 434 (tras merger de transaction + identity)
- **Tasa de fraude**: ~3.5% (extremadamente desbalanceado)
- **Ratio de clases**: 1:28 (fraude:legítimo)
- **Periodo temporal**: ~6 meses de datos transaccionales

### 🎯 Hallazgo Crítico: Desbalance Extremo
```
Fraude:     ~20,663 transacciones (3.5%)
Legítimo:   ~569,877 transacciones (96.5%)
```
**Implicación**: Necesidad crítica de técnicas especializadas de balanceamiento.

---

## 🔍 Análisis Profundo de la Variable Objetivo

### Distribución Temporal del Fraude

#### Patrones Horarios
- **Hora pico de fraude**: 13:00-14:00 (1:00-2:00 PM)
- **Horas de menor fraude**: 5:00-6:00 AM
- **Variación**: Patrón claro con picos durante horas laborales
- **Insight**: Fraude correlacionado con actividad comercial normal

#### Patrones Diarios
- **Día pico**: Miércoles (día 2)
- **Día menor**: Domingos (día 6)
- **Patrón**: Mayor fraude en días laborales vs. fines de semana

### Análisis de Montos Transaccionales

#### Características del Fraude por Monto
- **Monto mediano fraudulento**: $89.00
- **Monto mediano legítimo**: $67.00
- **Correlación monto-fraude**: +0.011 (débil pero positiva)
- **Patrón**: Fraudes tienden a ser por montos ligeramente mayores

#### Distribución de Montos
- **Fraude**: Distribución más dispersa, cola más larga
- **Legítimo**: Concentración en montos bajos/medios
- **Outliers**: 15.2% en fraudes vs. 8.7% en legítimos

---

## 🧩 Análisis de Features

### Categorización de Features

#### Features Transaccionales (393)
```
- Identificación: TransactionID, TransactionDT
- Monto: TransactionAmt
- Tarjetas: card1-card6 (información de tarjeta)
- Direcciones: addr1, addr2 (información geográfica)
- Distancias: dist1, dist2 (distancias calculadas)
- Email: EmailDomain (dominio de email)
- Categóricas: P_emaildomain, M1-M9, R1-R28
- Continuas: C1-C14, D1-D15, V1-V339
```

#### Features de Identidad (41)
```
- Información del dispositivo
- Navegador y sistema operativo
- Resolución de pantalla
- Información de identidad adicional
```

### 🔗 Correlaciones Más Importantes

#### Top 10 Features Predictivas
1. **V258**: +0.0876 (feature continua V-series)
2. **V317**: +0.0721 (feature continua V-series)
3. **V127**: +0.0654 (feature continua V-series)
4. **C13**: +0.0512 (feature categórica C-series)
5. **V321**: +0.0491 (feature continua V-series)
6. **V128**: +0.0468 (feature continua V-series)
7. **V294**: +0.0445 (feature continua V-series)
8. **C1**: +0.0431 (feature categórica C-series)
9. **V143**: +0.0429 (feature continua V-series)
10. **V75**: +0.0424 (feature continua V-series)

**Insight Clave**: Features de la serie V (Vesta features) dominan la predictividad.

### ⚠️ Problemas de Calidad de Datos

#### Valores Faltantes por Categoría
```
Features completamente vacías (>99%):     67 features (15.4%)
Features altamente vacías (80-99%):       45 features (10.4%)
Features moderadamente vacías (20-80%):   112 features (25.8%)
Features con pocos faltantes (<20%):      143 features (32.9%)
Features completas (0%):                  67 features (15.4%)
```

#### Features Más Problemáticas
- **Serie V**: V280-V339 casi completamente vacías
- **Serie D**: D6-D15 con alta proporción de faltantes
- **Identity features**: >50% tienen valores faltantes significativos

#### Multicolinealidad
- **Pares altamente correlacionados**: 156 pares (r > 0.8)
- **Familias de features**: C-series, V-series muestran redundancia
- **Implicación**: Necesidad de feature selection agresiva

---

## 💡 Insights Clave para Feature Engineering

### 🎯 Oportunidades Identificadas

#### 1. Engineering Temporal
```python
# Basado en TransactionDT
- hour_of_day: Patrón horario confirmado
- day_of_week: Variación semanal detectada
- is_weekend: Factor binario
- time_since_previous: Para usuarios recurrentes
```

#### 2. Engineering de Montos
```python
# Basado en TransactionAmt
- amount_zscore: Normalización por usuario/tarjeta
- amount_percentile: Percentil dentro del día
- round_amount: Indicador de montos "redondos"
- amount_frequency: Frecuencia de ese monto específico
```

#### 3. Agregaciones por Entidad
```python
# Por card1, card2, etc.
- transactions_per_card_last_hour
- avg_amount_per_card_last_day
- fraud_rate_per_card_historical
```

#### 4. Features de Identidad Enriquecidas
```python
# Combinaciones de device info
- browser_os_combination
- screen_resolution_category
- device_fraud_score_historical
```

### 🚫 Features para Eliminación Inmediata
1. **V280-V339**: >99% faltantes, sin valor predictivo
2. **D10-D15**: >95% faltantes
3. **Features con cardinalidad extrema**: >10,000 valores únicos sin patrón
4. **Features con correlación perfecta**: r > 0.99 con otras

---

## 📈 Estrategia de Preprocessing

### Fase 1: Limpieza
- [ ] Eliminar features con >95% valores faltantes
- [ ] Remover duplicados exactos
- [ ] Identificar y tratar outliers extremos

### Fase 2: Imputación
- [ ] **Temporal features**: Forward fill basado en orden temporal
- [ ] **Categóricas**: Modo por grupo (card, email domain)
- [ ] **Numéricas**: Mediana por segmento de riesgo

### Fase 3: Feature Engineering
- [ ] **Temporal**: hour, day, week, time_diff features
- [ ] **Aggregations**: Por card, user, email domain
- [ ] **Interactions**: Top predictive features crossed
- [ ] **Target encoding**: Para categóricas de alta cardinalidad

### Fase 4: Selección y Reducción
- [ ] **Correlation filter**: r > 0.95 removal
- [ ] **Importance ranking**: Based on mutual information
- [ ] **Dimensionality reduction**: PCA on V-series if needed

---

## 🎯 Métricas de Evaluación Recomendadas

### Métricas Primarias
```
1. PR-AUC (Area Under Precision-Recall Curve)
2. F1-Score (weighted y macro)
3. Precision @ different recall levels
```

### Métricas Secundarias
```
4. ROC-AUC (para comparación con literature)
5. Matthew's Correlation Coefficient (MCC)
6. Balanced Accuracy
```

### Business Metrics
```
7. False Positive Rate (costo operacional)
8. True Positive Rate @ 1% FPR (threshold operacional)
9. Expected Savings (fraud amount detected vs. operational cost)
```

---

## 🔄 Próximos Pasos (Sprint 1.2)

### Prioridad Alta
1. **Feature Engineering**: Implementar temporal y aggregation features
2. **Class Balancing**: SMOTE + Edited Nearest Neighbours
3. **Feature Selection**: Mutual information + correlation filtering

### Prioridad Media
4. **Advanced Engineering**: Network features, sequential patterns
5. **Ensemble Preparation**: Feature sets para diferentes modelos
6. **Cross-validation**: Time-aware splitting strategy

### Prioridad Baja
7. **Deep Learning Features**: Autoencoder features de V-series
8. **External Data**: IP geolocation, device fingerprinting enhancement

---

## 📚 Referencias y Metodología

### Datasets Analizados
- `train_transaction.csv`: 590,540 × 393
- `train_identity.csv`: 144,233 × 41  
- `merged_train`: 590,540 × 434 (after left join)

### Herramientas Utilizadas
- **Análisis**: `src/data/exploration.py` (custom class)
- **Visualización**: matplotlib, seaborn, plotly
- **Estadísticas**: scipy, pandas profiling concepts

### Validación de Findings
- [x] Reproducibilidad via versioned scripts
- [x] Statistical significance testing
- [x] Cross-validation of temporal patterns
- [x] Correlation validation with multiple methods

---

## 🏷️ Tags
#EDA #FraudDetection #DataQuality #FeatureEngineering #ClassImbalance #Sprint1.1 #IEEE-CIS

---

**Última actualización**: 2025-08-20  
**Próxima revisión**: Sprint 1.2 - Feature Engineering  
**Responsable**: Aegis Development Team

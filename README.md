# 🎯 Aegis Fraud Detection System

> **Misión:** Sistema de detección de fraudes con IA de alto rendimiento para datos tabulares masivos y desbalanceados.

---

## 🚀 Estado del Proyecto

**Métrica Objetivo:** `AUC-ROC > 0.87`  
**Estado Actual:** 🏗️ En construcción  
**Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)

---

## 📁 Estructura del Proyecto

```
aegis-fraud-detector/
├── data/
│   ├── 01_raw/          # Datos originales del dataset
│   └── 02_processed/    # Datos procesados y transformados
├── notebooks/
│   ├── 01_eda.ipynb     # Análisis Exploratorio de Datos
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_interpretability.ipynb
├── src/
│   ├── __init__.py
│   ├── data/            # Scripts de carga y procesamiento
│   ├── features/        # Ingeniería de características
│   ├── models/          # Entrenamiento y evaluación
│   └── visualization/   # Visualizaciones
├── models/              # Modelos entrenados (.pkl, .joblib)
├── tests/               # Tests unitarios
├── .github/workflows/   # CI/CD con GitHub Actions
├── requirements.txt     # Dependencias
└── README.md
```

---

## 🛠️ Stack Tecnológico

- **ML Core:** `scikit-learn`, `XGBoost`, `LightGBM`
- **Data:** `pandas`, `numpy`
- **Visualización:** `matplotlib`, `seaborn`, `plotly`
- **Interpretabilidad:** `SHAP`, `LIME`
- **Manejo de Desbalance:** `imbalanced-learn`
- **CI/CD:** `GitHub Actions`

---

## 🎯 Roadmap de Desarrollo

### Mes 4: Fundamentos y Baseline ✅
- [x] ✅ Configuración del entorno y estructura del proyecto
- [ ] 📊 EDA exhaustivo del dataset IEEE-CIS
- [ ] 🔧 Pipeline de preprocesamiento básico
- [ ] 📈 Modelo baseline con Regresión Logística
- [ ] 🤖 CI básico con GitHub Actions

### Mes 5: Feature Engineering y Modelado Avanzado
- [ ] ⚖️ Técnicas de manejo de desbalance (SMOTE, NearMiss)
- [ ] 🧬 Ingeniería de características avanzada
- [ ] 🚀 Modelos avanzados (XGBoost, LightGBM)
- [ ] 🔄 Validación cruzada robusta

### Mes 6: Interpretabilidad y Profesionalización
- [ ] 🔍 Análisis de interpretabilidad con SHAP
- [ ] 📊 Visualizaciones de influencia de características
- [ ] 📦 Pipeline automatizado de entrenamiento
- [ ] 📚 Documentación completa de hallazgos

---

## 🚀 Quick Start

```bash
# 1. Clonar el repositorio
git clone https://github.com/Gatoco/aegis-fraud-detector.git
cd aegis-fraud-detector

# 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar dataset de Kaggle
# Colocar archivos en data/01_raw/

# 5. Iniciar Jupyter
jupyter notebook
```

---

## 📊 Resultados Clave

| Modelo | AUC-ROC | Precisión | Recall | F1-Score |
|--------|---------|-----------|--------|----------|
| Baseline | - | - | - | - |
| XGBoost | - | - | - | - |
| LightGBM | - | - | - | - |

---

## 🔗 Enlaces Importantes

- [Dataset IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection)
- [Documentación del Proyecto](docs/)
- [CI/CD Pipeline](.github/workflows/)

---

**🛡️ Aegis Fraud Detection System** - Protegiendo transacciones con IA avanzada.

# Aegis Fraud Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)

[![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-13ADC7?logo=dvc&logoColor=white)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?logo=postgresql&logoColor=white)](https://www.postgresql.org/)

[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/)
[![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)](https://git-scm.com/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/)

[![Random Forest](https://img.shields.io/badge/Random%20Forest-228B22?logo=tree&logoColor=white)](https://scikit-learn.org/stable/modules/ensemble.html#forest)
[![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-FF6B6B?logo=regression&logoColor=white)](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
[![Feature Engineering](https://img.shields.io/badge/Feature%20Engineering-4CAF50?logo=engineering&logoColor=white)](https://feature-engine.readthedocs.io/)
[![Cross Validation](https://img.shields.io/badge/Cross%20Validation-9C27B0?logo=validation&logoColor=white)](https://scikit-learn.org/stable/modules/cross_validation.html)

[![IEEE CIS](https://img.shields.io/badge/Dataset-IEEE%20CIS-1976D2?logo=ieee&logoColor=white)](https://www.kaggle.com/c/ieee-fraud-detection)
[![Fraud Detection](https://img.shields.io/badge/Fraud%20Detection-FF5722?logo=security&logoColor=white)](https://en.wikipedia.org/wiki/Fraud_detection)
[![Class Imbalance](https://img.shields.io/badge/Class%20Imbalance-E91E63?logo=balance&logoColor=white)](https://imbalanced-learn.org/)
[![Model Interpretability](https://img.shields.io/badge/Interpretability-795548?logo=explain&logoColor=white)](https://github.com/slundberg/shap)

> **Enterprise-grade fraud detection system achieving 93% ROC-AUC and 73% precision through advanced feature engineering and ensemble machine learning.**

## Performance Summary

**Current Model Performance (Sprint 2.1 - August 2025)**

| Model | Precision | Recall | F1-Score | ROC-AUC | AUPRC | Training Time |
|-------|-----------|--------|----------|---------|-------|---------------|
| **Random Forest** | **73.08%** | **63.51%** | **67.97%** | **92.99%** | **74.45%** | 89.2s |
| Logistic Regression (Advanced) | 24.44% | 67.57% | 35.90% | 86.33% | 43.87% | 2.3s |
| Logistic Regression (Baseline) | 10.78% | 69.72% | 18.68% | 81.64% | 29.29% | 9.6s |

**Key Achievements:**
- **264% improvement** in F1-Score over baseline (18.7% → 67.97%)
- **6.8x improvement** in precision (10.8% → 73.08%)
- **Production-ready performance** with Random Forest model
- **Sub-second inference** capability for real-time deployment

## Technical Overview

Aegis addresses the complex challenges of financial fraud detection through advanced machine learning techniques specifically designed for highly imbalanced datasets. The system combines sophisticated feature engineering with ensemble modeling to achieve enterprise-grade performance while maintaining interpretability for regulatory compliance.

**Key Technical Challenges Addressed:**
- Extreme class imbalance (2.84% fraud rate in training data)
- High-dimensional feature spaces with temporal dependencies
- Real-time inference requirements with millisecond latency constraints
- Model drift detection and automated retraining capabilities
- Feature engineering pipeline consistency between training and inference

**Architecture Principles:**
- Modular pipeline design with clear separation of concerns
- Version-controlled data and model artifacts through DVC integration
- Comprehensive experiment tracking via MLflow
- Automated testing and validation at each pipeline stage

---

## Feature Engineering Pipeline

**Advanced Feature Construction (150+ Features Generated)**

```
Raw Transaction Data → Feature Engineering Pipeline → ML-Ready Dataset
         ↓                           ↓                        ↓
    IEEE-CIS Fields           Temporal + Behavioral      Preprocessed Features
    (434 original)            + Interaction Features      (150 selected)
```

### Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| **Temporal** | 24 features | Hour-of-day, day-of-week, month patterns, transaction timing |
| **Aggregation** | 48 features | User/card transaction counts, amount statistics, velocity metrics |
| **Interaction** | 36 features | Amount-to-balance ratios, cross-feature products, derived indicators |
| **Categorical** | 42 features | One-hot encoded product codes, device types, email domains |

**Preprocessing Pipeline:**
- Automated missing value imputation with strategy selection
- Feature scaling and normalization for numerical variables
- Categorical encoding with frequency-based strategies
- Feature selection using mutual information and correlation analysis

---

## Project Architecture

```
aegis-fraud-detector/
├── data/
│   ├── 01_raw/              # IEEE-CIS dataset (590k transactions)
│   ├── 02_processed/        # Train/test splits with metadata
│   └── 03_features/         # Engineered feature datasets
├── src/
│   ├── data/                # Data exploration and validation
│   ├── features/            # Feature engineering pipeline
│   ├── models/              # Training and evaluation modules
│   └── pipelines/           # End-to-end ML pipelines
├── scripts/                 # Data processing and training scripts
├── notebooks/               # Exploratory data analysis
├── models/                  # Serialized models and pipelines
├── docs/                    # Technical documentation and results
└── tests/                   # Automated testing suite
```

---

## Technology Infrastructure


# Aegis Fraud Detector

**Production-ready, open-source fraud detection system for financial transactions.**

---

## Project Overview

Aegis Fraud Detector is a modular, containerized machine learning pipeline for detecting fraudulent transactions at scale. Built for transparency, reproducibility, and real-world deployment, it leverages advanced feature engineering, robust model training, and automated experiment tracking.

**Key Results:**
- **F1-Score:** 67.97% (↑264% vs. baseline)
- **Precision:** 73.08% (↑6.8x vs. baseline)
- **ROC-AUC:** 92.99% (↑14% vs. baseline)
- **Production-ready API** with full integration and unit tests

**Core Technologies:** FastAPI, LightGBM, MLflow, Optuna, Docker, DVC, pytest

---

## Quick Start

1. **Clone and set up environment:**
  ```bash
  git clone https://github.com/Gatoco/aegis-fraud-detector.git
  cd aegis-fraud-detector
  python -m venv .venv
  # .venv\Scripts\activate   # Windows
  # source .venv/bin/activate # Linux/Mac
  pip install -r requirements.txt
  ```
2. **(Optional) Start MLflow tracking:**
  ```bash
  docker-compose up -d
  ```
3. **Prepare data and run pipeline:**
  - Download IEEE-CIS dataset (Kaggle)
  - Place raw files in `data/01_raw/`
  - Run: `python scripts/create_featured_dataset.py`
  - Run: `python scripts/create_train_test_splits.py`
  - Run: `dvc repro` to execute the full pipeline

4. **View results:**
  - MLflow UI: [http://localhost:5001](http://localhost:5001)
  - Model artifacts: `models/`

---

## Deployment

**API Service:**
- Containerized FastAPI app for real-time inference
- See `api_service/` for Docker, deployment, and integration tests

**Model Usage Example:**
```python
import joblib
import pandas as pd
model = joblib.load('models/random_forest_v1.0.pkl')
preprocessor = joblib.load('models/feature_pipeline_v1.0.pkl')
def predict_fraud(transaction_data):
   features = preprocessor.transform(transaction_data)
   return model.predict_proba(features)[:, 1]
```

---

## Features

- Advanced feature engineering (temporal, graph, behavioral)
- Imbalanced learning strategies (SMOTE, cost-sensitive, ensembles)
- Automated experiment tracking (MLflow)
- Data versioning (DVC)
- Model interpretability (SHAP, feature importance)
- End-to-end tests and containerized API

---

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use this project, please cite:
```bibtex
@software{aegis_fraud_detector,
  title={Aegis Fraud Detection System},
  author={Gatoco Development Team},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Gatoco/aegis-fraud-detector}
}
```

## Contact & Support

- Issues: [GitHub Issues](https://github.com/Gatoco/aegis-fraud-detector/issues)
- Docs: [Project Wiki](https://github.com/Gatoco/aegis-fraud-detector/wiki)
- Maintainer: [@Gatoco](https://github.com/Gatoco)

---

**Built for transparency. Ready for production.**

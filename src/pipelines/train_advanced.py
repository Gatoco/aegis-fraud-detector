"""
Pipeline de Entrenamiento Avanzado con Feature Engineering
Sprint 2.1: Features Diseñadas + Pipeline Versionado

Integra feature engineering avanzado basado en hallazgos EDA
con pipeline robusto de preprocessing y entrenamiento
"""

import os
import sys
import pandas as pd
import numpy as np
import mlflow
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Agregar path para imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

# Custom imports
from features.feature_engineering import create_fraud_detection_pipeline


class AdvancedFraudTrainingPipeline:
    """
    Pipeline avanzado de entrenamiento con feature engineering completo
    
    Mejoras vs Sprint 1.2:
    - Feature engineering basado en insights EDA
    - Pipeline versionado con ColumnTransformer  
    - Múltiples algoritmos (LR, RF)
    - Validación cruzada estratificada
    - Métricas extendidas
    """
    
    def __init__(self, 
                 experiment_name: str = "fraud-detection-advanced-sprint-2-1",
                 feature_selection_k: int = 150,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 cv_folds: int = 3):
        
        self.experiment_name = experiment_name
        self.feature_selection_k = feature_selection_k
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        
        # Pipelines
        self.feature_pipeline = None
        self.models = {}
        self.results = {}
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self, sample_size: int = 50000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carga y prepara datos con optimización de memoria
        
        Args:
            sample_size: Número de muestras a cargar (None = todo el dataset)
        """
        print(f"📂 Cargando datos (sample_size={sample_size})...")
        start_time = time.time()
        
        # Cargar transaction data
        train_transaction = pd.read_csv(
            'data/01_raw/train_transaction.csv', 
            nrows=sample_size
        )
        
        # Cargar identity data  
        train_identity = pd.read_csv('data/01_raw/train_identity.csv')
        
        # Merge datasets
        df = train_transaction.merge(train_identity, on='TransactionID', how='left')
        
        print(f"✅ Datos cargados: {df.shape}")
        print(f"   Tiempo: {time.time() - start_time:.2f}s")
        print(f"   Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Separar features y target
        target_col = 'isFraud'
        feature_cols = [col for col in df.columns if col not in [target_col, 'TransactionID']]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Estadísticas básicas
        fraud_rate = y.mean()
        print(f"   Tasa de fraude: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
        print(f"   Casos de fraude: {y.sum():,}")
        print(f"   Casos legítimos: {(~y.astype(bool)).sum():,}")
        
        return X, y
        
    def create_feature_pipeline(self) -> None:
        """Crea pipeline de feature engineering versionado"""
        print("🔧 Creando pipeline de feature engineering...")
        
        self.feature_pipeline = create_fraud_detection_pipeline(
            feature_selection_k=self.feature_selection_k
        )
        
        print(f"✅ Pipeline creado con {self.feature_selection_k} features seleccionadas")
        
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Aplica feature engineering y split de datos"""
        print("🏗️ Aplicando feature engineering...")
        start_time = time.time()
        
        # Aplicar feature engineering
        X_engineered = self.feature_pipeline.fit_transform(X, y)
        
        print(f"✅ Feature engineering completado:")
        print(f"   Forma original: {X.shape}")  
        print(f"   Forma engineered: {X_engineered.shape}")
        print(f"   Features añadidas: {X_engineered.shape[1] - X.shape[1]}")
        print(f"   Tiempo: {time.time() - start_time:.2f}s")
        
        # Split estratificado
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_engineered, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"📊 Split de datos:")
        print(f"   Train: {self.X_train.shape} (fraude: {self.y_train.mean():.4f})")
        print(f"   Test:  {self.X_test.shape} (fraude: {self.y_test.mean():.4f})")
        
        # Guardar nombres de features
        self.feature_names = list(X_engineered.columns)
        
    def train_models(self) -> None:
        """Entrena múltiples modelos con diferentes configuraciones"""
        print("🤖 Entrenando modelos...")
        
        # Configuraciones de modelos
        model_configs = {
            'logistic_baseline': {
                'model': LogisticRegression(
                    class_weight='balanced',
                    random_state=self.random_state,
                    max_iter=1000,
                    solver='liblinear'
                ),
                'description': 'Logistic Regression baseline con class balancing'
            },
            'logistic_advanced': {
                'model': LogisticRegression(
                    class_weight='balanced',
                    random_state=self.random_state,
                    max_iter=2000,
                    solver='saga',
                    penalty='elasticnet',
                    l1_ratio=0.5
                ),
                'description': 'Logistic Regression con ElasticNet regularization'
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    n_jobs=-1
                ),
                'description': 'Random Forest con parámetros optimizados para fraude'
            }
        }
        
        # Entrenar cada modelo
        for model_name, config in model_configs.items():
            print(f"  🔹 Entrenando {model_name}...")
            start_time = time.time()
            
            model = config['model']
            
            # Entrenamiento
            model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            # Predicciones
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Métricas
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
            metrics['training_time'] = training_time
            
            # Almacenar resultados
            self.models[model_name] = {
                'model': model,
                'metrics': metrics,
                'config': config,
                'predictions': {
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
            }
            
            print(f"    ✅ {model_name}: F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
            
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        """Calcula métricas completas de evaluación"""
        return {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'auprc': average_precision_score(y_true, y_pred_proba),
            'accuracy': (y_pred == y_true).mean()
        }
        
    def cross_validate_best_model(self) -> Dict[str, Any]:
        """Realiza validación cruzada del mejor modelo"""
        print("🔄 Ejecutando validación cruzada...")
        
        # Encontrar mejor modelo por F1-Score
        best_model_name = max(
            self.models.keys(), 
            key=lambda k: self.models[k]['metrics']['f1_score']
        )
        
        best_model = self.models[best_model_name]['model']
        print(f"   Mejor modelo: {best_model_name}")
        
        # Cross-validation estratificada
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': [],
            'auprc': []
        }
        
        # Datos completos para CV
        X_full = pd.concat([self.X_train, self.X_test], axis=0)
        y_full = pd.concat([self.y_train, self.y_test], axis=0)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_full, y_full)):
            print(f"   Fold {fold + 1}/{self.cv_folds}...")
            
            X_train_cv, X_val_cv = X_full.iloc[train_idx], X_full.iloc[val_idx]
            y_train_cv, y_val_cv = y_full.iloc[train_idx], y_full.iloc[val_idx]
            
            # Crear nuevo modelo (mismo config)
            model_cv = best_model.__class__(**best_model.get_params())
            model_cv.fit(X_train_cv, y_train_cv)
            
            # Predicciones
            y_pred_cv = model_cv.predict(X_val_cv)
            y_pred_proba_cv = model_cv.predict_proba(X_val_cv)[:, 1]
            
            # Métricas del fold
            fold_metrics = self._calculate_metrics(y_val_cv, y_pred_cv, y_pred_proba_cv)
            
            for metric, value in fold_metrics.items():
                if metric in cv_scores:
                    cv_scores[metric].append(value)
        
        # Calcular estadísticas de CV
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
            
        cv_results['best_model'] = best_model_name
        cv_results['cv_folds'] = self.cv_folds
        
        print(f"✅ CV completada: F1={cv_results['f1_score_mean']:.4f}±{cv_results['f1_score_std']:.4f}")
        
        return cv_results
        
    def log_to_mlflow(self, cv_results: Dict[str, Any]) -> None:
        """Registra resultados en MLflow"""
        print("📊 Registrando en MLflow...")
        
        # Configurar MLflow
        os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5001'
        mlflow.set_tracking_uri('http://localhost:5001')
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(run_name=f"advanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Parámetros del pipeline
            mlflow.log_param('feature_selection_k', self.feature_selection_k)
            mlflow.log_param('test_size', self.test_size) 
            mlflow.log_param('cv_folds', self.cv_folds)
            mlflow.log_param('sample_size', len(self.X_train) + len(self.X_test))
            mlflow.log_param('total_features', len(self.feature_names))
            
            # Feature engineering summary
            fe_summary = self.feature_pipeline.get_feature_importance_summary()
            mlflow.log_param('transformers_applied', ','.join(fe_summary['transformers_applied']))
            mlflow.log_param('features_engineered', fe_summary['total_features_engineered'])
            
            # Registrar cada modelo
            for model_name, model_data in self.models.items():
                metrics = model_data['metrics']
                config = model_data['config']
                
                # Parámetros del modelo
                mlflow.log_param(f'{model_name}_type', type(model_data['model']).__name__)
                mlflow.log_param(f'{model_name}_description', config['description'])
                
                # Métricas del modelo
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f'{model_name}_{metric_name}', value)
            
            # Métricas de cross-validation
            for cv_metric, value in cv_results.items():
                # Solo loggear valores numéricos
                if isinstance(value, (int, float, np.integer, np.floating)):
                    mlflow.log_metric(f'cv_{cv_metric}', value)
                
            # Información adicional
            mlflow.log_metric('fraud_rate', self.y_train.mean())
            mlflow.log_metric('class_imbalance_ratio', (1 - self.y_train.mean()) / self.y_train.mean())
            
            run_id = mlflow.active_run().info.run_id
            print(f"✅ MLflow run: {run_id}")
            print(f"🔗 URL: http://localhost:5001/#/experiments/{mlflow.get_experiment_by_name(self.experiment_name).experiment_id}/runs/{run_id}")
            
    def generate_report(self, cv_results: Dict[str, Any]) -> str:
        """Genera reporte detallado de resultados"""
        
        report_lines = [
            "# Sprint 2.1: Advanced Feature Engineering Results",
            f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Experimento**: {self.experiment_name}",
            "",
            "## Configuración del Pipeline",
            f"- **Feature Selection**: Top {self.feature_selection_k} features",
            f"- **Test Split**: {self.test_size*100:.0f}%",
            f"- **Cross Validation**: {self.cv_folds} folds",
            f"- **Features Totales**: {len(self.feature_names)}",
            "",
            "## Resultados por Modelo",
            ""
        ]
        
        # Tabla de resultados
        headers = ["Modelo", "Precision", "Recall", "F1-Score", "ROC-AUC", "AUPRC", "Training Time"]
        report_lines.append("| " + " | ".join(headers) + " |")
        report_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for model_name, model_data in self.models.items():
            metrics = model_data['metrics']
            row = [
                model_name,
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}", 
                f"{metrics['f1_score']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                f"{metrics['auprc']:.4f}",
                f"{metrics['training_time']:.2f}s"
            ]
            report_lines.append("| " + " | ".join(row) + " |")
            
        # Cross-validation results
        best_model = cv_results['best_model']
        report_lines.extend([
            "",
            "## Cross-Validation Results",
            f"**Mejor Modelo**: {best_model}",
            "",
            f"- **Precision**: {cv_results['precision_mean']:.4f} ± {cv_results['precision_std']:.4f}",
            f"- **Recall**: {cv_results['recall_mean']:.4f} ± {cv_results['recall_std']:.4f}",
            f"- **F1-Score**: {cv_results['f1_score_mean']:.4f} ± {cv_results['f1_score_std']:.4f}",
            f"- **ROC-AUC**: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}",
            f"- **AUPRC**: {cv_results['auprc_mean']:.4f} ± {cv_results['auprc_std']:.4f}",
            "",
            "## Insights y Próximos Pasos",
            "- [ ] Optimización de hiperparámetros",
            "- [ ] Ensemble methods",  
            "- [ ] Feature importance analysis",
            "- [ ] Threshold optimization",
            "",
            f"*Generado automáticamente por AdvancedFraudTrainingPipeline*"
        ])
        
        return "\n".join(report_lines)
        
    def run_complete_pipeline(self, sample_size: int = 50000) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de entrenamiento"""
        print("🚀 Iniciando Advanced Training Pipeline - Sprint 2.1")
        print("=" * 60)
        
        total_start_time = time.time()
        
        # 1. Cargar datos
        X, y = self.load_data(sample_size)
        
        # 2. Crear feature pipeline
        self.create_feature_pipeline()
        
        # 3. Feature engineering y preprocessing
        self.preprocess_data(X, y)
        
        # 4. Entrenar modelos
        self.train_models()
        
        # 5. Cross-validation
        cv_results = self.cross_validate_best_model()
        
        # 6. Log a MLflow
        self.log_to_mlflow(cv_results)
        
        # 7. Generar reporte
        report = self.generate_report(cv_results)
        
        total_time = time.time() - total_start_time
        
        print("=" * 60)
        print(f"✅ Pipeline completado en {total_time:.2f}s")
        print(f"🏆 Mejor modelo: {cv_results['best_model']}")
        print(f"📊 F1-Score CV: {cv_results['f1_score_mean']:.4f} ± {cv_results['f1_score_std']:.4f}")
        
        return {
            'models': self.models,
            'cv_results': cv_results,
            'report': report,
            'execution_time': total_time
        }


def main():
    """Función principal para ejecutar el pipeline"""
    
    # Configuración del pipeline
    pipeline = AdvancedFraudTrainingPipeline(
        experiment_name="fraud-detection-advanced-sprint-2-1",
        feature_selection_k=150,
        test_size=0.2,
        cv_folds=3
    )
    
    # Ejecutar pipeline completo
    results = pipeline.run_complete_pipeline(sample_size=50000)
    
    # Guardar reporte
    report_path = f"docs/sprints/Sprint_2_1_Advanced_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(results['report'])
    
    print(f"📄 Reporte guardado: {report_path}")
    
    return results


if __name__ == "__main__":
    results = main()

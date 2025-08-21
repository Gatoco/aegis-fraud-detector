"""
Feature Engineering Pipeline para Detección de Fraude
Sprint 2.1: Diseño de Features Avanzadas

Basado en los hallazgos del EDA Sprint 1.1:
- Patrones temporales identificados (hora pico: 13-14h)
- Montos de fraude ligeramente mayores (mediana $89 vs $67)
- Desbalance extremo 3.5% fraude
- 434 features base con alta colinealidad
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Extrae features temporales basados en TransactionDT
    
    Insights del EDA:
    - Hora pico fraude: 13-14h
    - Mayor fraude días laborales vs fines de semana
    - Patrones estacionales detectados
    """
    
    def __init__(self):
        self.reference_time = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Calcula tiempo de referencia para consistencia"""
        if 'TransactionDT' in X.columns:
            self.reference_time = X['TransactionDT'].min()
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Genera features temporales"""
        X_temp = X.copy()
        
        if 'TransactionDT' not in X_temp.columns:
            return X_temp
            
        # Features temporales básicas
        X_temp['hour'] = (X_temp['TransactionDT'] / 3600) % 24
        X_temp['day_of_week'] = ((X_temp['TransactionDT'] / (3600 * 24)) % 7).astype(int)
        X_temp['day_of_month'] = ((X_temp['TransactionDT'] / (3600 * 24)) % 30).astype(int)
        X_temp['week_of_year'] = ((X_temp['TransactionDT'] / (3600 * 24 * 7)) % 52).astype(int)
        
        # Features categóricas temporales
        X_temp['is_weekend'] = (X_temp['day_of_week'] >= 5).astype(int)
        X_temp['is_night'] = ((X_temp['hour'] >= 22) | (X_temp['hour'] <= 6)).astype(int)
        X_temp['is_business_hours'] = ((X_temp['hour'] >= 9) & (X_temp['hour'] <= 17) & 
                                      (X_temp['day_of_week'] < 5)).astype(int)
        X_temp['is_fraud_peak_hour'] = ((X_temp['hour'] >= 13) & (X_temp['hour'] <= 14)).astype(int)
        
        # Features cíclicas (encoding circular para preservar continuidad)
        X_temp['hour_sin'] = np.sin(2 * np.pi * X_temp['hour'] / 24)
        X_temp['hour_cos'] = np.cos(2 * np.pi * X_temp['hour'] / 24)
        X_temp['day_sin'] = np.sin(2 * np.pi * X_temp['day_of_week'] / 7)
        X_temp['day_cos'] = np.cos(2 * np.pi * X_temp['day_of_week'] / 7)
        X_temp['month_sin'] = np.sin(2 * np.pi * X_temp['day_of_month'] / 30)
        X_temp['month_cos'] = np.cos(2 * np.pi * X_temp['day_of_month'] / 30)
        
        return X_temp


class AmountFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Extrae features relacionadas con montos transaccionales
    
    Insights del EDA:
    - Mediana fraude: $89 vs legítimo: $67
    - Distribución fraude más dispersa
    - Correlación débil pero positiva: +0.011
    """
    
    def __init__(self):
        self.amount_stats = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Calcula estadísticas de montos para features relativas"""
        if 'TransactionAmt' in X.columns:
            self.amount_stats = {
                'mean': X['TransactionAmt'].mean(),
                'median': X['TransactionAmt'].median(),
                'std': X['TransactionAmt'].std(),
                'q25': X['TransactionAmt'].quantile(0.25),
                'q75': X['TransactionAmt'].quantile(0.75),
                'q95': X['TransactionAmt'].quantile(0.95),
                'q99': X['TransactionAmt'].quantile(0.99)
            }
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Genera features de montos"""
        X_amt = X.copy()
        
        if 'TransactionAmt' not in X_amt.columns:
            return X_amt
            
        amt = X_amt['TransactionAmt']
        
        # Features de distribución
        X_amt['amount_zscore'] = (amt - self.amount_stats['mean']) / self.amount_stats['std']
        X_amt['amount_above_median'] = (amt > self.amount_stats['median']).astype(int)
        X_amt['amount_above_q75'] = (amt > self.amount_stats['q75']).astype(int)
        X_amt['amount_above_q95'] = (amt > self.amount_stats['q95']).astype(int)
        X_amt['amount_outlier'] = (amt > self.amount_stats['q99']).astype(int)
        
        # Features de patrones
        X_amt['amount_is_round'] = (amt == amt.round()).astype(int)
        X_amt['amount_is_round_10'] = (amt % 10 == 0).astype(int)
        X_amt['amount_is_round_100'] = (amt % 100 == 0).astype(int)
        X_amt['amount_decimal_places'] = amt.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
        
        # Features logarítmicas (para normalizar distribución sesgada)
        X_amt['amount_log'] = np.log1p(amt)
        X_amt['amount_sqrt'] = np.sqrt(amt)
        
        # Features categóricas por rango
        X_amt['amount_category'] = pd.cut(amt, 
                                         bins=[0, 25, 67, 89, 200, 500, np.inf], 
                                         labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme'],
                                         include_lowest=True)
        
        return X_amt


class AggregationFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Crea features de agregación por entidades (cards, email, etc.)
    
    Insights del EDA:
    - Features de identidad con valores faltantes significativos
    - Necesidad de agregaciones para capturar comportamiento histórico
    """
    
    def __init__(self, lookback_hours: List[int] = [1, 6, 24, 168]):  # 1h, 6h, 1d, 1w
        self.lookback_hours = lookback_hours
        self.aggregation_stats = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Calcula estadísticas para agregaciones"""
        # Para entrenamiento, calculamos estadísticas globales
        # En producción, estas se actualizarían dinámicamente
        self._calculate_entity_stats(X, y)
        return self
        
    def _calculate_entity_stats(self, X: pd.DataFrame, y=None):
        """Calcula estadísticas por entidad para features de agregación"""
        entity_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                      'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
        
        for col in entity_cols:
            if col in X.columns:
                # Estadísticas básicas por entidad
                if y is not None:
                    # Si tenemos labels, calculamos tasa de fraude por entidad
                    fraud_rates = pd.DataFrame({'entity': X[col], 'fraud': y}).groupby('entity')['fraud'].agg([
                        'count', 'sum', 'mean'
                    ]).reset_index()
                    fraud_rates.columns = ['entity', 'count', 'fraud_count', 'fraud_rate']
                    self.aggregation_stats[f'{col}_fraud_stats'] = fraud_rates.set_index('entity').to_dict()
                
                # Estadísticas de montos por entidad
                if 'TransactionAmt' in X.columns:
                    amount_stats = X.groupby(col)['TransactionAmt'].agg([
                        'count', 'mean', 'std', 'min', 'max', 'median'
                    ]).reset_index()
                    self.aggregation_stats[f'{col}_amount_stats'] = amount_stats.set_index(col).to_dict()
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Genera features de agregación"""
        X_agg = X.copy()
        
        # Features básicas de conteo por entidad
        entity_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
        
        for col in entity_cols:
            if col in X_agg.columns:
                # Frecuencia de la entidad
                value_counts = X_agg[col].value_counts()
                X_agg[f'{col}_frequency'] = X_agg[col].map(value_counts).fillna(0)
                
                # Indicador de entidad rara
                X_agg[f'{col}_is_rare'] = (X_agg[f'{col}_frequency'] <= 5).astype(int)
                
                # Features de riesgo por entidad (si disponibles del fit)
                fraud_stats_key = f'{col}_fraud_stats'
                if fraud_stats_key in self.aggregation_stats:
                    fraud_rates = self.aggregation_stats[fraud_stats_key].get('fraud_rate', {})
                    X_agg[f'{col}_historical_fraud_rate'] = X_agg[col].map(fraud_rates).fillna(0.035)  # Global rate
                
                # Features de monto por entidad
                amount_stats_key = f'{col}_amount_stats'
                if amount_stats_key in self.aggregation_stats:
                    mean_amounts = self.aggregation_stats[amount_stats_key].get('mean', {})
                    X_agg[f'{col}_avg_amount'] = X_agg[col].map(mean_amounts).fillna(X_agg['TransactionAmt'].mean())
                    
                    # Ratio del monto actual vs promedio histórico
                    if 'TransactionAmt' in X_agg.columns:
                        X_agg[f'{col}_amount_ratio'] = X_agg['TransactionAmt'] / X_agg[f'{col}_avg_amount']
        
        return X_agg


class InteractionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Crea features de interacción entre variables importantes
    
    Basado en correlaciones identificadas en EDA
    """
    
    def __init__(self):
        self.feature_interactions = [
            ('TransactionAmt', 'hour'),
            ('card1', 'TransactionAmt'),
            ('addr1', 'TransactionAmt'),
            ('hour', 'day_of_week'),
            ('C1', 'C2'),  # Features C altamente correlacionadas según EDA
            ('D1', 'D2'),  # Features D con patrones
        ]
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Genera features de interacción"""
        X_int = X.copy()
        
        for feat1, feat2 in self.feature_interactions:
            if feat1 in X_int.columns and feat2 in X_int.columns:
                # Producto de features (captura interacciones multiplicativas)
                X_int[f'{feat1}_x_{feat2}'] = X_int[feat1] * X_int[feat2]
                
                # Ratio de features (captura relaciones proporcionales)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X_int[f'{feat1}_div_{feat2}'] = X_int[feat1] / (X_int[feat2] + 1e-8)
                
                # Diferencia absoluta (captura distancia)
                X_int[f'{feat1}_diff_{feat2}'] = np.abs(X_int[feat1] - X_int[feat2])
        
        return X_int


class FraudFeatureEngineeringPipeline:
    """
    Pipeline completo de feature engineering para detección de fraude
    
    Integra todos los transformadores en un pipeline versionado y reproducible
    """
    
    def __init__(self, 
                 use_temporal: bool = True,
                 use_amount: bool = True, 
                 use_aggregation: bool = True,
                 use_interactions: bool = True,
                 feature_selection_k: int = 200):
        
        self.use_temporal = use_temporal
        self.use_amount = use_amount
        self.use_aggregation = use_aggregation
        self.use_interactions = use_interactions
        self.feature_selection_k = feature_selection_k
        
        # Inicializar transformadores
        self.transformers = {}
        if use_temporal:
            self.transformers['temporal'] = TemporalFeatureEngineer()
        if use_amount:
            self.transformers['amount'] = AmountFeatureEngineer()
        if use_aggregation:
            self.transformers['aggregation'] = AggregationFeatureEngineer()
        if use_interactions:
            self.transformers['interactions'] = InteractionFeatureEngineer()
            
        self.preprocessing_pipeline = None
        self.feature_selector = None
        self.feature_names_out = None
        
    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Crea pipeline de preprocessing con ColumnTransformer
        Maneja diferentes tipos de features de manera consistente
        """
        
        # Identificar tipos de columnas
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remover target y ID si están presentes
        numeric_features = [f for f in numeric_features if f not in ['isFraud', 'TransactionID']]
        categorical_features = [f for f in categorical_features if f not in ['isFraud', 'TransactionID']]
        
        # Pipeline para features numéricas
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())  # Más robusto a outliers que StandardScaler
        ])
        
        # Pipeline para features categóricas
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False) if len(categorical_features) > 0 else 'passthrough')
        ])
        
        # Crear ColumnTransformer
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ], remainder='drop')
        
        return preprocessor
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FraudFeatureEngineeringPipeline':
        """Entrena el pipeline completo de feature engineering"""
        
        print("🔧 Iniciando Feature Engineering Pipeline...")
        X_processed = X.copy()
        
        # Aplicar transformadores de feature engineering
        for name, transformer in self.transformers.items():
            print(f"  📊 Aplicando {name} feature engineering...")
            transformer.fit(X_processed, y)
            X_processed = transformer.transform(X_processed)
            
        print(f"  ✅ Features después de engineering: {X_processed.shape[1]}")
        
        # Crear y entrenar pipeline de preprocessing
        print("  🏗️ Creando pipeline de preprocessing...")
        self.preprocessing_pipeline = self.create_preprocessing_pipeline(X_processed)
        
        # Preparar datos para feature selection
        X_preprocessed = self.preprocessing_pipeline.fit_transform(X_processed)
        
        # Feature selection si se especifica
        if self.feature_selection_k and self.feature_selection_k < X_preprocessed.shape[1] and y is not None:
            print(f"  🎯 Seleccionando top {self.feature_selection_k} features...")
            self.feature_selector = SelectKBest(
                score_func=f_classif, 
                k=min(self.feature_selection_k, X_preprocessed.shape[1])
            )
            self.feature_selector.fit(X_preprocessed, y)
            
        # Guardar nombres de features finales
        self._save_feature_names(X_processed)
        
        print("✅ Pipeline de Feature Engineering entrenado!")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica el pipeline completo de transformaciones"""
        
        X_processed = X.copy()
        
        # Aplicar feature engineering
        for name, transformer in self.transformers.items():
            X_processed = transformer.transform(X_processed)
            
        # Aplicar preprocessing
        X_preprocessed = self.preprocessing_pipeline.transform(X_processed)
        
        # Aplicar feature selection si está disponible
        if self.feature_selector:
            X_preprocessed = self.feature_selector.transform(X_preprocessed)
            
        # Convertir a DataFrame con nombres apropiados
        if self.feature_names_out:
            feature_names = self.feature_names_out[:X_preprocessed.shape[1]]
        else:
            feature_names = [f'feature_{i}' for i in range(X_preprocessed.shape[1])]
            
        return pd.DataFrame(X_preprocessed, columns=feature_names, index=X.index)
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Entrena y transforma en un solo paso"""
        return self.fit(X, y).transform(X)
        
    def _save_feature_names(self, X_processed: pd.DataFrame):
        """Guarda nombres de features para consistencia"""
        # Obtener nombres después de preprocessing
        numeric_features = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remover target y ID
        numeric_features = [f for f in numeric_features if f not in ['isFraud', 'TransactionID']]
        categorical_features = [f for f in categorical_features if f not in ['isFraud', 'TransactionID']]
        
        self.feature_names_out = numeric_features + categorical_features
        
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Retorna resumen de importancia de features si está disponible"""
        summary = {
            'total_features_engineered': len(self.feature_names_out) if self.feature_names_out else 0,
            'transformers_applied': list(self.transformers.keys()),
            'feature_selection_applied': self.feature_selector is not None
        }
        
        if self.feature_selector:
            summary['selected_features'] = self.feature_selection_k
            summary['feature_scores'] = self.feature_selector.scores_.tolist() if hasattr(self.feature_selector, 'scores_') else None
            
        return summary


# Importar Pipeline de sklearn para usar en ColumnTransformer
from sklearn.pipeline import Pipeline


def create_fraud_detection_pipeline(feature_selection_k: int = 200) -> FraudFeatureEngineeringPipeline:
    """
    Factory function para crear pipeline optimizado para detección de fraude
    
    Args:
        feature_selection_k: Número de features top a seleccionar
        
    Returns:
        Pipeline configurado para detección de fraude
    """
    
    pipeline = FraudFeatureEngineeringPipeline(
        use_temporal=True,      # Patrones temporales confirmados en EDA
        use_amount=True,        # Diferencias de monto detectadas  
        use_aggregation=True,   # Comportamiento por entidad crítico
        use_interactions=True,  # Capturar relaciones complejas
        feature_selection_k=feature_selection_k
    )
    
    return pipeline


if __name__ == "__main__":
    """
    Test básico del pipeline de feature engineering
    """
    print("🧪 Ejecutando test del Feature Engineering Pipeline...")
    
    # Datos de ejemplo
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'TransactionID': range(n_samples),
        'TransactionDT': np.random.randint(0, 60*60*24*30, n_samples),  # 30 días de segundos
        'TransactionAmt': np.random.lognormal(4, 1, n_samples),  # Distribución log-normal
        'card1': np.random.randint(1000, 9999, n_samples),
        'card2': np.random.randint(100, 999, n_samples),
        'addr1': np.random.randint(100, 500, n_samples),
        'C1': np.random.normal(0, 1, n_samples),
        'C2': np.random.normal(0, 1, n_samples),
        'D1': np.random.normal(10, 5, n_samples),
        'D2': np.random.normal(20, 8, n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.965, 0.035])
    })
    
    # Crear pipeline
    pipeline = create_fraud_detection_pipeline(feature_selection_k=50)
    
    # Separar features y target
    X = test_data.drop(['isFraud'], axis=1)
    y = test_data['isFraud']
    
    # Ejecutar pipeline
    try:
        X_transformed = pipeline.fit_transform(X, y)
        print(f"✅ Pipeline exitoso!")
        print(f"   Forma original: {X.shape}")
        print(f"   Forma transformada: {X_transformed.shape}")
        print(f"   Features creadas: {X_transformed.shape[1] - X.shape[1]}")
        
        # Mostrar resumen
        summary = pipeline.get_feature_importance_summary()
        print(f"   Resumen: {summary}")
        
    except Exception as e:
        print(f"❌ Error en pipeline: {e}")
        raise

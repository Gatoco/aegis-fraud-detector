"""
DVC Pipeline para Versionado de Datasets Procesados
Sprint 2.1: Sistema de Versionado con DVC

Versiona datasets en diferentes etapas:
- Raw data
- Feature engineered data  
- Preprocessed data
- Model artifacts
"""

import pandas as pd
import numpy as np
import pickle
import yaml
from pathlib import Path
import time
from datetime import datetime

# Configuración de paths
DATA_RAW_DIR = Path("data/01_raw")
DATA_PROCESSED_DIR = Path("data/02_processed")
DATA_FEATURES_DIR = Path("data/03_features") 
MODELS_DIR = Path("models")

# Crear directorios si no existen
for dir_path in [DATA_PROCESSED_DIR, DATA_FEATURES_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def create_feature_engineered_dataset(sample_size: int = 50000, version: str = "v1.0") -> dict:
    """
    Crea dataset con feature engineering aplicado y lo versiona
    
    Args:
        sample_size: Número de muestras a procesar
        version: Versión del dataset (para tracking)
        
    Returns:
        dict: Metadatos del dataset creado
    """
    print(f"🔧 Creando dataset con feature engineering (v{version})...")
    start_time = time.time()
    
    # Importar pipeline de feature engineering
    import sys
    sys.path.append('src')
    from features.feature_engineering import create_fraud_detection_pipeline
    
    # Cargar datos raw
    print("📂 Cargando datos raw...")
    train_transaction = pd.read_csv(DATA_RAW_DIR / "train_transaction.csv", nrows=sample_size)
    train_identity = pd.read_csv(DATA_RAW_DIR / "train_identity.csv")
    
    # Merge datasets
    df = train_transaction.merge(train_identity, on='TransactionID', how='left')
    
    # Separar features y target
    target_col = 'isFraud'
    feature_cols = [col for col in df.columns if col not in [target_col, 'TransactionID']]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"✅ Datos cargados: {X.shape}")
    
    # Aplicar feature engineering
    print("🔧 Aplicando feature engineering...")
    fe_pipeline = create_fraud_detection_pipeline(feature_selection_k=150)
    X_engineered = fe_pipeline.fit_transform(X, y)
    
    print(f"✅ Feature engineering completado: {X_engineered.shape}")
    
    # Crear dataset final
    dataset_engineered = X_engineered.copy()
    dataset_engineered['isFraud'] = y.values
    dataset_engineered['TransactionID'] = df['TransactionID'].values
    
    # Metadatos del dataset
    metadata = {
        'version': version,
        'created_at': datetime.now().isoformat(),
        'shape': dataset_engineered.shape,
        'sample_size': sample_size,
        'fraud_rate': y.mean(),
        'feature_count': len(X_engineered.columns),
        'original_feature_count': len(feature_cols),
        'processing_time_seconds': time.time() - start_time,
        'pipeline_config': {
            'feature_selection_k': 150,
            'transformers': ['temporal', 'amount', 'aggregation', 'interactions']
        }
    }
    
    # Guardar dataset
    output_path = DATA_FEATURES_DIR / f"fraud_dataset_engineered_{version}.parquet"
    dataset_engineered.to_parquet(output_path, index=False)
    
    # Guardar metadatos
    metadata_path = DATA_FEATURES_DIR / f"fraud_dataset_engineered_{version}_metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    # Guardar pipeline fitted
    pipeline_path = MODELS_DIR / f"feature_pipeline_{version}.pkl"
    with open(pipeline_path, 'wb') as f:
        pickle.dump(fe_pipeline, f)
    
    print(f"💾 Dataset guardado: {output_path}")
    print(f"📊 Metadatos: {metadata_path}")
    print(f"🔧 Pipeline: {pipeline_path}")
    
    return metadata


def create_train_test_splits(version: str = "v1.0", test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Crea splits de entrenamiento y prueba versionados
    
    Args:
        version: Versión del dataset base
        test_size: Proporción para test set
        random_state: Semilla para reproducibilidad
        
    Returns:
        dict: Metadatos de los splits creados
    """
    print(f"📊 Creando splits train/test (v{version})...")
    
    from sklearn.model_selection import train_test_split
    
    # Cargar dataset engineered
    dataset_path = DATA_FEATURES_DIR / f"fraud_dataset_engineered_{version}.parquet"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_path} no encontrado. Ejecutar feature engineering primero.")
    
    df = pd.read_parquet(dataset_path)
    
    # Separar features y target
    feature_cols = [col for col in df.columns if col not in ['isFraud', 'TransactionID']]
    X = df[feature_cols]
    y = df['isFraud']
    transaction_ids = df['TransactionID']
    
    # Split estratificado
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, transaction_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Crear datasets de train y test
    train_df = X_train.copy()
    train_df['isFraud'] = y_train
    train_df['TransactionID'] = ids_train
    
    test_df = X_test.copy()
    test_df['isFraud'] = y_test  
    test_df['TransactionID'] = ids_test
    
    # Guardar splits
    train_path = DATA_PROCESSED_DIR / f"fraud_train_{version}.parquet"
    test_path = DATA_PROCESSED_DIR / f"fraud_test_{version}.parquet"
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    # Metadatos de splits
    split_metadata = {
        'version': version,
        'created_at': datetime.now().isoformat(),
        'test_size': test_size,
        'random_state': random_state,
        'train_shape': train_df.shape,
        'test_shape': test_df.shape,
        'train_fraud_rate': y_train.mean(),
        'test_fraud_rate': y_test.mean(),
        'stratified': True
    }
    
    # Guardar metadatos
    split_metadata_path = DATA_PROCESSED_DIR / f"fraud_splits_{version}_metadata.yaml"
    with open(split_metadata_path, 'w') as f:
        yaml.dump(split_metadata, f, default_flow_style=False)
    
    print(f"💾 Train set: {train_path} - {train_df.shape}")
    print(f"💾 Test set: {test_path} - {test_df.shape}")
    print(f"📊 Split metadata: {split_metadata_path}")
    
    return split_metadata


def create_dvc_pipeline_yaml():
    """Crea archivo dvc.yaml para pipeline reproducible"""
    
    dvc_pipeline = {
        'stages': {
            'feature_engineering': {
                'cmd': 'python scripts/create_featured_dataset.py',
                'deps': [
                    'data/01_raw/train_transaction.csv',
                    'data/01_raw/train_identity.csv',
                    'src/features/feature_engineering.py'
                ],
                'outs': [
                    'data/03_features/fraud_dataset_engineered_v1.0.parquet',
                    'data/03_features/fraud_dataset_engineered_v1.0_metadata.yaml',
                    'models/feature_pipeline_v1.0.pkl'
                ],
                'params': [
                    'sample_size',
                    'feature_selection_k'
                ]
            },
            'create_splits': {
                'cmd': 'python scripts/create_train_test_splits.py',
                'deps': [
                    'data/03_features/fraud_dataset_engineered_v1.0.parquet'
                ],
                'outs': [
                    'data/02_processed/fraud_train_v1.0.parquet',
                    'data/02_processed/fraud_test_v1.0.parquet', 
                    'data/02_processed/fraud_splits_v1.0_metadata.yaml'
                ],
                'params': [
                    'test_size',
                    'random_state'
                ]
            }
        }
    }
    
    # Guardar dvc.yaml
    with open('dvc.yaml', 'w') as f:
        yaml.dump(dvc_pipeline, f, default_flow_style=False)
    
    print("✅ dvc.yaml creado")


def create_params_yaml():
    """Crea archivo params.yaml con parámetros del pipeline"""
    
    params = {
        'sample_size': 50000,
        'feature_selection_k': 150,
        'test_size': 0.2,
        'random_state': 42,
        'feature_engineering': {
            'use_temporal': True,
            'use_amount': True,
            'use_aggregation': True,
            'use_interactions': True
        },
        'model_training': {
            'cv_folds': 3,
            'models': ['logistic_baseline', 'logistic_advanced', 'random_forest']
        }
    }
    
    # Guardar params.yaml
    with open('params.yaml', 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    
    print("✅ params.yaml creado")


def main():
    """Ejecuta el pipeline completo de versionado"""
    print("🚀 Iniciando DVC Pipeline - Sprint 2.1")
    print("=" * 50)
    
    version = "v1.0"
    
    # 1. Crear dataset con feature engineering
    fe_metadata = create_feature_engineered_dataset(
        sample_size=50000, 
        version=version
    )
    
    print("\n" + "=" * 50)
    
    # 2. Crear splits train/test
    split_metadata = create_train_test_splits(
        version=version,
        test_size=0.2,
        random_state=42
    )
    
    print("\n" + "=" * 50)
    
    # 3. Crear archivos DVC
    create_dvc_pipeline_yaml()
    create_params_yaml()
    
    print("\n" + "=" * 50)
    print("✅ DVC Pipeline completado!")
    print(f"📦 Dataset version: {version}")
    print(f"📊 Features engineered: {fe_metadata['feature_count']}")
    print(f"📈 Fraud rate: {fe_metadata['fraud_rate']:.4f}")
    print(f"🔧 Processing time: {fe_metadata['processing_time_seconds']:.2f}s")
    
    print("\n🔄 Para ejecutar con DVC:")
    print("  dvc repro")
    print("\n📊 Para ver métricas:")
    print("  dvc metrics show")
    print("\n🔗 Para trackear cambios:")
    print("  dvc add data/03_features/fraud_dataset_engineered_v1.0.parquet")
    print("  git add data/03_features/fraud_dataset_engineered_v1.0.parquet.dvc")
    
    return {
        'feature_engineering': fe_metadata,
        'splits': split_metadata,
        'version': version
    }


if __name__ == "__main__":
    results = main()

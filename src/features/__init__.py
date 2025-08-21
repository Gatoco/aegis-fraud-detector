# Features Module
"""
Módulo de feature engineering para AEGIS Fraud Detector

Contiene transformadores especializados para crear features avanzadas
basadas en los hallazgos del EDA Sprint 1.1
"""

from .feature_engineering import (
    TemporalFeatureEngineer,
    AmountFeatureEngineer, 
    AggregationFeatureEngineer,
    InteractionFeatureEngineer,
    FraudFeatureEngineeringPipeline,
    create_fraud_detection_pipeline
)

__all__ = [
    'TemporalFeatureEngineer',
    'AmountFeatureEngineer',
    'AggregationFeatureEngineer', 
    'InteractionFeatureEngineer',
    'FraudFeatureEngineeringPipeline',
    'create_fraud_detection_pipeline'
]

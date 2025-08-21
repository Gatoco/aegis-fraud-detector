"""
Aegis Fraud Detector - Training Pipelines Module

This module contains training pipelines for different stages of the fraud detection project.

Modules:
- train_baseline: Baseline training pipeline with Logistic Regression
- (Future) train_advanced: Advanced pipeline with feature engineering
- (Future) train_ensemble: Ensemble training pipeline
"""

from .train_baseline import BaselineTrainingPipeline

__all__ = ['BaselineTrainingPipeline']

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Fraud Detection API - Sprint 3.1
=====================================

FastAPI service for real-time fraud detection using optimized LightGBM model
from Sprint 2.4 with MLflow integration.

Features:
- FastAPI application with automatic documentation
- Pydantic models for request/response validation
- Health check endpoint (/health)
- Prediction endpoint (/v1/predict)
- MLflow model integration
- Production-ready architecture

Author: AEGIS Fraud Detection Team
Sprint: 3.1 - API Service Creation
Date: 2025-08-21
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Pydantic models
from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict

# ML imports
try:
    import mlflow
    import mlflow.pyfunc
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App configuration
app = FastAPI(
    title="AEGIS Fraud Detection API",
    description="Real-time fraud detection service using optimized LightGBM model from Sprint 2.4",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
MODEL_CACHE = {
    "model": None,
    "scaler": None,
    "feature_names": None,
    "model_version": None,
    "loaded": False
}


class TransactionInput(BaseModel):
    """
    Pydantic model for transaction input validation
    
    Based on IEEE-CIS Fraud Detection dataset core features
    Optimized for production use with essential fraud indicators
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    # Core transaction features (required)
    TransactionAmt: float = Field(..., gt=0, le=1000000, description="Transaction amount in USD")
    ProductCD: str = Field(..., description="Product code (W, C, R, H, S)")
    
    # Card features (optional but important)
    card1: Optional[int] = Field(None, ge=0, description="Card identifier 1")
    card2: Optional[float] = Field(None, description="Card feature 2")
    card3: Optional[float] = Field(None, description="Card feature 3")
    card4: Optional[str] = Field(None, description="Card category (discover, mastercard, visa, american express)")
    card5: Optional[float] = Field(None, description="Card feature 5")
    card6: Optional[str] = Field(None, description="Card type (debit, credit)")
    
    # Address features
    addr1: Optional[float] = Field(None, description="Address feature 1")
    addr2: Optional[float] = Field(None, description="Address feature 2")
    
    # Distance features
    dist1: Optional[float] = Field(None, ge=0, description="Distance feature 1")
    dist2: Optional[float] = Field(None, ge=0, description="Distance feature 2")
    
    # Email domains
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    
    # Count features (C1-C14)
    C1: Optional[float] = Field(None, ge=0, description="Count feature 1")
    C2: Optional[float] = Field(None, ge=0, description="Count feature 2")
    C3: Optional[float] = Field(None, ge=0, description="Count feature 3")
    C4: Optional[float] = Field(None, ge=0, description="Count feature 4")
    C5: Optional[float] = Field(None, ge=0, description="Count feature 5")
    C6: Optional[float] = Field(None, ge=0, description="Count feature 6")
    C7: Optional[float] = Field(None, ge=0, description="Count feature 7")
    C8: Optional[float] = Field(None, ge=0, description="Count feature 8")
    C9: Optional[float] = Field(None, ge=0, description="Count feature 9")
    C10: Optional[float] = Field(None, ge=0, description="Count feature 10")
    C11: Optional[float] = Field(None, ge=0, description="Count feature 11")
    C12: Optional[float] = Field(None, ge=0, description="Count feature 12")
    C13: Optional[float] = Field(None, ge=0, description="Count feature 13")
    C14: Optional[float] = Field(None, ge=0, description="Count feature 14")
    
    # Timedelta features (D1-D15)
    D1: Optional[float] = Field(None, description="Timedelta feature 1 (days)")
    D2: Optional[float] = Field(None, description="Timedelta feature 2 (days)")
    D3: Optional[float] = Field(None, description="Timedelta feature 3 (days)")
    D4: Optional[float] = Field(None, description="Timedelta feature 4 (days)")
    D5: Optional[float] = Field(None, description="Timedelta feature 5 (days)")
    D6: Optional[float] = Field(None, description="Timedelta feature 6 (days)")
    D7: Optional[float] = Field(None, description="Timedelta feature 7 (days)")
    D8: Optional[float] = Field(None, description="Timedelta feature 8 (days)")
    D9: Optional[float] = Field(None, description="Timedelta feature 9 (days)")
    D10: Optional[float] = Field(None, description="Timedelta feature 10 (days)")
    D11: Optional[float] = Field(None, description="Timedelta feature 11 (days)")
    D12: Optional[float] = Field(None, description="Timedelta feature 12 (days)")
    D13: Optional[float] = Field(None, description="Timedelta feature 13 (days)")
    D14: Optional[float] = Field(None, description="Timedelta feature 14 (days)")
    D15: Optional[float] = Field(None, description="Timedelta feature 15 (days)")
    
    # Match features (M1-M9)
    M1: Optional[str] = Field(None, description="Match feature 1 (T/F)")
    M2: Optional[str] = Field(None, description="Match feature 2 (T/F)")
    M3: Optional[str] = Field(None, description="Match feature 3 (T/F)")
    M4: Optional[str] = Field(None, description="Match feature 4 (M0/M1/M2)")
    M5: Optional[str] = Field(None, description="Match feature 5 (T/F)")
    M6: Optional[str] = Field(None, description="Match feature 6 (T/F)")
    M7: Optional[str] = Field(None, description="Match feature 7 (T/F)")
    M8: Optional[str] = Field(None, description="Match feature 8 (T/F)")
    M9: Optional[str] = Field(None, description="Match feature 9 (T/F)")
    
    # Vesta features (simplified - top 5)
    V1: Optional[float] = Field(None, description="Vesta engineered feature 1")
    V2: Optional[float] = Field(None, description="Vesta engineered feature 2")
    V3: Optional[float] = Field(None, description="Vesta engineered feature 3")
    V4: Optional[float] = Field(None, description="Vesta engineered feature 4")
    V5: Optional[float] = Field(None, description="Vesta engineered feature 5")
    
    @validator('ProductCD')
    def validate_product_cd(cls, v):
        """Validate ProductCD against known values"""
        valid_products = {'W', 'C', 'R', 'H', 'S'}
        if v.upper() not in valid_products:
            raise ValueError(f'ProductCD must be one of {valid_products}')
        return v.upper()
    
    @validator('card4')
    def validate_card4(cls, v):
        """Validate card4 category"""
        if v is None:
            return v
        valid_cards = {'discover', 'mastercard', 'visa', 'american express'}
        if v.lower() not in valid_cards:
            logger.warning(f"Unknown card4 value: {v}")
        return v.lower()
    
    @validator('card6')
    def validate_card6(cls, v):
        """Validate card6 type"""
        if v is None:
            return v
        valid_types = {'debit', 'credit'}
        if v.lower() not in valid_types:
            logger.warning(f"Unknown card6 value: {v}")
        return v.lower()


class FraudPrediction(BaseModel):
    """
    Pydantic model for fraud prediction response
    
    Comprehensive response with probability, risk assessment, and metadata
    """
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    # Prediction results
    transaction_id: str = Field(..., description="Unique transaction identifier")
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability [0-1]")
    fraud_score: int = Field(..., ge=0, le=1000, description="Fraud score [0-1000]")
    risk_level: str = Field(..., description="Risk assessment: LOW, MEDIUM, HIGH, CRITICAL")
    prediction: str = Field(..., description="Binary prediction: FRAUD or LEGITIMATE")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    
    # Model metadata
    model_version: str = Field(..., description="Model version used for prediction")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    
    # Additional context
    risk_factors: List[str] = Field(default=[], description="Key risk factors identified")
    recommendation: str = Field(..., description="Recommended action")
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        """Validate risk level"""
        valid_levels = {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}
        if v not in valid_levels:
            raise ValueError(f'risk_level must be one of {valid_levels}')
        return v
    
    @validator('prediction')
    def validate_prediction(cls, v):
        """Validate prediction"""
        valid_predictions = {'FRAUD', 'LEGITIMATE'}
        if v not in valid_predictions:
            raise ValueError(f'prediction must be one of {valid_predictions}')
        return v


class HealthResponse(BaseModel):
    """
    Health check response model
    """
    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    timestamp: str = Field(..., description="Current timestamp (ISO format)")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Model loading status")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    dependencies: Dict[str, str] = Field(default={}, description="Dependency status")


def load_model_pipeline() -> bool:
    """
    Load the optimized LightGBM model pipeline from Sprint 2.4
    
    Attempts to load from:
    1. MLflow Model Registry (Production stage)
    2. Local model artifacts
    3. Fallback dummy model
    
    Returns:
        bool: True if model loaded successfully
    """
    try:
        logger.info("🔄 Loading fraud detection model...")
        
        # Method 1: Try MLflow Model Registry
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri("http://localhost:5001")
                model = mlflow.pyfunc.load_model("models:/fraud-detection-lightgbm/Production")
                MODEL_CACHE["model"] = model
                MODEL_CACHE["model_version"] = "mlflow-production"
                MODEL_CACHE["loaded"] = True
                logger.info("✅ Model loaded from MLflow Registry (Production)")
                return True
            except Exception as e:
                logger.warning(f"MLflow Registry unavailable: {e}")
        
        # Method 2: Try local model artifacts
        model_paths = [
            "../docs/sprints/optimization_results",
            "../../docs/sprints/optimization_results",
            "../../../docs/sprints/optimization_results",
            "models"
        ]
        
        for model_path in model_paths:
            try:
                path = Path(model_path)
                if path.exists():
                    model_files = list(path.glob("best_lightgbm_model_*.pkl"))
                    if model_files:
                        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                        logger.info(f"Loading model from: {latest_model}")
                        
                        with open(latest_model, 'rb') as f:
                            model = pickle.load(f)
                        
                        MODEL_CACHE["model"] = model
                        MODEL_CACHE["model_version"] = f"local-{latest_model.stem}"
                        MODEL_CACHE["loaded"] = True
                        logger.info("✅ Model loaded from local artifact")
                        return True
            except Exception as e:
                logger.debug(f"Failed to load from {model_path}: {e}")
        
        # Method 3: Create dummy model for testing
        logger.warning("🚨 Creating dummy model for testing purposes")
        MODEL_CACHE["model"] = DummyFraudModel()
        MODEL_CACHE["model_version"] = "dummy-v1.0"
        MODEL_CACHE["loaded"] = True
        logger.info("⚠️ Dummy model loaded - replace with real model for production")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load any model: {e}")
        return False


class DummyFraudModel:
    """
    Dummy model for testing when real model is not available
    """
    
    def predict_proba(self, X):
        """Return dummy probabilities based on transaction amount"""
        if hasattr(X, 'iloc'):
            # DataFrame
            amounts = X.iloc[:, 0] if len(X.columns) > 0 else [100]
        else:
            # Array
            amounts = X[:, 0] if X.shape[1] > 0 else [100]
        
        # Simple heuristic: higher amounts = higher fraud probability
        probabilities = []
        for amount in amounts:
            if amount > 1000:
                fraud_prob = min(0.8, amount / 10000)
            elif amount > 500:
                fraud_prob = 0.3
            else:
                fraud_prob = 0.1
            
            probabilities.append([1 - fraud_prob, fraud_prob])
        
        return np.array(probabilities)


def preprocess_transaction(transaction: TransactionInput) -> pd.DataFrame:
    """
    Preprocess transaction data for model prediction
    
    Args:
        transaction: Validated transaction input
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    # Convert to dictionary and DataFrame
    data = transaction.model_dump()
    df = pd.DataFrame([data])
    
    # Define feature order (should match training data)
    feature_order = [
        'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain'
    ] + [f'C{i}' for i in range(1, 15)] + [f'D{i}' for i in range(1, 16)] + \
    [f'M{i}' for i in range(1, 10)] + [f'V{i}' for i in range(1, 6)]
    
    # Add missing features with default values
    for feature in feature_order:
        if feature not in df.columns:
            if feature.startswith(('C', 'D', 'V')) or feature in ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2']:
                df[feature] = 0.0
            else:
                df[feature] = 'unknown'
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('unknown')
    
    # Reorder columns to match training
    df = df[feature_order]
    
    return df


def calculate_risk_assessment(probability: float, amount: float) -> Dict[str, Any]:
    """
    Calculate comprehensive risk assessment
    
    Args:
        probability: Fraud probability [0-1]
        amount: Transaction amount
        
    Returns:
        Dictionary with risk level, factors, and recommendation
    """
    risk_factors = []
    
    # Probability-based risk
    if probability >= 0.8:
        risk_level = "CRITICAL"
        risk_factors.append("Very high fraud probability")
    elif probability >= 0.6:
        risk_level = "HIGH"
        risk_factors.append("High fraud probability")
    elif probability >= 0.3:
        risk_level = "MEDIUM"
        risk_factors.append("Moderate fraud probability")
    else:
        risk_level = "LOW"
    
    # Amount-based risk
    if amount > 10000:
        risk_factors.append("Large transaction amount")
    elif amount > 1000:
        risk_factors.append("Above-average transaction amount")
    
    # Recommendation
    if risk_level == "CRITICAL":
        recommendation = "BLOCK transaction and investigate immediately"
    elif risk_level == "HIGH":
        recommendation = "REVIEW transaction manually before approval"
    elif risk_level == "MEDIUM":
        recommendation = "MONITOR transaction and user behavior"
    else:
        recommendation = "APPROVE transaction with standard monitoring"
    
    return {
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendation": recommendation
    }


def calculate_confidence(probability: float) -> float:
    """
    Calculate model confidence based on probability distance from 0.5
    
    Args:
        probability: Fraud probability [0-1]
        
    Returns:
        Confidence score [0-1]
    """
    return abs(probability - 0.5) * 2


# Track startup time for uptime calculation
import time
STARTUP_TIME = time.time()


@app.on_event("startup")
async def startup_event():
    """
    Initialize the API service on startup
    """
    logger.info("🚀 Initializing AEGIS Fraud Detection API...")
    
    success = load_model_pipeline()
    if success:
        logger.info("✅ API initialization completed successfully")
    else:
        logger.error("❌ API initialization failed - service will be degraded")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for service monitoring
    
    Returns:
        Comprehensive health status including model state and dependencies
    """
    from datetime import datetime
    
    # Calculate uptime
    uptime = time.time() - STARTUP_TIME
    
    # Check dependencies
    dependencies = {
        "mlflow": "available" if MLFLOW_AVAILABLE else "unavailable",
        "model": "loaded" if MODEL_CACHE["loaded"] else "not_loaded"
    }
    
    # Determine overall status
    if MODEL_CACHE["loaded"]:
        status = "healthy"
    else:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="3.1.0",
        model_loaded=MODEL_CACHE["loaded"],
        model_version=MODEL_CACHE["model_version"],
        uptime_seconds=uptime,
        dependencies=dependencies
    )


@app.post("/v1/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict fraud probability for a transaction
    
    This endpoint processes a transaction and returns comprehensive fraud assessment
    including probability, risk level, and recommended actions.
    
    Args:
        transaction: Transaction data for fraud detection
        
    Returns:
        Comprehensive fraud prediction with risk assessment
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    import uuid
    
    start_time = time.time()
    
    # Check if model is loaded
    if not MODEL_CACHE["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check /health endpoint for service status."
        )
    
    try:
        # Generate unique transaction ID
        transaction_id = str(uuid.uuid4())
        
        # Preprocess transaction data
        processed_data = preprocess_transaction(transaction)
        
        # Make prediction
        model = MODEL_CACHE["model"]
        
        if hasattr(model, 'predict_proba'):
            # Standard scikit-learn/lightgbm interface
            probabilities = model.predict_proba(processed_data)
            fraud_probability = float(probabilities[0][1])  # Probability of fraud class
        elif hasattr(model, 'predict'):
            # MLflow or other model interface
            prediction = model.predict(processed_data)
            if isinstance(prediction, np.ndarray):
                fraud_probability = float(prediction[0])
            else:
                fraud_probability = float(prediction)
        else:
            raise ValueError("Model does not support prediction interface")
        
        # Calculate derived metrics
        fraud_score = int(fraud_probability * 1000)  # Convert to 0-1000 scale
        prediction_label = "FRAUD" if fraud_probability >= 0.5 else "LEGITIMATE"
        confidence = calculate_confidence(fraud_probability)
        
        # Risk assessment
        risk_assessment = calculate_risk_assessment(fraud_probability, transaction.TransactionAmt)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        logger.info(f"Prediction completed: ID={transaction_id[:8]}, Probability={fraud_probability:.3f}, Risk={risk_assessment['risk_level']}")
        
        return FraudPrediction(
            transaction_id=transaction_id,
            fraud_probability=fraud_probability,
            fraud_score=fraud_score,
            risk_level=risk_assessment["risk_level"],
            prediction=prediction_label,
            confidence=confidence,
            model_version=MODEL_CACHE["model_version"] or "unknown",
            processing_time_ms=processing_time,
            risk_factors=risk_assessment["risk_factors"],
            recommendation=risk_assessment["recommendation"]
        )
        
    except Exception as e:
        logger.error(f"Prediction error for transaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint with API information and model performance metrics
    """
    return {
        "service": "AEGIS Fraud Detection API",
        "version": "3.1.0",
        "sprint": "3.1 - API Service Creation",
        "description": "Production-ready fraud detection using optimized LightGBM model",
        "endpoints": {
            "health": "/health - Service health check",
            "predict": "/v1/predict - Fraud prediction",
            "docs": "/docs - Interactive API documentation",
            "redoc": "/redoc - Alternative API documentation"
        },
        "model_info": {
            "type": "LightGBM Classifier (Optimized)",
            "optimization": "Optuna Bayesian Search (Sprint 2.4)",
            "performance": {
                "PR-AUC": 0.6347,
                "F1-Score": 0.6061,
                "improvement_vs_baseline": "+12.16% F1-Score, +5.40% PR-AUC"
            },
            "features": "51 engineered features from IEEE-CIS dataset"
        },
        "usage": {
            "authentication": "None required for demo",
            "rate_limits": "None configured",
            "response_time": "< 100ms typical"
        }
    }


# Additional utility endpoints for monitoring and debugging

@app.get("/v1/model/info")
async def model_info():
    """
    Get detailed model information
    """
    if not MODEL_CACHE["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_version": MODEL_CACHE["model_version"],
        "model_type": str(type(MODEL_CACHE["model"]).__name__),
        "loaded": MODEL_CACHE["loaded"],
        "features_expected": 51,
        "prediction_threshold": 0.5,
        "optimization_details": {
            "framework": "Optuna TPE",
            "trials_completed": 75,
            "best_params": "Available in Sprint 2.4 artifacts"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    logger.info("🚀 Starting AEGIS Fraud Detection API in development mode...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],
        log_level="info",
        access_log=True
    )

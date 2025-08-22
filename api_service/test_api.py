#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Fraud Detection API - Test Suite
=====================================

Comprehensive test suite for the FastAPI fraud detection service.
Tests API endpoints, model integration, and data validation.

Author: AEGIS Fraud Detection Team
Sprint: 3.1 - API Service Creation
Date: 2025-01-21
"""

import pytest
import json
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the main application
from main import app, MODEL_CACHE, DummyFraudModel

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Test suite for health check endpoint"""
    
    def test_health_check_success(self):
        """Test successful health check when model is loaded"""
        # Ensure model is loaded for test
        MODEL_CACHE["loaded"] = True
        MODEL_CACHE["model_version"] = "test-model-v1.0"
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "3.1.0"
        assert data["model_loaded"] is True
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "dependencies" in data
    
    def test_health_check_model_not_loaded(self):
        """Test health check when model is not loaded"""
        # Simulate model not loaded
        MODEL_CACHE["loaded"] = False
        MODEL_CACHE["model_version"] = None
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestRootEndpoint:
    """Test suite for root endpoint"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "AEGIS Fraud Detection API"
        assert data["version"] == "3.1.0"
        assert "endpoints" in data
        assert "model_info" in data
        assert "usage" in data


class TestModelInfoEndpoint:
    """Test suite for model info endpoint"""
    
    def test_model_info_success(self):
        """Test model info when model is loaded"""
        # Setup model for test
        MODEL_CACHE["loaded"] = True
        MODEL_CACHE["model"] = DummyFraudModel()
        MODEL_CACHE["model_version"] = "test-model-v1.0"
        
        response = client.get("/v1/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_version"] == "test-model-v1.0"
        assert data["loaded"] is True
        assert "model_type" in data
        assert data["features_expected"] == 51
    
    def test_model_info_not_loaded(self):
        """Test model info when model is not loaded"""
        MODEL_CACHE["loaded"] = False
        
        response = client.get("/v1/model/info")
        assert response.status_code == 503


class TestPredictionEndpoint:
    """Test suite for fraud prediction endpoint"""
    
    def setup_method(self):
        """Setup for each test - ensure dummy model is loaded"""
        MODEL_CACHE["loaded"] = True
        MODEL_CACHE["model"] = DummyFraudModel()
        MODEL_CACHE["model_version"] = "test-dummy-v1.0"
    
    def test_prediction_valid_transaction(self):
        """Test fraud prediction with valid transaction data"""
        transaction_data = {
            "TransactionAmt": 150.50,
            "ProductCD": "W",
            "card1": 13553,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com"
        }
        
        response = client.post("/v1/predict", json=transaction_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        assert "transaction_id" in data
        assert "fraud_probability" in data
        assert "fraud_score" in data
        assert "risk_level" in data
        assert "prediction" in data
        assert "confidence" in data
        assert "model_version" in data
        assert "processing_time_ms" in data
        assert "risk_factors" in data
        assert "recommendation" in data
        
        # Validate data types and ranges
        assert 0 <= data["fraud_probability"] <= 1
        assert 0 <= data["fraud_score"] <= 1000
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert data["prediction"] in ["FRAUD", "LEGITIMATE"]
        assert 0 <= data["confidence"] <= 1
        assert data["processing_time_ms"] >= 0
        assert isinstance(data["risk_factors"], list)
    
    def test_prediction_minimal_data(self):
        """Test prediction with minimal required data"""
        transaction_data = {
            "TransactionAmt": 50.0,
            "ProductCD": "C"
        }
        
        response = client.post("/v1/predict", json=transaction_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "fraud_probability" in data
        assert data["prediction"] in ["FRAUD", "LEGITIMATE"]
    
    def test_prediction_high_amount(self):
        """Test prediction with high transaction amount"""
        transaction_data = {
            "TransactionAmt": 15000.0,
            "ProductCD": "W"
        }
        
        response = client.post("/v1/predict", json=transaction_data)
        assert response.status_code == 200
        
        data = response.json()
        # High amounts should typically have higher fraud probability with dummy model
        assert data["fraud_probability"] > 0.3
        assert "Large transaction amount" in str(data["risk_factors"])
    
    def test_prediction_invalid_product_cd(self):
        """Test validation error for invalid ProductCD"""
        transaction_data = {
            "TransactionAmt": 100.0,
            "ProductCD": "INVALID"
        }
        
        response = client.post("/v1/predict", json=transaction_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_negative_amount(self):
        """Test validation error for negative transaction amount"""
        transaction_data = {
            "TransactionAmt": -50.0,
            "ProductCD": "W"
        }
        
        response = client.post("/v1/predict", json=transaction_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_missing_required_fields(self):
        """Test validation error for missing required fields"""
        transaction_data = {
            "card1": 12345
            # Missing TransactionAmt and ProductCD
        }
        
        response = client.post("/v1/predict", json=transaction_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_model_not_loaded(self):
        """Test prediction when model is not loaded"""
        MODEL_CACHE["loaded"] = False
        
        transaction_data = {
            "TransactionAmt": 100.0,
            "ProductCD": "W"
        }
        
        response = client.post("/v1/predict", json=transaction_data)
        assert response.status_code == 503  # Service unavailable
    
    def test_prediction_extra_fields_forbidden(self):
        """Test that extra fields are rejected"""
        transaction_data = {
            "TransactionAmt": 100.0,
            "ProductCD": "W",
            "invalid_field": "should_be_rejected"
        }
        
        response = client.post("/v1/predict", json=transaction_data)
        assert response.status_code == 422  # Validation error


class TestPydanticModels:
    """Test suite for Pydantic model validation"""
    
    def test_transaction_input_validation(self):
        """Test TransactionInput model validation"""
        from main import TransactionInput
        
        # Valid data
        valid_data = {
            "TransactionAmt": 100.0,
            "ProductCD": "w"  # Should be converted to uppercase
        }
        
        transaction = TransactionInput(**valid_data)
        assert transaction.ProductCD == "W"
        assert transaction.TransactionAmt == 100.0
    
    def test_card_validation(self):
        """Test card field validation"""
        from main import TransactionInput
        
        # Valid card data
        data = {
            "TransactionAmt": 100.0,
            "ProductCD": "W",
            "card4": "VISA",  # Should be converted to lowercase
            "card6": "CREDIT"  # Should be converted to lowercase
        }
        
        transaction = TransactionInput(**data)
        assert transaction.card4 == "visa"
        assert transaction.card6 == "credit"
    
    def test_fraud_prediction_validation(self):
        """Test FraudPrediction model validation"""
        from main import FraudPrediction
        
        valid_data = {
            "transaction_id": "test-123",
            "fraud_probability": 0.3,
            "fraud_score": 300,
            "risk_level": "MEDIUM",
            "prediction": "LEGITIMATE",
            "confidence": 0.6,
            "model_version": "test-v1.0",
            "processing_time_ms": 50.0,
            "recommendation": "Monitor transaction"
        }
        
        prediction = FraudPrediction(**valid_data)
        assert prediction.risk_level == "MEDIUM"
        assert prediction.prediction == "LEGITIMATE"


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_calculate_risk_assessment(self):
        """Test risk assessment calculation"""
        from main import calculate_risk_assessment
        
        # Test critical risk
        result = calculate_risk_assessment(0.9, 5000)
        assert result["risk_level"] == "CRITICAL"
        assert "Very high fraud probability" in result["risk_factors"]
        assert "BLOCK" in result["recommendation"]
        
        # Test low risk
        result = calculate_risk_assessment(0.1, 50)
        assert result["risk_level"] == "LOW"
        assert "APPROVE" in result["recommendation"]
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        from main import calculate_confidence
        
        # Test high confidence (far from 0.5)
        assert calculate_confidence(0.9) == 0.8
        assert calculate_confidence(0.1) == 0.8
        
        # Test low confidence (close to 0.5)
        assert calculate_confidence(0.5) == 0.0
        assert calculate_confidence(0.6) == 0.2
    
    def test_preprocess_transaction(self):
        """Test transaction preprocessing"""
        from main import preprocess_transaction, TransactionInput
        
        transaction = TransactionInput(
            TransactionAmt=100.0,
            ProductCD="W",
            card1=12345
        )
        
        df = preprocess_transaction(transaction)
        
        # Check DataFrame structure
        assert len(df) == 1  # Single row
        assert "TransactionAmt" in df.columns
        assert "ProductCD" in df.columns
        assert df.iloc[0]["TransactionAmt"] == 100.0


class TestDummyModel:
    """Test suite for dummy fraud model"""
    
    def test_dummy_model_prediction(self):
        """Test dummy model predictions"""
        model = DummyFraudModel()
        
        # Test with different amounts
        import pandas as pd
        
        # Low amount
        low_amount_data = pd.DataFrame({"TransactionAmt": [100.0]})
        probs = model.predict_proba(low_amount_data)
        assert probs[0][1] == 0.1  # Low fraud probability
        
        # High amount
        high_amount_data = pd.DataFrame({"TransactionAmt": [5000.0]})
        probs = model.predict_proba(high_amount_data)
        assert probs[0][1] == 0.5  # Higher fraud probability


class TestIntegration:
    """Integration tests for the complete API workflow"""
    
    def test_complete_workflow(self):
        """Test complete fraud detection workflow"""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Get model info (if model loaded)
        if health_response.json()["model_loaded"]:
            model_response = client.get("/v1/model/info")
            assert model_response.status_code == 200
        
        # 3. Make prediction
        MODEL_CACHE["loaded"] = True
        MODEL_CACHE["model"] = DummyFraudModel()
        
        transaction_data = {
            "TransactionAmt": 250.0,
            "ProductCD": "W",
            "card4": "visa",
            "card6": "debit"
        }
        
        prediction_response = client.post("/v1/predict", json=transaction_data)
        assert prediction_response.status_code == 200
        
        # Validate prediction response
        prediction_data = prediction_response.json()
        assert "transaction_id" in prediction_data
        assert 0 <= prediction_data["fraud_probability"] <= 1
        assert prediction_data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    def test_error_handling(self):
        """Test API error handling"""
        # Test with model not loaded
        MODEL_CACHE["loaded"] = False
        
        transaction_data = {
            "TransactionAmt": 100.0,
            "ProductCD": "W"
        }
        
        response = client.post("/v1/predict", json=transaction_data)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


# Performance tests
class TestPerformance:
    """Performance tests for the API"""
    
    def test_prediction_response_time(self):
        """Test prediction response time is reasonable"""
        MODEL_CACHE["loaded"] = True
        MODEL_CACHE["model"] = DummyFraudModel()
        
        transaction_data = {
            "TransactionAmt": 100.0,
            "ProductCD": "W"
        }
        
        start_time = time.time()
        response = client.post("/v1/predict", json=transaction_data)
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        assert response.status_code == 200
        assert response_time < 1000  # Should be under 1 second
        
        # Check reported processing time
        data = response.json()
        assert data["processing_time_ms"] < 500  # Should be under 500ms


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])

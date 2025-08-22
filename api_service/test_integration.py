"""
Aegis Fraud Detection API - Integration Tests (Sprint 3.2)
Comprehensive integration testing for containerized API service
"""

import pytest
import requests
import time
import json
import random
from typing import Dict, Any


class TestAPIIntegration:
    """Integration tests for the Aegis Fraud Detection API service"""
    
    BASE_URL = "http://localhost:8000"
    
    @classmethod
    def setup_class(cls):
        """Setup for the test class - wait for API to be ready"""
        print("Waiting for API service to be ready...")
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{cls.BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    print(f"API ready after {i+1} attempts")
                    break
            except requests.ConnectionError:
                if i == max_retries - 1:
                    pytest.fail("API service not available after maximum retries")
                time.sleep(2)
    
    def test_health_endpoint(self):
        """Test /health endpoint returns healthy status"""
        response = requests.get(f"{self.BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert data["service"] == "aegis-fraud-detector"
    
    def test_predict_endpoint_valid_transaction(self):
        """Test /v1/predict with valid transaction data"""
        # Valid transaction payload (based on Sprint 3.1 model)
        valid_transaction = {
            "type": 1,
            "amount": 150.75,
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4849.25,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 1150.75,
            "step": 24,
            "nameDest_C": 1,
            "nameDest_M": 0,
            "nameOrig_C": 1,
            "nameOrig_M": 0,
            "amount_log": 5.0162,
            "oldbalanceOrg_log": 8.5172,
            "newbalanceOrig_log": 8.4866,
            "oldbalanceDest_log": 6.9078,
            "newbalanceDest_log": 7.0488,
            "amount_oldbalanceOrg_ratio": 0.0302,
            "amount_oldbalanceDest_ratio": 0.1508,
            "balance_change_orig": -150.75,
            "balance_change_dest": 150.75,
            "orig_balance_after_transaction": 4849.25,
            "dest_balance_after_transaction": 1150.75,
            "is_amount_rounded": 0,
            "hour": 0,
            "is_weekend": 0,
            "amount_z_score": -0.2156,
            "orig_balance_z_score": 0.3412,
            "dest_balance_z_score": -0.1234,
            "transaction_frequency": 1,
            "avg_transaction_amount": 150.75,
            "balance_velocity_orig": -150.75,
            "balance_velocity_dest": 150.75,
            "is_high_risk_amount": 0,
            "amount_percentile": 25.0,
            "orig_balance_percentile": 60.0,
            "dest_balance_percentile": 40.0,
            "cross_border_indicator": 0,
            "merchant_category": 1,
            "payment_method": 1,
            "device_fingerprint": 0,
            "ip_risk_score": 0.1,
            "user_tenure": 365,
            "previous_failures": 0,
            "account_verification_status": 1,
            "transaction_context_score": 0.8,
            "behavioral_score": 0.7,
            "network_risk_score": 0.2,
            "temporal_risk_score": 0.1,
            "amount_deviation_score": 0.3,
            "frequency_score": 0.4,
            "pattern_anomaly_score": 0.2,
            "composite_risk_score": 0.35
        }
        
        response = requests.post(
            f"{self.BASE_URL}/v1/predict",
            json=valid_transaction,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert "recommendation" in data
        assert "request_id" in data
        assert "timestamp" in data
        
        # Validate data types
        assert isinstance(data["is_fraud"], bool)
        assert isinstance(data["fraud_probability"], float)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["risk_level"], str)
        assert isinstance(data["recommendation"], str)
        
        # Validate value ranges
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["risk_level"] in ["low", "medium", "high", "critical"]
    
    def test_predict_endpoint_invalid_data(self):
        """Test /v1/predict with invalid data"""
        invalid_transaction = {
            "amount": 100.0,
            # Missing many required fields
        }
        
        response = requests.post(
            f"{self.BASE_URL}/v1/predict",
            json=invalid_transaction,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    def test_api_performance(self):
        """Test API response time performance"""
        valid_transaction = {
            "type": 1,
            "amount": 100.0,
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 900.0,
            "oldbalanceDest": 500.0,
            "newbalanceDest": 600.0,
            "step": 1,
            "nameDest_C": 1,
            "nameDest_M": 0,
            "nameOrig_C": 1,
            "nameOrig_M": 0,
            "amount_log": 4.6052,
            "oldbalanceOrg_log": 6.9078,
            "newbalanceOrig_log": 6.8024,
            "oldbalanceDest_log": 6.2146,
            "newbalanceDest_log": 6.3969,
            "amount_oldbalanceOrg_ratio": 0.1,
            "amount_oldbalanceDest_ratio": 0.2,
            "balance_change_orig": -100.0,
            "balance_change_dest": 100.0,
            "orig_balance_after_transaction": 900.0,
            "dest_balance_after_transaction": 600.0,
            "is_amount_rounded": 1,
            "hour": 14,
            "is_weekend": 0,
            "amount_z_score": 0.0,
            "orig_balance_z_score": 0.0,
            "dest_balance_z_score": 0.0,
            "transaction_frequency": 1,
            "avg_transaction_amount": 100.0,
            "balance_velocity_orig": -100.0,
            "balance_velocity_dest": 100.0,
            "is_high_risk_amount": 0,
            "amount_percentile": 20.0,
            "orig_balance_percentile": 50.0,
            "dest_balance_percentile": 40.0,
            "cross_border_indicator": 0,
            "merchant_category": 1,
            "payment_method": 1,
            "device_fingerprint": 0,
            "ip_risk_score": 0.1,
            "user_tenure": 365,
            "previous_failures": 0,
            "account_verification_status": 1,
            "transaction_context_score": 0.8,
            "behavioral_score": 0.7,
            "network_risk_score": 0.2,
            "temporal_risk_score": 0.1,
            "amount_deviation_score": 0.2,
            "frequency_score": 0.3,
            "pattern_anomaly_score": 0.1,
            "composite_risk_score": 0.25
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.BASE_URL}/v1/predict",
            json=valid_transaction,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Response should be under 1 second
        print(f"API response time: {response_time:.3f} seconds")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])

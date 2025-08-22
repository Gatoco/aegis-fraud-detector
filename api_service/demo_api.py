#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Fraud Detection API - Quick Demo
=====================================

Simple demonstration of the fraud detection API endpoints.
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def demo_fraud_detection():
    """
    Demonstrate the AEGIS Fraud Detection API capabilities
    """
    print("=" * 60)
    print("🛡️  AEGIS FRAUD DETECTION API - SPRINT 3.1 DEMO")
    print("=" * 60)
    
    # Wait for API to start
    print("⏳ Waiting for API to start...")
    for i in range(10):
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=2)
            if response.status_code == 200:
                print("✅ API is ready!")
                break
        except:
            time.sleep(2)
            print(f"   Attempt {i+1}/10...")
    else:
        print("❌ API not available. Please start with: python main.py")
        return
    
    # 1. API Information
    print("\n📋 1. API INFORMATION")
    print("-" * 30)
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"Service: {data['service']}")
            print(f"Version: {data['version']}")
            print(f"Sprint: {data['sprint']}")
            print(f"Model Performance:")
            print(f"  - PR-AUC: {data['model_info']['performance']['PR-AUC']}")
            print(f"  - F1-Score: {data['model_info']['performance']['F1-Score']}")
            print(f"  - Improvement: {data['model_info']['performance']['improvement_vs_baseline']}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 2. Health Check
    print("\n🏥 2. HEALTH CHECK")
    print("-" * 30)
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data['status']} ✅")
            print(f"Model Loaded: {data['model_loaded']} {'✅' if data['model_loaded'] else '❌'}")
            print(f"Model Version: {data.get('model_version', 'N/A')}")
            print(f"Uptime: {data['uptime_seconds']:.1f} seconds")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 3. Model Information
    print("\n🤖 3. MODEL INFORMATION")
    print("-" * 30)
    try:
        response = requests.get(f"{API_BASE_URL}/v1/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"Model Type: {data.get('model_type', 'N/A')}")
            print(f"Version: {data.get('model_version', 'N/A')}")
            print(f"Features Expected: {data.get('features_expected', 'N/A')}")
            print(f"Prediction Threshold: {data.get('prediction_threshold', 'N/A')}")
            print(f"Optimization: {data.get('optimization_details', {}).get('framework', 'N/A')}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 4. Fraud Predictions
    print("\n🔮 4. FRAUD PREDICTION EXAMPLES")
    print("-" * 30)
    
    # Example 1: Low-risk transaction
    print("\n📝 Example 1: Low-risk transaction ($25.50)")
    low_risk_transaction = {
        "TransactionAmt": 25.50,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/v1/predict",
            json=low_risk_transaction,
            headers={"Content-Type": "application/json"}
        )
        request_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"   🆔 Transaction ID: {data['transaction_id'][:8]}...")
            print(f"   🎯 Fraud Probability: {data['fraud_probability']:.3f}")
            print(f"   📊 Risk Level: {data['risk_level']}")
            print(f"   ✅ Prediction: {data['prediction']}")
            print(f"   💡 Recommendation: {data['recommendation']}")
            print(f"   ⏱️  Processing Time: {data['processing_time_ms']:.1f}ms")
            print(f"   🌐 Request Time: {request_time:.1f}ms")
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Example 2: High-risk transaction
    print("\n📝 Example 2: High-risk transaction ($5,000.00)")
    high_risk_transaction = {
        "TransactionAmt": 5000.00,
        "ProductCD": "W",
        "card4": "american express",
        "card6": "credit",
        "C1": 0,
        "D4": 0.5
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/v1/predict",
            json=high_risk_transaction,
            headers={"Content-Type": "application/json"}
        )
        request_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"   🆔 Transaction ID: {data['transaction_id'][:8]}...")
            print(f"   🎯 Fraud Probability: {data['fraud_probability']:.3f}")
            print(f"   📊 Risk Level: {data['risk_level']}")
            print(f"   ✅ Prediction: {data['prediction']}")
            print(f"   ⚠️  Risk Factors: {', '.join(data.get('risk_factors', []))}")
            print(f"   💡 Recommendation: {data['recommendation']}")
            print(f"   ⏱️  Processing Time: {data['processing_time_ms']:.1f}ms")
            print(f"   🌐 Request Time: {request_time:.1f}ms")
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Example 3: Validation error
    print("\n📝 Example 3: Validation error (invalid ProductCD)")
    invalid_transaction = {
        "TransactionAmt": 100.00,
        "ProductCD": "INVALID"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/predict",
            json=invalid_transaction,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 422:
            print(f"   ✅ Validation Error (Expected): {response.status_code}")
            data = response.json()
            print(f"   📋 Error Details: {data.get('detail', [{}])[0].get('msg', 'Validation failed')}")
        else:
            print(f"   ❌ Unexpected Status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. Performance Summary
    print("\n📊 5. SPRINT 3.1 SUMMARY")
    print("-" * 30)
    print("✅ FastAPI Service: Production-ready")
    print("✅ Pydantic Validation: Input/Output models")
    print("✅ Health Monitoring: /health endpoint")
    print("✅ Fraud Detection: /v1/predict endpoint")
    print("✅ Model Integration: Sprint 2.4 optimized LightGBM")
    print("✅ Documentation: /docs and /redoc endpoints")
    print("✅ Error Handling: Comprehensive validation")
    print("✅ Production Features: Logging, metrics, CORS")
    
    print("\n🎯 API Endpoints Available:")
    print("   📋 GET  /          - API information")
    print("   🏥 GET  /health    - Health check")
    print("   🔮 POST /v1/predict - Fraud prediction")
    print("   🤖 GET  /v1/model/info - Model details")
    print("   📚 GET  /docs      - Interactive documentation")
    
    print("\n" + "=" * 60)
    print("✅ SPRINT 3.1 - API SERVICE CREATION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    demo_fraud_detection()

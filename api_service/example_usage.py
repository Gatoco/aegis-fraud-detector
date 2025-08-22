#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Fraud Detection API - Example Usage
========================================

Demonstration script showing how to interact with the fraud detection API.
Includes examples for health checks, predictions, and error handling.

Author: AEGIS Fraud Detection Team
Sprint: 3.1 - API Service Creation
Date: 2025-01-21
"""

import requests
import json
import time
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30


def check_api_health() -> Dict[str, Any]:
    """
    Check API health status
    
    Returns:
        Health check response data
    """
    try:
        print("🔍 Checking API health...")
        response = requests.get(f"{API_BASE_URL}/health", timeout=API_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ API Status: {data['status']}")
        print(f"📊 Model Loaded: {data['model_loaded']}")
        print(f"🏷️  Model Version: {data.get('model_version', 'N/A')}")
        print(f"⏱️  Uptime: {data['uptime_seconds']:.1f} seconds")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return {}


def get_model_info() -> Dict[str, Any]:
    """
    Get detailed model information
    
    Returns:
        Model information response data
    """
    try:
        print("\n🤖 Getting model information...")
        response = requests.get(f"{API_BASE_URL}/v1/model/info", timeout=API_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        print(f"📋 Model Type: {data.get('model_type', 'N/A')}")
        print(f"🏷️  Version: {data.get('model_version', 'N/A')}")
        print(f"🔢 Features Expected: {data.get('features_expected', 'N/A')}")
        print(f"🎯 Threshold: {data.get('prediction_threshold', 'N/A')}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info request failed: {e}")
        return {}


def predict_fraud(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a fraud prediction for a transaction
    
    Args:
        transaction_data: Transaction data dictionary
        
    Returns:
        Prediction response data
    """
    try:
        print(f"\n🔮 Predicting fraud for transaction: ${transaction_data.get('TransactionAmt', 'N/A')}")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/v1/predict",
            json=transaction_data,
            timeout=API_TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        request_time = (time.time() - start_time) * 1000
        
        data = response.json()
        
        # Display results
        print(f"🆔 Transaction ID: {data['transaction_id'][:8]}...")
        print(f"🎯 Fraud Probability: {data['fraud_probability']:.3f}")
        print(f"📊 Fraud Score: {data['fraud_score']}/1000")
        print(f"⚠️  Risk Level: {data['risk_level']}")
        print(f"✅ Prediction: {data['prediction']}")
        print(f"🔒 Confidence: {data['confidence']:.3f}")
        print(f"⏱️  Processing Time: {data['processing_time_ms']:.1f}ms")
        print(f"🌐 Request Time: {request_time:.1f}ms")
        
        if data.get('risk_factors'):
            print(f"⚠️  Risk Factors: {', '.join(data['risk_factors'])}")
        
        print(f"💡 Recommendation: {data['recommendation']}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                print(f"📋 Error Details: {error_data.get('detail', 'No details available')}")
            except:
                print(f"📋 Error Response: {e.response.text}")
        return {}


def demonstrate_api():
    """
    Demonstrate various API capabilities with example transactions
    """
    print("=" * 60)
    print("AEGIS Fraud Detection API - Demonstration")
    print("=" * 60)
    
    # 1. Health Check
    health_data = check_api_health()
    if not health_data or health_data.get('status') != 'healthy':
        print("\n❌ API is not healthy. Please check the service.")
        return
    
    # 2. Model Information
    model_info = get_model_info()
    
    # 3. Example Transactions
    print("\n" + "=" * 60)
    print("EXAMPLE TRANSACTIONS")
    print("=" * 60)
    
    # Example 1: Low-risk transaction
    print("\n📝 Example 1: Low-risk transaction")
    print("-" * 40)
    low_risk_transaction = {
        "TransactionAmt": 25.50,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "C1": 1,
        "C2": 1
    }
    predict_fraud(low_risk_transaction)
    
    # Example 2: Medium-risk transaction
    print("\n📝 Example 2: Medium-risk transaction")
    print("-" * 40)
    medium_risk_transaction = {
        "TransactionAmt": 850.00,
        "ProductCD": "C",
        "card4": "mastercard",
        "card6": "credit",
        "P_emaildomain": "hotmail.com",
        "C1": 3,
        "C4": 2,
        "D1": 5.0
    }
    predict_fraud(medium_risk_transaction)
    
    # Example 3: High-risk transaction
    print("\n📝 Example 3: High-risk transaction")
    print("-" * 40)
    high_risk_transaction = {
        "TransactionAmt": 5000.00,
        "ProductCD": "W",
        "card4": "american express",
        "card6": "credit",
        "dist1": 1000.0,
        "C1": 0,
        "C14": 5,
        "D4": 0.5,
        "V1": 0.5
    }
    predict_fraud(high_risk_transaction)
    
    # Example 4: Minimal data transaction
    print("\n📝 Example 4: Minimal data (required fields only)")
    print("-" * 40)
    minimal_transaction = {
        "TransactionAmt": 100.00,
        "ProductCD": "R"
    }
    predict_fraud(minimal_transaction)
    
    # 4. Demonstrate error handling
    print("\n" + "=" * 60)
    print("ERROR HANDLING EXAMPLES")
    print("=" * 60)
    
    # Example 5: Invalid data
    print("\n📝 Example 5: Invalid ProductCD")
    print("-" * 40)
    invalid_transaction = {
        "TransactionAmt": 100.00,
        "ProductCD": "INVALID"
    }
    predict_fraud(invalid_transaction)
    
    # Example 6: Missing required fields
    print("\n📝 Example 6: Missing required fields")
    print("-" * 40)
    incomplete_transaction = {
        "card1": 12345
        # Missing TransactionAmt and ProductCD
    }
    predict_fraud(incomplete_transaction)
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


def performance_test(num_requests: int = 10):
    """
    Simple performance test
    
    Args:
        num_requests: Number of requests to make
    """
    print(f"\n🚀 Performance Test: {num_requests} requests")
    print("-" * 40)
    
    test_transaction = {
        "TransactionAmt": 150.00,
        "ProductCD": "W",
        "card4": "visa"
    }
    
    times = []
    successes = 0
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/v1/predict",
                json=test_transaction,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            request_time = (time.time() - start_time) * 1000
            times.append(request_time)
            successes += 1
            
            if (i + 1) % 5 == 0:
                print(f"✅ Completed {i + 1}/{num_requests} requests...")
                
        except Exception as e:
            print(f"❌ Request {i + 1} failed: {e}")
    
    # Calculate statistics
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Performance Results:")
        print(f"   ✅ Success Rate: {successes}/{num_requests} ({100 * successes / num_requests:.1f}%)")
        print(f"   ⏱️  Average Time: {avg_time:.1f}ms")
        print(f"   🚀 Fastest Time: {min_time:.1f}ms")
        print(f"   🐌 Slowest Time: {max_time:.1f}ms")
        print(f"   🎯 Throughput: {1000 / avg_time:.1f} requests/second")


def batch_prediction_example():
    """
    Example of processing multiple transactions in sequence
    """
    print("\n📦 Batch Processing Example")
    print("-" * 40)
    
    transactions = [
        {"TransactionAmt": 50.00, "ProductCD": "W"},
        {"TransactionAmt": 250.00, "ProductCD": "C"},
        {"TransactionAmt": 1000.00, "ProductCD": "R"},
        {"TransactionAmt": 2500.00, "ProductCD": "H"},
        {"TransactionAmt": 100.00, "ProductCD": "S"}
    ]
    
    results = []
    total_time = 0
    
    for i, transaction in enumerate(transactions, 1):
        print(f"\n📝 Processing transaction {i}/{len(transactions)}")
        start_time = time.time()
        result = predict_fraud(transaction)
        processing_time = time.time() - start_time
        total_time += processing_time
        
        if result:
            results.append({
                "amount": transaction["TransactionAmt"],
                "product": transaction["ProductCD"],
                "fraud_probability": result.get("fraud_probability", 0),
                "risk_level": result.get("risk_level", "UNKNOWN"),
                "prediction": result.get("prediction", "UNKNOWN")
            })
    
    # Summary
    print(f"\n📊 Batch Processing Summary:")
    print(f"   📦 Transactions Processed: {len(results)}")
    print(f"   ⏱️  Total Time: {total_time:.2f} seconds")
    print(f"   📈 Average Time per Transaction: {total_time / len(transactions):.2f} seconds")
    
    fraud_count = sum(1 for r in results if r["prediction"] == "FRAUD")
    print(f"   🚨 Fraud Detected: {fraud_count}/{len(results)} transactions")
    
    if results:
        avg_fraud_prob = sum(r["fraud_probability"] for r in results) / len(results)
        print(f"   🎯 Average Fraud Probability: {avg_fraud_prob:.3f}")


if __name__ == "__main__":
    print("Starting AEGIS Fraud Detection API demonstration...")
    print("Make sure the API is running on http://localhost:8000")
    print()
    
    try:
        # Main demonstration
        demonstrate_api()
        
        # Performance test
        performance_test(10)
        
        # Batch processing example
        batch_prediction_example()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demonstration failed: {e}")
    
    print("\n🎯 Demonstration complete!")
    print("📚 For more information, visit: http://localhost:8000/docs")

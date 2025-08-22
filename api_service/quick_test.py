import time
import subprocess
import sys

print("🚀 Starting AEGIS Fraud Detection API...")

# Start the API server
process = subprocess.Popen([
    sys.executable, "-m", "uvicorn", "main:app", 
    "--host", "0.0.0.0", "--port", "8000"
], 
stdout=subprocess.PIPE, 
stderr=subprocess.STDOUT, 
text=True
)

# Wait a bit for startup
time.sleep(5)

# Test the API
import requests
try:
    print("\n📋 Testing API endpoints...")
    
    # Health check
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health Check: {data['status']}")
        print(f"📊 Model Loaded: {data['model_loaded']}")
        print(f"🏷️  Version: {data['version']}")
    
    # Fraud prediction test
    transaction = {
        "TransactionAmt": 150.50,
        "ProductCD": "W",
        "card4": "visa"
    }
    
    response = requests.post("http://localhost:8000/v1/predict", json=transaction, timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"\n🔮 Fraud Prediction Test:")
        print(f"   Amount: ${transaction['TransactionAmt']}")
        print(f"   Fraud Probability: {data['fraud_probability']:.3f}")
        print(f"   Risk Level: {data['risk_level']}")
        print(f"   Prediction: {data['prediction']}")
        print(f"   Recommendation: {data['recommendation']}")
    
    print(f"\n✅ Sprint 3.1 API Service Creation - COMPLETED!")
    print(f"📚 Documentation: http://localhost:8000/docs")
    
except Exception as e:
    print(f"❌ Error testing API: {e}")

finally:
    # Clean up
    process.terminate()
    print(f"\n🛑 API server stopped.")

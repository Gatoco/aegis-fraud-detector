#!/usr/bin/env python3
"""
Sprint 3.2 Automation Script
Comprehensive validation of containerized API service
"""

import subprocess
import time
import requests
import sys
import json
from pathlib import Path


class Sprint32Validator:
    """Automated validation for Sprint 3.2 deliverables"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.project_root = Path(__file__).parent.parent
        self.api_dir = Path(__file__).parent
        
    def log(self, message, level="INFO"):
        """Log with timestamp"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, command, cwd=None, check=True):
        """Run shell command with error handling"""
        try:
            if cwd is None:
                cwd = self.project_root
            
            self.log(f"Running: {command}")
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=check
            )
            
            if result.stdout:
                self.log(f"Output: {result.stdout.strip()}")
            if result.stderr and check:
                self.log(f"Error: {result.stderr.strip()}", "WARNING")
            
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e}", "ERROR")
            if e.stdout:
                self.log(f"STDOUT: {e.stdout}", "ERROR")
            if e.stderr:
                self.log(f"STDERR: {e.stderr}", "ERROR")
            raise
    
    def check_docker_available(self):
        """Check if Docker is available"""
        try:
            result = self.run_command("docker --version", check=False)
            if result.returncode == 0:
                self.log("✅ Docker is available")
                return True
            else:
                self.log("❌ Docker is not available", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Docker check failed: {e}", "ERROR")
            return False
    
    def check_docker_compose_available(self):
        """Check if Docker Compose is available"""
        try:
            result = self.run_command("docker-compose --version", check=False)
            if result.returncode == 0:
                self.log("✅ Docker Compose is available")
                return True
            else:
                # Try docker compose (new syntax)
                result = self.run_command("docker compose version", check=False)
                if result.returncode == 0:
                    self.log("✅ Docker Compose (new syntax) is available")
                    return True
                else:
                    self.log("❌ Docker Compose is not available", "ERROR")
                    return False
        except Exception as e:
            self.log(f"❌ Docker Compose check failed: {e}", "ERROR")
            return False
    
    def build_api_image(self):
        """Build the API Docker image"""
        self.log("Building API Docker image...")
        try:
            # Build using the specific Dockerfile for API
            result = self.run_command(
                "docker build -f api_service/Dockerfile.api -t aegis-fraud-api:latest .",
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                self.log("✅ API Docker image built successfully")
                return True
            else:
                self.log("❌ Failed to build API Docker image", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Docker build failed: {e}", "ERROR")
            return False
    
    def start_api_container(self):
        """Start the API container"""
        self.log("Starting API container...")
        try:
            # Stop any existing container
            self.run_command("docker stop aegis-fraud-api", check=False)
            self.run_command("docker rm aegis-fraud-api", check=False)
            
            # Start new container
            result = self.run_command(
                "docker run -d -p 8000:8000 --name aegis-fraud-api aegis-fraud-api:latest",
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                self.log("✅ API container started successfully")
                return True
            else:
                self.log("❌ Failed to start API container", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Container start failed: {e}", "ERROR")
            return False
    
    def wait_for_api_ready(self, max_wait=60):
        """Wait for API to be ready"""
        self.log("Waiting for API to be ready...")
        
        for i in range(max_wait):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    self.log(f"✅ API ready after {i+1} seconds")
                    return True
            except requests.ConnectionError:
                pass
            
            time.sleep(1)
        
        self.log("❌ API did not become ready within timeout", "ERROR")
        return False
    
    def test_health_endpoint(self):
        """Test the health endpoint"""
        self.log("Testing /health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log("✅ Health endpoint working correctly")
                    return True
                else:
                    self.log(f"❌ Health endpoint returned unexpected status: {data}", "ERROR")
                    return False
            else:
                self.log(f"❌ Health endpoint returned status code: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Health endpoint test failed: {e}", "ERROR")
            return False
    
    def test_predict_endpoint(self):
        """Test the predict endpoint with sample data"""
        self.log("Testing /v1/predict endpoint...")
        
        # Sample transaction data
        transaction = {
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
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/predict",
                json=transaction,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["is_fraud", "fraud_probability", "confidence", "risk_level", "recommendation"]
                
                if all(field in data for field in required_fields):
                    self.log("✅ Predict endpoint working correctly")
                    self.log(f"Sample prediction: fraud={data['is_fraud']}, prob={data['fraud_probability']:.3f}")
                    return True
                else:
                    self.log(f"❌ Predict endpoint missing required fields: {data}", "ERROR")
                    return False
            else:
                self.log(f"❌ Predict endpoint returned status code: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Predict endpoint test failed: {e}", "ERROR")
            return False
    
    def run_integration_tests(self):
        """Run the full integration test suite"""
        self.log("Running integration test suite...")
        try:
            result = self.run_command(
                "python -m pytest test_integration.py -v",
                cwd=self.api_dir,
                check=False
            )
            
            if result.returncode == 0:
                self.log("✅ All integration tests passed")
                return True
            else:
                self.log("❌ Some integration tests failed", "WARNING")
                return False
        except Exception as e:
            self.log(f"❌ Integration tests failed: {e}", "ERROR")
            return False
    
    def check_container_logs(self):
        """Check container logs for errors"""
        self.log("Checking container logs...")
        try:
            result = self.run_command("docker logs aegis-fraud-api", check=False)
            
            if "error" in result.stderr.lower() or "error" in result.stdout.lower():
                self.log("⚠️ Found errors in container logs", "WARNING")
            else:
                self.log("✅ No errors found in container logs")
            
            return True
        except Exception as e:
            self.log(f"❌ Failed to check container logs: {e}", "ERROR")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        self.log("Cleaning up...")
        try:
            self.run_command("docker stop aegis-fraud-api", check=False)
            self.run_command("docker rm aegis-fraud-api", check=False)
            self.log("✅ Cleanup completed")
        except Exception as e:
            self.log(f"❌ Cleanup failed: {e}", "ERROR")
    
    def validate_sprint_32(self):
        """Main validation method for Sprint 3.2"""
        self.log("🚀 Starting Sprint 3.2 validation...")
        
        # Track success/failure
        results = {}
        
        # Check prerequisites
        results["docker"] = self.check_docker_available()
        results["docker_compose"] = self.check_docker_compose_available()
        
        if not all([results["docker"], results["docker_compose"]]):
            self.log("❌ Prerequisites not met. Cannot continue.", "ERROR")
            return False
        
        try:
            # Build and deploy
            results["build"] = self.build_api_image()
            if not results["build"]:
                return False
            
            results["start"] = self.start_api_container()
            if not results["start"]:
                return False
            
            results["ready"] = self.wait_for_api_ready()
            if not results["ready"]:
                return False
            
            # Test endpoints
            results["health"] = self.test_health_endpoint()
            results["predict"] = self.test_predict_endpoint()
            
            # Run comprehensive tests
            results["integration"] = self.run_integration_tests()
            
            # Check logs
            results["logs"] = self.check_container_logs()
            
            # Summary
            passed = sum(1 for v in results.values() if v)
            total = len(results)
            
            self.log(f"📊 Sprint 3.2 Results: {passed}/{total} checks passed")
            
            if passed == total:
                self.log("🎉 Sprint 3.2 validation SUCCESSFUL!")
                self.log("✅ API service is properly containerized and functional")
                return True
            else:
                self.log("⚠️ Sprint 3.2 validation completed with issues")
                for check, success in results.items():
                    status = "✅" if success else "❌"
                    self.log(f"  {status} {check}")
                return False
        
        finally:
            # Always cleanup
            self.cleanup()


if __name__ == "__main__":
    validator = Sprint32Validator()
    success = validator.validate_sprint_32()
    sys.exit(0 if success else 1)

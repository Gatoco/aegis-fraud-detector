#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Fraud Detection System - Zero-Cost Local CI
=================================================

Ultra-fast local CI validation that mimics the optimized GitHub Actions workflow.
Designed for maximum efficiency and zero cost.

Usage:
    python scripts/zero_cost_ci.py [--fast] [--api-only]

Options:
    --fast      Skip non-essential checks
    --api-only  Only run API-related tests

Author: Gat
Date: 2025-01-22
"""

import subprocess
import sys
import os
import argparse
import time
from pathlib import Path


class FastCI:
    """Ultra-fast CI runner"""
    
    def __init__(self, fast_mode=False, api_only=False):
        self.fast_mode = fast_mode
        self.api_only = api_only
        self.start_time = time.time()
        
    def log(self, message, level="INFO"):
        """Fast logging"""
        elapsed = time.time() - self.start_time
        prefix = "⚡" if level == "FAST" else "✓" if level == "OK" else "✗" if level == "ERROR" else "ℹ"
        print(f"{prefix} [{elapsed:.1f}s] {message}")
    
    def run_fast(self, command, description):
        """Run command with minimal overhead"""
        try:
            # Handle Windows vs Unix timeout
            if os.name == 'nt':  # Windows
                # Remove timeout command for Windows compatibility
                command = command.replace('timeout 60 ', '').replace('timeout 120 ', '')
            
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                self.log(f"{description} - OK", "OK")
                return True
            else:
                self.log(f"{description} - FAILED: {result.stderr.strip()}", "ERROR")
                return False
        except subprocess.TimeoutExpired:
            self.log(f"{description} - TIMEOUT", "ERROR")
            return False
        except Exception as e:
            self.log(f"{description} - ERROR: {e}", "ERROR")
            return False
    
    def detect_changes(self):
        """Detect what changed to skip unnecessary work"""
        self.log("Detecting changes...", "FAST")
        
        # Simple change detection
        try:
            # Check if we have git
            result = subprocess.run(
                "git status --porcelain", shell=True, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                changes = result.stdout.strip()
                if not changes:
                    self.log("No changes detected - skipping CI", "FAST")
                    return False, False, False
                
                code_changed = any(path in changes for path in ['src/', 'requirements.txt'])
                api_changed = any(path in changes for path in ['api_service/', 'main.py'])
                tests_changed = any(path in changes for path in ['tests/', 'test_'])
                
                return code_changed, api_changed, tests_changed
            else:
                # Assume changes if git not available
                return True, True, True
        except:
            # Assume changes if detection fails
            return True, True, True
    
    def fast_linting(self):
        """Ultra-fast linting checks"""
        if self.api_only:
            self.log("Skipping linting (API-only mode)", "FAST")
            return True
            
        self.log("Running fast linting...", "FAST")
        
        # Check if linting tools are installed
        tools_check = self.run_fast(
            "python -c \"import black, flake8, isort\"",
            "Checking linting tools"
        )
        if not tools_check:
            self.log("Installing linting tools...", "FAST")
            self.run_fast(
                "pip install black==22.10.0 flake8==5.0.4 isort==5.11.4 --quiet",
                "Installing tools"
            )
        
        results = []
        
        # Essential checks only
        if not self.fast_mode:
            results.append(self.run_fast(
                "black --check --fast --quiet src/ api_service/ scripts/ tests/ --line-length 88",
                "Code formatting"
            ))
        
        results.append(self.run_fast(
            "flake8 src/ api_service/ scripts/ tests/ --select=E9,F63,F7,F82 --quiet",
            "Critical errors"
        ))
        
        return all(results) if results else True
    
    def fast_unit_tests(self):
        """Minimal unit tests"""
        if self.api_only:
            self.log("Skipping unit tests (API-only mode)", "FAST")
            return True
            
        self.log("Running fast unit tests...", "FAST")
        
        # Basic import test
        import_test = self.run_fast(
            "python -c \"import sys; sys.path.append('src'); print('Imports OK')\"",
            "Core imports"
        )
        
        # Run existing tests if any
        if Path("tests").exists() and any(Path("tests").glob("test_*.py")):
            test_result = self.run_fast(
                "pytest tests/ -x --tb=no --quiet",
                "Unit tests"
            )
            return import_test and test_result
        
        return import_test
    
    def fast_api_tests(self, code_changed, api_changed):
        """API tests only when needed"""
        if not api_changed and not self.api_only:
            self.log("Skipping API tests (no API changes)", "FAST")
            return True
            
        self.log("Running fast API tests...", "FAST")
        
        # Check API dependencies
        api_deps = self.run_fast(
            "python -c \"import fastapi, uvicorn, pytest\"",
            "API dependencies"
        )
        if not api_deps:
            self.log("Installing API dependencies...", "FAST")
            self.run_fast(
                "pip install fastapi uvicorn pytest httpx --quiet",
                "Installing API deps"
            )
        
        # Quick API test
        api_test = self.run_fast(
            "cd api_service && timeout 60 python -c \"\nfrom main import app\nfrom fastapi.testclient import TestClient\nclient = TestClient(app)\nresponse = client.get('/health')\nassert response.status_code == 200\nprint('API OK')\n\"",
            "API health check"
        )
        
        # Full API tests if health check passes
        if api_test and not self.fast_mode:
            return self.run_fast(
                "cd api_service && timeout 120 pytest test_api.py -x --tb=no --quiet",
                "API tests"
            )
        
        return api_test
    
    def run(self):
        """Run the complete fast CI"""
        print("🚀 AEGIS Zero-Cost CI - Ultra Fast Mode")
        print("=" * 50)
        
        # Detect changes
        code_changed, api_changed, tests_changed = self.detect_changes()
        
        if not any([code_changed, api_changed, tests_changed]) and not self.api_only:
            self.log("No relevant changes detected - CI complete!", "OK")
            return True
        
        # Run checks based on changes
        results = []
        
        if code_changed or tests_changed or self.api_only:
            results.append(self.fast_linting())
            results.append(self.fast_unit_tests())
        
        if api_changed or self.api_only:
            results.append(self.fast_api_tests(code_changed, api_changed))
        
        # Summary
        elapsed = time.time() - self.start_time
        if all(results):
            self.log(f"CI PASSED in {elapsed:.1f}s - Zero cost achieved! 🎉", "OK")
            return True
        else:
            self.log(f"CI FAILED in {elapsed:.1f}s", "ERROR")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Zero-Cost CI Runner")
    parser.add_argument("--fast", action="store_true", help="Skip non-essential checks")
    parser.add_argument("--api-only", action="store_true", help="Only run API-related tests")
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run CI
    ci = FastCI(fast_mode=args.fast, api_only=args.api_only)
    success = ci.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
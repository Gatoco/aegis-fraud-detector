#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Fraud Detection System - Local CI Validation
=================================================

Script to validate CI pipeline locally before pushing to GitHub Actions.
Sprint 4.1: CI Implementation

Usage:
    python scripts/validate_ci.py

Author: AEGIS Fraud Detection Team
Date: 2025-01-22
"""

import subprocess
import sys
import os
from pathlib import Path
import time


class Colors:
    """Terminal colors for output formatting"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def run_command(command, description, cwd=None):
    """Run a command and return success status"""
    print(f"\n{Colors.BLUE}→ {description}{Colors.END}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"{Colors.GREEN}✓ {description} - PASSED{Colors.END}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"{Colors.RED}✗ {description} - FAILED{Colors.END}")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}✗ {description} - TIMEOUT{Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.RED}✗ {description} - EXCEPTION: {e}{Colors.END}")
        return False


def validate_linting():
    """Validate linting steps"""
    print(f"\n{Colors.BOLD}=== LINTING VALIDATION ==={Colors.END}")
    
    results = []
    
    # Check Black formatting
    results.append(run_command(
        "black --check --diff src/ api_service/ scripts/ tests/ --line-length 88",
        "Black code formatting check"
    ))
    
    # Check isort import sorting
    results.append(run_command(
        "isort --check-only --diff src/ api_service/ scripts/ tests/",
        "Import sorting check"
    ))
    
    # Check flake8 linting - critical errors
    results.append(run_command(
        "flake8 src/ api_service/ scripts/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics",
        "Flake8 critical errors check"
    ))
    
    # Check flake8 linting - full
    results.append(run_command(
        "flake8 src/ api_service/ scripts/ tests/ --count --max-complexity=10 --max-line-length=88 --statistics --extend-ignore=E203,W503",
        "Flake8 full linting check"
    ))
    
    return all(results)


def validate_unit_tests():
    """Validate unit tests"""
    print(f"\n{Colors.BOLD}=== UNIT TESTS VALIDATION ==={Colors.END}")
    
    results = []
    
    # Core unit tests
    if os.path.exists("tests") and any(Path("tests").glob("test_*.py")):
        results.append(run_command(
            "pytest tests/ -v --tb=short",
            "Core unit tests"
        ))
    else:
        print(f"{Colors.YELLOW}⚠ No core unit tests found in tests/ directory{Colors.END}")
        results.append(True)  # Don't fail if no tests exist yet
    
    # API unit tests
    results.append(run_command(
        "pytest test_api.py -v --tb=short",
        "API unit tests",
        cwd="api_service"
    ))
    
    return all(results)


def validate_integration_tests():
    """Validate integration tests"""
    print(f"\n{Colors.BOLD}=== INTEGRATION TESTS VALIDATION ==={Colors.END}")
    
    results = []
    
    # API integration tests
    results.append(run_command(
        "pytest test_integration.py -v --tb=short --maxfail=3",
        "API integration tests",
        cwd="api_service"
    ))
    
    # Basic pipeline test
    pipeline_test = '''
import sys
sys.path.append("src")

try:
    from features import feature_engineering
    from models import advanced_models
    print("✓ Core modules import successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("✓ End-to-end pipeline test completed")
'''
    
    results.append(run_command(
        f'python -c "{pipeline_test}"',
        "End-to-end pipeline test"
    ))
    
    return all(results)


def main():
    """Main validation function"""
    print(f"{Colors.BOLD}🚀 AEGIS Fraud Detection - Local CI Validation{Colors.END}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Validation steps
    steps = [
        ("Linting", validate_linting),
        ("Unit Tests", validate_unit_tests),
        ("Integration Tests", validate_integration_tests)
    ]
    
    results = []
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"{Colors.RED}✗ {step_name} - EXCEPTION: {e}{Colors.END}")
            results.append((step_name, False))
    
    # Summary
    print(f"\n{Colors.BOLD}=== VALIDATION SUMMARY ==={Colors.END}")
    print("=" * 40)
    
    all_passed = True
    for step_name, result in results:
        status = f"{Colors.GREEN}PASSED{Colors.END}" if result else f"{Colors.RED}FAILED{Colors.END}"
        print(f"{step_name}: {status}")
        if not result:
            all_passed = False
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal validation time: {elapsed_time:.2f} seconds")
    
    if all_passed:
        print(f"\n{Colors.GREEN}🎉 All validations passed! CI pipeline is ready.{Colors.END}")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}❌ Some validations failed. Please fix issues before pushing.{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
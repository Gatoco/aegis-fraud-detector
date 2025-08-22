#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Fraud Detection System - Core Module Tests
================================================

Basic unit tests for core modules and functionality.
Sprint 4.1: CI Implementation

Author: AEGIS Fraud Detection Team
Date: 2025-01-22
"""

import pytest
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCoreModules:
    """Test core module imports and basic functionality"""
    
    def test_core_imports(self):
        """Test that core modules can be imported without errors"""
        try:
            # Test feature engineering imports
            from features import feature_engineering
            assert hasattr(feature_engineering, '__name__')
            
            # Test model imports
            from models import advanced_models
            assert hasattr(advanced_models, '__name__')
            
            # Test pipeline imports
            from pipelines import train_advanced, train_baseline
            assert hasattr(train_advanced, '__name__')
            assert hasattr(train_baseline, '__name__')
            
        except ImportError as e:
            pytest.fail(f"Failed to import core modules: {e}")
    
    def test_src_structure(self):
        """Test that src directory has expected structure"""
        src_path = Path(__file__).parent.parent / "src"
        
        # Check main directories exist
        expected_dirs = [
            "data", "features", "models", "pipelines", 
            "sampling", "optimization", "visualization"
        ]
        
        for dir_name in expected_dirs:
            dir_path = src_path / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist in src/"
            
            # Check each directory has __init__.py
            init_file = dir_path / "__init__.py"
            assert init_file.exists(), f"__init__.py should exist in src/{dir_name}/"


class TestProjectStructure:
    """Test project structure and configuration"""
    
    def test_required_files_exist(self):
        """Test that required project files exist"""
        project_root = Path(__file__).parent.parent
        
        required_files = [
            "requirements.txt",
            "README.md",
            "dvc.yaml",
            "params.yaml",
            "Makefile",
            "Dockerfile"
        ]
        
        for file_name in required_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"Required file {file_name} should exist"
    
    def test_api_service_structure(self):
        """Test API service has proper structure"""
        api_path = Path(__file__).parent.parent / "api_service"
        
        required_files = [
            "main.py",
            "requirements.txt",
            "test_api.py",
            "test_integration.py"
        ]
        
        for file_name in required_files:
            file_path = api_path / file_name
            assert file_path.exists(), f"API service file {file_name} should exist"


class TestConfiguration:
    """Test configuration and parameter files"""
    
    def test_params_yaml_exists(self):
        """Test that params.yaml exists and is readable"""
        params_path = Path(__file__).parent.parent / "params.yaml"
        assert params_path.exists(), "params.yaml should exist"
        
        # Try to read it
        with open(params_path, 'r') as f:
            content = f.read()
            assert len(content) > 0, "params.yaml should not be empty"
    
    def test_dvc_yaml_exists(self):
        """Test that dvc.yaml exists and is readable"""
        dvc_path = Path(__file__).parent.parent / "dvc.yaml"
        assert dvc_path.exists(), "dvc.yaml should exist"
        
        # Try to read it
        with open(dvc_path, 'r') as f:
            content = f.read()
            assert len(content) > 0, "dvc.yaml should not be empty"


class TestUtilities:
    """Test utility functions and helpers"""
    
    def test_python_version(self):
        """Test that we're running on supported Python version"""
        major, minor = sys.version_info[:2]
        
        # Project supports Python 3.10+
        assert major == 3, "Should be running Python 3.x"
        assert minor >= 10, "Should be running Python 3.10 or higher"
    
    def test_basic_imports(self):
        """Test that basic Python packages are available"""
        import pandas as pd
        import numpy as np
        import sklearn
        
        # Basic version checks
        assert hasattr(pd, '__version__')
        assert hasattr(np, '__version__')
        assert hasattr(sklearn, '__version__')


if __name__ == "__main__":
    # Run tests when called directly
    pytest.main([__file__, "-v"])
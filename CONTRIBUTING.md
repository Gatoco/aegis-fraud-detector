# Contributing to Aegis Fraud Detection System

Thank you for your interest in contributing to Aegis! This document provides guidelines and information for contributors.

## Development Setup

1. Fork the repository and clone your fork
2. Create a virtual environment: `python -m venv .venv`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Install pre-commit hooks: `pre-commit install`

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all functions and methods
- Maintain test coverage above 85%
- Document all public APIs using Google-style docstrings

## Testing

- Write unit tests for all new functionality
- Use pytest for testing framework
- Include integration tests for pipeline components
- Test edge cases and error conditions

## Pull Request Process

1. Create a feature branch from main
2. Make your changes with appropriate tests
3. Ensure all CI checks pass
4. Submit a pull request with clear description

## Issue Reporting

Please use the GitHub issue tracker to report bugs or request features. Include:
- Clear description of the issue
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details

## Code of Conduct

This project follows the Contributor Covenant Code of Conduct. Please be respectful and inclusive in all interactions.

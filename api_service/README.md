# AEGIS Fraud Detection API - Sprint 3.1

## Overview

Production-ready FastAPI service for real-time fraud detection using the optimized LightGBM model from Sprint 2.4.

## Features

- ✅ **FastAPI Application** with automatic OpenAPI documentation
- ✅ **Pydantic Models** for request/response validation
- ✅ **Health Check Endpoint** (`/health`) for monitoring
- ✅ **Prediction Endpoint** (`/v1/predict`) for fraud detection
- ✅ **MLflow Integration** for model serving
- ✅ **Production Architecture** with comprehensive error handling

## Model Performance

The API serves an optimized LightGBM model with the following performance:
- **PR-AUC**: 0.6347 (+5.40% improvement)
- **F1-Score**: 0.6061 (+12.16% improvement)
- **Optimization**: 75 Optuna trials completed
- **Features**: 51 engineered features from IEEE-CIS dataset

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API

```bash
# Development mode with auto-reload
python main.py

# Production mode with Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Access Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Info**: http://localhost:8000/

## API Endpoints

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-21T15:30:00Z",
  "version": "3.1.0",
  "model_loaded": true,
  "model_version": "local-best_lightgbm_model_20250121",
  "uptime_seconds": 3600.5,
  "dependencies": {
    "mlflow": "available",
    "model": "loaded"
  }
}
```

### Fraud Prediction
```
POST /v1/predict
```

**Request Body:**
```json
{
  "TransactionAmt": 150.50,
  "ProductCD": "W",
  "card1": 13553,
  "card4": "visa",
  "card6": "debit",
  "P_emaildomain": "gmail.com",
  "C1": 1,
  "C2": 1,
  "D1": 0,
  "V1": 1.0
}
```

**Response:**
```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "fraud_probability": 0.15,
  "fraud_score": 150,
  "risk_level": "LOW",
  "prediction": "LEGITIMATE",
  "confidence": 0.70,
  "model_version": "local-best_lightgbm_model_20250121",
  "processing_time_ms": 45.2,
  "risk_factors": [],
  "recommendation": "APPROVE transaction with standard monitoring"
}
```

## Model Integration

The API automatically attempts to load models in the following order:

1. **MLflow Model Registry** (Production stage)
2. **Local Model Artifacts** (from Sprint 2.4)
3. **Dummy Model** (for testing when real model unavailable)

### Model Paths
The API searches for model files in:
- `../docs/sprints/optimization_results/`
- `../../docs/sprints/optimization_results/`
- `models/`

## Input Validation

### Required Fields
- `TransactionAmt`: Transaction amount (> 0, ≤ 1,000,000)
- `ProductCD`: Product code (W, C, R, H, S)

### Optional Fields
- **Card Features**: card1-card6
- **Address Features**: addr1, addr2
- **Distance Features**: dist1, dist2
- **Email Domains**: P_emaildomain, R_emaildomain
- **Count Features**: C1-C14
- **Timedelta Features**: D1-D15
- **Match Features**: M1-M9
- **Vesta Features**: V1-V5

## Risk Assessment

The API provides comprehensive risk assessment:

### Risk Levels
- **LOW**: Fraud probability < 0.3
- **MEDIUM**: Fraud probability 0.3-0.6
- **HIGH**: Fraud probability 0.6-0.8
- **CRITICAL**: Fraud probability ≥ 0.8

### Recommendations
- **APPROVE**: Standard monitoring
- **MONITOR**: Enhanced tracking
- **REVIEW**: Manual approval required
- **BLOCK**: Immediate investigation

## Production Deployment

### Environment Variables
```bash
export MLFLOW_TRACKING_URI="http://localhost:5001"
export API_HOST="0.0.0.0"
export API_PORT="8000"
export LOG_LEVEL="info"
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Health Monitoring
```bash
# Check API health
curl http://localhost:8000/health

# Prometheus metrics (if configured)
curl http://localhost:8000/metrics
```

## Testing

### Unit Tests
```bash
pytest tests/
```

### API Testing
```bash
# Health check
curl -X GET http://localhost:8000/health

# Fraud prediction
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 1500.00,
    "ProductCD": "W",
    "card4": "visa"
  }'
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   FastAPI App    │───▶│   LightGBM      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   MLflow Model   │
                       │    Registry      │
                       └──────────────────┘
```

## Security Considerations

- **Input Validation**: Comprehensive Pydantic validation
- **Rate Limiting**: Configure for production use
- **Authentication**: Add JWT/API key authentication
- **CORS**: Configure allowed origins
- **HTTPS**: Use reverse proxy (nginx/traefik)

## Performance

- **Typical Response Time**: < 100ms
- **Throughput**: 1000+ requests/second
- **Memory Usage**: ~200MB base + model size
- **CPU Usage**: 1-2 cores recommended

## Monitoring

The API provides built-in monitoring:

- **Health Checks**: `/health` endpoint
- **Uptime Tracking**: Service start time
- **Model Status**: Loading and version info
- **Processing Metrics**: Request timing
- **Error Tracking**: Comprehensive logging

## Sprint 3.1 Completion

✅ **Completed Tasks:**
- FastAPI application with Pydantic models
- Health check endpoint implementation
- Fraud prediction endpoint with comprehensive response
- MLflow integration for model serving
- Production-ready architecture
- Comprehensive documentation

**Next Steps:**
- Deploy to production environment
- Configure monitoring and alerting
- Implement authentication/authorization
- Add batch prediction capabilities
- Integrate with existing fraud prevention system

## Support

For issues and questions:
- Check `/docs` endpoint for interactive documentation
- Review logs for debugging information
- Verify model artifacts are available
- Ensure all dependencies are installed

---

**AEGIS Fraud Detection Team**  
Sprint 3.1 - API Service Creation  
Version 3.1.0

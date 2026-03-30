"""
API tests for AgriPrice Prophet
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import app
from datetime import datetime, timedelta
import numpy as np
import io
import json

client = TestClient(app)

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "AgriPrice Prophet" in response.json()["message"]

def test_commodities_endpoint():
    """Test commodities listing"""
    response = client.get("/commodities")
    assert response.status_code == 200
    assert "commodities" in response.json()

def test_disease_classes_endpoint():
    """Test disease classes endpoint"""
    response = client.get("/disease-classes")
    assert response.status_code == 200
    assert "classes" in response.json()

def test_price_prediction():
    """Test price prediction endpoint"""
    test_data = {
        "commodity": "Potato",
        "market": "Delhi",
        "state": "Delhi",
        "date": (datetime.now() + timedelta(days=7)).date().isoformat(),
        "arrivals": 100.0,
        "min_price": 50.0,
        "max_price": 60.0
    }
    
    response = client.post("/predict/price", json=test_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "predicted_price" in result
    assert "confidence_interval" in result
    assert "latency_ms" in result

def test_price_prediction_invalid_input():
    """Test price prediction with invalid input"""
    test_data = {
        "commodity": "Potato",
        "market": "Delhi",
        "state": "Delhi",
        "date": "invalid-date",
        "arrivals": 100.0
    }
    
    response = client.post("/predict/price", json=test_data)
    assert response.status_code == 422  # Validation error

def test_upload_csv():
    """Test file upload with CSV"""
    # Create test CSV
    import pandas as pd
    df = pd.DataFrame({
        'date': ['2024-01-01'],
        'commodity': ['Potato'],
        'market': ['Delhi'],
        'state': ['Delhi'],
        'arrivals': [100],
        'min_price': [50],
        'max_price': [60],
        'modal_price': [55]
    })
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    files = {'file': ('test.csv', csv_buffer.getvalue(), 'text/csv')}
    response = client.post("/upload", files=files)
    
    assert response.status_code == 200
    assert response.json()["filename"] == "test.csv"
    assert response.json()["data_type"] == "tabular"

def test_upload_invalid_file():
    """Test upload with invalid file type"""
    files = {'file': ('test.txt', b'This is not an image', 'text/plain')}
    response = client.post("/upload", files=files)
    assert response.status_code == 400  # Bad request

def test_stats_endpoint():
    """Test stats endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    assert "total_predictions" in response.json()
    assert "model_status" in response.json()

def test_feature_explanations():
    """Test feature explanations endpoint"""
    response = client.get("/feature-explanations")
    assert response.status_code == 200
    assert "features" in response.json()
    assert len(response.json()["features"]) >= 3
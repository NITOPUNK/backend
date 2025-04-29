import pytest
from fastapi.testclient import TestClient
from main import app
import numpy as np
import joblib
import os

# Create a test client
client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "EV Battery Consumption Prediction API"}

def test_model_info_endpoint():
    """Test the model info endpoint"""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_type" in data
    assert "scaler_type" in data
    assert "feature_count" in data

@pytest.mark.parametrize("test_input,expected_status", [
    # Valid test cases
    ({
        "distance": 50.0,
        "duration": 60.0,
        "ambient_temp": 25.0,
        "weather": "sunny",
        "month": 6
    }, 200),
    # Edge cases
    ({
        "distance": 0.0,
        "duration": 0.0,
        "ambient_temp": -40.0,
        "weather": "rainy",
        "month": 12
    }, 200),
    # Invalid test cases
    ({
        "distance": -50.0,  # Negative distance
        "duration": 60.0,
        "ambient_temp": 25.0,
        "weather": "sunny",
        "month": 6
    }, 422),
    ({
        "distance": 50.0,
        "duration": 60.0,
        "ambient_temp": 25.0,
        "weather": "invalid_weather",  # Invalid weather
        "month": 13  # Invalid month
    }, 422)
])
def test_predict_endpoint(test_input, expected_status):
    """Test the predict endpoint with various inputs"""
    response = client.post("/predict", json=test_input)
    assert response.status_code == expected_status
    
    if expected_status == 200:
        data = response.json()
        assert "predicted_soc_difference" in data
        assert "estimated_range_impact" in data
        assert isinstance(data["predicted_soc_difference"], float)
        assert isinstance(data["estimated_range_impact"], float)

@pytest.mark.parametrize("test_input,expected_status", [
    # Valid test cases
    ({
        "distance": 50.0,
        "duration": 60.0,
        "ambient_temp": 25.0,
        "weather": "sunny",
        "month": 6
    }, 200),
    # Edge cases
    ({
        "distance": 0.0,
        "duration": 0.0,
        "ambient_temp": -40.0,
        "weather": "rainy",
        "month": 12
    }, 200)
])
def test_predict_simple_endpoint(test_input, expected_status):
    """Test the simplified predict endpoint"""
    response = client.get(
        "/predict-simple",
        params=test_input
    )
    assert response.status_code == expected_status
    
    if expected_status == 200:
        data = response.json()
        assert "predicted_soc_difference" in data
        assert "estimated_range_impact" in data
        assert isinstance(data["predicted_soc_difference"], float)
        assert isinstance(data["estimated_range_impact"], float)

def test_model_loading():
    """Test if model and scaler are loaded correctly"""
    from main import model, scaler
    assert model is not None
    assert scaler is not None
    assert hasattr(model, 'predict')
    assert hasattr(scaler, 'transform')

def test_weather_encoding():
    """Test weather encoding functionality"""
    from main import weather_categories
    assert weather_categories["sunny"] == 1.0
    assert weather_categories["cloudy"] == 0.7
    assert weather_categories["slightly cloudy"] == 0.5
    assert weather_categories["rainy"] == 0.3

def test_season_encoding():
    """Test season encoding functionality"""
    from main import predict_soc_difference
    # Test each season
    seasons = {
        3: 1,  # Spring
        6: 2,  # Summer
        9: 3,  # Fall
        12: 4  # Winter
    }
    
    for month, expected_season in seasons.items():
        request = {
            "distance": 50.0,
            "duration": 60.0,
            "ambient_temp": 25.0,
            "weather": "sunny",
            "month": month
        }
        response = client.post("/predict", json=request)
        assert response.status_code == 200 
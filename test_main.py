import pytest
from fastapi.testclient import TestClient
from main import app
import os
import sys

# Add the current directory to path so we can import the main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns correct message"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "EV Battery" in response.json()["message"]

def test_model_info_endpoint():
    """Test the model info endpoint returns model information"""
    response = client.get("/model-info")
    assert response.status_code == 200
    # Either we have model info or an error message if models aren't loaded
    assert "status" in response.json()

def test_predict_endpoint_valid_data():
    """Test the prediction endpoint with valid data"""
    test_data = {
        "distance": 50.0,
        "duration": 60.0,
        "ambient_temp": 22.0,
        "weather": "sunny",
        "month": 6
    }
    
    response = client.post("/predict", json=test_data)
    
    # Check if this is a valid response (either successful prediction or error about model not loaded)
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "predicted_soc_difference" in data
        assert "estimated_range_impact" in data
        
        # Check the types of returned values
        assert isinstance(data["predicted_soc_difference"], float)
        assert isinstance(data["estimated_range_impact"], float)
    else:
        # If model not loaded, there should be a detail message
        assert "detail" in response.json()

def test_predict_simple_endpoint():
    """Test the simplified prediction endpoint with query parameters"""
    query_params = {
        "distance": 50.0,
        "duration": 60.0,
        "ambient_temp": 22.0,
        "weather": "sunny",
        "month": 6
    }
    
    response = client.get("/predict-simple", params=query_params)
    
    # Check if this is a valid response (either successful prediction or error about model not loaded)
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "predicted_soc_difference" in data
        assert "estimated_range_impact" in data

def test_predict_endpoint_invalid_data():
    """Test the prediction endpoint with invalid data"""
    test_data = {
        "distance": "invalid",  # Should be a number
        "duration": 60.0,
        "ambient_temp": 22.0,
        "weather": "sunny",
        "month": 6
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_missing_data():
    """Test the prediction endpoint with missing required data"""
    test_data = {
        # Missing distance
        "duration": 60.0,
        "ambient_temp": 22.0,
        "weather": "sunny",
        "month": 6
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error 
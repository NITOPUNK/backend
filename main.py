from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Path to the model files
scaler_path = "ev_battery_scaler.pkl"
model_path = "ev_battery_predictor_ridge_regression.pkl"

# Make paths absolute to ensure they work in deployment
current_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(current_dir, scaler_path)
model_path = os.path.join(current_dir, model_path)

# Load the model and scaler
try:
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    
    logger.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)
    logger.info("Scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    # Create fallback model for testing if needed
    model = None
    scaler = None

# Request model classes
class BatteryRequest(BaseModel):
    distance: float  # in kilometers
    duration: float  # in minutes
    ambient_temp: float  # in Celsius
    weather: str = "slightly cloudy"  # Default value
    month: int = 6  # Default value (June)

class BatteryResponse(BaseModel):
    predicted_soc_difference: float  # state of charge difference
    estimated_range_impact: float  # estimated impact on range in km

# Weather encoding
weather_categories = {
    "sunny": 1.0, 
    "cloudy": 0.7, 
    "slightly cloudy": 0.5, 
    "rainy": 0.3
}

@app.get("/")
async def root():
    return {"message": "EV Battery Consumption Prediction API"}

@app.post("/predict")
async def predict_soc_difference(request: BatteryRequest):
    """
    Predict the State of Charge (SoC) difference based on input parameters.
    """
    try:
        # Handle missing model or scaler
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model or scaler not loaded properly")
        
        # Encode inputs
        weather_encoded = weather_categories.get(request.weather.lower(), 0.5)
        season = (
            1 if 3 <= request.month <= 5 else  # Spring
            2 if 6 <= request.month <= 8 else  # Summer
            3 if 9 <= request.month <= 11 else  # Fall
            4  # Winter
        )

        # Prepare features
        features = np.array([[
            request.distance, 
            request.duration, 
            request.ambient_temp, 
            weather_encoded, 
            season
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Calculate estimated range impact
        # Assuming average EV has 400 km range at 100% battery
        estimated_range_impact = prediction * 400  
        
        return {
            "predicted_soc_difference": round(prediction, 4),
            "estimated_range_impact": round(estimated_range_impact, 2)
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/model-info")
async def model_info():
    """
    Return information about the loaded model
    """
    if model is None or scaler is None:
        return {"status": "error", "message": "Model or scaler not loaded"}
    
    return {
        "status": "ok",
        "model_type": str(type(model).__name__),
        "scaler_type": str(type(scaler).__name__),
        "feature_count": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown"
    }

# Flutter integration API - simplified endpoint with query parameters
@app.get("/predict-simple")
async def predict_simple(
    distance: float = Query(..., description="Distance in kilometers"),
    duration: float = Query(..., description="Duration in minutes"),
    ambient_temp: float = Query(..., description="Ambient temperature in Celsius"),
    weather: str = Query("slightly cloudy", description="Weather condition"),
    month: int = Query(6, description="Month (1-12)")
):
    """
    Simplified prediction endpoint using query parameters for easier Flutter integration
    """
    request = BatteryRequest(
        distance=distance,
        duration=duration,
        ambient_temp=ambient_temp,
        weather=weather,
        month=month
    )
    return await predict_soc_difference(request)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)

# Ensure the app is configured to run on 0.0.0.0 in production
if os.environ.get("RENDER", False):
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)



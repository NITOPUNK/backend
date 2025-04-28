# EV Battery Consumption Prediction API

This FastAPI backend provides an API for predicting electric vehicle battery consumption based on various driving conditions and environmental factors. It's designed to be easily integrated with a Flutter mobile application.

## Features

- Predict State of Charge (SoC) difference for EV trips
- Return estimated range impact based on the prediction
- Include environmental factors like temperature and weather
- Easy to integrate with Flutter applications
- Deployment-ready for platforms like Render

## Installation and Setup

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd path/to/backend
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the server:
   ```bash
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload
   ```

5. The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access the auto-generated Swagger documentation at:
`http://localhost:8000/docs`

### Main Endpoints

#### GET /
Health check endpoint.

#### POST /predict
Predicts the battery SoC difference based on input parameters.

Request body:
```json
{
  "distance": 50.0,        // Trip distance in kilometers
  "duration": 60.0,        // Trip duration in minutes
  "ambient_temp": 22.0,    // Ambient temperature in Celsius
  "weather": "sunny",      // Weather condition (sunny, cloudy, slightly cloudy, rainy)
  "month": 6               // Month of the year (1-12)
}
```

Response:
```json
{
  "predicted_soc_difference": 0.1234,   // Predicted state of charge difference (0-1)
  "estimated_range_impact": 49.36       // Estimated impact on vehicle range in km
}
```

#### GET /predict-simple
A simplified version of the prediction endpoint that uses query parameters instead of a JSON body, which can be easier to work with in some Flutter HTTP libraries.

Query parameters:
- `distance`: Trip distance in kilometers
- `duration`: Trip duration in minutes
- `ambient_temp`: Ambient temperature in Celsius
- `weather`: Weather condition (default: "slightly cloudy")
- `month`: Month of the year (default: 6)

Example: `/predict-simple?distance=50&duration=60&ambient_temp=22&weather=sunny&month=6`

#### GET /model-info
Returns information about the loaded machine learning model.

## Deployment

### Deploying to Render

1. Create a new Web Service on Render
2. Link to your repository
3. Use the following settings:
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add the following environment variables if needed:
   - `PORT`: This will be set automatically by Render

### Troubleshooting Deployment

- **Model Loading Error**: If the model fails to load, check that the model path is correctly specified and that the model files are included in your repository or deployment package.
- **CORS Issues**: If you're having trouble connecting from your Flutter app, ensure the CORS middleware is properly configured in the FastAPI app.
- **Memory Issues**: Machine learning models can be memory-intensive. If you're experiencing crashes on deployment, consider upgrading to a higher-tier service with more memory.

## Integration with Flutter

### HTTP Requests

Here's a sample code snippet for integration with a Flutter application using the `http` package:

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> predictBatteryConsumption({
  required double distance,
  required double duration,
  required double ambientTemp,
  String weather = 'slightly cloudy',
  int month = 6,
}) async {
  final apiUrl = 'https://your-api-url.com/predict-simple';
  
  final response = await http.get(
    Uri.parse('$apiUrl?distance=$distance&duration=$duration&ambient_temp=$ambientTemp&weather=$weather&month=$month'),
  );

  if (response.statusCode == 200) {
    return jsonDecode(response.body);
  } else {
    throw Exception('Failed to predict battery consumption: ${response.body}');
  }
}
```

### Alternative Integration with JSON Body

```dart
Future<Map<String, dynamic>> predictBatteryConsumptionWithJson({
  required double distance,
  required double duration,
  required double ambientTemp,
  String weather = 'slightly cloudy',
  int month = 6,
}) async {
  final apiUrl = 'https://your-api-url.com/predict';
  
  final response = await http.post(
    Uri.parse(apiUrl),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'distance': distance,
      'duration': duration,
      'ambient_temp': ambientTemp,
      'weather': weather,
      'month': month,
    }),
  );

  if (response.statusCode == 200) {
    return jsonDecode(response.body);
  } else {
    throw Exception('Failed to predict battery consumption: ${response.body}');
  }
}
```

## Model Information

The prediction model is a Ridge Regression model trained on EV battery consumption data. It takes into account:

- Trip distance
- Trip duration
- Ambient temperature
- Weather conditions 
- Season (derived from month)

The model outputs the predicted battery state of charge difference, which is then used to estimate the impact on the vehicle's range.

## Testing

Run the included tests with:

```bash
pytest
```

## License

[Include your license information here] 
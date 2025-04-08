# Employee Performance Prediction API

This FastAPI application provides an endpoint for predicting employee performance based on various features.

## FastAPI UI Demo
![FastAPI UI Demo](https://github.com/inamdarmihir/Uniblox_Task_Submission/blob/main/FastAPI_Demo.png)

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your model files are in the correct location:
- `src/models/final_model.joblib`
- `src/models/preprocessor.joblib`

## Running the API

Start the API server:
```bash
python src/api.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
- **URL**: `/`
- **Method**: GET
- **Description**: Welcome message and available endpoints

### 2. Health Check
- **URL**: `/health`
- **Method**: GET
- **Description**: Check if the API and model are working properly

### 3. Prediction Endpoint
- **URL**: `/predict`
- **Method**: POST
- **Description**: Get performance prediction for an employee
- **Request Body**: JSON object with employee features
- **Response**: Prediction and probability

## Example Usage

```python
import requests
import json

url = "http://localhost:8000/predict"
data = {
    "age": 35,
    "education": 3,
    "job_level": 2,
    "department": "Sales",
    "distance_from_home": 10.5,
    "environment_satisfaction": 4,
    "job_involvement": 3,
    "job_satisfaction": 4,
    "relationship_satisfaction": 3,
    "work_life_balance": 3,
    "years_at_company": 5,
    "years_in_current_role": 3,
    "years_since_last_promotion": 2,
    "years_with_curr_manager": 3,
    "total_working_years": 8,
    "training_times_last_year": 2,
    "num_companies_worked": 2,
    "stock_option_level": 1,
    "monthly_rate": 5000.0,
    "daily_rate": 200.0,
    "hourly_rate": 25.0,
    "percentsalaryhike": 15.0,
    "overtime": "No",
    "marital_status": "Single",
    "job_role": "Sales Executive",
    "business_travel": "Travel Rarely",
    "education_field": "Marketing",
    "gender": "Male",
    "over18": "Y"
}

response = requests.post(url, json=data)
print(response.json())
```

## API Documentation

Once the API is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, Any
import os

# Initialize FastAPI app
app = FastAPI(
    title="Employee Performance Prediction API",
    description="API for predicting employee performance based on various features",
    version="1.0.0"
)

# Load model and preprocessor
try:
    model = joblib.load('src/models/final_model.joblib')
    preprocessor = joblib.load('src/models/preprocessor.joblib')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    preprocessor = None

# Define input data model
class EmployeeData(BaseModel):
    age: int
    education: int
    job_level: int
    department: str
    distance_from_home: float
    environment_satisfaction: int
    job_involvement: int
    job_satisfaction: int
    relationship_satisfaction: int
    work_life_balance: int
    years_at_company: int
    years_in_current_role: int
    years_since_last_promotion: int
    years_with_curr_manager: int
    total_working_years: int
    training_times_last_year: int
    num_companies_worked: int
    stock_option_level: int
    monthly_rate: float
    daily_rate: float
    hourly_rate: float
    percentsalaryhike: float
    overtime: str
    marital_status: str
    job_role: str
    business_travel: str
    education_field: str
    gender: str
    over18: str

@app.get("/")
async def root():
    """Root endpoint returning welcome message and available endpoints"""
    return {
        "message": "Welcome to Employee Performance Prediction API",
        "endpoints": {
            "root": "/",
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify if the model is loaded"""
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(data: EmployeeData) -> Dict[str, Any]:
    """Prediction endpoint that accepts employee data and returns performance prediction"""
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Convert input data to dictionary
        input_data = data.dict()
        
        # Create a DataFrame with a single row
        import pandas as pd
        df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        X = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability[1]),
            "message": "High Performance" if prediction == 1 else "Low Performance"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
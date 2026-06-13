import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

# 1. SETUP
app = FastAPI(title="Telco Churn Predictor", description="XGBoost Model API")

# Update path to read from the 'models' folder in your repository
MODEL_PATH = "models/churn_xgb_pipeline_v1.pkl"

try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 2. SCHEMA VALIDATION
class CustomerRequest(BaseModel):
    gender: Literal['Male', 'Female']
    SeniorCitizen: int
    Partner: Literal['Yes', 'No']
    Dependents: Literal['Yes', 'No']
    tenure: int
    PhoneService: Literal['Yes', 'No']
    MultipleLines: Literal['Yes', 'No', 'No phone service']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: Literal['Yes', 'No', 'No internet service']
    OnlineBackup: Literal['Yes', 'No', 'No internet service']
    DeviceProtection: Literal['Yes', 'No', 'No internet service']
    TechSupport: Literal['Yes', 'No', 'No internet service']
    StreamingTV: Literal['Yes', 'No', 'No internet service']
    StreamingMovies: Literal['Yes', 'No', 'No internet service']
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: Literal['Yes', 'No']
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# 3. PREDICTION ENDPOINT
@app.post("/predict")
def predict_churn(data: CustomerRequest):
    df = pd.DataFrame([data.dict()])

    # Feature Engineering
    services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['Num_Services'] = (df[services] == 'Yes').sum(axis=1) + \
                         (df['InternetService'].isin(['DSL', 'Fiber optic'])).astype(int)
    
    df['Avg_Price_Per_Service'] = df['MonthlyCharges'] / df['Num_Services'].replace(0, 1)
    
    def bin_tenure(months):
        if months < 12: return 'New'
        elif months < 24: return 'Little_Established'
        elif months < 48: return 'Established'
        else: return 'Loyal'
    
    df['Tenure_Group'] = df['tenure'].apply(bin_tenure)

    # Prediction
    try:
        prediction = model_pipeline.predict(df)[0]
        probability = model_pipeline.predict_proba(df)[0][1]

        return {
            "prediction": int(prediction),
            "churn_probability": float(probability),
            "risk_level": "High" if probability > 0.5 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Telecom Churn API is running."}
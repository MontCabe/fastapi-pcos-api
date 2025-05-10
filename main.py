from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model and label encoders
model = joblib.load("pcos_model.pkl")
label_encoders = joblib.load("pcos_label_encoders.pkl")  # Check this exists

# Define the expected features
expected_features = list(model.feature_names_in_)  # Works if you used a pipeline or sklearn >=1.0

# Define input schema
class PCOSInput(BaseModel):
    Age: str
    Weight_kg: float
    Height_ft: float
    Marital_Status: str
    Family_History_PCOS: str
    Menstrual_Irregularity: str
    Hormonal_Imbalance: str
    Hyperandrogenism: str
    Hirsutism: str
    Mental_Health: str
    Conception_Difficulty: str
    Insulin_Resistance: str
    Diabetes: str
    Childhood_Trauma: str
    Cardiovascular_Disease: str
    Vegetarian: str
    Exercise_Frequency: str
    Sleep_Hours: str
    Stress_Level: str
    Smoking: str
    PCOS_Medication: str

@app.post("/predict")
async def predict(data: PCOSInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Encode categorical features
        for col in label_encoders:
            if col in input_df.columns:
                le = label_encoders[col]
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid value for column '{col}'")

        # Ensure all expected features are present (even if dummy)
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0  # or np.nan, or another default

        # Reorder to match training
        input_df = input_df[expected_features]

        # Predict
        prediction = model.predict(input_df)[0]
        result = "Prone to PCOS" if prediction == 1 else "Not Prone to PCOS"
        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

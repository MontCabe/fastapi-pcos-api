import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("pcos_model.pkl")
label_encoders = joblib.load("pcos_label_encoders.pkl")

# Sample user input
input_data = {
    'Age': '20-25',
    'Weight_kg': 66,
    'Height_ft': 157.48,
    'Marital_Status': 'Unmarried',
    'Family_History_PCOS': 'No',
    'Menstrual_Irregularity': 'Yes',
    'Hormonal_Imbalance': 'No',
    'Hyperandrogenism': 'No',
    'Hirsutism': 'No',
    'Mental_Health': 'Yes',
    'Conception_Difficulty': 'No',
    'Insulin_Resistance': 'No',
    'Diabetes': 'No',
    'Childhood_Trauma': 'No',
    'Cardiovascular_Disease': 'No',
    'Vegetarian': 'No',
    'Exercise_Frequency': 'Rarely',
    'Sleep_Hours': 'Less than 6 hours',
    'Stress_Level': 'No',
    'Smoking': 'No',
    'PCOS_Medication': 'No.'
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical columns if they exist in the input
for col, le in label_encoders.items():
    if col in input_df.columns:
        try:
            input_df[col] = le.transform(input_df[col].astype(str))
        except ValueError as e:
            print(f"Encoding error for column '{col}': {e}")
            exit()

# Make prediction
prediction = model.predict(input_df)[0]
result = "Prone to PCOS" if prediction == 1 else "Not Prone to PCOS"
print("✅ Prediction:", result)

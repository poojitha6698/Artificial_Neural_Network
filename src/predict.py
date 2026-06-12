import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "artifacts", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "artifacts", "scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "artifacts", "feature_columns.pkl"))

def predict(input_df):
    input_df = input_df[feature_columns]
    scaled = scaler.transform(input_df)
    probability = model.predict_proba(scaled)[0][1]
    return probability

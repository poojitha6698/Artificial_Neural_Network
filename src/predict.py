import os
import pandas as pd
import joblib
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "artifacts", "model.h5"),
    compile=False
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "artifacts", "scaler.pkl")
)

feature_columns = joblib.load(
    os.path.join(BASE_DIR, "artifacts", "feature_columns.pkl")
)

def predict(input_df):

    input_df = input_df[feature_columns]

    scaled = scaler.transform(input_df)

    prediction = model.predict(
        scaled,
        verbose=0
    )[0][0]

    return prediction

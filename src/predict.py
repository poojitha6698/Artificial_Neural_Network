import pandas as pd
import joblib
import tensorflow as tf

model = tf.keras.models.load_model(
    "artifacts/model.h5"
)

scaler = joblib.load(
    "artifacts/scaler.pkl"
)

feature_columns = joblib.load(
    "artifacts/feature_columns.pkl"
)

def predict(input_df):

    input_df = input_df[feature_columns]

    scaled = scaler.transform(input_df)

    prediction = model.predict(
        scaled,
        verbose=0
    )[0][0]

    return prediction
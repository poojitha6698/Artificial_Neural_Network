import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils import save_object

def preprocess(df):

    df = df.copy()

    # Target column
    df["Attrition"] = df["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    encoders = {}

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    save_object("artifacts/scaler.pkl", scaler)
    save_object("artifacts/encoder.pkl", encoders)
    save_object("artifacts/feature_columns.pkl", X.columns.tolist())

    return X_scaled, y

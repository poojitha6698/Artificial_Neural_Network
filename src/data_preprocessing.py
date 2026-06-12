"""Data preprocessing module for Customer Churn Prediction.
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def preprocess_data(data_path):
    """
    Load dataset, preprocess data and save artifacts.

    Parameters
    ----------
    data_path : str

    Returns
    -------
    tuple
    """

    try:

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset not found: {data_path}"
            )

        df = pd.read_csv(data_path)

        df.drop_duplicates(inplace=True)

        numeric_cols = df.select_dtypes(
            include=["number"]
        ).columns

        df[numeric_cols] = df[numeric_cols].fillna(
            df[numeric_cols].median()
        )

        object_cols = df.select_dtypes(
            include=["object"]
        ).columns

        for col in object_cols:
            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

        gender_encoder = LabelEncoder()

        df["Gender"] = gender_encoder.fit_transform(
            df["Gender"]
        )

        joblib.dump(
            gender_encoder,
            os.path.join(
                ARTIFACT_DIR,
                "label_encoder.pkl"
            )
        )

        df = pd.get_dummies(
            df,
            columns=["Geography"],
            drop_first=True
        )

        drop_columns = [
            "RowNumber",
            "CustomerId",
            "Surname",
            "Exited"
        ]

        X = df.drop(
            columns=drop_columns,
            errors="ignore"
        )

        y = df["Exited"]

        feature_columns = X.columns.tolist()

        joblib.dump(
            feature_columns,
            os.path.join(
                ARTIFACT_DIR,
                "feature_columns.pkl"
            )
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        joblib.dump(
            scaler,
            os.path.join(
                ARTIFACT_DIR,
                "scaler.pkl"
            )
        )

        return (
            X_train,
            X_test,
            y_train.values,
            y_test.values,
            feature_columns
        )

    except Exception as e:
        raise Exception(
            f"Preprocessing Error: {str(e)}"
        )



"""
Prediction Module for Customer Churn Prediction
"""

import os
import joblib
import torch
import pandas as pd

from src.model import CustomerChurnANN


# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "model.pth"
)

SCALER_PATH = os.path.join(
    ARTIFACT_DIR,
    "scaler.pkl"
)

ENCODER_PATH = os.path.join(
    ARTIFACT_DIR,
    "label_encoder.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    ARTIFACT_DIR,
    "feature_columns.pkl"
)


# =====================================================
# LOAD ARTIFACTS
# =====================================================

def load_artifacts():
    """
    Load required artifacts.

    Returns
    -------
    tuple
        scaler, encoder, feature_columns
    """

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        ENCODER_PATH,
        FEATURE_COLUMNS_PATH
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Required artifact not found: {file_path}"
            )

    scaler = joblib.load(
        SCALER_PATH
    )

    encoder = joblib.load(
        ENCODER_PATH
    )

    feature_columns = joblib.load(
        FEATURE_COLUMNS_PATH
    )

    return (
        scaler,
        encoder,
        feature_columns
    )


# =====================================================
# PREPARE INPUT DATA
# =====================================================

def prepare_input_data(
    dataframe,
    encoder,
    scaler,
    feature_columns
):
    """
    Prepare input data for prediction.

    Parameters
    ----------
    dataframe : pd.DataFrame
    encoder : LabelEncoder
    scaler : StandardScaler
    feature_columns : list

    Returns
    -------
    numpy.ndarray
    """

    try:

        if dataframe.empty:
            raise ValueError(
                "Input dataframe is empty."
            )

        required_columns = [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary"
        ]

        missing_cols = [
            col
            for col in required_columns
            if col not in dataframe.columns
        ]

        if missing_cols:

            raise ValueError(
                f"Missing columns: {missing_cols}"
            )

        dataframe = dataframe.copy()

        # =====================================
        # Encode Gender
        # =====================================

        dataframe["Gender"] = encoder.transform(
            dataframe["Gender"]
        )

        # =====================================
        # One Hot Encode Geography
        # =====================================

        dataframe = pd.get_dummies(
            dataframe,
            columns=["Geography"],
            drop_first=True
        )

        # =====================================
        # Align Columns
        # =====================================

        dataframe = dataframe.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # =====================================
        # Scale Data
        # =====================================

        scaled_data = scaler.transform(
            dataframe
        )

        return scaled_data

    except Exception as e:

        raise Exception(
            f"Data Preparation Error: {str(e)}"
        )


# =====================================================
# LOAD MODEL
# =====================================================

def load_model(input_dim):
    """
    Load trained PyTorch model.
    """

    try:

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        model = CustomerChurnANN(
            input_dim=input_dim
        )

        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=torch.device("cpu")
            )
        )

        model.eval()

        return model

    except Exception as e:

        raise Exception(
            f"Model Loading Error: {str(e)}"
        )


# =====================================================
# PREDICT
# =====================================================

def predict(dataframe):
    """
    Predict customer churn.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    tuple
        (probability, prediction)
    """

    try:

        if not isinstance(
            dataframe,
            pd.DataFrame
        ):

            raise ValueError(
                "Input must be a pandas DataFrame."
            )

        (
            scaler,
            encoder,
            feature_columns
        ) = load_artifacts()

        processed_data = prepare_input_data(
            dataframe=dataframe,
            encoder=encoder,
            scaler=scaler,
            feature_columns=feature_columns
        )

        model = load_model(
            input_dim=len(feature_columns)
        )

        input_tensor = torch.FloatTensor(
            processed_data
        )

        with torch.no_grad():

            probability = model(
                input_tensor
            ).item()

        prediction = (
            "Leave"
            if probability >= 0.50
            else "Stay"
        )

        return (
            float(probability),
            prediction
        )

    except Exception as e:

        raise Exception(
            f"Prediction Error: {str(e)}"
        )


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    sample = pd.DataFrame({
        "CreditScore": [650],
        "Geography": ["France"],
        "Gender": ["Male"],
        "Age": [35],
        "Tenure": [5],
        "Balance": [50000],
        "NumOfProducts": [2],
        "HasCrCard": [1],
        "IsActiveMember": [1],
        "EstimatedSalary": [60000]
    })

    probability, prediction = predict(
        sample
    )

    print(
        f"Probability: {probability:.4f}"
    )

    print(
        f"Prediction: {prediction}"
    )

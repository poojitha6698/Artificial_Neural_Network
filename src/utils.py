"""
Utility Functions
"""

import os
import joblib
import torch


def save_object(file_path, obj):
    """
    Save object using joblib.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)

    except Exception as e:
        raise Exception(f"Error saving object: {e}")


def load_object(file_path):
    """
    Load object using joblib.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        return joblib.load(file_path)

    except Exception as e:
        raise Exception(f"Error loading object: {e}")


def load_model(model, model_path):
    """
    Load PyTorch model.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device("cpu")
            )
        )

        model.eval()

        return model

    except Exception as e:
        raise Exception(f"Error loading model: {e}")
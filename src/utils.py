import os
import joblib

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def save_object(file_path, obj):
    create_directory(os.path.dirname(file_path))
    joblib.dump(obj, file_path)

def load_object(file_path):
    return joblib.load(file_path)

def load_model(model_path):
    return joblib.load(model_path)

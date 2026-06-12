
"""
Training Script for Customer Churn Prediction
"""

import os
import json
import numpy as np
import torch

from torch import nn
from torch import optim

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from src.model import CustomerChurnANN
from src.data_preprocessing import preprocess_data


# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "churn.csv"
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "model.pth"
)

METRICS_PATH = os.path.join(
    ARTIFACT_DIR,
    "metrics.json"
)

CONFUSION_MATRIX_PATH = os.path.join(
    ARTIFACT_DIR,
    "confusion_matrix.npy"
)

ROC_PATH = os.path.join(
    ARTIFACT_DIR,
    "roc_data.npz"
)


# =====================================================
# TRAIN FUNCTION
# =====================================================

def train_model():
    """
    Train ANN model.
    """

    try:

        print("=" * 60)
        print("Starting Training...")
        print("=" * 60)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            feature_columns
        ) = preprocess_data(DATA_PATH)

        X_train_tensor = torch.FloatTensor(
            X_train
        )

        y_train_tensor = torch.FloatTensor(
            y_train
        ).reshape(-1, 1)

        X_test_tensor = torch.FloatTensor(
            X_test
        )

        train_dataset = TensorDataset(
            X_train_tensor,
            y_train_tensor
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )

        model = CustomerChurnANN(
            input_dim=X_train.shape[1]
        )

        criterion = nn.BCELoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001
        )

        epochs = 100

        # =====================================
        # TRAIN LOOP
        # =====================================

        for epoch in range(epochs):

            model.train()

            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:

                optimizer.zero_grad()

                outputs = model(batch_X)

                loss = criterion(
                    outputs,
                    batch_y
                )

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:

                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Loss: {epoch_loss/len(train_loader):.4f}"
                )

        # =====================================
        # SAVE MODEL
        # =====================================

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"\nModel saved at:\n{MODEL_PATH}"
        )

        # =====================================
        # EVALUATION
        # =====================================

        model.eval()

        with torch.no_grad():

            probabilities = model(
                X_test_tensor
            ).numpy()

        predictions = (
            probabilities > 0.5
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        cm = confusion_matrix(
            y_test,
            predictions
        )

        fpr, tpr, thresholds = roc_curve(
            y_test,
            probabilities
        )

        # =====================================
        # SAVE METRICS
        # =====================================

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        }

        with open(
            METRICS_PATH,
            "w"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        # =====================================
        # SAVE CONFUSION MATRIX
        # =====================================

        np.save(
            CONFUSION_MATRIX_PATH,
            cm
        )

        # =====================================
        # SAVE ROC DATA
        # =====================================

        np.savez(
            ROC_PATH,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds
        )

        # =====================================
        # PRINT RESULTS
        # =====================================

        print("\n")
        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        print(
            f"ROC AUC   : {roc_auc:.4f}"
        )

        print("\nConfusion Matrix\n")
        print(cm)

        print("\nArtifacts Saved Successfully")

    except Exception as e:

        print(
            f"\nTraining Failed: {str(e)}"
        )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    train_model()

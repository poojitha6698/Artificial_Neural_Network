import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

selected_features = [
    "Age",
    "MonthlyIncome",
    "DistanceFromHome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "JobLevel",
    "PercentSalaryHike",
    "NumCompaniesWorked"
]

X = df[selected_features]

y = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(selected_features, "artifacts/feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ANN using sklearn MLPClassifier (same architecture: 64 -> 32 neurons, ReLU)
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    random_state=42
)

model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Accuracy: {acc:.4f}")

joblib.dump(model, "artifacts/model.pkl")

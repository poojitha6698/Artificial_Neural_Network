import pandas as pd
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

joblib.dump(
    scaler,
    "artifacts/scaler.pkl"
)

joblib.dump(
    selected_features,
    "artifacts/feature_columns.pkl"
)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Accuracy: {acc:.4f}")

model.save(
    "artifacts/model.keras"
)
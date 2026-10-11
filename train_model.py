
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load the dataset
df = pd.read_csv("data/ai4i2020.csv")

# 2. Select features and target
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

# 3. Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Test failure cases:", int(y_test.sum()))

# 4. Preprocess categorical machine type
preprocessor = ColumnTransformer(
    transformers=[
        (
            "machine_type",
            OneHotEncoder(handle_unknown="ignore"),
            ["Type"]
        )
    ],
    remainder="passthrough"
)

# 5. Build the Random Forest pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# 6. Train the model
model.fit(X_train, y_train)

# 7. Evaluate on the held-out test set
predictions = model.predict(X_test)

print("\n--- Random Forest Results ---")
print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["No Failure", "Failure"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 8. Compare with a simple majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

print("\n--- Baseline Results ---")
print(f"Accuracy: {accuracy_score(y_test, baseline_predictions):.4f}")
print(f"Recall:   {recall_score(y_test, baseline_predictions, zero_division=0):.4f}")
print(f"F1-score: {f1_score(y_test, baseline_predictions, zero_division=0):.4f}")

# 9. Save the entire preprocessing + model pipeline
joblib.dump(model, "model.pkl")

print("\nModel and preprocessing pipeline saved to model.pkl")

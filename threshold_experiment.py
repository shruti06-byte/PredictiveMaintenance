
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

# 1. Load the dataset
df = pd.read_csv("data/ai4i2020.csv")

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

X = df[features]
y = df["Machine failure"]

# 2. Use the same stratified split as previous experiments
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 3. Preprocessing
preprocessor = ColumnTransformer([
    (
        "type",
        OneHotEncoder(handle_unknown="ignore"),
        ["Type"],
    ),
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        features[1:],
    ),
])

# 4. Train the original Random Forest
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

model.fit(X_train, y_train)

# 5. Get failure probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# 6. Evaluate multiple thresholds
thresholds = [0.50, 0.40, 0.30, 0.20, 0.10]
results = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Failure Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Failure Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Failure F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "False Alarms": int(
            ((predictions == 1) & (y_test == 0)).sum()
        ),
        "Missed Failures": int(
            ((predictions == 0) & (y_test == 1)).sum()
        ),
    })

    print(f"\nThreshold: {threshold:.2f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

# 7. Display and save results
results_df = pd.DataFrame(results)

print("\n=== THRESHOLD TUNING RESULTS ===")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("threshold_results.csv", index=False)

print("\nResults saved to threshold_results.csv")
print("Existing model.pkl has not been changed.")

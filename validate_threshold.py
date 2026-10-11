
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

# 1. Load dataset
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

# 2. Create separate training, validation and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=123,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    random_state=123,
    stratify=y_trainval,
)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))

print("\nFailure counts:")
print("Training:", int(y_train.sum()))
print("Validation:", int(y_val.sum()))
print("Test:", int(y_test.sum()))

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

# 4. Train Random Forest on training data only
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

# 5. Choose threshold using validation data
val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = [0.50, 0.40, 0.30, 0.20, 0.10]
validation_results = []

for threshold in thresholds:
    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    validation_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "False Alarms": int(
            ((predictions == 1) & (y_val == 0)).sum()
        ),
        "Missed Failures": int(
            ((predictions == 0) & (y_val == 1)).sum()
        ),
    })

validation_df = pd.DataFrame(validation_results)

print("\n=== VALIDATION RESULTS ===")
print(validation_df.round(4).to_string(index=False))

# Select the threshold with the highest validation F1
best_index = validation_df["F1"].idxmax()
best_threshold = float(
    validation_df.loc[best_index, "Threshold"]
)

print(f"\nSelected threshold: {best_threshold:.2f}")

# 6. Evaluate once on the untouched test set
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n=== FINAL TEST RESULTS ===")
print(f"Threshold: {best_threshold:.2f}")
print(f"Accuracy: {accuracy_score(y_test, test_predictions):.4f}")
print(
    "Failure Precision:",
    round(precision_score(y_test, test_predictions, zero_division=0), 4)
)
print(
    "Failure Recall:",
    round(recall_score(y_test, test_predictions, zero_division=0), 4)
)
print(
    "Failure F1:",
    round(f1_score(y_test, test_predictions, zero_division=0), 4)
)
print(
    "PR-AUC:",
    round(average_precision_score(y_test, test_probabilities), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

# 7. Save results separately
validation_df.to_csv(
    "threshold_validation_results.csv",
    index=False,
)

pd.DataFrame([{
    "Selected Threshold": best_threshold,
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Failure Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Failure Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "Failure F1": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "PR-AUC": average_precision_score(
        y_test, test_probabilities
    ),
}]).to_csv("threshold_final_test.csv", index=False)

print("\nSaved threshold_validation_results.csv")
print("Saved threshold_final_test.csv")
print("Existing model.pkl has not been changed.")


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight

# 1. Load the existing dataset
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
y = df["Machine failure"].astype(int)

# 2. Use the same stratified split as compare_models.py
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training class distribution:")
print(y_train.value_counts().sort_index())
print("\nTest class distribution:")
print(y_test.value_counts().sort_index())

# 3. Preprocess numerical and categorical features
numeric_features = features[1:]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "type",
            OneHotEncoder(handle_unknown="ignore"),
            ["Type"],
        ),
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
    ]
)

# 4. Compare imbalance-handling strategies
models = {
    "Random Forest - Original": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    "Random Forest - Balanced": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Random Forest - Balanced Subsample": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

for name, classifier in models.items():
    model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
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
        "PR-AUC (Average Precision)": average_precision_score(
            y_test, probabilities
        ),
    })

    print(f"\n--- {name} ---")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

# 5. Display and save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "Failure F1", ascending=False
)

print("\n=== CLASS IMBALANCE EXPERIMENT ===")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("imbalance_results.csv", index=False)

print("\nResults saved to imbalance_results.csv")
print("Existing model.pkl has not been changed.")

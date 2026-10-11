
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load data
df = pd.read_csv("data/ai4i2020.csv")

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

# 2. Identical stratified split for all models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Shared preprocessing
numeric_features = features[1:]

def make_preprocessor(scale_numeric=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    from sklearn.pipeline import Pipeline as SkPipeline

    numeric_pipeline = SkPipeline(numeric_steps)

    return ColumnTransformer([
        (
            "type",
            OneHotEncoder(handle_unknown="ignore"),
            ["Type"]
        ),
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    ])

# 4. Models
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", make_preprocessor(True)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# 5. Majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

results = []

def evaluate(name, estimator):
    predictions = estimator.predict(X_test)

    row = {
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
        )
    }

    results.append(row)

    print(f"\n--- {name} ---")
    print(pd.Series(row).to_string())
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

evaluate("Majority Baseline", baseline)

for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    evaluate(name, estimator)

# 6. Display and save comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "Failure F1", ascending=False
)

print("\n=== MODEL COMPARISON ===")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

print("\nSaved comparison to model_comparison.csv")
print("All models used the same held-out test records.")

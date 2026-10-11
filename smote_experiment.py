
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline


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

# 2. Same stratified split for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Original training distribution:")
print(y_train.value_counts().sort_index())

# 3. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "type",
            OneHotEncoder(handle_unknown="ignore"),
            ["Type"],
        ),
        (
            "numeric",
            SklearnPipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            features[1:],
        ),
    ]
)

# 4. Original Random Forest baseline
original_model = ImbPipeline([
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

original_model.fit(X_train, y_train)

# 5. SMOTENC Random Forest
# SMOTENC preserves the categorical machine Type feature.
smote_model = ImbPipeline([
    (
        "sampler",
        SMOTENC(
            categorical_features=["Type"],
            random_state=42,
        ),
    ),
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

smote_model.fit(X_train, y_train)

# 6. Evaluate both models on the SAME untouched test set
results = []

for name, model in [
    ("Original Random Forest", original_model),
    ("SMOTENC Random Forest", smote_model),
]:
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
        "PR-AUC": average_precision_score(
            y_test, probabilities
        ),
    })

    print(f"\n{name}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "Failure F1", ascending=False
)

print("\n=== SMOTE EXPERIMENT RESULTS ===")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("smote_results.csv", index=False)

print("\nResults saved to smote_results.csv")
print("Existing model.pkl has not been changed.")

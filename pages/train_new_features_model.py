
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ---------- File Paths ----------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "synthetic_new_features_test.csv"
MODEL_PATH = BASE_DIR / "model_new_features.pkl"

# ---------- Load Dataset ----------

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "Vibration_mm_s",
    "Pressure_bar",
    "Humidity_percent",
    "Motor_Current_A",
    "Acoustic_Level_dB",
    "Bearing_Temperature_C"
]

TARGET = "Machine_Failure"

# ---------- Validate Dataset ----------

required_columns = FEATURES + [TARGET]
missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET], errors="coerce")

valid_rows = y.notna() & y.isin([0, 1])
X = X.loc[valid_rows]
y = y.loc[valid_rows].astype(int)

if y.nunique() < 2:
    raise ValueError(
        "The dataset must contain both failure (1) "
        "and non-failure (0) labels."
    )

# ---------- Split Dataset ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ---------- Build Model Pipeline ----------

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])

# ---------- Train Model ----------

model.fit(X_train, y_train)

# ---------- Evaluate Model ----------

y_pred = model.predict(X_test)

print("\nMODEL EVALUATION")
print("-------------------------")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(
    f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}"
)
print(
    f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}"
)
print(
    f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.3f}"
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------- Save Model ----------

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully at: {MODEL_PATH}")
print("Original model.pkl has not been changed.")

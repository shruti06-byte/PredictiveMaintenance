
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Model Performance | Predictive Maintenance AI",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
NEW_MODEL_PATH = BASE_DIR / "model_new_features.pkl"
NEW_DATA_PATH = BASE_DIR / "data" / "synthetic_new_features_test.csv"

# ---------- Navy and White Theme ----------
st.markdown("""
<style>
.stApp {
    background-color: #F3F6FA;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10243A 0%, #183B55 100%);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF;
}
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #10243A;
}
.page-subtitle {
    color: #7B8798;
    font-size: 1rem;
    margin-top: -0.5rem;
    margin-bottom: 1.5rem;
}
.info-banner {
    background: #E5EDF7;
    border: 1px solid #D5E0EF;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #243B53;
    margin: 1.2rem 0 1.8rem 0;
}
.section-caption {
    color: #7B8798;
    margin-bottom: 1rem;
}
div[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid #DFE6EF;
    border-radius: 10px;
    overflow: hidden;
}
div[data-testid="stDownloadButton"] button {
    background: #10243A !important;
    color: white !important;
    border: 1px solid #10243A !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
div[data-testid="stDownloadButton"] button:hover {
    background: #1C3550 !important;
    color: white !important;
    border-color: #1C3550 !important;
}
hr {
    border-color: #D8E0E9;
}
</style>
""", unsafe_allow_html=True)

# ---------- Page Heading ----------
st.title("Model Performance")
st.markdown(
    '<div class="page-subtitle">'
    'Evaluate machine-learning models for predictive maintenance.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-banner">
    This page presents saved experiments and a separate evaluation
    of the new sensor model. Evaluation scores depend on the dataset,
    class distribution, and decision threshold.
</div>
""", unsafe_allow_html=True)

# ---------- Load Saved Results ----------
@st.cache_data
def load_results(filename):
    path = BASE_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def format_metric_table(data):
    display_data = data.copy()

    for column in display_data.select_dtypes(include="number").columns:
        if any(word in column.lower() for word in [
            "accuracy", "precision", "recall", "f1",
            "specificity", "auc", "roc"
        ]):
            valid = display_data[column].dropna()
            if not valid.empty and valid.between(0, 1).all():
                display_data[column] = display_data[column].map(
                    lambda value: (
                        f"{value:.2%}" if pd.notna(value) else ""
                    )
                )

    return display_data


# ---------- Original Model Comparison ----------
results = load_results("model_comparison.csv")

if results is not None and not results.empty:
    st.subheader("Original AI4I Model Comparison")
    st.caption(
        "These are the saved results from the original AI4I experiments."
    )

    st.dataframe(
        format_metric_table(results),
        use_container_width=True,
        hide_index=True
    )

    numeric_columns = results.select_dtypes(
        include="number"
    ).columns.tolist()

    label_candidates = [
        column for column in results.columns
        if results[column].dtype == "object"
        or column.lower() in ["model", "model_name", "classifier"]
    ]

    if numeric_columns and label_candidates:
        st.divider()
        st.subheader("Original Model Metric Comparison")

        col1, col2 = st.columns(2)

        with col1:
            metric = st.selectbox(
                "Evaluation Metric",
                numeric_columns,
                key="original_metric"
            )

        with col2:
            model_column = st.selectbox(
                "Model Name Column",
                label_candidates,
                key="original_model_column"
            )

        chart_data = results[[model_column, metric]].copy()
        chart_data[metric] = pd.to_numeric(
            chart_data[metric], errors="coerce"
        )
        chart_data = chart_data.dropna()

        if not chart_data.empty:
            # Convert percentage-like proportions for the chart.
            if chart_data[metric].between(0, 1).all():
                chart_data[metric] = chart_data[metric] * 100

            st.bar_chart(
                chart_data.set_index(model_column)[metric],
                horizontal=True,
                x_label=f"{metric} (%)",
                y_label="Model",
                color="#10243A",
            )
else:
    st.warning(
        "model_comparison.csv is missing or empty. "
        "Check that it exists in the main project folder."
    )

# ---------- New Sensor Model Evaluation ----------
st.divider()
st.subheader("New Sensor Model Evaluation")

st.markdown(
    '<div class="section-caption">'
    'Evaluation of the model trained on the additional sensor features.'
    '</div>',
    unsafe_allow_html=True
)

st.warning(
    "The current sensor dataset is synthetic demonstration data. "
    "These metrics demonstrate the evaluation workflow and should not "
    "be interpreted as validated real-world failure-detection performance."
)

NEW_FEATURES = [
    "Vibration_mm_s",
    "Pressure_bar",
    "Humidity_percent",
    "Motor_Current_A",
    "Acoustic_Level_dB",
    "Bearing_Temperature_C",
]
TARGET = "Machine_Failure"

if not NEW_MODEL_PATH.exists():
    st.error(
        "model_new_features.pkl was not found in the project root. "
        "Run: python pages/train_new_features_model.py"
    )
elif not NEW_DATA_PATH.exists():
    st.error(
        "synthetic_new_features_test.csv was not found inside the data folder."
    )
else:
    try:
        sensor_data = pd.read_csv(NEW_DATA_PATH)

        missing_columns = [
            column for column in NEW_FEATURES + [TARGET]
            if column not in sensor_data.columns
        ]

        if missing_columns:
            st.error(
                "The sensor dataset is missing required columns: "
                + ", ".join(missing_columns)
            )
        else:
            X = sensor_data[NEW_FEATURES].copy()
            y = pd.to_numeric(
                sensor_data[TARGET], errors="coerce"
            )

            valid_rows = y.notna() & y.isin([0, 1])
            X = X.loc[valid_rows].copy()
            y = y.loc[valid_rows].astype(int)

            if X.empty or y.nunique() < 2:
                st.error(
                    "The dataset must contain valid sensor readings "
                    "and both failure classes (0 and 1)."
                )
            else:
                model = joblib.load(NEW_MODEL_PATH)

                predictions = model.predict(X)
                probabilities = model.predict_proba(X)

                classes = list(model.classes_)
                if 1 not in classes:
                    st.error(
                        "The loaded model does not contain failure class 1."
                    )
                    st.stop()

                failure_index = classes.index(1)
                failure_probabilities = probabilities[:, failure_index]

                accuracy = accuracy_score(y, predictions)
                precision = precision_score(
                    y, predictions, zero_division=0
                )
                recall = recall_score(
                    y, predictions, zero_division=0
                )
                f1 = f1_score(
                    y, predictions, zero_division=0
                )

                st.caption(
                    f"Evaluation rows: {len(y)} | "
                    f"Actual failures: {int(y.sum())} | "
                    f"Actual non-failures: {int((y == 0).sum())}"
                )

                metric_cols = st.columns(4)
                metric_cols[0].metric("Accuracy", f"{accuracy:.1%}")
                metric_cols[1].metric("Precision", f"{precision:.1%}")
                metric_cols[2].metric("Recall", f"{recall:.1%}")
                metric_cols[3].metric("F1-score", f"{f1:.1%}")

                st.caption(
                    "These metrics are calculated against the target "
                    "labels in the supplied synthetic CSV."
                )

                # ---------- Confusion Matrix ----------
                st.divider()
                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y, predictions, labels=[0, 1]
                )

                fig, ax = plt.subplots(figsize=(5, 4))
                display = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=["No Failure", "Failure"]
                )
                display.plot(
                    ax=ax,
                    cmap="Blues",
                    values_format="d",
                    colorbar=False
                )
                ax.set_title("New Sensor Model")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                tn, fp, fn, tp = cm.ravel()

                st.markdown(
                    f"""
                    - **True negatives:** {tn}
                    - **False positives:** {fp}
                    - **False negatives:** {fn}
                    - **True positives:** {tp}
                    """
                )

                # ---------- Classification Report ----------
                with st.expander("View detailed classification report"):
                    report = classification_report(
                        y,
                        predictions,
                        labels=[0, 1],
                        target_names=["No Failure", "Failure"],
                        output_dict=True,
                        zero_division=0
                    )

                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(
                        report_df.round(3),
                        use_container_width=True
                    )

                # ---------- Feature Importance ----------
                st.divider()
                st.subheader("Sensor Feature Importance")

                classifier = model.named_steps.get("classifier")

                if classifier is not None and hasattr(
                    classifier, "feature_importances_"
                ):
                    importance_df = pd.DataFrame({
                        "Sensor Feature": NEW_FEATURES,
                        "Importance": classifier.feature_importances_
                    }).sort_values(
                        "Importance", ascending=True
                    )

                    st.caption(
                        "Feature importance indicates how the fitted "
                        "Random Forest used each feature. It does not "
                        "prove causation."
                    )

                    st.bar_chart(
                        importance_df.set_index("Sensor Feature"),
                        horizontal=True,
                        x_label="Importance",
                        color="#29445F",
                    )
                else:
                    st.info(
                        "Feature importance is unavailable for this model."
                    )

                # ---------- Download Evaluation Predictions ----------
                evaluation_results = X.copy()
                evaluation_results["Actual_Failure"] = y
                evaluation_results["Predicted_Failure"] = predictions
                evaluation_results["Failure_Probability"] = (
                    failure_probabilities
                )
                evaluation_results["Prediction_Correct"] = (
                    predictions == y.to_numpy()
                )

                st.download_button(
                    "Download Sensor Evaluation Results",
                    data=evaluation_results.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="new_sensor_model_evaluation.csv",
                    mime="text/csv"
                )

    except Exception as error:
        st.error(
            f"Could not evaluate the new sensor model: {error}"
        )
        st.caption(
            "Check that the model and dataset were generated using "
            "the expected sensor feature columns."
        )

# ---------- Additional Experiments ----------
st.divider()
st.subheader("Additional Experiments")
st.caption(
    "Saved experiments on class imbalance, SMOTE oversampling, "
    "and decision-threshold selection."
)

experiment_files = {
    "Class Imbalance": "imbalance_results.csv",
    "SMOTE Experiment": "smote_results.csv",
    "Threshold Experiment": "threshold_results.csv",
    "Threshold Validation": "threshold_validation_results.csv",
    "Final Threshold Test": "threshold_final_test.csv",
}

available = []

for label, filename in experiment_files.items():
    data = load_results(filename)
    if data is not None and not data.empty:
        available.append((label, filename, data))

if available:
    selected = st.selectbox(
        "Select an experiment",
        [item[0] for item in available],
        key="experiment_selection"
    )

    for label, filename, data in available:
        if label == selected:
            st.caption(f"Source file: {filename}")
            st.dataframe(
                format_metric_table(data),
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                label="Download Experiment Results",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name=filename,
                mime="text/csv",
                key=f"download_{filename}"
            )
            break
else:
    st.warning("No additional experiment result files were found.")

# ---------- Footer ----------
st.divider()
st.caption(
    "Predictive Maintenance AI | AI4I 2020 Dataset "
    "and synthetic sensor demonstration | Academic prototype"
)


import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Model Explainability",
    page_icon="🧠",
    layout="wide"
)

# -------------------- THEME AND PATHS --------------------
NAVY = "#10243A"
MID_BLUE = "#365D83"
TEXT = "#25364A"
GREY = "#64748B"

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_CONFIG = {
    "AI4I Model": {
        "model_path": BASE_DIR / "model.pkl",
        "data_path": BASE_DIR / "data" / "ai4i2020.csv",
        "features": [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ],
        "target": "Machine failure",
        "description": "Original AI4I 2020 machine-failure model and dataset.",
        "synthetic": False,
    },
    "New Sensor Model": {
        "model_path": BASE_DIR / "model_new_features.pkl",
        "data_path": BASE_DIR / "data" / "synthetic_new_features_test.csv",
        "features": [
            "Vibration_mm_s",
            "Pressure_bar",
            "Humidity_percent",
            "Motor_Current_A",
            "Acoustic_Level_dB",
            "Bearing_Temperature_C"
        ],
        "target": "Machine_Failure",
        "description": "Additional sensor-feature model trained on synthetic demonstration data.",
        "synthetic": True,
    },
}

# -------------------- PAGE STYLING --------------------
st.markdown(f"""
<style>
.stApp {{
    background-color: #F5F7FA;
    color: {TEXT};
}}
[data-testid="stSidebar"] {{
    background-color: {NAVY};
}}
[data-testid="stSidebar"] * {{
    color: white;
}}
.block-container {{
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}}
.explain-header {{
    background: linear-gradient(110deg, {NAVY}, #243F5B);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 22px;
    color: white;
}}
.explain-header h1 {{
    color: white;
    font-size: 30px;
    margin: 0 0 8px 0;
}}
.explain-header p {{
    color: #DCE6F0;
    font-size: 15px;
    margin: 0;
}}
.section-heading {{
    color: {NAVY};
    font-size: 21px;
    font-weight: 700;
    margin-top: 8px;
    margin-bottom: 5px;
}}
div[data-testid="stMetric"] {{
    background: white;
    border: 1px solid #DFE7EF;
    border-radius: 12px;
    padding: 18px;
}}
div[data-testid="stMetricLabel"] {{
    color: {GREY};
}}
div[data-testid="stMetricValue"] {{
    color: {NAVY};
}}
div[data-testid="stDataFrame"] {{
    border: 1px solid #DFE7EF;
    border-radius: 10px;
    overflow: hidden;
}}
div[data-testid="stNumberInput"] label {{
    color: {NAVY};
    font-weight: 600;
}}
hr {{
    border-color: #DCE4ED;
}}
</style>
""", unsafe_allow_html=True)


# -------------------- CACHED LOADERS --------------------
@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# -------------------- SHAP HELPERS --------------------
def style_chart(ax):
    ax.set_facecolor("white")
    ax.title.set_color(NAVY)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.tick_params(colors=TEXT)
    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)


def transform_features(model, frame):
    """Return preprocessed model inputs, classifier, and feature names."""
    if hasattr(model, "named_steps"):
        transformed = model[:-1].transform(frame)
        classifier = model.steps[-1][1]
        preprocessing = model[:-1]

        try:
            names = preprocessing.get_feature_names_out()
        except Exception:
            names = [
                f"Feature {i + 1}"
                for i in range(transformed.shape[1])
            ]
    else:
        transformed = frame
        classifier = model
        names = list(frame.columns)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return transformed, classifier, list(names)


def get_failure_shap_values(shap_values):
    """Handle common SHAP output formats for binary classification."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]

    if getattr(shap_values, "ndim", 0) == 3:
        return shap_values[:, :, 1]

    return shap_values


def get_failure_expected_value(explainer):
    value = explainer.expected_value

    if hasattr(value, "__len__") and not isinstance(value, str):
        return value[1] if len(value) > 1 else value[0]

    return value


def make_explainer(classifier, transformed_sample):
    """Create a tree explainer and calculate SHAP values."""
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed_sample)

    return explainer, get_failure_shap_values(values)


# -------------------- PAGE HEADER --------------------
st.markdown("""
<div class="explain-header">
    <h1>Model Explainability</h1>
    <p>Understand which machine parameters influence the model's failure predictions.</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "SHAP helps explain how input features influence model predictions. "
    "These explanations describe model behaviour, not guaranteed causes of failure."
)


# -------------------- MODEL SELECTOR --------------------
st.markdown(
    '<p class="section-heading">Choose a Model</p>',
    unsafe_allow_html=True
)

selected_model_name = st.selectbox(
    "Model and dataset",
    list(MODEL_CONFIG.keys()),
    help="Each model uses its own matching feature set and dataset."
)

config = MODEL_CONFIG[selected_model_name]

st.caption(config["description"])

if config["synthetic"]:
    st.warning(
        "The New Sensor Model uses synthetic demonstration data. "
        "Its explanations illustrate model behaviour and are not "
        "validated real-world failure causes."
    )


# -------------------- LOAD SELECTED MODEL --------------------
try:
    if not config["model_path"].exists():
        st.error(f"Model file not found: {config['model_path'].name}")
        st.stop()

    if not config["data_path"].exists():
        st.error(f"Dataset file not found: {config['data_path'].name}")
        st.stop()

    df = load_data(str(config["data_path"]))
    model = load_model(str(config["model_path"]))

    features = config["features"]
    target = config["target"]

    missing_columns = [
        column for column in features + [target]
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            "The selected dataset is missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    if df.empty:
        st.warning("The selected dataset contains no records to explain.")
        st.stop()

    # Prepare the displayed data.
    display_df = df.copy()

    if "Type" in features:
        display_df["Type"] = display_df["Type"].astype(str)


    # -------------------- DATASET OVERVIEW --------------------
    st.markdown(
        '<p class="section-heading">Dataset Overview</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Input Features", len(features))

    with col3:
        failure_count = int(
            pd.to_numeric(
                df[target], errors="coerce"
            ).fillna(0).sum()
        )
        st.metric("Recorded Failures", f"{failure_count:,}")

    st.divider()


    # -------------------- GLOBAL FEATURE IMPORTANCE --------------------
    st.markdown(
        '<p class="section-heading">Global Feature Importance</p>',
        unsafe_allow_html=True
    )

    st.caption(
        "Mean absolute SHAP values estimate the average contribution "
        "of each model input to the failure-class output across a sample."
    )

    sample_size = min(100, len(display_df))

    sample = display_df[features].sample(
        n=sample_size,
        random_state=42
    ).copy()

    transformed_sample, classifier, feature_names = transform_features(
        model, sample
    )

    explainer, values = make_explainer(
        classifier, transformed_sample
    )

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP Value": abs(values).mean(axis=0)
    }).sort_values(
        "Mean Absolute SHAP Value",
        ascending=False
    )

    col1, col2 = st.columns([0.9, 1.1], gap="large")

    with col1:
        st.markdown("**Feature Importance Values**")

        display_importance = importance.copy()

        display_importance["Mean Absolute SHAP Value"] = (
            display_importance["Mean Absolute SHAP Value"].round(5)
        )

        st.dataframe(
            display_importance,
            hide_index=True,
            width="stretch"
        )

    with col2:
        st.markdown("**Importance Ranking**")

        plot_data = importance.sort_values(
            "Mean Absolute SHAP Value",
            ascending=True
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")

        ax.barh(
            plot_data["Feature"],
            plot_data["Mean Absolute SHAP Value"],
            color=MID_BLUE
        )

        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_title("Global Feature Importance", pad=14)

        style_chart(ax)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "Larger mean absolute SHAP values indicate greater average "
        "influence on model output. They do not show whether a feature "
        "increases or decreases risk and do not prove causation."
    )

    st.divider()


    # -------------------- INDIVIDUAL PREDICTION EXPLANATION --------------------
    st.markdown(
        '<p class="section-heading">Explain an Individual Prediction</p>',
        unsafe_allow_html=True
    )

    st.caption(
        "Select a dataset record to inspect its input values, estimated "
        "failure probability, and SHAP explanation."
    )

    record_index = st.number_input(
        "Dataset record index",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    record = display_df.iloc[[int(record_index)]][features].copy()

    st.markdown("**Selected Machine Record**")

    st.dataframe(
        record,
        hide_index=True,
        width="stretch"
    )

    if st.button("Explain Selected Record", type="primary"):
        probabilities = model.predict_proba(record)[0]
        classes = list(model.classes_)

        if 1 not in classes:
            st.error(
                "The selected model does not expose a failure class labelled 1."
            )
            st.stop()

        failure_probability = float(
            probabilities[classes.index(1)]
        )

        prediction = int(model.predict(record)[0])

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Estimated Failure Probability",
                f"{failure_probability:.2%}"
            )

        with col2:
            st.metric(
                "Model Prediction",
                "Failure Risk Detected"
                if prediction == 1
                else "Normal Operation"
            )

        st.caption(
            "This is the model's estimated probability, not a guarantee "
            "that the machine will or will not fail."
        )

        st.markdown("**Factors Influencing This Prediction**")

        st.write(
            "The SHAP waterfall chart shows how each transformed feature "
            "moves the model output away from its expected output for "
            "the selected record."
        )

        transformed_record, _, record_feature_names = transform_features(
            model, record
        )

        individual_explainer, individual_values = make_explainer(
            classifier, transformed_record
        )

        explanation = shap.Explanation(
            values=individual_values[0],
            base_values=get_failure_expected_value(
                individual_explainer
            ),
            data=transformed_record[0],
            feature_names=record_feature_names
        )

        plt.figure(figsize=(10, 5))

        shap.plots.waterfall(
            explanation,
            max_display=10,
            show=False
        )

        st.pyplot(plt.gcf(), clear_figure=True)
        plt.close("all")


    # -------------------- FOOTER --------------------
    st.divider()

    st.caption(
        "Important: SHAP explains the behaviour of the selected trained "
        "model. It does not prove that a feature causes machine failure. "
        "Each model is explained using its matching dataset."
    )

except Exception as e:
    st.error("The explainability page could not be loaded.")

    st.warning(
        "Please share the terminal error so we can identify the exact issue."
    )

    st.code(str(e))

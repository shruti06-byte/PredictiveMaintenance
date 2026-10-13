
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Model Explainability | PredictiveAI",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

AI4I_MODEL_PATH = BASE_DIR / "model.pkl"
NEW_MODEL_PATH = BASE_DIR / "model_new_features.pkl"

AI4I_DATA_PATH = DATA_DIR / "ai4i2020.csv"
NEW_DATA_PATH = DATA_DIR / "synthetic_new_features_test.csv"

AI4I_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

NEW_FEATURES = [
    "Vibration_mm_s",
    "Pressure_bar",
    "Humidity_percent",
    "Motor_Current_A",
    "Acoustic_Level_dB",
    "Bearing_Temperature_C"
]

MODEL_CONFIG = {
    "AI4I Model": {
        "model_path": AI4I_MODEL_PATH,
        "data_path": AI4I_DATA_PATH,
        "features": AI4I_FEATURES,
        "target": "Machine failure",
        "threshold": 0.30,
        "description": "Original AI4I 2020 machine-failure model."
    },
    "New Sensor Model": {
        "model_path": NEW_MODEL_PATH,
        "data_path": NEW_DATA_PATH,
        "features": NEW_FEATURES,
        "target": "Machine_Failure",
        "threshold": 0.50,
        "description": (
            "Demonstration model trained on synthetic sensor data. "
            "It is not validated for real-world industrial use."
        )
    }
}

NAVY = "#10243A"
TEXT = "#243B53"
RED = "#DC2626"
GREEN = "#16A34A"


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


def readable_feature(name):
    name = str(name)

    for prefix in (
        "remainder__",
        "num__",
        "cat__",
        "onehotencoder__",
        "ordinalencoder__"
    ):
        name = name.replace(prefix, "")

    name = name.replace("machine_type_Type_", "Machine type ")
    name = name.replace("_", " ")
    return name


def get_failure_probability(model, record):
    probabilities = model.predict_proba(record)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The model does not contain a failure class labelled 1."
        )

    return float(probabilities[classes.index(1)])


def get_transformed_feature_names(model, features):
    if hasattr(model, "get_feature_names_out"):
        try:
            return list(model.get_feature_names_out())
        except Exception:
            pass

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "get_feature_names_out"):
                try:
                    return list(step.get_feature_names_out(features))
                except Exception:
                    pass

    return list(features)


def calculate_shap_values(model, record, background):
    """
    Explain the failure-class output of the fitted pipeline.
    Use TreeExplainer first, with a model-agnostic fallback.
    """
    transformed_record = record
    transformed_background = background

    if hasattr(model, "named_steps"):
        step_names = list(model.named_steps.keys())

        if len(step_names) >= 2:
            preprocessing = model[:-1]

            transformed_record = preprocessing.transform(record)
            transformed_background = preprocessing.transform(background)

            if hasattr(transformed_record, "toarray"):
                transformed_record = transformed_record.toarray()

            if hasattr(transformed_background, "toarray"):
                transformed_background = transformed_background.toarray()

    feature_names = get_transformed_feature_names(
        model, record.columns
    )

    final_estimator = model

    if hasattr(model, "named_steps") and len(model.named_steps) > 0:
        final_estimator = list(model.named_steps.values())[-1]

    try:
        explainer = shap.TreeExplainer(final_estimator)
        shap_values = explainer.shap_values(transformed_record)

        if isinstance(shap_values, list):
            classes = list(
                getattr(final_estimator, "classes_", [])
            )

            if 1 in classes:
                shap_values = shap_values[classes.index(1)]
            else:
                shap_values = shap_values[-1]

        values = np.asarray(shap_values)

        if values.ndim == 3:
            classes = list(
                getattr(final_estimator, "classes_", [])
            )
            class_index = (
                classes.index(1)
                if 1 in classes
                else values.shape[-1] - 1
            )
            values = values[:, :, class_index]

        values = values.reshape(1, -1)[0]

        if len(values) == len(feature_names):
            expected = explainer.expected_value

            if isinstance(expected, (list, np.ndarray)):
                expected_array = np.asarray(expected).reshape(-1)
                classes = list(
                    getattr(final_estimator, "classes_", [])
                )

                expected = (
                    expected_array[classes.index(1)]
                    if len(expected_array) > 1 and 1 in classes
                    else expected_array[-1]
                )

            return (
                values,
                feature_names,
                float(np.asarray(expected).reshape(-1)[0])
            )

    except Exception:
        pass

    background_sample = background.sample(
        n=min(30, len(background)),
        random_state=42
    )

    try:
        explainer = shap.Explainer(
            model.predict_proba,
            background_sample
        )

        explanation = explainer(record)
        values = np.asarray(explanation.values)

        classes = list(model.classes_)
        failure_index = classes.index(1)

        if values.ndim == 3:
            values = values[0, :, failure_index]
        elif values.ndim == 2:
            values = values[0]

        feature_names = list(record.columns)

        return values.reshape(-1), feature_names, float("nan")

    except Exception as error:
        raise RuntimeError(
            "SHAP could not explain this model. "
            "Check the model preprocessing and feature configuration. "
            f"Details: {error}"
        ) from error


# ---------- Header ----------

st.markdown(
    """
    <style>
        .stApp {
            background-color: #F5F7FA;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg, #10243A 0%, #16324A 100%
            );
        }

        [data-testid="stSidebar"] * {
            color: #F0F6FC;
        }

        .explain-header {
            background: linear-gradient(
                110deg, #10243A, #29445F
            );
            padding: 28px 32px;
            border-radius: 16px;
            margin-bottom: 22px;
        }

        .explain-header h1 {
            color: white;
            font-size: 30px;
        }

        .explain-header p {
            color: #E2E8F0;
            margin-bottom: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="explain-header">
        <h1>Model Explainability</h1>
        <p>
            Understand which machine parameters influence the model's
            failure predictions and why a particular record was classified.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "SHAP explains how the model uses input features. "
    "It does not prove that a feature physically causes a machine failure."
)


# ---------- Model Selection ----------

uploaded_context = st.session_state.get("explain_uploaded_record")

preferred_model = (
    uploaded_context.get("model_key")
    if isinstance(uploaded_context, dict)
    else None
)

model_names = list(MODEL_CONFIG.keys())

default_index = (
    model_names.index(preferred_model)
    if preferred_model in model_names
    else 0
)

selected_model_name = st.selectbox(
    "Model and dataset",
    model_names,
    index=default_index
)

config = MODEL_CONFIG[selected_model_name]
st.caption(config["description"])

if not config["model_path"].exists():
    st.error(f"Model file not found: {config['model_path'].name}")
    st.stop()

if not config["data_path"].exists():
    st.error(f"Dataset file not found: {config['data_path'].name}")
    st.stop()

try:
    model = load_model(str(config["model_path"]))
    df = load_dataset(str(config["data_path"]))
except Exception as error:
    st.error(f"Could not load model or dataset: {error}")
    st.stop()

features = config["features"]

missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:
    st.error(
        "The reference dataset is missing required features: "
        + ", ".join(missing_features)
    )
    st.stop()

display_df = df[features].copy()


# ---------- Dataset Overview ----------

st.markdown("### Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Input Features", len(features))

with col3:
    if config["target"] in df.columns:
        st.metric(
            "Recorded Failures",
            int((df[config["target"]] == 1).sum())
        )
    else:
        st.metric("Recorded Failures", "Not available")


# ---------- Select Record ----------

uploaded_context_matches = (
    isinstance(uploaded_context, dict)
    and uploaded_context.get("model_key") == selected_model_name
    and uploaded_context.get("source") == "uploaded_csv"
)

st.markdown("### Select a Machine Record")

if uploaded_context_matches:
    st.success(
        f"Explaining uploaded CSV row "
        f"{uploaded_context.get('row_index')}."
    )

    uploaded_features = uploaded_context.get("features", {})

    try:
        record = pd.DataFrame(
            [{
                feature: uploaded_features[feature]
                for feature in features
            }],
            columns=features
        )
    except KeyError as error:
        st.error(
            f"The uploaded record is missing a required feature: {error}"
        )
        st.stop()

    st.dataframe(
        record,
        hide_index=True,
        use_container_width=True
    )

else:
    record_index = st.number_input(
        "Dataset record index",
        min_value=0,
        max_value=len(display_df) - 1,
        value=0,
        step=1
    )

    record = display_df.iloc[[int(record_index)]].copy()

    st.dataframe(
        record,
        hide_index=True,
        use_container_width=True
    )


# ---------- Prediction ----------

st.markdown("### Prediction Result")

if st.button(
    "Generate Predictions",
    type="primary",
    use_container_width=True
):
    try:
        failure_probability = get_failure_probability(model, record)
        prediction = int(
            failure_probability >= config["threshold"]
        )

        st.session_state["explain_manual_prediction"] = {
            "model_key": selected_model_name,
            "probability": failure_probability,
            "prediction": prediction,
            "record": record.copy()
        }

    except Exception as error:
        st.error(f"Prediction failed: {error}")

manual_result = st.session_state.get("explain_manual_prediction")

if (
    isinstance(manual_result, dict)
    and manual_result.get("model_key") == selected_model_name
    and manual_result.get("record") is not None
    and manual_result["record"].equals(record)
):
    failure_probability = manual_result["probability"]
    prediction = manual_result["prediction"]

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

    if prediction == 1:
        st.error(
            "The model classified this record as having a failure risk."
        )
    else:
        st.success(
            "The model classified this record as normal operation."
        )

    st.caption(
        "This is a model-estimated probability, not a guarantee "
        "of the machine's actual condition."
    )

else:
    st.info(
        "Click Generate Predictions to evaluate the selected machine."
    )
    st.stop()


# ---------- SHAP Explanation ----------

st.markdown("### Factors Influencing This Prediction")

st.write(
    "The graph shows how each feature moves the model's failure-class "
    "output up or down for the selected record."
)

try:
    background = display_df.sample(
        n=min(100, len(display_df)),
        random_state=42
    )

    shap_values, feature_names, expected_value = calculate_shap_values(
        model,
        record,
        background
    )

    if len(shap_values) != len(feature_names):
        raise ValueError(
            "The number of SHAP values does not match the feature names."
        )

    contribution_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP value": shap_values
    })

    contribution_df["Absolute impact"] = (
        contribution_df["SHAP value"].abs()
    )

    contribution_df = contribution_df.sort_values(
        "Absolute impact",
        ascending=True
    ).tail(10)

    chart_colors = [
        RED if value > 0 else GREEN
        for value in contribution_df["SHAP value"]
    ]

    fig, ax = plt.subplots(
        figsize=(10, max(4.5, 0.48 * len(contribution_df)))
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Bars with numerical SHAP labels
    bars = ax.barh(
        [readable_feature(name) for name in contribution_df["Feature"]],
        contribution_df["SHAP value"],
        color=chart_colors
    )

    ax.bar_label(
        bars,
        labels=[
            f"{value:+.4f}"
            for value in contribution_df["SHAP value"]
        ],
        padding=5,
        fontsize=9,
        color=TEXT
    )

    # Expand both sides so numeric labels remain visible
    min_value = min(
        0.0,
        float(contribution_df["SHAP value"].min())
    )
    max_value = max(
        0.0,
        float(contribution_df["SHAP value"].max())
    )

    value_range = max(max_value - min_value, 0.01)

    ax.set_xlim(
        min_value - value_range * 0.25,
        max_value + value_range * 0.25
    )

    ax.axvline(0, color="#64748B", linewidth=1)

    ax.set_xlabel(
        "SHAP contribution to failure-class output",
        color=TEXT
    )

    ax.set_title(
        "Feature Contributions for This Machine",
        color=NAVY,
        pad=12
    )

    ax.tick_params(axis="both", colors=TEXT)
    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"<span style='color:{RED};font-weight:700'>"
            "■ Towards failure</span>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<span style='color:{GREEN};font-weight:700'>"
            "■ Away from failure</span>",
            unsafe_allow_html=True
        )

    st.caption(
        "Red features increase the model's failure-class output; "
        "green features decrease it. This colour convention refers to "
        "SHAP contributions, not directly to the machine's final status."
    )


    # ---------- Record-Specific Explanation ----------

    st.markdown("### Why did the model make this prediction?")

    sorted_pairs = sorted(
        zip(feature_names, shap_values),
        key=lambda pair: abs(float(pair[1])),
        reverse=True
    )

    pushing_failure = [
        (name, float(value))
        for name, value in sorted_pairs
        if value > 1e-8
    ]

    pushing_normal = [
        (name, float(value))
        for name, value in sorted_pairs
        if value < -1e-8
    ]

    if prediction == 1:
        st.error(
            f"**Failure risk detected.** The model estimated a failure "
            f"probability of {failure_probability:.2%}. Its final output "
            "crossed the configured classification threshold. "
            "The positive SHAP contributions below show which features "
            "pushed the model towards failure."
        )
    else:
        st.success(
            f"**Normal operation predicted.** The model estimated a failure "
            f"probability of {failure_probability:.2%}, which is below the "
            f"{config['threshold']:.0%} classification threshold. "
            "Some features may still increase the failure-class output, "
            "but the final prediction remained normal."
        )

    if pushing_failure:
        st.markdown("#### Features pushing towards failure")

        for name, value in pushing_failure[:3]:
            st.write(
                f"- **{readable_feature(name)}** pushed the model's "
                f"failure-class output upwards "
                f"(SHAP contribution: +{value:.4f})."
            )
    else:
        st.write(
            "No positive SHAP contributions were identified for this record."
        )

    if pushing_normal:
        st.markdown("#### Features pushing away from failure")

        for name, value in pushing_normal[:3]:
            st.write(
                f"- **{readable_feature(name)}** pushed the model's "
                f"failure-class output downwards "
                f"(SHAP contribution: {value:.4f})."
            )
    else:
        st.write(
            "No negative SHAP contributions were identified for this record."
        )

    if sorted_pairs:
        strongest_name, strongest_value = sorted_pairs[0]

        st.markdown("#### Overall interpretation")

        if strongest_value > 0:
            st.write(
                f"The strongest individual contributor was "
                f"**{readable_feature(strongest_name)}**, which pushed "
                "the model towards the failure class."
            )
        elif strongest_value < 0:
            st.write(
                f"The strongest individual contributor was "
                f"**{readable_feature(strongest_name)}**, which pushed "
                "the model away from the failure class."
            )
        else:
            st.write(
                "The displayed features had no non-zero SHAP contribution."
            )

    st.info(
        "These explanations describe the behaviour of the trained model. "
        "They do not prove that a sensor value physically caused or "
        "prevented a machine failure. Confirm suspected issues through "
        "inspection and appropriate maintenance procedures."
    )

except Exception as error:
    st.error(
        "The prediction was generated, but its SHAP explanation could "
        "not be calculated."
    )

    st.code(str(error))

    st.info(
        "Check the model's preprocessing pipeline, feature configuration, "
        "and installed SHAP version."
    )

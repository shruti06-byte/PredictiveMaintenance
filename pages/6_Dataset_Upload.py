
import streamlit as st
import pandas as pd
import joblib
import re
from pathlib import Path

st.set_page_config(
    page_title="Dataset Upload | PredictiveAI",
    page_icon="📂",
    layout="wide"
)

# ---------- Paths and Configuration ----------

BASE_DIR = Path(__file__).resolve().parent.parent

AI4I_MODEL_PATH = BASE_DIR / "model.pkl"
NEW_FEATURES_MODEL_PATH = BASE_DIR / "model_new_features.pkl"

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

FAILURE_THRESHOLD = 0.30

# ---------- Page Styling ----------

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

        .upload-header {
            background: linear-gradient(
                110deg, #10243A, #243F5B
            );
            padding: 28px 32px;
            border-radius: 16px;
            margin-bottom: 22px;
        }

        .upload-header h1 {
            color: white;
            font-size: 30px;
            margin-bottom: 8px;
        }

        .upload-header p {
            color: #DCE6F0;
            font-size: 15px;
            margin-bottom: 0;
        }

        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #DFE7EF;
            border-radius: 12px;
            padding: 16px;
        }

        [data-testid="stMetricValue"] {
            color: #10243A;
        }

        h2, h3 {
            color: #10243A;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Model Loading ----------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# ---------- Column Name Matching ----------

def normalize_name(name):
    """Ignore spaces, punctuation, and capitalization."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# ---------- AI4I Feature Aliases ----------

AI4I_ALIASES = {
    "type": ("Type", "none"),
    "machinetype": ("Type", "none"),
    "producttype": ("Type", "none"),

    "airtemperaturek": ("Air temperature [K]", "none"),
    "airtempk": ("Air temperature [K]", "none"),
    "airtemperaturekelvin": ("Air temperature [K]", "none"),
    "airtemperaturec": ("Air temperature [K]", "celsius"),
    "airtemperaturecelsius": ("Air temperature [K]", "celsius"),
    "airtempc": ("Air temperature [K]", "celsius"),
    "airtemperaturef": ("Air temperature [K]", "fahrenheit"),
    "airtemperaturefahrenheit": ("Air temperature [K]", "fahrenheit"),

    "processtemperaturek": ("Process temperature [K]", "none"),
    "processtempk": ("Process temperature [K]", "none"),
    "processtemperaturekelvin": ("Process temperature [K]", "none"),
    "processtemperaturec": ("Process temperature [K]", "celsius"),
    "processtemperaturecelsius": ("Process temperature [K]", "celsius"),
    "processtempc": ("Process temperature [K]", "celsius"),
    "processtemperaturef": ("Process temperature [K]", "fahrenheit"),
    "processtemperaturefahrenheit": ("Process temperature [K]", "fahrenheit"),

    "rotationalspeedrpm": ("Rotational speed [rpm]", "none"),
    "rotationspeedrpm": ("Rotational speed [rpm]", "none"),
    "speedrpm": ("Rotational speed [rpm]", "none"),
    "rpm": ("Rotational speed [rpm]", "none"),

    "torquenm": ("Torque [Nm]", "none"),
    "torque": ("Torque [Nm]", "none"),
    "torquenewtonmeter": ("Torque [Nm]", "none"),

    "toolwearmin": ("Tool wear [min]", "none"),
    "toolwearminutes": ("Tool wear [min]", "none"),
    "toolwear": ("Tool wear [min]", "none"),
}

# ---------- New Sensor Feature Aliases ----------

NEW_FEATURES_ALIASES = {
    "vibrationmms": "Vibration_mm_s",
    "vibration": "Vibration_mm_s",

    "pressurebar": "Pressure_bar",
    "pressure": "Pressure_bar",

    "humiditypercent": "Humidity_percent",
    "humidity": "Humidity_percent",

    "motorcurrenta": "Motor_Current_A",
    "motorcurrent": "Motor_Current_A",
    "currenta": "Motor_Current_A",

    "acousticleveldb": "Acoustic_Level_dB",
    "acousticlevel": "Acoustic_Level_dB",
    "noiseleveldb": "Acoustic_Level_dB",

    "bearingtemperaturec": "Bearing_Temperature_C",
    "bearingtemperature": "Bearing_Temperature_C",
    "bearingtempc": "Bearing_Temperature_C",
}


# ---------- Temperature Conversion ----------

def convert_temperature(series, unit):
    values = pd.to_numeric(series, errors="coerce")

    if unit == "celsius":
        return values + 273.15

    if unit == "fahrenheit":
        return (values - 32) * 5 / 9 + 273.15

    return values


# ---------- Prepare AI4I Dataset ----------

def prepare_ai4i_dataset(uploaded_df):
    result = pd.DataFrame(index=uploaded_df.index)
    matched_columns = {}
    conversions = []
    errors = []

    for original_column in uploaded_df.columns:
        normalized = normalize_name(original_column)

        if normalized not in AI4I_ALIASES:
            continue

        target, unit = AI4I_ALIASES[normalized]

        if target in matched_columns:
            errors.append(
                f"More than one column matches '{target}': "
                f"'{matched_columns[target]}' and '{original_column}'."
            )
            continue

        matched_columns[target] = original_column

        if target == "Type":
            result[target] = (
                uploaded_df[original_column]
                .astype("string")
                .str.strip()
                .str.upper()
            )
        else:
            result[target] = convert_temperature(
                uploaded_df[original_column], unit
            )

            if unit != "none":
                conversions.append(
                    f"{original_column} → {target}"
                )

    missing = [
        feature for feature in AI4I_FEATURES
        if feature not in result.columns
    ]

    if missing:
        errors.append(
            "Missing required AI4I features: "
            + ", ".join(missing)
        )

    if errors:
        return None, matched_columns, conversions, errors

    for feature in AI4I_FEATURES[1:]:
        invalid = result[feature].isna()

        if invalid.any():
            errors.append(
                f"'{feature}' contains {int(invalid.sum())} "
                "missing or non-numeric value(s)."
            )

    invalid_types = ~result["Type"].isin({"L", "M", "H"})

    if invalid_types.any():
        errors.append(
            f"'Type' contains unsupported values: "
            f"{result.loc[invalid_types, 'Type'].unique().tolist()}. "
            "Expected L, M, or H."
        )

    if errors:
        return None, matched_columns, conversions, errors

    return result[AI4I_FEATURES].copy(), matched_columns, conversions, []


# ---------- Prepare New Sensor Dataset ----------

def prepare_new_features_dataset(uploaded_df):
    result = pd.DataFrame(index=uploaded_df.index)
    matched_columns = {}
    errors = []

    for original_column in uploaded_df.columns:
        normalized = normalize_name(original_column)

        if normalized not in NEW_FEATURES_ALIASES:
            continue

        target = NEW_FEATURES_ALIASES[normalized]

        if target in matched_columns:
            errors.append(
                f"More than one column matches '{target}': "
                f"'{matched_columns[target]}' and '{original_column}'."
            )
            continue

        matched_columns[target] = original_column

        result[target] = pd.to_numeric(
            uploaded_df[original_column],
            errors="coerce"
        )

    missing = [
        feature for feature in NEW_FEATURES
        if feature not in result.columns
    ]

    if missing:
        errors.append(
            "Missing required new-sensor features: "
            + ", ".join(missing)
        )

    if errors:
        return None, matched_columns, errors

    for feature in NEW_FEATURES:
        invalid = result[feature].isna()

        if invalid.any():
            errors.append(
                f"'{matched_columns[feature]}' contains "
                f"{int(invalid.sum())} missing or non-numeric value(s)."
            )

    if errors:
        return None, matched_columns, errors

    return result[NEW_FEATURES].copy(), matched_columns, []


# ---------- Header ----------

st.markdown(
    """
    <div class="upload-header">
        <h1>Dataset Upload & Prediction</h1>
        <p>
            Upload machine readings, identify the compatible trained model,
            estimate failure risk, and export the results.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "Two feature sets are supported: the original AI4I machine features "
    "and the new synthetic sensor features. Upload a CSV matching one "
    "complete feature set."
)

st.markdown("## 1. Upload Machine Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV containing one supported set of machine readings.",
    key="dataset_upload_csv"
)

if uploaded_file is None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### AI4I model features")

        st.dataframe(
            pd.DataFrame({
                "Required feature": AI4I_FEATURES,
                "Example": [
                    "M", "298.1", "308.6", "1551", "42.8", "100"
                ]
            }),
            hide_index=True,
            use_container_width=True
        )

    with col2:
        st.markdown("### New sensor model features")

        st.dataframe(
            pd.DataFrame({
                "Required feature": NEW_FEATURES,
                "Example": [
                    "2.77", "5.37", "34.5", "11.62", "69.8", "77.2"
                ]
            }),
            hide_index=True,
            use_container_width=True
        )

    st.caption(
        "The new sensor model was trained on synthetic demonstration data. "
        "Its predictions demonstrate the workflow and are not validated "
        "for real-world machine safety."
    )

    st.stop()


# ---------- Read Uploaded CSV ----------

try:
    uploaded_df = pd.read_csv(uploaded_file)

    if uploaded_df.empty:
        st.error("The uploaded CSV contains no data rows.")
        st.stop()

    if len(uploaded_df.columns) == 0:
        st.error("The uploaded CSV contains no columns.")
        st.stop()

except Exception as e:
    st.error(f"Could not read the uploaded CSV: {e}")
    st.stop()


# ---------- Clear Stale Results ----------

current_upload_signature = (
    uploaded_file.name,
    uploaded_file.size,
    tuple(uploaded_df.columns)
)

if st.session_state.get("upload_signature") != current_upload_signature:
    st.session_state.pop("upload_prediction_results", None)
    st.session_state["upload_signature"] = current_upload_signature


# ---------- File Preview ----------

st.markdown("## 2. Review Uploaded Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Uploaded Rows", f"{len(uploaded_df):,}")

with col2:
    st.metric("Uploaded Columns", len(uploaded_df.columns))

with col3:
    st.metric("Supported Model Types", 2)

st.dataframe(
    uploaded_df.head(10),
    hide_index=True,
    use_container_width=True
)


# ---------- Identify Model and Validate ----------

st.markdown("## 3. Validate Model Compatibility")

ai4i_prepared, ai4i_mapping, conversions, ai4i_errors = (
    prepare_ai4i_dataset(uploaded_df)
)

new_prepared, new_mapping, new_errors = (
    prepare_new_features_dataset(uploaded_df)
)

selected_model_type = None
prepared_df = None
model_path = None
mapping = {}

if ai4i_prepared is not None:
    selected_model_type = "AI4I machine features"
    prepared_df = ai4i_prepared
    model_path = AI4I_MODEL_PATH
    mapping = ai4i_mapping

elif new_prepared is not None:
    selected_model_type = "New sensor features"
    prepared_df = new_prepared
    model_path = NEW_FEATURES_MODEL_PATH
    mapping = new_mapping

else:
    st.error("This file does not match either supported model.")

    with st.expander("AI4I feature validation details"):
        for error in ai4i_errors:
            st.write(f"- {error}")

    with st.expander("New sensor feature validation details"):
        for error in new_errors:
            st.write(f"- {error}")

    st.warning(
        "No predictions were generated. Upload a CSV containing all six "
        "AI4I features or all six new sensor features."
    )

    st.stop()


if not model_path.exists():
    st.error(
        f"The trained model file was not found: {model_path.name}"
    )

    st.info(
        "For the new sensor dataset, run "
        "`python pages/train_new_features_model.py` "
        "from the project root to create `model_new_features.pkl`."
    )

    st.stop()


st.success(
    f"Compatible feature set detected: **{selected_model_type}**"
)

st.markdown("**Feature mapping**")

mapping_rows = []

for feature in prepared_df.columns:
    source_column = mapping[feature]

    status = (
        "Unit converted"
        if any(source_column in item for item in conversions)
        else "Matched"
    )

    mapping_rows.append({
        "Model feature": feature,
        "Uploaded column": source_column,
        "Status": status
    })

st.dataframe(
    pd.DataFrame(mapping_rows),
    hide_index=True,
    use_container_width=True
)

if conversions:
    st.info(
        "Temperature conversions applied:\n\n"
        + "\n".join(f"- {item}" for item in conversions)
    )

with st.expander("Preview prepared model inputs"):
    st.dataframe(
        prepared_df.head(10),
        hide_index=True,
        use_container_width=True
    )


# ---------- Generate Predictions ----------

st.markdown("## 4. Generate Predictions")

st.write(
    f"Selected model: **{selected_model_type}**. "
    f"Failure classification threshold: **{FAILURE_THRESHOLD:.0%}**."
)

if selected_model_type == "New sensor features":
    st.warning(
        "This model was trained on synthetic data with artificially "
        "generated labels. Use its output only to demonstrate the "
        "upload-and-prediction workflow."
    )

if st.button(
    "Generate Predictions",
    type="primary",
    use_container_width=True
):
    try:
        model = load_model(str(model_path))

        probabilities_all = model.predict_proba(prepared_df)

        classes = (
            list(model.classes_)
            if hasattr(model, "classes_")
            else [0, 1]
        )

        if 1 not in classes:
            raise ValueError(
                "The selected model does not contain failure class 1."
            )

        failure_class_index = classes.index(1)
        probabilities = probabilities_all[:, failure_class_index]

        results_df = uploaded_df.copy()

        results_df["Estimated Failure Probability (%)"] = (
            probabilities * 100
        ).round(2)

        results_df["Predicted Status"] = [
            "Failure Risk Detected"
            if probability >= FAILURE_THRESHOLD
            else "Normal Operation"
            for probability in probabilities
        ]

        results_df["Model Used"] = selected_model_type

        st.session_state["upload_prediction_results"] = results_df

    except Exception as e:
        st.session_state.pop("upload_prediction_results", None)

        st.error(
            "Prediction failed. No results were generated. "
            "Check the model file and input compatibility."
        )

        st.code(str(e))


# ---------- Results ----------

if "upload_prediction_results" in st.session_state:
    results_df = st.session_state["upload_prediction_results"]

    risk_count = int(
        (
            results_df["Predicted Status"]
            == "Failure Risk Detected"
        ).sum()
    )

    normal_count = len(results_df) - risk_count

    st.markdown("## 5. Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Processed Rows", f"{len(results_df):,}")

    with col2:
        st.metric("Failure Risk Detected", f"{risk_count:,}")

    with col3:
        st.metric("Normal Operation", f"{normal_count:,}")

    st.dataframe(
        results_df,
        hide_index=True,
        use_container_width=True
    )

    csv_data = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Results (CSV)",
        data=csv_data,
        file_name="predictive_maintenance_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "Predictions are model estimates, not guarantees of machine failure. "
        "The new sensor model uses synthetic demonstration data and is not "
        "validated for real industrial use."
    )

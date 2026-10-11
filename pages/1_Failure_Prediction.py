
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Failure Prediction | Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

# ---------- Paths ----------
PROJECT_DIR = Path(__file__).resolve().parent.parent

AI4I_MODEL_PATH = PROJECT_DIR / "model.pkl"
SENSOR_MODEL_PATH = PROJECT_DIR / "model_new_features.pkl"

AI4I_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

SENSOR_FEATURES = [
    "Vibration_mm_s",
    "Pressure_bar",
    "Humidity_percent",
    "Motor_Current_A",
    "Acoustic_Level_dB",
    "Bearing_Temperature_C"
]

# The original AI4I threshold is preserved.
AI4I_THRESHOLD = 0.30

# This is a demo threshold, not a validated operational threshold.
SENSOR_THRESHOLD = 0.50


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# ---------- Styling ----------
st.markdown("""
<style>
.stApp {
    background-color: #F3F6FA;
    color: #26374A;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10243A 0%, #183B55 100%);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #10243A;
}
.page-subtitle {
    color: #8490A0;
    font-size: 1rem;
    margin-top: -12px;
    margin-bottom: 28px;
}
.section-label {
    color: #64748B;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.result-card {
    background: #FFFFFF;
    border: 1px solid #E1E8F0;
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    box-shadow: 0 4px 14px rgba(16, 36, 58, 0.035);
}
.result-label {
    color: #64748B;
    font-size: 0.9rem;
    margin-bottom: 8px;
}
.result-value {
    color: #10243A;
    font-size: 2rem;
    font-weight: 750;
}
.muted-text {
    color: #8490A0;
    font-size: 0.9rem;
}
div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E1E8F0;
    border-radius: 16px;
    padding: 24px;
}
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E1E8F0;
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #10243A;
}
div[data-testid="stFormSubmitButton"] button {
    background: #10243A !important;
    color: #FFFFFF !important;
    border: 1px solid #10243A !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    min-height: 44px !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: #1B3A56 !important;
    border-color: #1B3A56 !important;
}
div[data-testid="stDownloadButton"] button {
    background: #10243A !important;
    color: #FFFFFF !important;
    border: 1px solid #10243A !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
div[data-testid="stDownloadButton"] button p {
    color: #FFFFFF !important;
}
hr {
    border-color: #D8E1EB;
}
</style>
""", unsafe_allow_html=True)


# ---------- Page Header ----------
st.markdown("""
<div class="hero-banner">
    <h1>Machine Failure Prediction</h1>
    <p>Evaluate machine operating conditions and estimate failure risk.</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "This academic prototype estimates failure risk; it cannot guarantee "
    "machine condition. Use predictions alongside inspections and "
    "manufacturer maintenance guidance."
)


# ---------- Model Selector ----------
st.markdown("### Choose Prediction Model")

model_choice = st.selectbox(
    "Select model",
    ["AI4I Model", "New Sensor Model"]
)

if model_choice == "AI4I Model":
    model_path = AI4I_MODEL_PATH
    features = AI4I_FEATURES
    threshold = AI4I_THRESHOLD

    st.caption(
        "Uses the original AI4I 2020 model and six operating-condition features."
    )
else:
    model_path = SENSOR_MODEL_PATH
    features = SENSOR_FEATURES
    threshold = SENSOR_THRESHOLD

    st.warning(
        "The New Sensor Model uses synthetic demonstration data. "
        "Its predictions illustrate the workflow and are not validated "
        "real-world failure estimates. The 50% threshold is a demo setting."
    )


# ---------- Load Selected Model ----------
if not model_path.exists():
    st.error(f"Model file not found: {model_path.name}")
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as error:
    st.error(f"Could not load the selected model: {error}")
    st.stop()


# ---------- Input Form ----------
st.markdown("### Machine Operating Conditions")

with st.form("prediction_form"):

    if model_choice == "AI4I Model":

        st.markdown(
            '<p class="section-label">Machine Configuration</p>',
            unsafe_allow_html=True
        )

        machine_type = st.selectbox(
            "Machine Type",
            options=["L", "M", "H"],
            format_func=lambda x: {
                "L": "L — Low",
                "M": "M — Medium",
                "H": "H — High"
            }[x]
        )

        st.divider()

        st.markdown(
            '<p class="section-label">Temperature Readings</p>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            air_temp_c = st.number_input(
                "Air Temperature (°C)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=0.5
            )

        with col2:
            process_temp_c = st.number_input(
                "Process Temperature (°C)",
                min_value=0.0,
                max_value=150.0,
                value=35.0,
                step=0.5
            )

        st.divider()

        st.markdown(
            '<p class="section-label">Mechanical Readings</p>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            rotational_speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=0,
                max_value=10000,
                value=1200,
                step=10
            )

        with col2:
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=500.0,
                value=60.0,
                step=1.0
            )

        with col3:
            tool_wear = st.number_input(
                "Tool Wear (minutes)",
                min_value=0,
                max_value=500,
                value=220,
                step=1
            )

    else:

        st.markdown(
            '<p class="section-label">Sensor Readings</p>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            vibration = st.number_input(
                "Vibration (mm/s)",
                min_value=0.0,
                max_value=100.0,
                value=1.5,
                step=0.1
            )

        with col2:
            pressure = st.number_input(
                "Pressure (bar)",
                min_value=0.0,
                max_value=100.0,
                value=8.0,
                step=0.1
            )

        with col3:
            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            motor_current = st.number_input(
                "Motor Current (A)",
                min_value=0.0,
                max_value=500.0,
                value=10.0,
                step=0.5
            )

        with col2:
            acoustic_level = st.number_input(
                "Acoustic Level (dB)",
                min_value=0.0,
                max_value=200.0,
                value=75.0,
                step=1.0
            )

        with col3:
            bearing_temperature = st.number_input(
                "Bearing Temperature (°C)",
                min_value=-50.0,
                max_value=250.0,
                value=80.0,
                step=0.5
            )

    st.divider()

    submitted = st.form_submit_button(
        "Run Failure Assessment",
        type="primary",
        use_container_width=True
    )


# ---------- Prediction ----------
if submitted:

    if model_choice == "AI4I Model":
        input_df = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp_c + 273.15,
            "Process temperature [K]": process_temp_c + 273.15,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])

    else:
        input_df = pd.DataFrame([{
            "Vibration_mm_s": vibration,
            "Pressure_bar": pressure,
            "Humidity_percent": humidity,
            "Motor_Current_A": motor_current,
            "Acoustic_Level_dB": acoustic_level,
            "Bearing_Temperature_C": bearing_temperature
        }])

    try:
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        if 1 not in classes:
            st.error("The selected model does not contain failure class 1.")
            st.stop()

        failure_index = classes.index(1)
        failure_probability = float(probabilities[failure_index])

        prediction = int(failure_probability >= threshold)

        st.divider()
        st.markdown("### Assessment Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Estimated Failure Probability
                    </div>
                    <div class="result-value">
                        {failure_probability:.1%}
                    </div>
                    <div class="muted-text">
                        Model-estimated probability
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Decision Threshold
                    </div>
                    <div class="result-value">
                        {threshold:.0%}
                    </div>
                    <div class="muted-text">
                        Current classification threshold
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("**Failure Probability Scale**")

        st.progress(
            min(max(failure_probability, 0.0), 1.0)
        )

        st.caption(
            "This scale represents estimated failure probability, "
            "not model accuracy."
        )

        # ---------- Risk Status ----------
        if prediction == 1:
            st.error(
                "### Elevated Failure Risk\n\n"
                "The estimated probability meets or exceeds the configured "
                "threshold. Review the operating readings and maintenance "
                "history, and arrange an inspection where appropriate."
            )
        else:
            st.success(
                "### Threshold Not Exceeded\n\n"
                "The estimated probability is below the configured "
                "threshold. Continue routine monitoring and follow the "
                "manufacturer's maintenance schedule."
            )

        if model_choice == "New Sensor Model":
            st.caption(
                "This is a synthetic-data demonstration. The displayed "
                "risk should not be used as a real maintenance decision."
            )

        # ---------- Input Summary ----------
        st.divider()
        st.markdown("### Input Summary")

        summary = input_df.copy()
        summary.insert(0, "Selected Model", model_choice)
        summary["Failure Probability"] = failure_probability
        summary["Decision Threshold"] = threshold
        summary["Predicted Failure"] = prediction
        summary["Prediction Status"] = (
            "Elevated Failure Risk"
            if prediction == 1
            else "Threshold Not Exceeded"
        )

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download Assessment",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="machine_assessment.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")

else:
    st.markdown("### Ready for Assessment")

    st.markdown(
        "Select a model, enter the machine readings, and choose "
        "**Run Failure Assessment** to view the estimated risk."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Input Parameters", "6")
    col2.metric("Prediction Type", "Binary Classification")
    col3.metric("Decision Threshold", f"{threshold:.0%}")

st.divider()

st.caption(
    "Predictive Maintenance AI | AI4I 2020 Dataset and synthetic sensor "
    "demonstration | Machine-learning academic prototype"
)

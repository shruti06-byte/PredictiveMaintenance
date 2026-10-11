
import streamlit as st
import pandas as pd
from pathlib import Path

# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="About Project | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# -------------------- PATHS --------------------

BASE_DIR = Path(__file__).resolve().parent.parent

AI4I_PATH = BASE_DIR / "data" / "ai4i2020.csv"
SENSOR_PATH = BASE_DIR / "data" / "synthetic_new_features_test.csv"

# -------------------- THEME --------------------

NAVY = "#10243A"
MID_BLUE = "#29445F"
PALE_BLUE = "#EAF0F6"
TEXT = "#25364A"
GREY = "#64748B"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #F5F7FB;
        color: {TEXT};
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }}

    .about-header {{
        background: linear-gradient(110deg, {NAVY}, #243F5B);
        padding: 30px 32px;
        border-radius: 16px;
        margin-bottom: 22px;
        color: white;
        box-shadow: 0 6px 20px rgba(16, 36, 58, 0.10);
    }}

    .about-header h1 {{
        color: white !important;
        font-size: 30px;
        margin: 0 0 10px 0;
    }}

    .about-header p {{
        color: #DCE6F0 !important;
        font-size: 15px;
        line-height: 1.7;
        margin: 0;
    }}

    .section-heading {{
        color: {NAVY};
        font-size: 22px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 8px;
    }}

    .info-card {{
        background: #FFFFFF;
        border: 1px solid #DFE7EF;
        border-radius: 12px;
        padding: 20px;
        min-height: 145px;
        margin-bottom: 12px;
        box-sizing: border-box;
    }}

    .info-card h3 {{
        color: {NAVY} !important;
        font-size: 18px;
        margin-top: 0;
        margin-bottom: 10px;
    }}

    .info-card p {{
        color: {TEXT} !important;
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 0;
    }}

    .step-number {{
        color: {MID_BLUE};
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }}

    div[data-testid="stMetric"] {{
        background: #FFFFFF;
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

    h1, h2, h3 {{
        color: {NAVY};
    }}

    hr {{
        border-color: #DCE4ED;
    }}

    .footer {{
        text-align: center;
        color: {GREY};
        padding: 14px 0 8px 0;
        font-size: 13px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------- DATASET LOADERS --------------------

@st.cache_data
def load_ai4i_data():
    return pd.read_csv(AI4I_PATH)


@st.cache_data
def load_sensor_data():
    return pd.read_csv(SENSOR_PATH)


# -------------------- PAGE HEADER --------------------

st.markdown(
    """
    <div class="about-header">
        <h1>About Predictive Maintenance</h1>
        <p>
            Explainable ML-Based Predictive Maintenance and Failure Risk
            Detection System — an interactive application that estimates
            machine failure risk, analyses operating conditions, compares
            model performance, and explains model predictions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- PROJECT OVERVIEW --------------------

st.markdown(
    '<p class="section-heading">Project Overview</p>',
    unsafe_allow_html=True
)

st.write(
    """
    Predictive maintenance is a data-driven approach used to identify
    potential equipment problems before they lead to unexpected downtime.

    This project provides two separate machine-learning workflows.
    The first uses the AI4I 2020 Predictive Maintenance benchmark dataset.
    The second demonstrates prediction using a different set of sensor
    features and a separately trained model.

    Users can enter machine readings, upload compatible CSV files,
    view estimated failure probabilities, analyse model performance,
    explore dataset patterns, and use SHAP explanations to understand
    which features influence predictions.
    """
)

st.divider()

# -------------------- PROJECT OBJECTIVES --------------------

st.markdown(
    '<p class="section-heading">Project Objectives</p>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="info-card">
            <h3>Prediction and Risk Detection</h3>
            <p>
                • Estimate machine failure probability from input readings.<br>
                • Support identification of potentially risky conditions.<br>
                • Accept compatible machine datasets for batch predictions.<br>
                • Demonstrate predictive maintenance workflows.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>Analysis and Explainability</h3>
            <p>
                • Compare machine-learning models using evaluation metrics.<br>
                • Examine failure patterns and feature distributions.<br>
                • Interpret predictions using SHAP explanations.<br>
                • Highlight the limitations of model predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# -------------------- TWO DATASETS AND MODELS --------------------

st.markdown(
    '<p class="section-heading">Datasets and Prediction Models</p>',
    unsafe_allow_html=True
)

st.write(
    "The application supports two distinct feature sets. "
    "Each feature set must be processed by its corresponding trained model."
)

dataset_col1, dataset_col2 = st.columns(2, gap="large")

with dataset_col1:
    st.markdown(
        """
        <div class="info-card">
            <h3>01. AI4I 2020 Dataset</h3>
            <p>
                The benchmark dataset contains machine operating
                measurements and machine failure labels. The original
                model uses six input features:
                <br><br>
                • Type<br>
                • Air temperature [K]<br>
                • Process temperature [K]<br>
                • Rotational speed [rpm]<br>
                • Torque [Nm]<br>
                • Tool wear [min]
                <br><br>
                <b>Target:</b> Machine failure
                <br>
                <b>Model:</b> Original trained AI4I pipeline
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with dataset_col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>02. Additional Sensor Dataset</h3>
            <p>
                A synthetic dataset used to demonstrate prediction with
                additional machine sensor measurements. Its six input
                features are:
                <br><br>
                • Vibration [mm/s]<br>
                • Pressure [bar]<br>
                • Humidity [%]<br>
                • Motor current [A]<br>
                • Acoustic level [dB]<br>
                • Bearing temperature [°C]
                <br><br>
                <b>Target:</b> Machine_Failure
                <br>
                <b>Model:</b> Separate sensor-feature pipeline
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.warning(
    "The additional sensor dataset contains synthetic readings and "
    "artificially generated failure labels. It is included to demonstrate "
    "how the application can support a different feature set. Its results "
    "should not be interpreted as proof of real-world predictive accuracy."
)

# -------------------- DATASET STATISTICS --------------------

st.markdown(
    '<p class="section-heading">Dataset Statistics</p>',
    unsafe_allow_html=True
)

try:
    ai4i_df = load_ai4i_data()
    sensor_df = load_sensor_data()

    ai4i_records = len(ai4i_df)
    ai4i_failures = int(ai4i_df["Machine failure"].sum())

    sensor_records = len(sensor_df)
    sensor_failures = int(sensor_df["Machine_Failure"].sum())

    ai4i_rate = (
        ai4i_failures / ai4i_records * 100
        if ai4i_records else 0
    )

    sensor_rate = (
        sensor_failures / sensor_records * 100
        if sensor_records else 0
    )

    st.markdown("### AI4I 2020 Benchmark Dataset")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Records", f"{ai4i_records:,}")

    with c2:
        st.metric("Recorded Failures", f"{ai4i_failures:,}")

    with c3:
        st.metric("Failure Rate", f"{ai4i_rate:.2f}%")

    st.markdown("### Synthetic Sensor Dataset")

    c4, c5, c6 = st.columns(3)

    with c4:
        st.metric("Total Records", f"{sensor_records:,}")

    with c5:
        st.metric("Failure-Labelled Records", f"{sensor_failures:,}")

    with c6:
        st.metric("Failure-Labelled Rate", f"{sensor_rate:.2f}%")

except FileNotFoundError:
    st.warning(
        "Dataset statistics are unavailable because one or both CSV files "
        "could not be found. Check the project's data folder."
    )

except KeyError as e:
    st.warning(
        f"Dataset statistics could not be calculated. "
        f"Expected target column is missing: {e}"
    )

except Exception as e:
    st.warning(f"Could not load dataset statistics: {e}")

st.divider()

# -------------------- TECHNOLOGY STACK --------------------

st.markdown(
    '<p class="section-heading">Technology Stack</p>',
    unsafe_allow_html=True
)

st.caption(
    "Tools and libraries used to develop the predictive maintenance system."
)

technologies = [
    ("Python", "Core programming language"),
    ("Scikit-learn", "Model training and evaluation"),
    ("Streamlit", "Interactive web application"),
    ("Pandas", "Dataset processing and analysis"),
    ("Matplotlib", "Charts and visualizations"),
    ("SHAP", "Model explainability"),
    ("Joblib", "Saving and loading trained models"),
    ("NumPy", "Numerical operations")
]

tech_columns = st.columns(3, gap="medium")

for i, (name, description) in enumerate(technologies):
    with tech_columns[i % 3]:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>{name}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()

# -------------------- MACHINE LEARNING WORKFLOW --------------------

st.markdown(
    '<p class="section-heading">Machine Learning Workflow</p>',
    unsafe_allow_html=True
)

st.caption(
    "The application follows separate input-feature workflows for its two models."
)

steps = [
    (
        "01",
        "Data Collection",
        "Load the AI4I benchmark data or the synthetic sensor dataset."
    ),
    (
        "02",
        "Preprocessing",
        "Prepare the features and target labels required by each model."
    ),
    (
        "03",
        "Model Training",
        "Train classification models using the corresponding feature set."
    ),
    (
        "04",
        "Evaluation",
        "Review accuracy, precision, recall, F1-score, and confusion matrices."
    ),
    (
        "05",
        "Prediction",
        "Generate a failure probability using the matching trained model."
    ),
    (
        "06",
        "Explainability",
        "Use SHAP to examine global feature importance and individual predictions."
    )
]

for i in range(0, len(steps), 2):
    col1, col2 = st.columns(2, gap="medium")

    for column, step in zip((col1, col2), steps[i:i + 2]):
        number, title, description = step

        with column:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="step-number">STEP {number}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.divider()

# -------------------- MODEL EVALUATION --------------------

st.markdown(
    '<p class="section-heading">Model Evaluation</p>',
    unsafe_allow_html=True
)

st.write(
    """
    Classification performance is explored using several evaluation
    measures. Each measure highlights a different aspect of the model's
    behaviour.
    """
)

metric_col1, metric_col2 = st.columns(2, gap="large")

with metric_col1:
    st.markdown(
        """
        <div class="info-card">
            <h3>Evaluation Metrics</h3>
            <p>
                <b>Accuracy:</b> Overall proportion of correct predictions.<br><br>
                <b>Precision:</b> Proportion of predicted failures that
                are actually labelled as failures.<br><br>
                <b>Recall:</b> Proportion of labelled failures detected
                by the model.<br><br>
                <b>F1-score:</b> Harmonic mean of precision and recall.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>Interpreting the Results</h3>
            <p>
                • A confusion matrix helps identify correct and incorrect
                classifications.<br><br>
                • Failure recall is important because missed failures can
                be costly.<br><br>
                • A classification threshold affects the balance between
                missed failures and false alarms.<br><br>
                • Results from synthetic data do not establish real-world
                model performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.info(
    "Evaluation caution: the synthetic sensor dataset has also been used "
    "for model training. Metrics calculated on those same records are "
    "resubstitution results, not independent test-set performance, and "
    "can overestimate generalization."
)

st.divider()

# -------------------- EXPLAINABILITY --------------------

st.markdown(
    '<p class="section-heading">Explainability Using SHAP</p>',
    unsafe_allow_html=True
)

st.write(
    """
    SHAP (SHapley Additive exPlanations) helps explain how input features
    contribute to a model's output.

    The Explainability page provides global feature-importance analysis
    and individual-record explanations. These visualizations help users
    understand model behaviour, but they do not establish that a feature
    causes machine failure.
    """
)

# -------------------- LIMITATIONS --------------------

st.markdown(
    '<p class="section-heading">Limitations</p>',
    unsafe_allow_html=True
)

limitations = [
    (
        "Dataset Dependence",
        "Predictions depend on the quality, distribution, and coverage "
        "of the data used to train the corresponding model."
    ),
    (
        "Synthetic Sensor Data",
        "The additional sensor dataset uses artificially generated "
        "readings and labels and is intended for demonstration."
    ),
    (
        "Feature Compatibility",
        "Each model requires its expected input features. New, unrelated "
        "features cannot automatically be interpreted by an existing model."
    ),
    (
        "Evaluation Limitations",
        "Metrics measured on training records do not independently "
        "demonstrate performance on unseen machines or operating conditions."
    ),
    (
        "Probability Interpretation",
        "Predicted probabilities are model estimates and are not guaranteed "
        "to be calibrated estimates of real-world failure likelihood."
    ),
    (
        "Decision Support",
        "The application is an educational decision-support prototype, "
        "not a replacement for engineering inspection or maintenance expertise."
    )
]

for i in range(0, len(limitations), 2):
    col1, col2 = st.columns(2, gap="medium")

    for column, item in zip((col1, col2), limitations[i:i + 2]):
        title, description = item

        with column:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.divider()

# -------------------- FUTURE SCOPE --------------------

st.markdown(
    '<p class="section-heading">Future Scope</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        <h3>Possible Improvements</h3>
        <p>
            • Train and validate the sensor model using real industrial sensor data.<br><br>
            • Evaluate models on independent, unseen datasets and machines.<br><br>
            • Calibrate prediction probabilities and tune decision thresholds
            according to maintenance requirements.<br><br>
            • Integrate real-time IoT sensor readings and alert notifications.<br><br>
            • Monitor model performance as operating conditions change.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------- FOOTER --------------------

st.markdown(
    f"""
    <div class="footer">
        Predictive Maintenance System |
        AI4I 2020 + Synthetic Sensor Dataset |
        Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)

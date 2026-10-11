
import streamlit as st
import pandas as pd
from pathlib import Path

# ---------- Page Configuration ----------

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Paths ----------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
SENSOR_DATA_PATH = BASE_DIR / "data" / "synthetic_new_features_test.csv"


# ---------- Theme ----------

NAVY = "#10243A"
MID_BLUE = "#29445F"
TEAL = "#168C8C"
DARK_TEAL = "#117272"

st.markdown(
    """
    <style>
        .stApp {
            background-color: #F5F7FB;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }

        /* Hide Streamlit default navigation */
        [data-testid="stSidebarNav"] {
            display: none;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #10243A 0%,
                #16324A 100%
            );
        }

        [data-testid="stSidebar"] * {
            color: #F0F6FC;
        }

        /* Sidebar branding */
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 0 4px 22px 4px;
            border-bottom: 1px solid #38516B;
            margin-bottom: 28px;
        }

        .brand-icon {
            min-width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #29445F;
            border-radius: 12px;
            font-size: 28px;
        }

        .brand-title {
            color: #FFFFFF;
            font-size: 22px;
            font-weight: 750;
            letter-spacing: -0.5px;
        }

        .brand-subtitle {
            color: #B8C9D9;
            font-size: 11px;
            margin-top: 5px;
        }

        .menu-label {
            color: #9FB4CA;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
            margin: 0 0 14px 0;
        }

        /* Sidebar page links */
        [data-testid="stSidebar"] [data-testid="stPageLink"] a {
            display: flex;
            align-items: center;
            padding: 10px 12px;
            border-radius: 9px;
            border: 1px solid transparent;
            background: transparent;
            color: #F0F6FC;
            font-weight: 500;
        }

        [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
            background: #203B55;
            border-color: transparent;
            color: #FFFFFF;
        }

        [data-testid="stSidebar"] [data-testid="stPageLink"] a p,
        [data-testid="stSidebar"] [data-testid="stPageLink"] a span {
            color: #F0F6FC !important;
        }

        /* Headings */
        h1, h2, h3 {
            color: #142D4E;
            letter-spacing: -0.4px;
        }

        /* Hero banner */
        .hero {
            padding: 2.8rem 3rem;
            border-radius: 22px;
            background: linear-gradient(
                120deg,
                #10243A 0%,
                #163B56 100%
            );
            color: white;
            margin: 1.3rem 0 2rem 0;
            box-shadow: 0 10px 30px rgba(16, 36, 58, 0.14);
        }

        .hero h1 {
            color: #FFFFFF !important;
            font-size: 2.7rem;
            font-weight: 750;
            line-height: 1.2;
            margin-bottom: 0.8rem;
        }

        .hero p {
            color: #D7E3EF !important;
            font-size: 1.08rem;
            line-height: 1.8;
            max-width: 850px;
            margin-bottom: 0;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E1E8EF;
            border-radius: 14px;
            padding: 1.15rem 1.3rem;
            box-shadow: 0 2px 8px rgba(16, 36, 58, 0.035);
        }

        [data-testid="stMetricLabel"] {
            color: #64748B;
            font-size: 0.9rem;
        }

        [data-testid="stMetricValue"] {
            color: #10243A;
            font-weight: 750;
        }

        /* Dashboard cards */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF;
            border: 1px solid #E1E8EF;
            border-radius: 14px;
            box-shadow: 0 2px 8px rgba(16, 36, 58, 0.035);
        }

        [data-testid="stVerticalBlockBorderWrapper"] h2,
        [data-testid="stVerticalBlockBorderWrapper"] h3 {
            color: #153B56;
        }

        [data-testid="stVerticalBlockBorderWrapper"] p {
            line-height: 1.65;
        }

        /* Teal buttons inside dashboard cards */
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stPageLink"] a {
            background-color: #168C8C !important;
            color: #FFFFFF !important;
            border: 1px solid #168C8C !important;
            border-radius: 8px;
            padding: 10px 14px;
            font-weight: 600;
            text-decoration: none;
            transition: background-color 0.2s ease;
        }

        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stPageLink"] a p,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stPageLink"] a span,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stPageLink"] a svg {
            color: #FFFFFF !important;
            fill: #FFFFFF;
        }

        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stPageLink"] a:hover {
            background-color: #117272 !important;
            border-color: #117272 !important;
            color: #FFFFFF !important;
        }

        /* Notice styling */
        [data-testid="stAlert"] {
            border-radius: 10px;
        }

        hr {
            border-color: #DCE5EF;
            margin-top: 2rem;
            margin-bottom: 2rem;
        }

        .footer {
            text-align: center;
            color: #718096;
            font-size: 0.85rem;
            padding: 1.2rem 0 0.5rem 0;
        }

        @media (max-width: 768px) {
            .hero {
                padding: 1.6rem;
            }

            .hero h1 {
                font-size: 2rem;
            }

            .hero p {
                font-size: 0.95rem;
            }
        }
        
        
        /* Main content navigation buttons */
        [data-testid="stMain"] [data-testid="stPageLink"] a {
            background-color: #29445F !important;
            color: #FFFFFF !important;
            border: 1px solid #29445F !important;
            border-radius: 8px !important;
            padding: 9px 12px !important;
            min-height: 40px;
            box-sizing: border-box;
            font-weight: 600;
            text-decoration: none !important;
        }

        [data-testid="stMain"] [data-testid="stPageLink"] a p,
        [data-testid="stMain"] [data-testid="stPageLink"] a span,
        [data-testid="stMain"] [data-testid="stPageLink"] a svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }

        [data-testid="stMain"] [data-testid="stPageLink"] a:hover {
            background-color: #163B56 !important;
            border-color: #163B56 !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------- Dataset Loaders ----------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_sensor_data():
    return pd.read_csv(SENSOR_DATA_PATH)


# ---------- Dashboard Page ----------

def dashboard():

    st.title("Predictive Maintenance AI")

    st.caption(
        "Machine learning for smarter equipment maintenance"
    )

    # ---------- Hero Section ----------

    st.markdown(
        """
        <div class="hero">
            <h1>Predict failures before they happen.</h1>
            <p>
                An interactive machine-learning application that estimates
                machine failure risk, compares predictive models, and
                explains factors associated with equipment failure.
                Explore two different machine-data workflows through
                prediction, analytics, and explainability.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------- Dataset Overview ----------

    st.subheader("Dataset Overview")

    try:
        ai4i_df = load_data()
        sensor_df = load_sensor_data()

        ai4i_records = len(ai4i_df)
        ai4i_failures = int(ai4i_df["Machine failure"].sum())

        sensor_records = len(sensor_df)
        sensor_failures = int(sensor_df["Machine_Failure"].sum())

        ai4i_failure_rate = (
            ai4i_failures / ai4i_records * 100
            if ai4i_records > 0 else 0
        )

        sensor_failure_rate = (
            sensor_failures / sensor_records * 100
            if sensor_records > 0 else 0
        )

        # Real-world benchmark dataset
        st.markdown("### AI4I 2020 — Benchmark Dataset")

        st.caption(
            "Established machine predictive-maintenance benchmark data"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Records",
                f"{ai4i_records:,}"
            )

        with col2:
            st.metric(
                "Recorded Failures",
                f"{ai4i_failures:,}"
            )

        with col3:
            st.metric(
                "Failure Rate",
                f"{ai4i_failure_rate:.2f}%"
            )

        st.divider()

        # Synthetic sensor dataset
        st.markdown("### Additional Sensor Dataset")

        st.caption(
            "Synthetic data created for demonstrating predictions with "
            "a different set of machine sensor features"
        )

        sensor_col1, sensor_col2, sensor_col3 = st.columns(3)

        with sensor_col1:
            st.metric(
                "Total Records",
                f"{sensor_records:,}"
            )

        with sensor_col2:
            st.metric(
                "Failure-Labelled Records",
                f"{sensor_failures:,}"
            )

        with sensor_col3:
            st.metric(
                "Failure-Labelled Rate",
                f"{sensor_failure_rate:.2f}%"
            )

        st.info(
            "Dataset limitation: The additional sensor dataset is synthetic "
            "and its failure labels were generated artificially. Results "
            "from this dataset demonstrate the application workflow; they "
            "do not establish validated real-world model performance."
        )

    except FileNotFoundError as e:
        st.error(
            "A required dataset file was not found. Check that "
            "'ai4i2020.csv' and 'synthetic_new_features_test.csv' "
            "are present in the data folder."
        )
        st.caption(str(e))

    except (KeyError, ValueError) as e:
        st.error(
            "A dataset is missing an expected target column or contains "
            "an invalid target value. Check the CSV column names."
        )
        st.caption(str(e))

    except Exception as e:
        st.error(f"Could not load the dataset overview: {e}")

    # ---------- Two Prediction Workflows ----------

    st.subheader("Two Prediction Workflows")

    st.caption(
        "Each model requires its own compatible set of input features."
    )

    model_col1, model_col2 = st.columns(2, gap="large")

    with model_col1:
        with st.container(border=True):

            st.subheader("01 · AI4I Machine Data")

            st.write(
                "Predict machine failure using six operating features: "
                "machine type, air temperature, process temperature, "
                "rotational speed, torque, and tool wear."
            )

            st.markdown(
                "**Model:** Original AI4I-trained pipeline"
            )

            st.caption("Dataset: AI4I 2020 benchmark data")

            st.page_link(
                prediction_page,
                label="Try Failure Prediction →",
                use_container_width=True
            )

    with model_col2:
        with st.container(border=True):

            st.subheader("02 · Additional Sensor Data")

            st.write(
                "Demonstrate failure-risk prediction using vibration, "
                "pressure, humidity, motor current, acoustic level, "
                "and bearing temperature."
            )

            st.markdown(
                "**Model:** Separate sensor-feature pipeline"
            )

            st.caption("Dataset: Synthetic demonstration data")

            st.page_link(
                upload_page,
                label="Upload Compatible Sensor Data →",
                use_container_width=True
            )

    # ---------- Explore Application ----------

    st.divider()

    st.subheader("Explore the Application")

    st.caption(
        "Choose a section to explore the predictive maintenance system."
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:

        with st.container(border=True):

            st.subheader("Failure Prediction")

            st.write(
                "Enter machine operating conditions or sensor readings "
                "to obtain a failure-risk estimate from the corresponding "
                "model."
            )

            st.page_link(
                prediction_page,
                label="Open Failure Prediction →",
                use_container_width=True
            )

        with st.container(border=True):

            st.subheader("Explainability")

            st.write(
                "Explore global feature importance and individual SHAP "
                "explanations to understand how input features influence "
                "a model's predictions."
            )

            st.page_link(
                explainability_page,
                label="Explore Explainability →",
                use_container_width=True
            )

        with st.container(border=True):

            st.subheader("About the Project")

            st.write(
                "Learn about the project objective, datasets, features, "
                "machine-learning workflow, technologies, and limitations."
            )

            st.page_link(
                about_page,
                label="View Project Details →",
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader("Model Performance")

            st.write(
                "Review model evaluation metrics, classification reports, "
                "confusion matrices, and feature importance."
            )

            st.page_link(
                performance_page,
                label="View Model Performance →",
                use_container_width=True
            )

        with st.container(border=True):

            st.subheader("Analytics & Insights")

            st.write(
                "Explore dataset distributions, failure patterns, machine "
                "parameters, and differences between normal and "
                "failure-labelled records."
            )

            st.page_link(
                analytics_page,
                label="Explore Analytics →",
                use_container_width=True
            )

        with st.container(border=True):

            st.subheader("Dataset Upload")

            st.write(
                "Upload a CSV containing compatible machine readings. "
                "The application identifies the supported feature set, "
                "generates predictions, and lets you download the results."
            )

            st.page_link(
                upload_page,
                label="Upload Dataset →",
                use_container_width=True
            )

    # ---------- Workflow Summary ----------

    st.divider()

    st.subheader("How the System Works")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("### 01")
        st.markdown("**Input Data**")
        st.write(
            "Enter machine readings or upload a compatible CSV file."
        )

    with step2:
        st.markdown("### 02")
        st.markdown("**Feature Matching**")
        st.write(
            "The application identifies the supported feature set."
        )

    with step3:
        st.markdown("### 03")
        st.markdown("**Risk Prediction**")
        st.write(
            "The corresponding trained model estimates failure probability."
        )

    with step4:
        st.markdown("### 04")
        st.markdown("**Interpret Results**")
        st.write(
            "Explore prediction outputs, evaluation metrics, and explanations."
        )

    # ---------- Footer ----------

    st.divider()

    st.markdown(
        """
        <div class="footer">
            Predictive Maintenance System |
            AI4I 2020 + Synthetic Sensor Dataset |
            Machine Learning Project
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------- Register Native Pages ----------

dashboard_page = st.Page(
    dashboard,
    title="Dashboard",
    icon=":material/dashboard:",
    default=True,
    url_path="dashboard"
)

prediction_page = st.Page(
    "pages/1_Failure_Prediction.py",
    title="Failure Prediction",
    icon=":material/speed:"
)

performance_page = st.Page(
    "pages/2_Model_Performance.py",
    title="Model Performance",
    icon=":material/monitoring:"
)

explainability_page = st.Page(
    "pages/3_Explainability.py",
    title="Explainability",
    icon=":material/psychology:"
)

analytics_page = st.Page(
    "pages/4_Analytics.py",
    title="Analytics",
    icon=":material/bar_chart:"
)

about_page = st.Page(
    "pages/5_About_Project.py",
    title="About Project",
    icon=":material/info:"
)

upload_page = st.Page(
    "pages/6_Dataset_Upload.py",
    title="Dataset Upload",
    icon=":material/upload_file:"
)


# ---------- Native Navigation ----------

pages = [
    dashboard_page,
    prediction_page,
    performance_page,
    explainability_page,
    analytics_page,
    about_page,
    upload_page
]

current_page = st.navigation(
    pages,
    position="hidden"
)


# ---------- Custom Sidebar ----------

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">⌁</div>
            <div>
                <div class="brand-title">PredictiveAI</div>
                <div class="brand-subtitle">MACHINE INTELLIGENCE</div>
            </div>
        </div>

        <div class="menu-label">MAIN MENU</div>
        """,
        unsafe_allow_html=True
    )

    st.page_link(
        dashboard_page,
        label="Dashboard",
        icon=":material/dashboard:",
        use_container_width=True
    )

    st.page_link(
        prediction_page,
        label="Failure Prediction",
        icon=":material/speed:",
        use_container_width=True
    )

    st.page_link(
        performance_page,
        label="Model Performance",
        icon=":material/monitoring:",
        use_container_width=True
    )

    st.page_link(
        explainability_page,
        label="Explainability",
        icon=":material/psychology:",
        use_container_width=True
    )

    st.page_link(
        analytics_page,
        label="Analytics",
        icon=":material/bar_chart:",
        use_container_width=True
    )

    st.page_link(
        upload_page,
        label="Dataset Upload",
        icon=":material/upload_file:",
        use_container_width=True
    )

    st.page_link(
        about_page,
        label="About Project",
        icon=":material/info:",
        use_container_width=True
    )

    st.markdown(
        """
        <div style="
            border-top: 1px solid #38516B;
            margin-top: 24px;
            padding-top: 18px;
            color: #9FB4CA;
            font-size: 11px;
            line-height: 1.6;
        ">
            AI-POWERED PREDICTIVE MAINTENANCE
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------- Run Selected Page ----------

current_page.run()


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Analytics | Predictive Maintenance AI",
    page_icon="📈",
    layout="wide"
)

# -------------------- THEME --------------------
NAVY = "#10243A"
PALE_BLUE = "#EAF0F6"
MID_BLUE = "#365D83"
TEXT = "#25364A"
GREY = "#64748B"

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "AI4I 2020 Dataset": {
        "path": BASE_DIR / "data" / "ai4i2020.csv",
        "target": "Machine failure",
        "features": [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ],
        "label": "AI4I 2020",
        "synthetic": False
    },
    "New Sensor Dataset": {
        "path": BASE_DIR / "data" / "synthetic_new_features_test.csv",
        "target": "Machine_Failure",
        "features": [
            "Vibration_mm_s",
            "Pressure_bar",
            "Humidity_percent",
            "Motor_Current_A",
            "Acoustic_Level_dB",
            "Bearing_Temperature_C"
        ],
        "label": "New Sensor Dataset",
        "synthetic": True
    }
}

# -------------------- PAGE STYLING --------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #F5F7FA;
        color: {TEXT};
    }}
    [data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    [data-testid="stSidebar"] * {{
        color: #FFFFFF;
    }}
    .analytics-header {{
        background: linear-gradient(110deg, {NAVY}, #243F5B);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 22px;
        color: white;
    }}
    .analytics-header h1 {{
        color: white;
        font-size: 30px;
        margin: 0 0 8px 0;
    }}
    .analytics-header p {{
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
    div[data-testid="stSelectbox"] label {{
        color: {NAVY};
        font-weight: 600;
    }}
    hr {{
        border-color: #DCE4ED;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------- DATA LOADING --------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


def style_chart(ax):
    """Apply consistent styling to Matplotlib charts."""
    ax.set_facecolor("white")
    ax.title.set_color(NAVY)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.tick_params(colors=TEXT)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)


# -------------------- PAGE HEADER --------------------
st.markdown(
    """
    <div class="analytics-header">
        <h1>Machine Analytics & Insights</h1>
        <p>Explore operating conditions, dataset statistics, and machine failure patterns.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------- DATASET SELECTOR --------------------
st.markdown(
    '<p class="section-heading">Select Dataset</p>',
    unsafe_allow_html=True
)

selected_dataset = st.selectbox(
    "Dataset to analyse",
    list(DATASETS.keys())
)

config = DATASETS[selected_dataset]

try:
    df = load_data(str(config["path"]))

    target = config["target"]
    features = config["features"]

    required_columns = [target] + features

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            "The selected dataset is missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    if df.empty:
        st.warning("The selected dataset contains no records.")
        st.stop()

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[df[target].isin([0, 1])].copy()

    if df.empty:
        st.warning("No valid failure labels (0 or 1) were found.")
        st.stop()

    if config["synthetic"]:
        st.warning(
            "This is synthetic demonstration data. Its distributions "
            "and failure patterns are illustrative, not validated "
            "real-world machine behaviour."
        )
    else:
        st.info(
            "This dashboard analyses the AI4I 2020 dataset. "
            "The charts describe dataset patterns, not proven causes of failure."
        )

    # -------------------- DATASET OVERVIEW --------------------
    st.markdown(
        '<p class="section-heading">Dataset Overview</p>',
        unsafe_allow_html=True
    )

    total_records = len(df)
    failure_count = int(df[target].sum())
    normal_count = total_records - failure_count

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{total_records:,}")

    with col2:
        st.metric("Normal Records", f"{normal_count:,}")

    with col3:
        st.metric("Machine Failures", f"{failure_count:,}")

    failure_rate = failure_count / total_records * 100
    st.caption(f"Failure rate in this dataset: {failure_rate:.2f}%")

    st.divider()

    # -------------------- FAILURE DISTRIBUTION --------------------
    st.markdown(
        '<p class="section-heading">Machine Failure Distribution</p>',
        unsafe_allow_html=True
    )

    failure_data = pd.DataFrame({
        "Status": ["Normal", "Failure"],
        "Records": [normal_count, failure_count]
    })

    col1, col2 = st.columns([0.9, 1.1], gap="large")

    with col1:
        st.markdown("**Record Summary**")
        st.dataframe(
            failure_data,
            hide_index=True,
            width="stretch"
        )

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("white")

        bars = ax.bar(
            failure_data["Status"],
            failure_data["Records"],
            color=[MID_BLUE, NAVY],
            width=0.55
        )

        ax.set_ylabel("Number of Records")
        ax.set_title("Normal vs Failure Records", pad=14)
        style_chart(ax)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height):,}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=NAVY
            )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # -------------------- FEATURE DISTRIBUTIONS --------------------
    st.markdown(
        '<p class="section-heading">Machine Parameter Distributions</p>',
        unsafe_allow_html=True
    )

    st.caption(
        "Select a sensor or operating parameter to examine its distribution."
    )

    selected_feature = st.selectbox(
        "Select a machine parameter",
        features,
        key="distribution_feature"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")

    ax.hist(
        pd.to_numeric(
            df[selected_feature], errors="coerce"
        ).dropna(),
        bins=30,
        color=MID_BLUE,
        edgecolor="white",
        alpha=0.95
    )

    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {selected_feature}", pad=14)
    style_chart(ax)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # -------------------- NORMAL VS FAILURE COMPARISON --------------------
    st.markdown(
        '<p class="section-heading">Operating Conditions by Failure Status</p>',
        unsafe_allow_html=True
    )

    st.caption(
        "Compare the average reading of a selected parameter for "
        "normal and failure records."
    )

    selected_comparison = st.selectbox(
        "Choose a parameter to compare",
        features,
        key="comparison_feature"
    )

    comparison_df = df.copy()
    comparison_df[selected_comparison] = pd.to_numeric(
        comparison_df[selected_comparison], errors="coerce"
    )

    comparison = (
        comparison_df.groupby(target)[selected_comparison]
        .mean()
        .reindex([0, 1])
        .rename(index={0: "Normal", 1: "Failure"})
        .reset_index()
    )

    comparison.columns = ["Machine Status", selected_comparison]

    col1, col2 = st.columns([0.9, 1.1], gap="large")

    with col1:
        st.markdown("**Average Values**")

        display_comparison = comparison.copy()
        display_comparison[selected_comparison] = (
            display_comparison[selected_comparison].round(2)
        )

        st.dataframe(
            display_comparison,
            hide_index=True,
            width="stretch"
        )

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("white")

        plot_comparison = comparison.dropna()

        bars = ax.bar(
            plot_comparison["Machine Status"],
            plot_comparison[selected_comparison],
            color=[MID_BLUE if status == "Normal" else NAVY
                   for status in plot_comparison["Machine Status"]],
            width=0.55
        )

        ax.set_ylabel(f"Average {selected_comparison}")
        ax.set_title("Average Reading by Machine Status", pad=14)
        style_chart(ax)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=NAVY
            )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # -------------------- OPTIONAL RECORD FILTER --------------------
    st.markdown(
        '<p class="section-heading">Filter Records</p>',
        unsafe_allow_html=True
    )

    status_filter = st.selectbox(
        "Show records",
        ["All records", "Normal only", "Failures only"]
    )

    filtered_df = df.copy()

    if status_filter == "Normal only":
        filtered_df = filtered_df[filtered_df[target] == 0]
    elif status_filter == "Failures only":
        filtered_df = filtered_df[filtered_df[target] == 1]

    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} records.")

    with st.expander("View Dataset Sample"):
        st.dataframe(
            filtered_df.head(100),
            hide_index=True,
            width="stretch"
        )

    st.download_button(
        "Download Filtered Dataset",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name=(
            "filtered_ai4i_data.csv"
            if not config["synthetic"]
            else "filtered_sensor_data.csv"
        ),
        mime="text/csv"
    )

    st.caption(
        "Note: These charts describe patterns in the selected dataset. "
        "They do not establish that a particular parameter causes machine failure."
    )

except Exception as e:
    st.error(f"Could not load or analyse the dataset: {e}")
    st.write(
        "Check that the selected CSV exists inside the project's data folder "
        "and contains the expected columns."
    )

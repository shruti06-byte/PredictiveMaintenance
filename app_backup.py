
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/ai4i2020.csv")

model = load_model()
df = load_data()

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)

st.title("⚙️ Predictive Maintenance AI")
st.caption(
    "Explainable ML-Based Predictive Maintenance "
    "and Failure Risk Detection System"
)

st.info(
    "This application uses a Random Forest model trained on "
    "the UCI AI4I 2020 dataset. Predictions are decision-support "
    "signals, not guarantees of equipment condition."
)

# Model evaluation
st.header("Model Performance — Held-Out Test Set")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
c2.metric(
    "Failure Precision",
    f"{precision_score(y_test, y_pred, zero_division=0):.1%}"
)
c3.metric(
    "Failure Recall",
    f"{recall_score(y_test, y_pred, zero_division=0):.1%}"
)
c4.metric(
    "Failure F1-score",
    f"{f1_score(y_test, y_pred, zero_division=0):.1%}"
)

st.caption(
    "Metrics are calculated on the same stratified test split "
    "used during model development. They are not real-world "
    "field-performance estimates."
)

matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

st.subheader("Confusion Matrix")
matrix_df = pd.DataFrame(
    matrix,
    index=["Actual: No Failure", "Actual: Failure"],
    columns=["Predicted: No Failure", "Predicted: Failure"]
)
st.dataframe(matrix_df, use_container_width=True)

# Prediction form
st.header("Machine Failure Risk Assessment")
st.write("Enter operating conditions to obtain a model prediction.")

with st.form("prediction_form"):
    machine_type = st.selectbox(
        "Machine type",
        ["L", "M", "H"],
        format_func=lambda x: {
            "L": "L — Low quality variant",
            "M": "M — Medium quality variant",
            "H": "H — High quality variant"
        }[x]
    )

    col1, col2 = st.columns(2)

    with col1:
        air_temp_c = st.number_input(
            "Air temperature (°C)",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=0.5
        )
        process_temp_c = st.number_input(
            "Process temperature (°C)",
            min_value=0.0,
            max_value=150.0,
            value=35.0,
            step=0.5
        )
        rotational_speed = st.number_input(
            "Rotational speed (RPM)",
            min_value=100,
            max_value=5000,
            value=1200,
            step=50
        )

    with col2:
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=200.0,
            value=60.0,
            step=1.0
        )
        tool_wear = st.number_input(
            "Tool wear (minutes)",
            min_value=0,
            max_value=300,
            value=220,
            step=5
        )

    submitted = st.form_submit_button(
        "Assess Machine Risk",
        use_container_width=True
    )

if submitted:
    if process_temp_c <= air_temp_c:
        st.error(
            "Process temperature should be higher than air "
            "temperature for this dataset's operating conditions."
        )
    else:
        input_df = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp_c + 273.15,
            "Process temperature [K]": process_temp_c + 273.15,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }], columns=features)

        
       
        probabilities = model.predict_proba(input_df)[0]
        failure_probability = float(probabilities[1])

        # Use the selected threshold from validation
        selected_threshold = 0.3
        prediction = int(failure_probability >= selected_threshold)


        st.subheader("Prediction Result")

        st.metric(
            "Model-estimated failure probability",
            f"{failure_probability:.1%}"
        )

        st.caption(
            "This is the model's probability estimate, not a "
            "calibrated real-world risk or a time-to-failure estimate."
        )

        if prediction == 1:
            st.error("The model predicts a machine failure.")
            st.write(
                "**Suggested action:** Inspect the equipment and "
                "review its operating conditions and maintenance "
                "history. Follow the manufacturer's safety procedures."
            )
        else:
            st.success("The model predicts no machine failure.")
            st.write(
                "**Suggested action:** Continue routine monitoring "
                "and scheduled maintenance. A negative prediction "
                "does not guarantee safe operation."
            )

        
        st.subheader("Why did the model make this prediction?")

        st.write(
            "SHAP explains how each machine parameter influences "
            "the model's predicted failure probability."
        )

    try:
        with st.spinner("Generating SHAP explanation..."):
            st.write("SHAP calculation started.")
            # Explain the model's probability predictions
            background_data = X.sample(
                    n=min(50, len(X)),
                    random_state=42
            )
            

            
            # Separate preprocessing from the classifier
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            background_transformed = preprocessor.transform(
                    background_data
            )
            input_transformed = preprocessor.transform(input_df)

        if hasattr(background_transformed, "toarray"):
            background_transformed = background_transformed.toarray()

        if hasattr(input_transformed, "toarray"):
            input_transformed = input_transformed.toarray()

            
            transformed_features = preprocessor.get_feature_names_out(
                features
            )
            st.write("Feature names retrieved successfully.")

            explainer = shap.TreeExplainer(
                classifier,
                data=background_transformed,
                feature_perturbation="interventional",
                model_output="probability"
            )

            st.write("Running SHAP explainer...")
            shap_values = explainer(input_transformed)

            st.write("SHAP values calculated successfully.")
            st.write("SHAP values shape:", shap_values.values.shape)
            st.write("SHAP calculation completed successfully.")

            # Select SHAP values for the failure class
            values = shap_values.values

            if values.ndim == 3:
                failure_shap = values[0, :, 1]
            else:
                failure_shap = values[0]

            # Create the explanation table
            explanation = pd.DataFrame({
                "Machine parameter": transformed_features,
                "SHAP impact": failure_shap
            })

            explanation["Absolute impact"] = (
                explanation["SHAP impact"].abs()
            )

            explanation = explanation.sort_values(
                "Absolute impact",
                ascending=True
            )

            st.caption(
                "Positive impacts increase the predicted failure "
                "probability; negative impacts decrease it."
            )

            # Plot individual prediction explanation
            fig, ax = plt.subplots(figsize=(8, 4))

            ax.barh(
                explanation["Machine parameter"],
                explanation["SHAP impact"]
            )

            ax.set_xlabel("SHAP impact on model output")
            ax.set_ylabel("Machine parameter")
            ax.set_title("SHAP Feature Impact")
            ax.axvline(0, linewidth=1)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    except Exception as e:
        st.error(f"SHAP error: {type(e).__name__}: {e}")
        st.exception(e)

    # Input Summary should be outside the try/except block
    st.subheader("Input Summary")

    summary = input_df.copy()
    summary["Air temperature (°C)"] = air_temp_c
    summary["Process temperature (°C)"] = process_temp_c

    st.dataframe(
        summary[
            [
                "Type",
                "Air temperature (°C)",
                "Process temperature (°C)",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]
        ],
        use_container_width=True
    )


# Global feature importance
st.header("Global Feature Importance")
st.caption(
    "Shows which features the Random Forest model relies on most overall. "
    "This is different from the SHAP explanation for an individual prediction."
)

classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

encoded_type_names = (
    preprocessor.named_transformers_["machine_type"]
    .get_feature_names_out(["Type"])
    .tolist()
)

feature_names = encoded_type_names + features[1:]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_
}).sort_values("Importance", ascending=False)

st.bar_chart(
    importance_df.set_index("Feature"),
    y="Importance",
    horizontal=True
)

st.caption(
    "Feature importance shows how much features contribute to "
    "the fitted Random Forest's splits. It does not prove causation "
    "or explain every individual prediction."
)

st.divider()
st.caption(
    "Academic prototype | UCI AI4I 2020 dataset | "
    "Random Forest Classifier"
)

# Model comparison
st.header("Algorithm Comparison")

comparison = pd.read_csv("model_comparison.csv")

metric_columns = [
    "Accuracy",
    "Failure Precision",
    "Failure Recall",
    "Failure F1"
]

st.write(
    "All algorithms were evaluated on the same held-out test set. "
    "Failure-class metrics are more informative than accuracy alone "
    "because machine failures are rare."
)

st.dataframe(
    comparison.style.format({
        column: "{:.2%}" for column in metric_columns
    }),
    use_container_width=True,
    hide_index=True
)

st.subheader("Failure F1-score by Model")

chart_data = comparison.set_index("Model")[["Failure F1"]]
st.bar_chart(chart_data, y="Failure F1")

best_model = comparison.loc[
    comparison["Failure F1"].idxmax()
]

st.info(
    f"Highest failure F1-score in this comparison: "
    f"{best_model['Model']} "
    f"({best_model['Failure F1']:.2%})."
)

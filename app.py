import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from pycaret.classification import ClassificationExperiment
from sklearn.datasets import make_classification

# Set up Streamlit layout
st.set_page_config(page_title="Credit Underwriting Compliance Portal", layout="wide")
st.title("🏦 Credit Underwriting & Risk Compliance Dashboard")
st.markdown(
    "Evaluate loan applicants, calculate default risk probabilities, and generate "
    "immutable, legally compliant **Adverse Action Notice** audit trails via SHAP values."
)


# ==========================================
# 1. DATA AND MODEL INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Training baseline models and optimization matrices...")
def initialize_ml_pipeline():
    # Generate reproducible loan profile data
    X, y = make_classification(
        n_samples=2500,  # Balanced for application speed
        n_features=12,
        n_classes=2,
        weights=[0.93, 0.07],
        random_state=42,
    )
    feature_names = [
        "Debt-to-Income Ratio", "Loan-to-Value Ratio", "Credit Score", "Annual Income",
        "Employment Length", "Savings Balance", "Utility Payment Score",
        "Recent Credit Inquiries", "Property Value", "Existing Debts Total", "Applicant Age", "Dependents Count"
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df["loan_default_status"] = y

    # Run PyCaret Auto-ML Setup
    exp = ClassificationExperiment()
    exp.setup(
        data=df, target="loan_default_status", train_size=0.8, session_id=42, verbose=False, html=False
    )

    # Train and extract the top-performing ensemble booster
    winning_model = exp.compare_models(
        include=["xgboost", "lightgbm", "catboost"], sort="F1", verbose=False
    )
    final_production_model = exp.finalize_model(winning_model)

    # Isolate underlying booster from the PyCaret wrapping pipeline
    raw_model = final_production_model.named_steps["trained_model"]

    # Extract structural engineering dataset matrices
    X_transformed = exp.get_config("X_train_transformed")
    if "loan_default_status" in X_transformed.columns:
        X_transformed = X_transformed.drop(columns=["loan_default_status"])

    # Generate global tree explainer engine
    explainer = shap.TreeExplainer(raw_model)

    return X_transformed, explainer


# Run the cached ingestion pipeline
X_transformed, explainer = initialize_ml_pipeline()

# ==========================================
# 2. APPLICATION SIDEBAR NAVIGATION
# ==========================================
st.sidebar.header("📋 Applicant Index Configuration")
applicant_idx = st.sidebar.number_input(
    "Select Applicant ID for Audit:",
    min_value=0,
    max_value=len(X_transformed) - 1,
    value=0,
    step=1
)

# Isolate target applicant row
applicant_data = X_transformed.iloc[[applicant_idx]]
applicant_shap = explainer(applicant_data)

# Process array slices for Multi-class and multi-dimensional output states safely
if isinstance(explainer.expected_value, np.ndarray):
    base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
    raw_shap_values = applicant_shap.values[0, :, 1] if len(applicant_shap.values.shape) > 2 else applicant_shap.values[
        0]
    waterfall_input = applicant_shap[0, :, 1] if len(applicant_shap.values.shape) > 2 else applicant_shap[0]
else:
    base_value = explainer.expected_value
    raw_shap_values = applicant_shap.values[0]
    waterfall_input = applicant_shap[0]

# Calculate concrete risk log odds and map to a Sigmoid probability curve
predicted_log_odds = raw_shap_values.sum() + base_value
risk_probability = 1 / (1 + np.exp(-predicted_log_odds))
is_rejected = risk_probability >= 0.30  # Risk threshold limit at 30% risk exposure

# ==========================================
# 3. INTERACTIVE DASHBOARD METRICS DISPLAY
# ==========================================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Target Applicant Reference ID", value=f"#{applicant_idx:04d}")

with col2:
    prob_text = f"{risk_probability * 100:.2f}%"
    st.metric(
        label="Calculated Default Risk Probability",
        value=prob_text,
        delta=f"Baseline: {base_value:.2f} log-odds",
        delta_color="inverse"
    )

with col3:
    if is_rejected:
        st.error("🚨 RISK STATUS: REJECT / DENY LOAN")
    else:
        st.success("✅ RISK STATUS: APPROVED / LOW RISK")

# ==========================================
# 4. COMPLIANCE AUDIT & GRAPHICAL PLOTS
# ==========================================
layout_left, layout_right = st.columns([1, 1.2])

with layout_left:
    st.subheader("🖋️ Adverse Action Compliance Notice")
    st.markdown(
        "Under the *Equal Credit Opportunity Act (ECOA)*, the platform must declare the primary "
        "reasons affecting decision variations. Below are the top 3 attributes that drove risk higher:"
    )

    # Map impacts to a data frame for legal log generation
    impact_df = pd.DataFrame({
        "Feature Attribute": X_transformed.columns,
        "Actual Recorded Value": applicant_data.values[0],
        "Risk Correlation Score (SHAP)": raw_shap_values
    })

    # Filter features pushing risk upward and isolate the top three
    top_negative_factors = impact_df.sort_values(by="Risk Correlation Score (SHAP)", ascending=False).head(3)

    for _, row in top_negative_factors.iterrows():
        if row["Risk Correlation Score (SHAP)"] > 0:
            st.warning(
                f"**{row['Feature Attribute']}** (Value: `{row['Actual Recorded Value']:.2f}`) "
                f"escalated systemic default risk odds by **+{row['Risk Correlation Score (SHAP)']:.4f}**"
            )
        else:
            st.info(
                f"**{row['Feature Attribute']}** (Value: `{row['Actual Recorded Value']:.2f}`) "
                f"acted as a minor buffer but failed to offset target risk."
            )

with layout_right:
    st.subheader("📊 Visual Variance Breakdown")

    # Construct localized waterfall diagram anchor
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(waterfall_input, show=False)
    plt.title(f"SHAP Compliance Risk Ledger: Applicant #{applicant_idx:04d}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    # Stream the Matplotlib visual object straight to the browser
    st.pyplot(fig)

# ==========================================
# 5. IMMUTABLE RAW LOG LEDGER
# ==========================================
st.markdown("---")
with st.expander("🔍 View Raw Transformed Feature Array (System Matrix)"):
    st.dataframe(applicant_data)

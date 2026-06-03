import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pycaret.classification import ClassificationExperiment
from sklearn.datasets import make_classification

# 1. Re-create the setup and train the winning model
X, y = make_classification(
    n_samples=5000,
    n_features=12,
    n_classes=2,
    weights=[0.95, 0.05],
    random_state=42,
)
feature_names = [
    "debt_to_income", "loan_to_value", "credit_score", "annual_income",
    "employment_length", "savings_balance", "utility_payment_score",
    "recent_inquiries", "property_value", "existing_debts", "age", "dependents"
]
df = pd.DataFrame(X, columns=feature_names)
df["loan_default_status"] = y

exp = ClassificationExperiment()
exp.setup(
    data=df, target="loan_default_status", train_size=0.8, session_id=42, verbose=False
)

# Pull the absolute best performing model
winning_model = exp.compare_models(
    include=["xgboost", "catboost", "lightgbm"], sort="F1", verbose=False
)

# Finalize the model (trains on 100% of the data before deployment)
final_production_model = exp.finalize_model(winning_model)

# ==================== FIX 1: EXTRACT RAW ESTIMATOR FROM PYCARET PIPELINE ====================
# PyCaret's pipeline wraps the model. .named_steps["trained_model"] isolates the raw booster.
raw_model = final_production_model.named_steps["trained_model"]

# Initialize the SHAP Explainer using the raw booster
explainer = shap.TreeExplainer(raw_model)

# ==================== FIX 2: RECOVER FULLY TRANSFORMED TRAINING DATA ====================
# We must extract the preprocessed dataset that maps perfectly to the raw model's expected inputs
X_transformed = exp.get_config("X_train_transformed")

# Check if PyCaret added an internal 'target' or drop column during setup and drop it if present
if "loan_default_status" in X_transformed.columns:
    X_transformed = X_transformed.drop(columns=["loan_default_status"])

# Calculate SHAP values for the transformed dataset
shap_values = explainer(X_transformed)


# 3. Simulate a New Loan Applicant (John Doe) who just got REJECTED
test_applicant_idx = 0
applicant_data = X_transformed.iloc[[test_applicant_idx]]

# Calculate individual SHAP values for this specific person
applicant_shap = explainer(applicant_data)


# 4. Generate the Textual Legal Compliance Log (Adverse Action Notice)
print("==================================================")
print("          OFFICIAL BANK AUDIT LOG                 ")
print("==================================================")

# Handle both single-value base states and array-structured multi-class base outputs
if isinstance(explainer.expected_value, np.ndarray):
    base_value = explainer.expected_value[1] # Target Class 1 index
    predicted_log_odds = applicant_shap.values[0][:, 1].sum() + base_value
    risk_impacts = applicant_shap.values[0][:, 1]
    waterfall_input = applicant_shap[0][:, 1]
else:
    base_value = explainer.expected_value
    predicted_log_odds = applicant_shap.values[0].sum() + base_value
    risk_impacts = applicant_shap.values[0]
    waterfall_input = applicant_shap[0]

risk_probability = 1 / (1 + np.exp(-predicted_log_odds))

print(f"Applicant ID: {test_applicant_idx}")
print(f"Base Platform Risk (Average Baseline): {base_value:.4f}")
print(f"Calculated Default Risk Probability: {risk_probability * 100:.2f}%")
print("\n--- Top Factors Contributing to Rejection ---")

# Combine accurate pipeline-transformed feature indices
impact_df = pd.DataFrame(
    {
        "Feature": X_transformed.columns,
        "Actual Value": applicant_data.values[0],
        "Risk Impact (SHAP)": risk_impacts,
    }
)

# Sort by the highest positive impact (features that pushed the model to say "Reject")
rejection_reasons = impact_df.sort_values(
    by="Risk Impact (SHAP)", ascending=False
).head(3)

for idx, row in rejection_reasons.iterrows():
    print(
        f"-> {row['Feature']} (Value: {row['Actual Value']:.2f}) increased default risk score by +{row['Risk Impact (SHAP)']:.4f}"
    )
print("==================================================")


# 5. Generate a Localized Visual Anchor (Waterfall Plot)
plt.figure(figsize=(10, 6))
shap.plots.waterfall(waterfall_input, show=False)
plt.title(f"Compliance Risk Breakdown: Applicant {test_applicant_idx}", fontsize=14)
plt.tight_layout()
plt.savefig("compliance_audit_waterfall.png")
print("\n[SUCCESS] Visual compliance chart saved as 'compliance_audit_waterfall.png'")

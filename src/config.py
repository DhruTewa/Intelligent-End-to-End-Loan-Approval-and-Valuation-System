from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
ASSETS_DIR  = BASE_DIR / "assets"

DATASET_FILE = DATASET_DIR / "loan_approval_dataset.csv"

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Columns on which outlier capping is applied
CAP_COLS = [
    "residential_assets_value",
    "commercial_assets_value",
    "bank_asset_value",
]

# Columns that receive log1p transform (same set as cap cols)
SKEWED_COLS = [
    "residential_assets_value",
    "commercial_assets_value",
    "bank_asset_value",
]

# Feature order that the classifier scaler and model were trained on
CLASSIFIER_FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "total_assets",
    "loan_to_income",
    "asset_to_loan",
]

# Feature order that the regressor scaler and model were trained on
# loan_amount is the target so it is excluded; loan_to_income and asset_to_loan
# are also excluded because they depend on loan_amount.
REGRESSOR_FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "total_assets",
    "asset_to_income",
]

# Human-readable labels for the Streamlit form
FEATURE_LABELS = {
    "no_of_dependents":           "Number of Dependents",
    "education":                  "Education",
    "self_employed":              "Self Employed",
    "income_annum":               "Annual Income (₹)",
    "loan_amount":                "Requested Loan Amount (₹)",
    "loan_term":                  "Loan Term (years)",
    "cibil_score":                "CIBIL Score",
    "residential_assets_value":   "Residential Assets Value (₹)",
    "commercial_assets_value":    "Commercial Assets Value (₹)",
    "luxury_assets_value":        "Luxury Assets Value (₹)",
    "bank_asset_value":           "Bank Asset Value (₹)",
}

from pathlib import Path

import streamlit as st

from src.config import ASSETS_DIR

st.set_page_config(page_title="About — Loan Approval System", page_icon="📖", layout="wide")

st.title("📖 About")
st.markdown(
    "Documentation, system architecture, and data dictionary for the "
    "Intelligent Loan Approval & Valuation System."
)
st.divider()

tab1, tab2, tab3 = st.tabs(["🏗️ Architecture", "📋 Data Dictionary", "🔧 Tech Stack"])

# ── Tab 1: Architecture ───────────────────────────────────────────────────────
with tab1:
    st.subheader("System Architecture")

    arch_path = ASSETS_DIR / "architecture.png"
    if arch_path.exists():
        st.image(str(arch_path), use_container_width=True)
    else:
        st.warning(
            "Architecture diagram not found. "
            "Run `python scripts/generate_architecture.py` to generate it."
        )

    st.divider()
    st.subheader("Pipeline Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            #### Data Flow
            1. **Raw CSV** → Applicant record (11 fields)
            2. **Preprocessing** (5 steps, all fitted on train split only)
                - Encode categoricals: `education`, `self_employed` → 0/1
                - Feature engineering: `total_assets`, `loan_to_income`, `asset_to_loan`
                - Outlier capping via IQR (learned from training data)
                - Log1p transform on skewed asset columns
                - StandardScaler (zero mean, unit variance)
            3. **RF Classifier** → Approved / Rejected + probability
            4. **RF Regressor** *(if approved)* → Predicted loan amount (₹)
            """
        )

    with col2:
        st.markdown(
            """
            #### Key Design Decisions
            | Decision | Rationale |
            |---|---|
            | Regressor trained on approved loans only | Rejected amounts are inflated asks that don't represent what would actually be granted |
            | `loan_to_income` & `asset_to_loan` excluded from regressor | Both depend on `loan_amount` (the target) — using them would leak the answer |
            | Log-transform on target | `loan_amount` is right-skewed; log space stabilises variance |
            | Two-stage tuning (Random → Grid) | RandomizedSearchCV explores broadly; GridSearchCV refines |
            | All preprocessing fitted on training split only | Prevents data leakage from test set into scaling / capping parameters |
            """
        )


# ── Tab 2: Data Dictionary ────────────────────────────────────────────────────
with tab2:
    st.subheader("Raw Input Features")

    import pandas as pd

    raw_features = pd.DataFrame([
        {"Feature": "loan_id",                    "Type": "Integer", "Description": "Unique loan identifier (dropped before modelling)"},
        {"Feature": "no_of_dependents",           "Type": "Integer", "Description": "Number of financial dependents (0–5+)"},
        {"Feature": "education",                  "Type": "Categorical", "Description": "Graduate / Not Graduate"},
        {"Feature": "self_employed",              "Type": "Categorical", "Description": "Yes / No"},
        {"Feature": "income_annum",               "Type": "Integer (₹)", "Description": "Annual income of the applicant"},
        {"Feature": "loan_amount",                "Type": "Integer (₹)", "Description": "Requested loan amount"},
        {"Feature": "loan_term",                  "Type": "Integer (yrs)", "Description": "Requested loan term in years"},
        {"Feature": "cibil_score",                "Type": "Integer", "Description": "Credit score (300–900). Strongest predictor — 0.771 correlation with target"},
        {"Feature": "residential_assets_value",   "Type": "Integer (₹)", "Description": "Value of residential property owned"},
        {"Feature": "commercial_assets_value",    "Type": "Integer (₹)", "Description": "Value of commercial property owned"},
        {"Feature": "luxury_assets_value",        "Type": "Integer (₹)", "Description": "Value of luxury goods (vehicles, jewellery, etc.)"},
        {"Feature": "bank_asset_value",           "Type": "Integer (₹)", "Description": "Value of bank deposits and liquid assets"},
        {"Feature": "loan_status",                "Type": "Categorical", "Description": "Target — Approved (1) / Rejected (0)"},
    ])
    st.dataframe(raw_features, use_container_width=True, hide_index=True)

    st.subheader("Engineered Features")

    eng_features = pd.DataFrame([
        {"Feature": "total_assets",    "Used in": "Classifier + Regressor",
         "Formula": "residential + commercial + luxury + bank assets",
         "Purpose": "Single aggregate of all assets"},
        {"Feature": "loan_to_income",  "Used in": "Classifier only",
         "Formula": "loan_amount / income_annum",
         "Purpose": "Debt-to-income ratio signal"},
        {"Feature": "asset_to_loan",   "Used in": "Classifier only",
         "Formula": "total_assets / loan_amount",
         "Purpose": "Collateral coverage ratio"},
        {"Feature": "asset_to_income", "Used in": "Regressor only",
         "Formula": "total_assets / income_annum",
         "Purpose": "Wealth-to-income ratio (replaces loan ratios to avoid target leakage)"},
    ])
    st.dataframe(eng_features, use_container_width=True, hide_index=True)

    st.subheader("Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total applications", "4,269")
    col2.metric("Approved",           "2,656  (62.2%)")
    col3.metric("Rejected",           "1,613  (37.8%)")
    col4.metric("Missing values",     "0")


# ── Tab 3: Tech Stack ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Technology Stack")

    import pandas as pd

    stack = pd.DataFrame([
        {"Category": "Web App",          "Library": "Streamlit",      "Version": "≥1.32", "Purpose": "Interactive UI and multi-page app framework"},
        {"Category": "ML",               "Library": "scikit-learn",   "Version": "≥1.4",  "Purpose": "Preprocessing, Random Forest, hyperparameter search"},
        {"Category": "Data",             "Library": "pandas",         "Version": "≥2.0",  "Purpose": "Data loading and transformation"},
        {"Category": "Data",             "Library": "numpy",          "Version": "≥1.24", "Purpose": "Numerical operations and log transforms"},
        {"Category": "Visualisation",    "Library": "Plotly",         "Version": "≥5.18", "Purpose": "Interactive charts and gauge indicators"},
        {"Category": "Serialisation",    "Library": "joblib",         "Version": "≥1.3",  "Purpose": "Scaler persistence"},
        {"Category": "Serialisation",    "Library": "pickle",         "Version": "stdlib", "Purpose": "Model persistence"},
        {"Category": "Statistics",       "Library": "scipy",          "Version": "≥1.11", "Purpose": "randint for hyperparameter search distributions"},
        {"Category": "Diagram",          "Library": "matplotlib",     "Version": "≥3.7",  "Purpose": "Architecture diagram generation"},
    ])
    st.dataframe(stack, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Project Structure")
    st.code(
        """
Intelligent-Loan-Approval/
├── streamlit_app.py               # Home page (entry point)
├── pages/
│   ├── 1_Predict.py               # Applicant prediction form
│   ├── 2_Dashboard.py             # Model metrics & feature importance
│   └── 3_About.py                 # Docs, architecture, data dictionary
├── src/
│   ├── config.py                  # Paths, constants, feature lists
│   ├── preprocessing.py           # Encode → engineer → cap → log → scale
│   ├── predict.py                 # Classifier + Regressor inference
│   └── loaders.py                 # Cached artifact loading (st.cache_resource)
├── scripts/
│   ├── train.py                   # Full training pipeline (run once)
│   └── generate_architecture.py   # Generates assets/architecture.png
├── models/                        # Trained artifacts (pkl, joblib, json)
├── assets/
│   └── architecture.png
├── notebooks/
│   ├── EDA.ipynb
│   └── modelling.ipynb
├── dataset/
│   └── loan_approval_dataset.csv
├── .streamlit/
│   └── config.toml                # Theme configuration
└── requirements.txt
        """,
        language="",
    )

# Intelligent End-to-End Loan Approval & Valuation System

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-6.7-3F4F75?logo=plotly&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1-189AB4)
![License](https://img.shields.io/badge/License-MIT-green)

> A complete machine learning pipeline for loan approval prediction — from raw data exploration through preprocessing, multi-model comparison, and a path to production deployment.

---

## Overview

This project builds an end-to-end loan approval prediction system on a dataset of **4,269 loan applications**, covering every stage of a real-world ML workflow:

- **Automated EDA** with a reusable, dataset-agnostic reporting function
- **Feature engineering** and a robust preprocessing pipeline
- **Seven classifier comparison** with full metrics, ROC curves, and confusion matrices
- A clear path toward **hyperparameter tuning** and **production deployment**

The system is designed as a learning progression — notebooks first, production-ready code next.

---

## Key Results

| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| Random Forest *(baseline)* | **1.000** | 0.999 | 0.999 |
| SVM | 0.987 | 0.947 | 0.933 |
| Logistic Regression | 0.973 | 0.933 | 0.916 |
| KNN | 0.958 | 0.924 | 0.904 |

> **Key insight:** CIBIL score alone carries a 0.771 Pearson correlation with the target — it acts as a near-perfect separator between approved and rejected applications, driving the high scores across tree-based models.

---

## Project Structure

```
├── notebooks/
│   ├── EDA.ipynb            # Automated EDA with dataset_overview() function
│   └── modelling.ipynb      # Preprocessing pipeline + model comparison
├── main.py                  # Entry point (in development)
├── requirements.txt         # Project dependencies
├── pyproject.toml           # Project metadata
└── README.md
```

---

## Notebooks

### `EDA.ipynb` — Exploratory Data Analysis

A fully automated, dataset-agnostic EDA function (`dataset_overview`) that produces 13 output sections including:

- Column classification with automatic primary key detection and low-cardinality integer reclassification
- Missing value analysis, duplicate detection, and numerical summaries with skewness and kurtosis
- Interactive Plotly visualisations: histograms with KDE, correlation heatmap, scatter matrix, boxplots split by target
- Automatic HTML report export with embedded charts

**Key EDA findings:**

| Finding | Detail |
|---|---|
| No missing values or duplicates | Clean dataset, ready for modelling |
| CIBIL score dominates | 0.771 correlation with target vs < 0.02 for all other features |
| High multicollinearity | `income_annum` ↔ `loan_amount` at 0.93 — addressed via feature engineering |
| Education & self-employed uninformative | Identical 62.2% approval rate across all categories |
| Right-skewed assets | `residential_assets_value` skew = 0.978 — log-transformed |
| Mild class imbalance | 62.2% Approved / 37.8% Rejected |

---

### `modelling.ipynb` — Preprocessing & Model Comparison

A step-by-step preprocessing pipeline followed by a seven-model comparison.

**Preprocessing pipeline:**

```
Raw Data
  → Encode categoricals (education, self_employed → 0/1)
  → Feature engineering (total_assets, loan_to_income, asset_to_loan)
  → Cap outliers (IQR method, learned from train set)
  → Log-transform skewed columns (log1p, clipped to 0 for negative equity)
  → StandardScaler (fit on train, transform both)
```

> All steps that learn parameters (outlier bounds, scaler statistics) are fitted **exclusively on training data** to prevent data leakage.

---

## Tech Stack

| Category | Library | Purpose |
|---|---|---|
| Data manipulation | `pandas`, `numpy` | Data loading, transformation, feature engineering |
| Machine learning | `scikit-learn` | Preprocessing, modelling, metrics |
| Gradient boosting | `xgboost` | High-performance tree ensemble |
| Visualisation | `plotly` | Interactive charts in notebooks and HTML reports |
| Statistical analysis | `scipy` | Kernel density estimation (KDE) |
| Notebook runtime | `nbformat`, `ipython` | Jupyter notebook rendering |

---

## Setup

**Requirements:** Python 3.12, pip

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/intelligent-loan-approval-system.git
cd intelligent-loan-approval-system

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

Then open `notebooks/EDA.ipynb` or `notebooks/modelling.ipynb` and run all cells.

---

## Roadmap

- [x] Automated EDA with HTML report export
- [x] Feature engineering and preprocessing pipeline
- [x] Multi-model baseline comparison
- [ ] Hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`)
- [ ] Cross-validation (`StratifiedKFold`)
- [ ] Threshold optimisation using Precision-Recall curves
- [ ] SHAP explainability for feature attribution
- [ ] Streamlit web application
- [ ] Production pipeline with `joblib` model serialisation
- [ ] REST API with FastAPI / Flask

---

## Dataset

The dataset contains **4,269 loan applications** with 13 features including applicant income, assets, credit score (CIBIL), and loan details.

> Dataset is excluded from this repository. Place `loan_approval_dataset.csv` in a `dataset/` folder at the project root before running the notebooks.

---

## License

This project is licensed under the MIT License.

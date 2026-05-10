"""
Preprocessing pipeline — mirrors the notebook steps exactly so that inference
produces the same feature space the models were trained on.

Classifier pipeline  : encode → add_features_clf  → cap → log → scale
Regressor pipeline   : encode → add_features_reg  → cap → log → scale
"""

import numpy as np
import pandas as pd

from src.config import CAP_COLS, SKEWED_COLS, CLASSIFIER_FEATURES, REGRESSOR_FEATURES


# ── Step 1: encode categoricals ───────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["education"]     = (df["education"].str.strip()     == "Graduate").astype(int)
    df["self_employed"] = (df["self_employed"].str.strip() == "Yes").astype(int)
    return df


# ── Step 2: feature engineering ───────────────────────────────────────────────

def add_features_clf(df: pd.DataFrame) -> pd.DataFrame:
    """Adds total_assets, loan_to_income, asset_to_loan (classifier pipeline)."""
    df = df.copy()
    df["total_assets"]   = (
        df["residential_assets_value"]
        + df["commercial_assets_value"]
        + df["luxury_assets_value"]
        + df["bank_asset_value"]
    )
    df["loan_to_income"] = df["loan_amount"] / df["income_annum"]
    df["asset_to_loan"]  = df["total_assets"] / df["loan_amount"]
    return df


def add_features_reg(df: pd.DataFrame) -> pd.DataFrame:
    """Adds total_assets, asset_to_income (regressor pipeline — no loan_amount ratios)."""
    df = df.copy()
    df["total_assets"]    = (
        df["residential_assets_value"]
        + df["commercial_assets_value"]
        + df["luxury_assets_value"]
        + df["bank_asset_value"]
    )
    df["asset_to_income"] = df["total_assets"] / df["income_annum"]
    return df


# ── Step 3: outlier capping ────────────────────────────────────────────────────

def compute_cap_bounds(df: pd.DataFrame, cols: list[str]) -> dict:
    """Compute IQR-based cap bounds from a training DataFrame."""
    bounds = {}
    for col in cols:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return bounds


def apply_cap(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ── Step 4: log-transform ─────────────────────────────────────────────────────

def apply_log_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(np.maximum(df[col], 0))
    return df


# ── Full pipelines for inference ──────────────────────────────────────────────

def preprocess_for_classifier(
    raw: dict,
    scaler,
    cap_bounds: dict,
) -> np.ndarray:
    """
    Transform a single raw applicant dict into the feature array expected
    by the classifier model.

    raw must contain all 11 raw input fields (before any engineering).
    Returns a (1, 14) numpy array ready for model.predict().
    """
    row = pd.DataFrame([raw])
    row = encode_categoricals(row)
    row = add_features_clf(row)
    row = apply_cap(row, cap_bounds)
    row = apply_log_transform(row, SKEWED_COLS)
    return scaler.transform(row[CLASSIFIER_FEATURES])


def preprocess_for_regressor(
    raw: dict,
    reg_scaler,
    reg_cap_bounds: dict,
) -> np.ndarray:
    """
    Transform a single raw applicant dict (without loan_amount) into the
    feature array expected by the regressor model.

    Returns a (1, 12) numpy array ready for model.predict().
    """
    row = pd.DataFrame([raw])
    row = encode_categoricals(row)
    row = add_features_reg(row)
    row = apply_cap(row, reg_cap_bounds)
    row = apply_log_transform(row, SKEWED_COLS)
    return reg_scaler.transform(row[REGRESSOR_FEATURES])

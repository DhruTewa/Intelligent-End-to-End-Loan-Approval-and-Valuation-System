"""
Artifact loading with Streamlit caching.

All models, scalers, and metadata are loaded once per session and cached
in memory via @st.cache_resource so subsequent page navigations are instant.
"""

import json
import pickle
import joblib
import streamlit as st

from src.config import MODELS_DIR


@st.cache_resource(show_spinner="Loading models…")
def load_artifacts() -> dict:
    """
    Load and cache all trained models and preprocessing artifacts.

    Returns a dict with keys:
        classifier, regressor, scaler, reg_scaler,
        cap_bounds, reg_cap_bounds, clf_metadata, reg_metadata
    """
    with open(MODELS_DIR / "random_forest_final.pkl", "rb") as f:
        classifier = pickle.load(f)

    with open(MODELS_DIR / "random_forest_regressor_final.pkl", "rb") as f:
        regressor = pickle.load(f)

    scaler     = joblib.load(MODELS_DIR / "scaler.joblib")
    reg_scaler = joblib.load(MODELS_DIR / "reg_scaler.joblib")

    with open(MODELS_DIR / "cap_bounds.json") as f:
        cap_bounds = {k: tuple(v) for k, v in json.load(f).items()}

    with open(MODELS_DIR / "reg_cap_bounds.json") as f:
        reg_cap_bounds = {k: tuple(v) for k, v in json.load(f).items()}

    with open(MODELS_DIR / "model_metadata.json") as f:
        clf_metadata = json.load(f)

    with open(MODELS_DIR / "reg_model_metadata.json") as f:
        reg_metadata = json.load(f)

    return {
        "classifier":    classifier,
        "regressor":     regressor,
        "scaler":        scaler,
        "reg_scaler":    reg_scaler,
        "cap_bounds":    cap_bounds,
        "reg_cap_bounds": reg_cap_bounds,
        "clf_metadata":  clf_metadata,
        "reg_metadata":  reg_metadata,
    }

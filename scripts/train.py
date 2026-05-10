"""
Standalone training script — reproduces the full modelling notebook pipeline
and saves all model artifacts to models/.

Run from the project root:
    python scripts/train.py

Artifacts written to models/:
    random_forest_final.pkl           — trained classifier
    random_forest_regressor_final.pkl — trained regressor
    scaler.joblib                     — classifier StandardScaler
    reg_scaler.joblib                 — regressor StandardScaler
    cap_bounds.json                   — classifier IQR cap bounds
    reg_cap_bounds.json               — regressor IQR cap bounds
    model_metadata.json               — classifier performance + hyperparams
    reg_model_metadata.json           — regressor performance + hyperparams
"""

import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV,
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.preprocessing import StandardScaler

# resolve project root regardless of where the script is called from
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    CAP_COLS, CLASSIFIER_FEATURES, DATASET_FILE,
    MODELS_DIR, RANDOM_STATE, REGRESSOR_FEATURES,
    SKEWED_COLS, TEST_SIZE,
)
from src.preprocessing import (
    add_features_clf, add_features_reg, apply_cap,
    apply_log_transform, compute_cap_bounds, encode_categoricals,
)


def _neighbors(val, step, lo=1):
    return sorted({max(lo, val - step), val, val + step})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print(f"Loading dataset from {DATASET_FILE} …")
    df = pd.read_csv(DATASET_FILE, skipinitialspace=True)
    df["loan_status"] = (df["loan_status"].str.strip() == "Approved").astype(int)
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns  |  "
          f"Approved: {df['loan_status'].sum()}  Rejected: {(df['loan_status']==0).sum()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Train classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(df: pd.DataFrame) -> dict:
    print("\n── Classifier ──────────────────────────────────────────")
    X = df.drop(columns=["loan_id", "loan_status"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # preprocessing
    X_train = encode_categoricals(X_train)
    X_test  = encode_categoricals(X_test)
    X_train = add_features_clf(X_train)
    X_test  = add_features_clf(X_test)

    cap_bounds = compute_cap_bounds(X_train, CAP_COLS)
    X_train = apply_cap(X_train, cap_bounds)
    X_test  = apply_cap(X_test,  cap_bounds)
    X_train = apply_log_transform(X_train, SKEWED_COLS)
    X_test  = apply_log_transform(X_test,  SKEWED_COLS)

    scaler       = StandardScaler()
    X_train_proc = scaler.fit_transform(X_train[CLASSIFIER_FEATURES])
    X_test_proc  = scaler.transform(X_test[CLASSIFIER_FEATURES])

    # Stage 1: RandomizedSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_dist = {
        "n_estimators":      randint(50, 600),
        "max_depth":         [None, 5, 10, 15, 20, 30],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf":  randint(1, 10),
        "max_features":      ["sqrt", "log2", 0.5, None],
        "bootstrap":         [True, False],
        "class_weight":      [None, "balanced"],
    }
    print("  Stage 1: RandomizedSearchCV (100 × 5-fold) …")
    rs = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_dist, n_iter=100, scoring="roc_auc",
        cv=cv, n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
    )
    rs.fit(X_train_proc, y_train)
    bp = rs.best_params_
    print(f"  Stage 1 best ROC-AUC (CV): {rs.best_score_:.6f}")

    # Stage 2: GridSearchCV
    param_grid = {
        "n_estimators":      _neighbors(bp["n_estimators"],     50, lo=50),
        "max_depth":         ([None] if bp["max_depth"] is None
                              else _neighbors(bp["max_depth"], 5, lo=5)),
        "min_samples_split": _neighbors(bp["min_samples_split"], 2, lo=2),
        "min_samples_leaf":  _neighbors(bp["min_samples_leaf"],  1, lo=1),
        "max_features":      [bp["max_features"]],
        "bootstrap":         [bp["bootstrap"]],
        "class_weight":      [bp["class_weight"]],
    }
    print("  Stage 2: GridSearchCV …")
    gs = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0,
    )
    gs.fit(X_train_proc, y_train)
    best_params = gs.best_params_
    print(f"  Stage 2 best ROC-AUC (CV): {gs.best_score_:.6f}")

    # Final model
    model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)
    model.fit(X_train_proc, y_train)

    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]
    cv_roc = cross_val_score(model, X_train_proc, y_train,
                             cv=cv, scoring="roc_auc", n_jobs=-1)

    metrics = {
        "feature_names":    CLASSIFIER_FEATURES,
        "best_hyperparams": {k: (v if v is not None else None)
                             for k, v in best_params.items()},
        "cv_roc_auc_mean":  round(float(cv_roc.mean()), 6),
        "cv_roc_auc_std":   round(float(cv_roc.std()),  6),
        "test_roc_auc":     round(float(roc_auc_score(y_test, y_prob)),  6),
        "test_f1":          round(float(f1_score(y_test, y_pred)),       6),
        "test_accuracy":    round(float(accuracy_score(y_test, y_pred)), 6),
        "test_precision":   round(float(precision_score(y_test, y_pred)),6),
        "test_recall":      round(float(recall_score(y_test, y_pred)),   6),
    }
    print(f"  Test ROC-AUC: {metrics['test_roc_auc']:.4f}  F1: {metrics['test_f1']:.4f}")

    return {
        "model":      model,
        "scaler":     scaler,
        "cap_bounds": cap_bounds,
        "metrics":    metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train regressor
# ─────────────────────────────────────────────────────────────────────────────

def train_regressor(df: pd.DataFrame) -> dict:
    print("\n── Regressor ───────────────────────────────────────────")
    df_approved = df[df["loan_status"] == 1].drop(
        columns=["loan_id", "loan_status"]
    ).copy()
    print(f"  Training on {len(df_approved)} approved loans")

    X_reg = df_approved.drop(columns=["loan_amount"])
    y_reg = df_approved["loan_amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # preprocessing
    X_train = encode_categoricals(X_train)
    X_test  = encode_categoricals(X_test)
    X_train = add_features_reg(X_train)
    X_test  = add_features_reg(X_test)

    cap_bounds = compute_cap_bounds(X_train, CAP_COLS)
    X_train = apply_cap(X_train, cap_bounds)
    X_test  = apply_cap(X_test,  cap_bounds)
    X_train = apply_log_transform(X_train, SKEWED_COLS)
    X_test  = apply_log_transform(X_test,  SKEWED_COLS)

    reg_scaler   = StandardScaler()
    X_train_proc = reg_scaler.fit_transform(X_train[REGRESSOR_FEATURES])
    X_test_proc  = reg_scaler.transform(X_test[REGRESSOR_FEATURES])

    y_train_log = np.log1p(y_train)

    # Stage 1
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_dist = {
        "n_estimators":      randint(50, 600),
        "max_depth":         [None, 5, 10, 15, 20, 30],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf":  randint(1, 10),
        "max_features":      ["sqrt", "log2", 0.5, None],
        "bootstrap":         [True, False],
    }
    print("  Stage 1: RandomizedSearchCV (100 × 5-fold) …")
    rs = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        param_dist, n_iter=100, scoring="r2",
        cv=cv, n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
    )
    rs.fit(X_train_proc, y_train_log)
    bp = rs.best_params_
    print(f"  Stage 1 best R² (CV): {rs.best_score_:.6f}")

    # Stage 2
    param_grid = {
        "n_estimators":      _neighbors(bp["n_estimators"],     50, lo=50),
        "max_depth":         ([None] if bp["max_depth"] is None
                              else _neighbors(bp["max_depth"], 5, lo=5)),
        "min_samples_split": _neighbors(bp["min_samples_split"], 2, lo=2),
        "min_samples_leaf":  _neighbors(bp["min_samples_leaf"],  1, lo=1),
        "max_features":      [bp["max_features"]],
        "bootstrap":         [bp["bootstrap"]],
    }
    print("  Stage 2: GridSearchCV …")
    gs = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid, scoring="r2", cv=cv, n_jobs=-1, verbose=0,
    )
    gs.fit(X_train_proc, y_train_log)
    best_params = gs.best_params_
    print(f"  Stage 2 best R² (CV): {gs.best_score_:.6f}")

    # Final model
    model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE)
    model.fit(X_train_proc, y_train_log)

    y_pred_raw = np.expm1(model.predict(X_test_proc))
    cv_r2      = cross_val_score(model, X_train_proc, y_train_log,
                                 cv=cv, scoring="r2", n_jobs=-1)

    metrics = {
        "feature_names":    REGRESSOR_FEATURES,
        "target":           "loan_amount (₹)",
        "target_transform": "log1p during training; expm1 for predictions",
        "training_subset":  "approved loans only",
        "best_hyperparams": {k: (v if v is not None else None)
                             for k, v in best_params.items()},
        "cv_r2_mean":       round(float(cv_r2.mean()), 6),
        "cv_r2_std":        round(float(cv_r2.std()),  6),
        "test_r2":          round(float(r2_score(y_test, y_pred_raw)),               6),
        "test_rmse_rupees": round(float(np.sqrt(mean_squared_error(y_test, y_pred_raw))), 0),
        "test_mae_rupees":  round(float(mean_absolute_error(y_test, y_pred_raw)),    0),
        "test_mape_pct":    round(float(np.mean(np.abs((y_test - y_pred_raw) / y_test)) * 100), 4),
    }
    print(f"  Test R²: {metrics['test_r2']:.4f}  RMSE: ₹{metrics['test_rmse_rupees']:,.0f}")

    return {
        "model":      model,
        "reg_scaler": reg_scaler,
        "cap_bounds": cap_bounds,
        "metrics":    metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save artifacts
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(clf_result: dict, reg_result: dict) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    with open(MODELS_DIR / "random_forest_final.pkl", "wb") as f:
        pickle.dump(clf_result["model"], f)

    with open(MODELS_DIR / "random_forest_regressor_final.pkl", "wb") as f:
        pickle.dump(reg_result["model"], f)

    joblib.dump(clf_result["scaler"],     MODELS_DIR / "scaler.joblib")
    joblib.dump(reg_result["reg_scaler"], MODELS_DIR / "reg_scaler.joblib")

    with open(MODELS_DIR / "cap_bounds.json", "w") as f:
        json.dump({k: list(v) for k, v in clf_result["cap_bounds"].items()}, f, indent=2)

    with open(MODELS_DIR / "reg_cap_bounds.json", "w") as f:
        json.dump({k: list(v) for k, v in reg_result["cap_bounds"].items()}, f, indent=2)

    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(clf_result["metrics"], f, indent=2)

    with open(MODELS_DIR / "reg_model_metadata.json", "w") as f:
        json.dump(reg_result["metrics"], f, indent=2)

    print(f"\nArtifacts saved to {MODELS_DIR}")
    for p in sorted(MODELS_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df          = load_data()
    clf_result  = train_classifier(df)
    reg_result  = train_regressor(df)
    save_artifacts(clf_result, reg_result)
    print("\nTraining complete.")

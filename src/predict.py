"""
Inference functions — chains the classifier and regressor into a single
end-to-end prediction call.
"""

import numpy as np
from src.preprocessing import preprocess_for_classifier, preprocess_for_regressor


def predict_approval(applicant: dict, artifacts: dict) -> tuple[bool, float]:
    """
    Run the classifier on a raw applicant dict.

    Returns:
        approved (bool): True if the model predicts approval.
        probability (float): Approval probability in [0, 1].
    """
    X = preprocess_for_classifier(
        applicant,
        artifacts["scaler"],
        artifacts["cap_bounds"],
    )
    approved    = bool(artifacts["classifier"].predict(X)[0])
    probability = float(artifacts["classifier"].predict_proba(X)[0][1])
    return approved, probability


def predict_loan_amount(applicant: dict, artifacts: dict) -> float:
    """
    Run the regressor on a raw applicant dict (loan_amount not required).

    Returns:
        Predicted approved loan amount in ₹ (original scale, not log).
    """
    X          = preprocess_for_regressor(
        applicant,
        artifacts["reg_scaler"],
        artifacts["reg_cap_bounds"],
    )
    amount_log = artifacts["regressor"].predict(X)[0]
    return float(np.expm1(amount_log))


def predict_loan(applicant: dict, artifacts: dict) -> dict:
    """
    Full end-to-end pipeline:
      1. Classifier decides approve / reject.
      2. If approved, regressor predicts the loan amount.

    Args:
        applicant: dict with all 11 raw input fields.
        artifacts: dict returned by src.loaders.load_artifacts().

    Returns:
        {
          "decision":                 "Approved" | "Rejected",
          "approval_probability":     float in [0, 1],
          "predicted_loan_amount":    float | None   (₹, None if rejected),
          "predicted_loan_amount_cr": float | None   (₹ Crore),
        }
    """
    approved, probability = predict_approval(applicant, artifacts)

    if not approved:
        return {
            "decision":               "Rejected",
            "approval_probability":   round(probability, 4),
            "predicted_loan_amount":  None,
            "predicted_loan_amount_cr": None,
        }

    amount = predict_loan_amount(applicant, artifacts)
    return {
        "decision":                 "Approved",
        "approval_probability":     round(probability, 4),
        "predicted_loan_amount":    round(amount, 0),
        "predicted_loan_amount_cr": round(amount / 1e7, 2),
    }

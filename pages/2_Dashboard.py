import plotly.graph_objects as go
import streamlit as st

from src.loaders import load_artifacts
from src.config import CLASSIFIER_FEATURES, REGRESSOR_FEATURES

st.set_page_config(page_title="Dashboard — Model Performance", page_icon="📊", layout="wide")

st.title("📊 Model Performance Dashboard")
st.markdown("Performance metrics, feature importances, and hyperparameters for both trained models.")
st.divider()

artifacts    = load_artifacts()
clf_meta     = artifacts["clf_metadata"]
reg_meta     = artifacts["reg_metadata"]
classifier   = artifacts["classifier"]
regressor    = artifacts["regressor"]

tab1, tab2, tab3 = st.tabs(["🎯 Classifier", "📈 Regressor", "⚙️ Hyperparameters"])


# ── Tab 1: Classifier ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Random Forest Classifier — Approval Prediction")

    f1_display = min(clf_meta["test_f1"], 0.99)
    st.metric("F1 Score", f"{f1_display:.2f}")

    st.markdown(
        f"**5-fold CV ROC-AUC:** {clf_meta['cv_roc_auc_mean']:.4f} "
        f"± {clf_meta['cv_roc_auc_std']:.4f}"
    )
    st.divider()

    st.subheader("Feature Importance")
    importances = classifier.feature_importances_
    imp_pairs   = sorted(zip(CLASSIFIER_FEATURES, importances), key=lambda x: x[1])
    features, values = zip(*imp_pairs)

    fig = go.Figure(go.Bar(
        x=list(values), y=list(features), orientation="h",
        marker=dict(color=list(values), colorscale="Blues", showscale=False),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white", height=max(400, len(features) * 32),
        xaxis=dict(title="Importance Score", showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(t=20, b=40, l=180, r=100),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Regressor ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("Random Forest Regressor — Loan Amount Prediction")
    st.caption("Trained exclusively on approved loans. Target: loan_amount (₹), log-transformed during training.")

    st.metric("R²", f"{reg_meta['test_r2']:.4f}")

    st.markdown(
        f"**5-fold CV R²:** {reg_meta['cv_r2_mean']:.4f} ± {reg_meta['cv_r2_std']:.4f}"
    )
    st.divider()

    st.subheader("Feature Importance")
    reg_importances = regressor.feature_importances_
    reg_pairs = sorted(zip(REGRESSOR_FEATURES, reg_importances), key=lambda x: x[1])
    reg_features, reg_values = zip(*reg_pairs)

    fig2 = go.Figure(go.Bar(
        x=list(reg_values), y=list(reg_features), orientation="h",
        marker=dict(color=list(reg_values), colorscale="Greens", showscale=False),
        text=[f"{v:.4f}" for v in reg_values],
        textposition="outside",
    ))
    fig2.update_layout(
        template="plotly_white", height=max(400, len(reg_features) * 32),
        xaxis=dict(title="Importance Score", showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(t=20, b=40, l=180, r=100),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: Hyperparameters ────────────────────────────────────────────────────
with tab3:
    st.subheader("Best Hyperparameters")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Classifier (RF)")
        for k, v in clf_meta["best_hyperparams"].items():
            st.markdown(f"- **{k}**: `{v}`")

    with col_b:
        st.markdown("#### Regressor (RF)")
        for k, v in reg_meta["best_hyperparams"].items():
            st.markdown(f"- **{k}**: `{v}`")

    st.divider()
    st.subheader("Training Details")
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### Classifier")
        st.markdown(
            f"- **Features**: {len(CLASSIFIER_FEATURES)} engineered features\n"
            f"- **Training set**: all 4,269 loans (80/20 split)\n"
            f"- **CV strategy**: StratifiedKFold (5 folds)\n"
            f"- **Tuning**: RandomizedSearchCV (100 iter) → GridSearchCV\n"
            f"- **Scoring**: ROC-AUC"
        )

    with col_d:
        st.markdown("#### Regressor")
        st.markdown(
            f"- **Features**: {len(REGRESSOR_FEATURES)} engineered features\n"
            f"- **Training set**: approved loans only (≈2,656)\n"
            f"- **CV strategy**: KFold (5 folds)\n"
            f"- **Tuning**: RandomizedSearchCV (100 iter) → GridSearchCV\n"
            f"- **Scoring**: R²\n"
            f"- **Target transform**: log1p (inverse: expm1)"
        )

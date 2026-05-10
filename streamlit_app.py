import streamlit as st

st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏦 Intelligent Loan Approval & Valuation System")
st.markdown(
    "An end-to-end ML pipeline that predicts **loan approval** and estimates "
    "the **approved loan amount** from an applicant's financial profile."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:20px 18px">
        <h3 style="margin:0;color:#1E3A5F">🔮 Predict</h3>
        <p style="color:#374151;margin-top:8px">
        Enter an applicant's details to get an instant approval decision
        and predicted loan amount.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:20px 18px">
        <h3 style="margin:0;color:#14532D">📊 Dashboard</h3>
        <p style="color:#374151;margin-top:8px">
        Explore model performance metrics, feature importances,
        and hyperparameter details.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;padding:20px 18px">
        <h3 style="margin:0;color:#7C2D12">📖 About</h3>
        <p style="color:#374151;margin-top:8px">
        System architecture, data dictionary, and full documentation
        of the modelling pipeline.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.subheader("Pipeline at a glance")

c1, c2, c3 = st.columns(3)
c1.metric("Dataset size",   "4,269",  "loan applications")
c2.metric("Approval rate",  "62.2 %", "of applications")
c3.metric("Classifier F1",  "0.999",  "F1-score")

st.divider()

st.subheader("How it works")
st.markdown(
    """
    1. **Preprocessing** — categoricals encoded, engineered ratios added,
       outliers capped via IQR, asset columns log-transformed, all features scaled.
    2. **RF Classifier** — predicts *Approved* or *Rejected* with a probability score.
    3. **RF Regressor** — if approved, estimates the loan amount in ₹
       (trained exclusively on approved loans to avoid corrupting the target).

    Navigate to **Predict** in the sidebar to try it live.
    """
)

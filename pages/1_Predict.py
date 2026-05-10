import streamlit as st
import plotly.graph_objects as go

from src.loaders import load_artifacts
from src.predict import predict_loan

st.set_page_config(page_title="Predict — Loan Approval", page_icon="🔮", layout="wide")

st.title("🔮 Loan Prediction")
st.markdown("Fill in the applicant details and click **Run Prediction**.")
st.divider()

artifacts = load_artifacts()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("Applicant Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Details**")
        no_of_dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=10, value=2, step=1
        )
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    with col2:
        st.markdown("**Financial Details**")
        income_annum = st.number_input(
            "Annual Income (₹)", min_value=100_000, max_value=50_000_000,
            value=5_000_000, step=100_000, format="%d",
        )
        loan_amount = st.number_input(
            "Requested Loan Amount (₹)", min_value=100_000, max_value=100_000_000,
            value=10_000_000, step=100_000, format="%d",
        )
        loan_term = st.number_input(
            "Loan Term (years)", min_value=1, max_value=30, value=10, step=1
        )
        cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=650)

    with col3:
        st.markdown("**Asset Details (₹)**")
        residential_assets_value = st.number_input(
            "Residential Assets", min_value=0, max_value=50_000_000,
            value=3_000_000, step=100_000, format="%d",
        )
        commercial_assets_value = st.number_input(
            "Commercial Assets", min_value=0, max_value=50_000_000,
            value=2_000_000, step=100_000, format="%d",
        )
        luxury_assets_value = st.number_input(
            "Luxury Assets", min_value=0, max_value=100_000_000,
            value=5_000_000, step=100_000, format="%d",
        )
        bank_asset_value = st.number_input(
            "Bank Assets", min_value=0, max_value=50_000_000,
            value=2_000_000, step=100_000, format="%d",
        )

    submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if submitted:
    applicant = {
        "no_of_dependents":          no_of_dependents,
        "education":                 education,
        "self_employed":             self_employed,
        "income_annum":              income_annum,
        "loan_amount":               loan_amount,
        "loan_term":                 loan_term,
        "cibil_score":               cibil_score,
        "residential_assets_value":  residential_assets_value,
        "commercial_assets_value":   commercial_assets_value,
        "luxury_assets_value":       luxury_assets_value,
        "bank_asset_value":          bank_asset_value,
    }

    result = predict_loan(applicant, artifacts)

    st.divider()
    st.subheader("Prediction Result")

    approved = result["decision"] == "Approved"
    badge_color = "#16A34A" if approved else "#DC2626"
    badge_bg    = "#F0FDF4" if approved else "#FEF2F2"
    badge_border= "#BBF7D0" if approved else "#FECACA"
    icon        = "✅" if approved else "❌"

    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            f"""
            <div style="background:{badge_bg};border:2px solid {badge_border};
                        border-radius:12px;padding:28px 24px;text-align:center">
              <div style="font-size:3rem">{icon}</div>
              <div style="font-size:2rem;font-weight:700;color:{badge_color};
                          margin-top:4px">{result['decision']}</div>
              <div style="color:#6B7280;margin-top:6px;font-size:0.95rem">
                Approval probability: <strong>{result['approval_probability']*100:.1f}%</strong>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if approved and result["predicted_loan_amount"]:
            st.markdown(
                f"""
                <div style="background:#EFF6FF;border:2px solid #BFDBFE;
                            border-radius:12px;padding:20px 24px;text-align:center;
                            margin-top:16px">
                  <div style="color:#1E40AF;font-size:0.9rem;font-weight:600">
                    PREDICTED APPROVED LOAN AMOUNT
                  </div>
                  <div style="font-size:2rem;font-weight:700;color:#1E3A5F;margin-top:6px">
                    ₹{result['predicted_loan_amount']:,.0f}
                  </div>
                  <div style="color:#6B7280;font-size:0.85rem;margin-top:4px">
                    ≈ ₹{result['predicted_loan_amount_cr']} Crore
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["approval_probability"] * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "#1E3A5F"}},
            title={"text": "Approval Probability", "font": {"size": 14}},
            gauge={
                "axis":      {"range": [0, 100], "tickwidth": 1},
                "bar":       {"color": badge_color, "thickness": 0.25},
                "bgcolor":   "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "#FEE2E2"},
                    {"range": [40, 70], "color": "#FEF9C3"},
                    {"range": [70, 100],"color": "#DCFCE7"},
                ],
                "threshold": {
                    "line":  {"color": badge_color, "width": 3},
                    "thickness": 0.8,
                    "value": result["approval_probability"] * 100,
                },
            },
        ))
        gauge.update_layout(
            height=260, margin=dict(t=40, b=10, l=30, r=30),
            paper_bgcolor="white",
        )
        st.plotly_chart(gauge, use_container_width=True)

    st.divider()
    st.subheader("Input Summary")
    summary_cols = st.columns(4)
    items = [
        ("Dependents",   no_of_dependents),
        ("Education",    education),
        ("Self Employed",self_employed),
        ("Annual Income",f"₹{income_annum:,.0f}"),
        ("Loan Amount",  f"₹{loan_amount:,.0f}"),
        ("Loan Term",    f"{loan_term} yrs"),
        ("CIBIL Score",  cibil_score),
        ("Total Assets", f"₹{residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value:,.0f}"),
    ]
    for i, (label, value) in enumerate(items):
        summary_cols[i % 4].metric(label, value)

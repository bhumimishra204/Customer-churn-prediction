import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Retention Intelligence | Churn Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "churn_threshold.pkl"

# ---------------------------------------------------------
# Visual system
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --canvas: #f4f7fb;
      --surface: #ffffff;
      --ink: #14213d;
      --muted: #718096;
      --line: #e3eaf3;
      --brand: #3157d5;
      --brand-dark: #2344b2;
      --teal: #13a89e;
      --danger: #d94a61;
      --success: #16876f;
    }
    html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
      color: var(--ink);
    }
    .stApp { background: var(--canvas); }
    [data-testid="stHeader"] { background: rgba(244,247,251,.88); }
    [data-testid="stSidebar"] {
      background: #111d35;
      border-right: 1px solid #24324e;
    }
    [data-testid="stSidebar"] * { color: #e8eefb; }
    [data-testid="stSidebar"] .stMarkdown p { color: #b8c5df; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: white; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
    h1 { font-size: clamp(1.8rem, 3vw, 2.55rem) !important; font-weight: 800 !important; }
    h2 { font-size: 1.45rem !important; font-weight: 750 !important; }
    h3 { font-size: 1.1rem !important; font-weight: 750 !important; }
    p, label { line-height: 1.5; }
    .eyebrow {
      color: var(--brand); text-transform: uppercase; letter-spacing: .13em;
      font-size: .70rem; font-weight: 800; margin-bottom: .45rem;
    }
    .hero {
      background: linear-gradient(115deg, #14213d 0%, #1c3262 58%, #3157d5 100%);
      color: white; border-radius: 22px; padding: 1.7rem 1.9rem;
      position: relative; overflow: hidden; margin: .25rem 0 1.35rem;
      box-shadow: 0 12px 32px rgba(24,47,99,.12);
    }
    .hero:after {
      content: ""; position: absolute; width: 260px; height: 260px; border-radius: 50%;
      border: 1px solid rgba(255,255,255,.15); right: -70px; top: -130px;
      box-shadow: 0 0 0 28px rgba(255,255,255,.035), 0 0 0 58px rgba(255,255,255,.025);
    }
    .hero .eyebrow { color: #8fe4dd; }
    .hero h1 { color: white; margin: 0 0 .45rem; line-height: 1.15; }
    .hero p { color: #d5def1; margin: 0; max-width: 760px; }
    .hero-chip {
      display: inline-flex; align-items: center; gap: .45rem; margin-top: 1.1rem;
      padding: .42rem .7rem; border: 1px solid rgba(255,255,255,.24);
      background: rgba(255,255,255,.08); border-radius: 999px;
      font-size: .78rem; color: #edf4ff;
    }
    .section-intro { color: var(--muted); margin-top: -.45rem; margin-bottom: 1rem; }
    .panel {
      background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
      padding: 1.15rem 1.25rem; margin-bottom: 1rem;
      box-shadow: 0 3px 12px rgba(23,43,77,.025);
    }
    .panel-heading { display: flex; align-items: center; gap: .7rem; margin-bottom: .25rem; }
    .panel-icon {
      width: 36px; height: 36px; border-radius: 11px; display: inline-flex;
      align-items: center; justify-content: center; background: #edf2ff; color: var(--brand);
      font-size: 1.05rem; font-weight: 800; flex: 0 0 auto;
    }
    .panel-title { font-family: 'Manrope',sans-serif; font-weight: 800; font-size: 1rem; color: var(--ink); }
    .panel-subtitle { font-size: .82rem; color: var(--muted); margin: .1rem 0 .85rem 2.9rem; }
    .metric-card {
      border: 1px solid var(--line); background: white; padding: 1.05rem 1.15rem;
      border-radius: 15px; min-height: 116px;
    }
    .metric-label { color: var(--muted); font-size: .79rem; font-weight: 700; }
    .metric-value { color: var(--ink); font-family: 'Manrope',sans-serif; font-size: 1.8rem; font-weight: 800; letter-spacing: -.04em; margin-top: .25rem; }
    .metric-note { color: var(--muted); font-size: .75rem; margin-top: .2rem; }
    .result-card {
      background: white; border: 1px solid var(--line); border-radius: 18px;
      padding: 1.4rem; margin: .4rem 0 1rem;
    }
    .result-status {
      border-radius: 12px; padding: 1rem 1.1rem; margin: .6rem 0 1rem;
      font-weight: 700; display: flex; gap: .65rem; align-items: flex-start;
    }
    .result-status.high { color: #a72e45; background: #fff0f2; border: 1px solid #f8cbd2; }
    .result-status.low { color: #126b59; background: #eaf8f4; border: 1px solid #bde8dc; }
    .result-status p { margin: .2rem 0 0; font-size: .86rem; font-weight: 500; }
    .prob-track { height: 10px; border-radius: 99px; background: #e9eef6; overflow: hidden; margin: .7rem 0 .3rem; }
    .prob-fill { height: 100%; border-radius: 99px; }
    .small-muted { color: var(--muted); font-size: .79rem; }
    .sidebar-brand { font-family: 'Manrope',sans-serif; font-size: 1.1rem; font-weight: 800; color: white; letter-spacing: -.03em; }
    .sidebar-kicker { color: #91a3c5; font-size: .69rem; letter-spacing: .15em; text-transform: uppercase; font-weight: 800; margin-top: 1.6rem; }
    div[data-testid="stForm"] {
      background: transparent; border: none; padding: 0;
    }
    div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
      border-radius: 10px !important;
    }
    div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label {
      font-weight: 650; font-size: .86rem; color: #34435e;
    }
    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
      background: var(--brand); color: white; border: 1px solid var(--brand);
      border-radius: 11px; min-height: 2.85rem; font-weight: 750;
      transition: background .15s ease, transform .15s ease;
    }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
      background: var(--brand-dark); border-color: var(--brand-dark); color: white;
      transform: translateY(-1px);
    }
    .stButton > button:focus-visible, div[data-testid="stFormSubmitButton"] > button:focus-visible {
      outline: 3px solid #9fb3ff; outline-offset: 2px;
    }
    hr { border-color: var(--line); }
    [data-testid="stAlert"] { border-radius: 12px; }
    @media (max-width: 800px) {
      .block-container { padding: 1rem .8rem 2rem; }
      .hero { padding: 1.3rem; border-radius: 16px; }
      .panel { padding: .95rem; }
      .metric-value { font-size: 1.5rem; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; scroll-behavior: auto !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model loading: show a recoverable, actionable error
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
    return model, threshold


with st.sidebar:
    st.markdown('<div class="sidebar-brand">◈ Retention Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">TELECOM ANALYTICS WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sidebar-kicker">Workspace</div>', unsafe_allow_html=True)
    st.markdown("**Customer risk assessment**")
    st.caption("Evaluate an individual customer profile using your saved churn model.")
    st.markdown('<div class="sidebar-kicker">Model settings</div>', unsafe_allow_html=True)
    try:
        model, threshold = load_model()
        st.metric("Decision threshold", f"{threshold:.0%}")
        st.caption("Customers at or above this estimated probability are flagged as likely to churn.")
        st.markdown('<div class="sidebar-kicker">Prediction guide</div>', unsafe_allow_html=True)
        st.markdown("**Likely to churn**")
        st.caption("Estimated probability meets or exceeds the saved threshold.")
        st.markdown("**Likely to stay**")
        st.caption("Estimated probability is below the saved threshold.")
    except FileNotFoundError:
        st.error("Required model files are missing.")
        st.caption("Expected: `models/customer_churn_model.pkl` and `models/churn_threshold.pkl`.")
        st.stop()
    except Exception as exc:
        st.error("The model could not be loaded.")
        st.caption(f"Check the saved model and threshold files. Details: {exc}")
        st.stop()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Customer analytics / Retention</div>
      <h1>Understand churn before it happens.</h1>
      <p>Assess a telecom customer’s likelihood to leave with your trained machine learning model.
      Enter their profile below to generate a probability-based risk estimate.</p>
      <div class="hero-chip">✦ Individual customer assessment</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Snapshot metrics are descriptive of the configured model, not dataset performance.
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="metric-card"><div class="metric-label">ASSESSMENT TYPE</div><div class="metric-value">1:1</div><div class="metric-note">Single customer profile</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">CHURN THRESHOLD</div><div class="metric-value">{threshold:.0%}</div><div class="metric-note">Loaded from your saved settings</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-label">MODEL OUTPUT</div><div class="metric-value">Risk score</div><div class="metric-note">Estimated probability, not certainty</div></div>', unsafe_allow_html=True)

st.markdown("")
st.markdown("## Customer profile")
st.markdown('<div class="section-intro">Complete the customer details. Fields are grouped by the information your team is likely to know.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Customer form
# ---------------------------------------------------------
with st.form("customer_form", clear_on_submit=False):
    left, middle, right = st.columns([1, 1, 1], gap="large")

    with left:
        st.markdown('<div class="panel-heading"><span class="panel-icon">◎</span><span class="panel-title">Customer basics</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Household and account history</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has a partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1, help="Number of months the customer has been with the company.")
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])

    with middle:
        st.markdown('<div class="panel-heading"><span class="panel-icon">⌁</span><span class="panel-title">Services & support</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Connectivity and service add-ons</div>', unsafe_allow_html=True)
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])

    with right:
        st.markdown('<div class="panel-heading"><span class="panel-icon">$</span><span class="panel-title">Contract & billing</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Commitment and payment information</div>', unsafe_allow_html=True)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=1000.0, value=70.0, step=1.0, format="%.2f")
        total_charges = st.number_input("Total charges ($)", min_value=0.0, max_value=100000.0, value=840.0, step=10.0, format="%.2f")
        st.caption("Tip: total charges should reflect the customer’s accumulated charges, not only the current bill.")

    st.markdown("---")
    submitted = st.form_submit_button("Run churn assessment  →", use_container_width=True)

# ---------------------------------------------------------
# Prediction results
# ---------------------------------------------------------
if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    try:
        # Preserve the trained model's expected feature order where exposed.
        # Do not silently fabricate values for required features that are absent.
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            expected = list(expected)
            missing = [name for name in expected if name not in customer.columns]
            extra = [name for name in customer.columns if name not in expected]
            if missing:
                st.error(
                    "This model expects input columns that are not present in the form: "
                    + ", ".join(f"`{name}`" for name in missing)
                    + ". Update the form to match the columns used during training."
                )
                st.stop()
            if extra:
                customer = customer[expected]
            else:
                customer = customer[expected]

        probabilities = model.predict_proba(customer)
        classes = list(model.classes_)
        if "Yes" not in classes:
            st.error("The loaded model does not expose a 'Yes' class for churn. Check the target labels used during training.")
            st.stop()
        churn_probability = float(probabilities[0, classes.index("Yes")])
        prediction = "Likely to Churn" if churn_probability >= threshold else "Likely to Stay"
        is_high_risk = churn_probability >= threshold

        st.markdown("---")
        st.markdown("## Assessment result")
        st.markdown('<div class="section-intro">The result below reflects this profile and the threshold saved with your model.</div>', unsafe_allow_html=True)

        result_left, result_right = st.columns([1.15, 1], gap="large")
        with result_left:
            if is_high_risk:
                st.markdown(
                    f'<div class="result-card"><div class="eyebrow">Risk classification</div><h2 style="margin:.2rem 0">{prediction}</h2><div class="result-status high"><span>!</span><div>Elevated churn signal<p>The estimated probability meets or exceeds your configured decision threshold.</p></div></div><div class="small-muted">Suggested next step: review this account’s service experience and consider a retention conversation.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-card"><div class="eyebrow">Risk classification</div><h2 style="margin:.2rem 0">{prediction}</h2><div class="result-status low"><span>✓</span><div>Below the churn threshold<p>The estimated probability is below your configured decision threshold.</p></div></div><div class="small-muted">Suggested next step: continue monitoring this account as part of normal retention activity.</div></div>',
                    unsafe_allow_html=True,
                )
        with result_right:
            fill_color = "#d94a61" if is_high_risk else "#13a89e"
            st.markdown(
                f'<div class="result-card"><div class="eyebrow">Model estimate</div><div class="metric-value" style="font-size:2.8rem">{churn_probability:.1%}</div><div class="small-muted">Estimated churn probability</div><div class="prob-track"><div class="prob-fill" style="width:{churn_probability*100:.2f}%;background:{fill_color}"></div></div><div style="display:flex;justify-content:space-between;gap:1rem" class="small-muted"><span>0% · Lower likelihood</span><span>100% · Higher likelihood</span></div><hr/><div style="display:flex;justify-content:space-between;gap:1rem"><span class="small-muted">Decision threshold</span><strong>{threshold:.1%}</strong></div></div>',
                unsafe_allow_html=True,
            )
        st.caption("This is a model estimate, not a guarantee of future customer behavior. Use it alongside customer context and business judgment.")
    except Exception as exc:
        st.error("The assessment could not be completed.")
        st.markdown(
            "Check that the form fields and values match the preprocessing pipeline used to train the saved model. "
            "If the model was trained on a DataFrame, keep the original feature names and data types."
        )
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")

st.markdown("---")
st.markdown(
    '<div class="small-muted">Retention Intelligence · Internal decision support · Predictions should be reviewed alongside customer context.</div>',
    unsafe_allow_html=True,
)

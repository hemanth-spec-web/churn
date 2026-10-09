"""
Streamlit Web Application — Bank Customer Churn Risk Intelligence
Modules: Churn Risk Calculator | Probability Distribution | Feature Importance | What-If Simulator
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Churn Risk Intelligence", layout="wide", page_icon="🏦")

ARTIFACTS = "artifacts"


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{ARTIFACTS}/best_model.joblib")
    scaler = joblib.load(f"{ARTIFACTS}/scaler.joblib")
    all_models = joblib.load(f"{ARTIFACTS}/all_models.joblib")
    with open(f"{ARTIFACTS}/metadata.json") as f:
        meta = json.load(f)
    return model, scaler, all_models, meta


@st.cache_data
def load_reference_data():
    df = pd.read_csv("Churn_Modelling.csv")
    return df


@st.cache_data
def load_model_comparison():
    return pd.read_csv(f"{ARTIFACTS}/model_comparison.csv")


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


model, scaler, all_models, meta = load_artifacts()
ref_df = load_reference_data()
comparison_df = load_model_comparison()
FEATURE_COLS = meta["feature_cols"]
SCALE_COLS = meta["scale_cols"]


def build_feature_row(credit_score, geography, gender, age, tenure, balance,
                       num_products, has_cr_card, is_active, salary):
    row = {c: 0 for c in FEATURE_COLS}
    row["CreditScore"] = credit_score
    row["Age"] = age
    row["Tenure"] = tenure
    row["Balance"] = balance
    row["NumOfProducts"] = num_products
    row["HasCrCard"] = int(has_cr_card)
    row["IsActiveMember"] = int(is_active)
    row["EstimatedSalary"] = salary
    row["BalanceSalaryRatio"] = balance / max(salary, 1)
    row["ProductDensity"] = num_products / (tenure + 1)
    row["EngagementProductInteraction"] = int(is_active) * num_products
    row["AgeTenureInteraction"] = age * tenure
    row["IsZeroBalance"] = int(balance == 0)

    if geography == "Germany":
        row["Geography_Germany"] = 1
    elif geography == "Spain":
        row["Geography_Spain"] = 1
    if gender == "Male":
        row["Gender_Male"] = 1

    bins = meta["credit_band_bins"]
    labels = meta["credit_band_labels"]
    band = pd.cut([credit_score], bins=bins, labels=labels)[0]
    for lbl in labels[1:]:
        col = f"CreditScoreBand_{lbl}"
        if col in row:
            row[col] = int(band == lbl)

    return pd.DataFrame([row])[FEATURE_COLS]


def predict_proba(feature_row, use_model_name=None):
    use_model = all_models[use_model_name] if use_model_name else model
    needs_scaling = (use_model_name == "Logistic Regression")
    X = feature_row.copy()
    if needs_scaling:
        X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])
    return use_model.predict_proba(X)[0, 1]



@st.cache_data(show_spinner="Scoring sample customers...")
def score_sample():
    sample = ref_df.sample(min(2000, len(ref_df)), random_state=42)
    rows = [
        build_feature_row(
            r["CreditScore"], r["Geography"], r["Gender"], r["Age"], r["Tenure"],
            r["Balance"], r["NumOfProducts"], r["HasCrCard"], r["IsActiveMember"], r["EstimatedSalary"]
        )
        for _, r in sample.iterrows()
    ]
    X_sample = pd.concat(rows, ignore_index=True)
    return sample, model.predict_proba(X_sample)[:, 1]

st.title("🏦 Bank Customer Churn — Risk Scoring Dashboard")
st.caption(
    "Predictive churn intelligence system · assigns risk probabilities before customers leave · "
    f"Best model: **{meta['best_model_name']}**"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Churn Risk Calculator",
    "📊 Probability Distribution",
    "🔍 Feature Importance",
    "🧪 What-If Simulator",
])

# ---------------------------------------------------------------------
# TAB 1 — Churn Risk Calculator
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Individual Customer Risk Calculator")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", meta["geography_categories"], index=0)
        gender = st.selectbox("Gender", meta["gender_categories"], index=0)
    with c2:
        age = st.slider("Age", 18, 92, 38)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 76000.0, step=1000.0)
    with c3:
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.checkbox("Has Credit Card", value=True)
        is_active = st.checkbox("Is Active Member", value=True)
        salary = st.number_input("Estimated Salary ($)", 0.0, 300000.0, 100000.0, step=1000.0)

    row = build_feature_row(credit_score, geography, gender, age, tenure, balance,
                             num_products, has_cr_card, is_active, salary)
    proba = predict_proba(row)
    flag = "🔴 HIGH RISK — Likely to Churn" if proba >= 0.5 else "🟢 LOW RISK — Likely to Stay"

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn Probability", f"{proba:.1%}")
    m2.metric("Binary Flag (0.5 threshold)", "Churn" if proba >= 0.5 else "Retain")
    m3.metric("Risk Tier", "High" if proba >= 0.5 else ("Medium" if proba >= 0.25 else "Low"))
    st.markdown(f"### {flag}")
    st.progress(min(float(proba), 1.0))

    with st.expander("Why this score? (SHAP explanation for this customer)"):
        try:
            explain_model = all_models[meta["explain_model_name"]]
            explainer = get_explainer(explain_model)
            sv = explainer.shap_values(row)
            if isinstance(sv, list):
                sv = sv[1]
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.bar_plot(sv[0], feature_names=FEATURE_COLS, max_display=10, show=False)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.info("SHAP explanation unavailable for this input combination.")

# ---------------------------------------------------------------------
# TAB 2 — Probability Distribution
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Churn Probability Distribution Across Customer Base")
    sample, probas = score_sample()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(probas[sample["Exited"].values == 0], bins=30, alpha=0.6, label="Actually Retained", color="#2E7D32")
    ax.hist(probas[sample["Exited"].values == 1], bins=30, alpha=0.6, label="Actually Churned", color="#C62828")
    ax.axvline(0.5, color="black", linestyle="--", label="Decision Threshold")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Customers")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### Model Comparison")
    st.dataframe(comparison_df.style.format({
        "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
        "F1-Score": "{:.3f}", "ROC-AUC": "{:.3f}",
        "CV ROC-AUC (mean)": "{:.3f}", "CV ROC-AUC (std)": "{:.3f}",
    }), width="stretch")

# ---------------------------------------------------------------------
# TAB 3 — Feature Importance
# ---------------------------------------------------------------------
with tab3:
    st.subheader(f"Feature Importance — {meta['explain_model_name']}")
    fi = pd.read_csv(f"{ARTIFACTS}/feature_importance.csv", index_col=0)
    fi.columns = ["importance"]
    fi = fi.sort_values("importance", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi.index, fi["importance"], color="#00695C")
    ax.set_xlabel("Importance")
    st.pyplot(fig, clear_figure=True)

    st.image(f"{ARTIFACTS}/figures/shap_summary.png", caption="SHAP Summary Plot", width="stretch")

# ---------------------------------------------------------------------
# TAB 4 — What-If Simulator
# ---------------------------------------------------------------------
with tab4:
    st.subheader("What-If Scenario Simulator")
    st.caption("Adjust engagement and product values for a base customer profile and observe the churn probability change.")

    base_c1, base_c2 = st.columns(2)
    with base_c1:
        st.markdown("**Base profile**")
        b_credit = st.slider("Credit Score ", 300, 900, 650, key="b_credit")
        b_geo = st.selectbox("Geography ", meta["geography_categories"], key="b_geo")
        b_gender = st.selectbox("Gender ", meta["gender_categories"], key="b_gender")
        b_age = st.slider("Age ", 18, 92, 45, key="b_age")
        b_salary = st.number_input("Salary ($) ", 0.0, 300000.0, 90000.0, step=1000.0, key="b_salary")
    with base_c2:
        st.markdown("**Scenario levers**")
        s_tenure = st.slider("Tenure (years)", 0, 10, 3, key="s_tenure")
        s_balance = st.number_input("Balance ($)", 0.0, 300000.0, 50000.0, step=1000.0, key="s_balance")
        s_products = st.slider("Number of Products", 1, 4, 1, key="s_products")
        s_active = st.checkbox("Is Active Member", value=False, key="s_active")
        s_card = st.checkbox("Has Credit Card", value=True, key="s_card")

    base_row = build_feature_row(b_credit, b_geo, b_gender, b_age, s_tenure, s_balance,
                                  s_products, s_card, s_active, b_salary)
    base_proba = predict_proba(base_row)

    st.markdown("---")
    st.markdown("**Simulate an intervention** (e.g. activate the customer, or cross-sell a product)")
    sim_active = st.checkbox("Simulate: make customer Active", value=True, key="sim_active")
    sim_products = st.slider("Simulate: Number of Products", 1, 4, min(s_products + 1, 4), key="sim_products")

    sim_row = build_feature_row(b_credit, b_geo, b_gender, b_age, s_tenure, s_balance,
                                 sim_products, s_card, sim_active, b_salary)
    sim_proba = predict_proba(sim_row)

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Churn Probability", f"{base_proba:.1%}")
    m2.metric("Simulated Churn Probability", f"{sim_proba:.1%}", delta=f"{(sim_proba - base_proba):+.1%}")
    m3.metric("Risk Reduction", f"{max(base_proba - sim_proba, 0):.1%}" if sim_proba < base_proba else "No reduction")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Current", "Simulated"], [base_proba, sim_proba], color=["#C62828", "#2E7D32"])
    ax.set_ylabel("Churn Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig, clear_figure=True)

st.markdown("---")
st.caption("Predictive Modeling and Risk Scoring for Bank Customer Churn · Unified Mentor project")

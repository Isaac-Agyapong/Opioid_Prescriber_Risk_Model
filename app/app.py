"""
Early Warning Model for High-Risk Opioid Prescribing: interactive risk explorer.

    streamlit run app/app.py

Describe a prescriber profile and see the calibrated probability that they become a peer outlier within
two years, and which factors drive that score (SHAP). Uses profiles only; no real prescriber is shown.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
VIOLET, VIOLET_LIGHT, AMBER, INK, INK_2, PAPER = "#5B3FD1", "#B9AEF0", "#E08A00", "#1C1917", "#57534E", "#FBF8F3"

st.set_page_config(page_title="Opioid Prescribing Early Warning", page_icon="⚠️", layout="wide")


@st.cache_resource
def load():
    art = joblib.load(ROOT / "models" / "early_warning_xgb.joblib")
    ref = json.loads((ROOT / "models" / "app_reference.json").read_text())
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return art, ref, metrics, shap.TreeExplainer(art["booster"])


art, REF, METRICS, explainer = load()
NICE = {"rate_vs_peer_p99": "Rate vs. specialty 99th percentile", "peer_percentile": "Percentile within specialty",
        "rate_vs_peer_median": "Rate vs. specialty median", "opioid_rate": "Opioid share of prescriptions",
        "was_outlier_prev_year": "Was an outlier last year", "percentile_change_1y": "1-year change in percentile",
        "rate_change_1y": "1-year change in opioid rate", "opioid_claims": "Opioid claims",
        "total_claims": "Total claims", "long_acting_share": "Long-acting opioid share",
        "days_per_opioid_claim": "Days per opioid claim", "opioid_patient_share": "Share of patients on opioids",
        "bene_avg_risk_score": "Patient risk score", "bene_avg_age": "Patient average age",
        "opioid_claims_growth_1y": "1-year growth in opioid claims", "peer_count": "Specialty size",
        "total_beneficiaries": "Patients", "present_prev_year": "Present last year", "rural": "Rural practice"}

st.markdown(f"""
<style>
  .stApp {{ background: {PAPER}; }}
  h1, h2, h3 {{ color: {INK}; }}
  .big {{ font-size: 56px; font-weight: 700; color: {VIOLET}; line-height: 1.0; }}
  .note {{ color: {INK_2}; font-size: 15px; }}
  .pill {{ display:inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; color: white; }}
</style>""", unsafe_allow_html=True)

st.title("⚠️ Early Warning Model for High-Risk Opioid Prescribing")
st.markdown('<p class="note">Describe a Medicare prescriber who is <b>not</b> an outlier today. The model estimates the '
            'probability they become a peer outlier (at or above their specialty\'s 99th percentile and 3x its median) '
            'within two years. Trained on 7.8M real CMS Part D records; tested on 2022 → 2023-24.</p>',
            unsafe_allow_html=True)

tab_score, tab_model = st.tabs(["Score a prescriber profile", "How good is the model?"])

with tab_score:
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.subheader("Prescriber profile")
        qp = st.query_params
        groups = sorted(REF["groups"])
        start_group = qp.get("group", "Primary Care")
        group = st.selectbox("Prescriber group", groups,
                             index=groups.index(start_group) if start_group in groups else groups.index("Primary Care"))
        g = REF["groups"][group]
        d = dict(g["defaults"])
        start_rate = float(qp.get("rate", round(d["opioid_rate"], 1)))
        start_prev = float(qp.get("prev", start_rate))
        c1, c2 = st.columns(2)
        region = c1.selectbox("Region", ["South", "West", "Midwest", "Northeast"])
        rural = c2.toggle("Rural practice", value=False)
        rate = st.slider("Opioid share of their prescriptions (%)", 0.0, 60.0, start_rate, 0.1)
        prev_rate = st.slider("Same measure one year earlier (%)", 0.0, 60.0, start_prev, 0.1)
        total_claims = st.number_input("Total Part D claims per year", 100, 50000, int(d["total_claims"]), step=100)
        c3, c4 = st.columns(2)
        patients = c3.number_input("Medicare patients", 11, 5000, int(d["total_beneficiaries"]), step=10)
        opioid_pat = c4.slider("Patients on opioids (%)", 0.0, 100.0, float(round(d["opioid_patient_share"] * 100, 1)), 0.5)
        c5, c6 = st.columns(2)
        days = c5.slider("Days supplied per opioid claim", 1.0, 35.0, float(round(d["days_per_opioid_claim"], 1)), 0.5)
        la = c6.slider("Long-acting opioid share (%)", 0.0, 80.0, float(round(d["long_acting_share"] * 100, 1)), 0.5)
        c7, c8 = st.columns(2)
        age = c7.slider("Patient average age", 30.0, 90.0, float(round(d["bene_avg_age"], 1)), 0.5)
        risk = c8.slider("Patient risk score (HCC)", 0.3, 4.0, float(round(d["bene_avg_risk_score"], 2)), 0.05)
        was_outlier = st.toggle("Was a peer outlier last year", value=False)

    q = np.array(g["rate_quantiles"])
    percentile = lambda r: float(np.searchsorted(q, r, side="right") - 1) / 100
    opioid_claims = total_claims * rate / 100
    prev_claims = total_claims * prev_rate / 100
    row = pd.DataFrame([{
        "specialty_group": group, "census_region": region, "rural": float(rural), "peer_count": g["peer_count"],
        "total_claims": total_claims, "opioid_claims": opioid_claims, "opioid_rate": rate,
        "peer_percentile": percentile(rate), "rate_vs_peer_median": rate / g["peer_median"] if g["peer_median"] else np.nan,
        "rate_vs_peer_p99": rate / g["peer_p99"], "long_acting_share": la / 100,
        "days_per_opioid_claim": days, "total_beneficiaries": patients, "opioid_patient_share": opioid_pat / 100,
        "bene_avg_age": age, "bene_avg_risk_score": risk, "present_prev_year": 1.0,
        "rate_change_1y": rate - prev_rate, "percentile_change_1y": percentile(rate) - percentile(prev_rate),
        "opioid_claims_growth_1y": opioid_claims / prev_claims - 1 if prev_claims else 0.0,
        "was_outlier_prev_year": float(was_outlier)}])
    X = pd.DataFrame(art["prep"].transform(row[art["categorical"] + art["numeric"]]), columns=art["features"])
    raw = float(art["booster"].predict_proba(X)[:, 1][0])
    prob = float(art["calibrator"].predict([raw])[0])
    multiple = prob / REF["base_rate"]
    if rate / g["peer_p99"] >= 1 and rate >= 3 * g["peer_median"]:
        tier, colour = "Already at outlier level", "#57534E"
    elif multiple >= 10:
        tier, colour = "High risk: review", "#B42318"
    elif multiple >= 2:
        tier, colour = "Elevated: monitor", AMBER
    else:
        tier, colour = "Low risk", "#15803D"

    with right:
        st.subheader("Early-warning score")
        k1, k2 = st.columns([1.2, 1])
        k1.markdown(f'<div class="big">{prob:.2%}</div><p class="note">probability of becoming a peer outlier '
                    f'in the next two years</p>', unsafe_allow_html=True)
        k2.markdown(f'<span class="pill" style="background:{colour}">{tier}</span><p class="note" style="margin-top:10px">'
                    f'<b>{multiple:,.1f}x</b> the average prescriber ({REF["base_rate"]:.2%})<br>'
                    f'Percentile within {group}: <b>{percentile(rate):.0%}</b></p>', unsafe_allow_html=True)

        sv = explainer.shap_values(X)[0]
        contrib = pd.Series(sv, index=X.columns)
        grouped = contrib.groupby(lambda c: "Specialty" if c.startswith("specialty_group") else
                                  "Region" if c.startswith("census_region") else NICE.get(c, c)).sum()
        top = grouped.reindex(grouped.abs().sort_values(ascending=False).index[:8])[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=[AMBER if v > 0 else VIOLET_LIGHT for v in top.values],
                               text=[("+" if v > 0 else "") + f"{v:.2f}" for v in top.values], textposition="auto",
                               insidetextanchor="middle", textfont=dict(size=12)))
        fig.update_layout(title=dict(text="Why: what pushes this score up (amber) or down (violet)", font=dict(size=16)),
                          paper_bgcolor=PAPER, plot_bgcolor=PAPER, height=420, margin=dict(l=10, r=20, t=50, b=20),
                          xaxis=dict(title="SHAP contribution (log-odds)", zeroline=True, zerolinecolor="#A8A29E",
                                     gridcolor="#E7E2DA"), font=dict(color=INK, size=13))
        st.plotly_chart(fig, width="stretch")
        st.caption("A flag is a reason for a supportive review (education, prescription-monitoring check), "
                   "not evidence of inappropriate prescribing.")

with tab_model:
    test = pd.DataFrame(METRICS["results"]["test"]).set_index("model")
    m = test.loc["XGBoost"]
    rule = test.loc["Rule: closeness to peer 99th pct"]
    a, b, c, e = st.columns(4)
    a.metric("Precision, top 1,000", f"{m['precision_top_1000']:.0%}", f"{m['precision_top_1000'] - rule['precision_top_1000']:+.0%} vs rule")
    b.metric("Lift over random", f"{m['lift_top_1000']:.0f}x", f"{m['lift_top_1000'] - rule['lift_top_1000']:+.0f}x vs rule")
    c.metric("Recall in top 1%", f"{m['recall_top_1pct']:.0%}", f"{m['recall_top_1pct'] - rule['recall_top_1pct']:+.0%} vs rule")
    e.metric("PR-AUC", f"{m['pr_auc']:.3f}", f"{m['pr_auc'] - rule['pr_auc']:+.3f} vs rule")
    img = ROOT / "Image"
    i1, i2 = st.columns(2)
    i1.image(str(img / "01_model_comparison.png"))
    i2.image(str(img / "02_capture_curve.png"))
    i3, i4 = st.columns(2)
    i3.image(str(img / "04_feature_importance.png"))
    i4.image(str(img / "03_calibration.png"))
    st.caption("Test year: 2022 snapshot, outcomes 2023-2024, never used for training or tuning.")

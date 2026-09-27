"""
Prescribing Insights: early-warning model for high-risk opioid prescribing (Streamlit app).

    streamlit run app/app.py

Pages: Overview, Prescriber assessment (score a profile + SHAP explanation), Model evidence, Data & methods.
Runs from the files committed in models/ and Image/, so it deploys to Streamlit Community Cloud without a database.
Works on prescriber *profiles*; no real prescriber is shown.
"""
import base64
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"
NAVY, NAVY_2, TEAL, TEAL_SOFT = "#0A1F3A", "#0F2A47", "#0E9AA7", "#E6F6F7"
VIOLET, AMBER, INK, INK_2, BG = "#5B3FD1", "#D97706", "#0A1F3A", "#5B6B80", "#F3F6F9"

st.set_page_config(page_title="Prescribing Insights | Early Warning Model", page_icon="💠", layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------------------------- data
@st.cache_resource
def load():
    art = joblib.load(ROOT / "models" / "early_warning_xgb.joblib")
    ref = json.loads((ROOT / "models" / "app_reference.json").read_text())
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return art, ref, metrics, shap.TreeExplainer(art["booster"])


@st.cache_data
def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


art, REF, METRICS, explainer = load()
TEST = pd.DataFrame(METRICS["results"]["test"]).set_index("model")
XGB, RULE = TEST.loc["XGBoost"], TEST.loc["Rule: closeness to peer 99th pct"]
CI = METRICS.get("test_bootstrap_95ci", {})
NICE = {"rate_vs_peer_p99": "Rate vs. specialty 99th percentile", "peer_percentile": "Percentile within specialty",
        "rate_vs_peer_median": "Rate vs. specialty median", "opioid_rate": "Opioid share of prescriptions",
        "was_outlier_prev_year": "Was an outlier last year", "percentile_change_1y": "1-year change in percentile",
        "rate_change_1y": "1-year change in opioid rate", "opioid_claims": "Opioid claims",
        "total_claims": "Total claims", "long_acting_share": "Long-acting opioid share",
        "days_per_opioid_claim": "Days per opioid claim", "opioid_patient_share": "Share of patients on opioids",
        "bene_avg_risk_score": "Patient risk score", "bene_avg_age": "Patient average age",
        "opioid_claims_growth_1y": "1-year growth in opioid claims", "peer_count": "Specialty size",
        "total_beneficiaries": "Patients", "present_prev_year": "Present last year", "rural": "Rural practice"}

# --------------------------------------------------------------------------------------------- style
ICON = {  # inline SVG icons (stroke = currentColor)
    "db": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>',
    "doc": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>',
    "clock": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    "info": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/></svg>',
    "user": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/></svg>',
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {BG}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {NAVY} 0%, #0C2440 100%);
    min-width: 290px !important; max-width: 290px !important; }}
section[data-testid="stSidebar"] * {{ color: #D6E2F0; }}
[data-testid="stSidebarNav"] a {{ border-radius: 10px; padding: 10px 12px; margin: 3px 6px; }}
[data-testid="stSidebarNav"] a span {{ font-size: 15.5px; font-weight: 500; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: #133A57; box-shadow: inset 3px 0 0 {TEAL}; }}
[data-testid="stSidebarNav"] a:hover {{ background: #11324E; }}
[data-testid="stSidebarNavSeparator"] {{ display: none; }}
.side-footer {{ position: fixed; bottom: 18px; width: 230px; border-top: 1px solid #22415F; padding-top: 14px;
               display: flex; gap: 12px; align-items: center; }}
.side-footer b {{ color: #FFFFFF; font-size: 15px; }} .side-footer span {{ color: #9FB4CC; font-size: 13px; }}

/* hero */
.hero {{ border-radius: 18px; padding: 30px 36px 30px 36px; color: white; margin-bottom: 18px;
        background: url(data:image/png;base64,{b64(ASSETS / "hero.png")}) center/cover no-repeat;
        box-shadow: 0 10px 30px rgba(10, 31, 58, .18); position: relative; }}
.hero .crumb {{ letter-spacing: .08em; font-size: 12.5px; color: #B9D3E6; font-weight: 600; }}
.hero h1 {{ color: white; font-size: 44px; font-weight: 800; margin: 6px 0 4px 0; line-height: 1.1; padding: 0; }}
.hero .lead {{ font-size: 20px; color: #E8F1F8; font-weight: 500; }}
.hero .sub {{ font-size: 15px; color: #A9C1D6; margin-top: 4px; }}
.hero .pill {{ position: absolute; top: 24px; right: 28px; border: 1px solid #5EEAD4; border-radius: 999px;
              padding: 6px 14px; font-size: 13px; color: #E6FFFB; background: rgba(8, 47, 60, .45); }}
.hero .pill:before {{ content: "●"; color: #2DD4BF; margin-right: 8px; }}

/* cards */
[class*="st-key-card-"] {{ background: white !important; border-radius: 16px !important;
    border: 1px solid #E3E9F0 !important; box-shadow: 0 4px 18px rgba(10, 31, 58, .06); padding: 18px 20px !important; }}
.kpi {{ min-height: 118px; }}
.strip {{ background: white; border: 1px solid #E3E9F0; border-radius: 14px; display: flex;
          box-shadow: 0 4px 18px rgba(10, 31, 58, .05); margin-bottom: 18px; }}
.strip div {{ flex: 1; display: flex; gap: 14px; align-items: center; justify-content: center; padding: 16px;
              color: {INK}; font-size: 17px; font-weight: 600; }}
.strip div + div {{ border-left: 1px solid #E3E9F0; }}
.strip svg {{ color: {INK_2}; }}
.card-title {{ font-size: 26px; font-weight: 800; color: {INK}; margin: 2px 0 0 0; }}
.card-sub {{ color: {INK_2}; font-size: 15px; margin: 2px 0 10px 0; }}
.big {{ font-size: 76px; font-weight: 800; color: {NAVY}; line-height: 1; letter-spacing: -2px; margin: 18px 0 6px 0; }}
.caption {{ color: {INK_2}; font-size: 15px; }}
.badge {{ display: inline-flex; gap: 8px; align-items: center; padding: 7px 16px; border-radius: 999px;
          font-weight: 600; font-size: 15px; }}
.stat {{ display: flex; gap: 0; margin-top: 18px; }}
.stat > div {{ flex: 1; padding-right: 16px; }} .stat > div + div {{ border-left: 1px solid #E3E9F0; padding-left: 22px; }}
.stat .v {{ font-size: 32px; font-weight: 800; color: {TEAL}; }}
.stat .l {{ font-size: 15px; color: {INK}; font-weight: 600; }} .stat .d {{ font-size: 13.5px; color: {INK_2}; }}
.interp {{ background: white; border: 1px solid #E3E9F0; border-radius: 16px; padding: 18px 22px; display: flex;
           gap: 18px; align-items: center; box-shadow: 0 4px 18px rgba(10, 31, 58, .05); }}
.interp .ic {{ background: {TEAL_SOFT}; color: {TEAL}; border-radius: 999px; width: 52px; height: 52px;
               display: flex; align-items: center; justify-content: center; flex: none; }}
.interp b {{ font-size: 19px; color: {INK}; }} .interp p {{ margin: 2px 0 0 0; color: {INK_2}; }}
.kpi .v {{ font-size: 40px; font-weight: 800; color: {NAVY}; line-height: 1.1; }}
.kpi .l {{ font-size: 15px; font-weight: 600; color: {INK}; }} .kpi .d {{ font-size: 13.5px; color: {INK_2}; }}
.foot {{ text-align: right; color: #8A99AB; font-size: 12.5px; margin-top: 10px; }}
</style>""", unsafe_allow_html=True)


def hero(crumb, title, lead, sub):
    st.markdown(f"""<div class="hero"><div class="pill">Portfolio demonstration</div>
        <div class="crumb">HEALTHCARE ANALYTICS &nbsp;/&nbsp; {crumb.upper()}</div>
        <h1>{title}</h1><div class="lead">{lead}</div><div class="sub">{sub}</div></div>""", unsafe_allow_html=True)


def strip():
    st.markdown(f"""<div class="strip"><div>{ICON['db']} CMS Medicare Part D</div>
        <div>{ICON['doc']} 7.8M prescriber records</div><div>{ICON['clock']} Two-year horizon</div></div>""",
                unsafe_allow_html=True)


def card_head(title, sub):
    st.markdown(f'<div class="card-title">{title}</div><div class="card-sub">{sub}</div>', unsafe_allow_html=True)


def kpi(col, value, label, detail):
    with col.container(border=True, key=card_key()):
        st.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div>'
                    f'<div class="d">{detail}</div></div>', unsafe_allow_html=True)


def ci(key, fmt="{:.0%}"):
    lo, hi = CI.get(key, [None, None])
    return f"95% CI {fmt.format(lo)}–{fmt.format(hi)}" if lo is not None else ""


_cards = [0]


def card_key():
    _cards[0] += 1
    return f"card-{_cards[0]}"


# --------------------------------------------------------------------------------------------- pages
def overview():
    hero("Overview", "Prescribing Analytics", "Prioritize prescriber review with predictive insights.",
         "Two-year outlook for opioid prescribing peer-outlier risk.")
    strip()
    c = st.columns(4, gap="medium")
    kpi(c[0], f"{XGB['precision_top_1000']:.0%}", "Correct flags in a 1,000-prescriber review list",
        f"vs {RULE['precision_top_1000']:.0%} for the best simple rule · {ci('precision_top')}")
    kpi(c[1], f"{XGB['lift_top_1000']:.0f}x", "Better than random selection",
        f"Random review finds {XGB['base_rate'] * 1000:.1f} per 1,000")
    kpi(c[2], f"{XGB['recall_top_1pct']:.0%}", "Future outliers caught in the top 1%", ci("recall_top_1pct"))
    kpi(c[3], f"{METRICS['calibration']['mean_predicted']:.2%}", "Average predicted risk",
        f"vs {METRICS['calibration']['observed_rate']:.2%} that actually happened: predictions match reality")
    st.write("")
    left, right = st.columns([1.15, 1], gap="large")
    with left.container(border=True, key=card_key()):
        card_head("The question", "Early warning, not hindsight")
        st.markdown(
            "About 1% of Medicare prescribers are extreme opioid outliers compared with their own specialty, and "
            "**78% of them stay outliers year after year**. Flagging them again is easy.\n\n"
            "The harder and more useful question: **among prescribers who are not outliers today, who will become "
            "one within two years?** Only 0.45% do (about 2,000 of 460,000 a year). A payer or state monitoring "
            "program can only review a short list, so the model is judged on **how many of its top-ranked "
            "prescribers really become outliers**.")
    with right.container(border=True, key=card_key()):
        card_head("How it works", "From public data to a review list")
        for n, (t, d) in enumerate([
                ("Data", "7.8M CMS Part D prescriber-years (2019-2024) modelled in PostgreSQL"),
                ("Inputs", "How a prescriber compares with others in their field, recent trend, volume and patient mix"),
                ("Model", "A machine learning model, checked against a simple rule of thumb and two other methods"),
                ("Honest test", "Tested on newer data the model had never seen, as in real use"),
                ("Explain", "Every score comes with the reasons behind it")], 1):
            st.markdown(f"**{n}. {t}** · <span style='color:{INK_2}'>{d}</span>", unsafe_allow_html=True)
    st.markdown('<div class="foot">Illustrative assessment · Not a clinical recommendation</div>', unsafe_allow_html=True)


def assessment():
    hero("Prescriber review", "Prescribing Analytics", "Prioritize prescriber review with predictive insights.",
         "Two-year outlook for opioid prescribing peer-outlier risk.")
    strip()
    qp = st.query_params
    groups = sorted(REF["groups"])
    start_group = qp.get("group", "Primary Care")
    left, right = st.columns([1, 1.05], gap="medium")
    with left.container(border=True, key=card_key()):
        card_head("Prescriber profile", "Set the characteristics for the prescriber to assess.")
        group = st.selectbox("Prescriber group", groups,
                             index=groups.index(start_group) if start_group in groups else groups.index("Primary Care"))
        g = REF["groups"][group]
        d = dict(g["defaults"])
        start_rate = float(qp.get("rate", round(d["opioid_rate"], 1)))
        c1, c2 = st.columns([1.6, 1])
        region = c1.selectbox("Region", ["South", "West", "Midwest", "Northeast"])
        c2.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        rural = c2.toggle("Rural practice", value=False)
        rate = st.slider("Opioid prescription share (%)", 0.0, 60.0, start_rate, 0.1)
        prev_rate = st.slider("Same measure one year earlier (%)", 0.0, 60.0, float(qp.get("prev", start_rate)), 0.1)
        with st.expander("More profile details", expanded=False):
            total_claims = st.number_input("Total Part D claims per year", 100, 50000, int(d["total_claims"]), step=100)
            e1, e2 = st.columns(2)
            patients = e1.number_input("Medicare patients", 11, 5000, int(d["total_beneficiaries"]), step=10)
            opioid_pat = e2.slider("Patients on opioids (%)", 0.0, 100.0, float(round(d["opioid_patient_share"] * 100, 1)), 0.5)
            e3, e4 = st.columns(2)
            days = e3.slider("Days supplied per opioid claim", 1.0, 35.0, float(round(d["days_per_opioid_claim"], 1)), 0.5)
            la = e4.slider("Long-acting opioid share (%)", 0.0, 80.0, float(round(d["long_acting_share"] * 100, 1)), 0.5)
            e5, e6 = st.columns(2)
            age = e5.slider("Patient average age", 30.0, 90.0, float(round(d["bene_avg_age"], 1)), 0.5)
            risk = e6.slider("Patient risk score (HCC)", 0.3, 4.0, float(round(d["bene_avg_risk_score"], 2)), 0.05)
            was_outlier = st.toggle("Was already a high prescriber last year", value=False)

    q = np.array(g["rate_quantiles"])
    percentile = lambda r: float(np.searchsorted(q, r, side="right") - 1) / 100
    opioid_claims, prev_claims = total_claims * rate / 100, total_claims * prev_rate / 100
    row = pd.DataFrame([{
        "specialty_group": group, "census_region": region, "rural": float(rural), "peer_count": g["peer_count"],
        "total_claims": total_claims, "opioid_claims": opioid_claims, "opioid_rate": rate,
        "peer_percentile": percentile(rate),
        "rate_vs_peer_median": rate / g["peer_median"] if g["peer_median"] else np.nan,
        "rate_vs_peer_p99": rate / g["peer_p99"], "long_acting_share": la / 100, "days_per_opioid_claim": days,
        "total_beneficiaries": patients, "opioid_patient_share": opioid_pat / 100, "bene_avg_age": age,
        "bene_avg_risk_score": risk, "present_prev_year": 1.0, "rate_change_1y": rate - prev_rate,
        "percentile_change_1y": percentile(rate) - percentile(prev_rate),
        "opioid_claims_growth_1y": opioid_claims / prev_claims - 1 if prev_claims else 0.0,
        "was_outlier_prev_year": float(was_outlier)}])
    X = pd.DataFrame(art["prep"].transform(row[art["categorical"] + art["numeric"]]), columns=art["features"])
    prob = float(art["calibrator"].predict([float(art["booster"].predict_proba(X)[:, 1][0])])[0])
    multiple = prob / REF["base_rate"]
    if rate / g["peer_p99"] >= 1 and (not g["peer_median"] or rate >= 3 * g["peer_median"]):
        tier, fg, bgc = "Already at outlier level", "#475569", "#E2E8F0"
    elif multiple >= 10:
        tier, fg, bgc = "Review priority: high", "#9A3412", "#FDE7D3"
    elif multiple >= 2:
        tier, fg, bgc = "Review priority: elevated", "#92400E", "#FEF3C7"
    else:
        tier, fg, bgc = "Review priority: low", "#166534", "#DCFCE7"

    with right.container(border=True, key=card_key()):
        top_l, top_r = st.columns([1.5, 1])
        with top_l:
            card_head("Assessment result", "Estimated risk based on the prescriber's profile.")
        top_r.markdown(f'<div style="text-align:right;margin-top:14px"><span class="badge" style="color:{fg};'
                       f'background:{bgc}">● {tier}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big">{prob:.2%}</div><div class="caption">Chance this prescriber will prescribe far more opioids '
                    f'than others in their field within two years.</div>'
                    f'<div class="stat"><div><div class="v">{multiple:,.1f}x</div><div class="l">Baseline</div>'
                    f'<div class="d">vs. the average prescriber ({REF["base_rate"]:.2%})</div></div>'
                    f'<div><div class="v">{percentile(rate) * 100:.0f}th</div><div class="l">Percentile within specialty</div>'
                    f'<div class="d">{group}</div></div></div>', unsafe_allow_html=True)

    with st.container(border=True, key=card_key()):
        card_head("Why this score", "What pushes the risk up (amber) or down (teal)")
        sv = explainer.shap_values(X)[0]
        grouped = pd.Series(sv, index=X.columns).groupby(
            lambda c: "Specialty" if c.startswith("specialty_group") else
            "Region" if c.startswith("census_region") else NICE.get(c, c)).sum()
        top = grouped.reindex(grouped.abs().sort_values(ascending=False).index[:8])[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=[AMBER if v > 0 else TEAL for v in top.values],
                               text=[f"{v:+.2f}" for v in top.values], textposition="auto",
                               insidetextanchor="middle", textfont=dict(size=13, color="white")))
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=330, margin=dict(l=10, r=20, t=10, b=30),
                          xaxis=dict(title="Effect on the risk score", zeroline=True,
                                     zerolinecolor="#94A3B8", gridcolor="#EEF2F6"),
                          yaxis=dict(tickfont=dict(size=13.5)), font=dict(family="Inter", color=INK, size=13))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown(f"""<div class="interp"><div class="ic">{ICON['info']}</div><div><b>Interpretation</b>
        <p>Use this score to prioritize further, supportive review (education, prescription-monitoring checks).</p>
        <p>A prescribing outlier score does not establish inappropriate care.</p></div></div>
        <div class="foot">Illustrative assessment · Not a clinical recommendation</div>""", unsafe_allow_html=True)


def evidence():
    hero("Model evidence", "Model Evidence", "How the early-warning model was tested.",
         "Tuned on 2020 only · tested once on the 2022 snapshot (outcomes 2023-2024) · 95% bootstrap intervals.")
    c = st.columns(4, gap="medium")
    kpi(c[0], f"{XGB['precision_top_1000']:.0%}", "Precision, top 1,000", ci("precision_top"))
    kpi(c[1], f"+{XGB['precision_top_1000'] - RULE['precision_top_1000']:.0%}", "Gain over the simple rule",
        ci("gain_vs_rule"))
    kpi(c[2], f"{XGB['recall_top_1pct']:.0%}", "Recall in the top 1%", ci("recall_top_1pct"))
    kpi(c[3], f"{XGB['pr_auc']:.3f}", "PR-AUC", ci("pr_auc", "{:.3f}"))
    st.write("")
    with st.container(border=True, key=card_key()):
        card_head("All models on the test year", "Same features and population; the rule needs no training")
        tbl = TEST[["precision_top_1000", "lift_top_1000", "recall_top_1pct", "pr_auc", "roc_auc"]].copy()
        tbl.index = ["Simple rule", "Logistic regression", "Random forest", "XGBoost (selected)"]
        tbl.columns = ["Precision, top 1,000", "Lift", "Recall, top 1%", "PR-AUC", "ROC-AUC"]
        st.dataframe(tbl.style.format({"Precision, top 1,000": "{:.1%}", "Lift": "{:.0f}x", "Recall, top 1%": "{:.1%}",
                                       "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}"}), width="stretch")
        bt = pd.DataFrame(METRICS["results"].get("backtest_2021", [])).set_index("model")
        if len(bt):
            st.caption(f"Out-of-time backtest (2021 snapshot, also untouched): XGBoost precision in the top 1,000 = "
                       f"{bt.loc['XGBoost', 'precision_top_1000']:.1%} vs {bt.loc['Rule: closeness to peer 99th pct', 'precision_top_1000']:.1%} for the rule.")
    img = ROOT / "Image"
    for a, b in [("01_model_comparison", "02_capture_curve"), ("04_feature_importance", "03_calibration")]:
        l, r = st.columns(2, gap="medium")
        with l.container(border=True, key=card_key()):
            st.image(str(img / f"{a}.png"), width="stretch")
        with r.container(border=True, key=card_key()):
            st.image(str(img / f"{b}.png"), width="stretch")


def methods():
    hero("Data & methods", "Data & Methods", "What the model sees, and what it must not be used for.",
         "Model card for the early-warning model.")
    l, r = st.columns(2, gap="medium")
    with l.container(border=True, key=card_key()):
        card_head("Data and label", "Real, public CMS data")
        st.markdown(
            "- **Source:** CMS Medicare Part D Prescribers by Provider, 2019-2024 (7.8M prescriber-years), modelled in PostgreSQL\n"
            "- **Population:** individual prescribers with 100+ Part D claims and unsuppressed opioid counts, **not** an outlier in year T\n"
            "- **Label:** becomes a peer outlier (at or above the specialty's 99th percentile and 3x its median) in T+1 or T+2\n"
            "- **Features:** information available at the end of year T only: position within specialty, one-year trend, "
            "volume, long-acting share, days supplied, patient mix, specialty, region, rural practice")
    with r.container(border=True, key=card_key()):
        card_head("Evaluation protocol", "No look-ahead at any step")
        st.markdown(
            "- **Fit:** 80% of the 2020 snapshot · **Tune** (model choice, early stopping, calibration): the other 20%\n"
            "- **Backtest:** 2021 snapshot, untouched · **Test:** 2022 snapshot → outcomes 2023-24, reported once\n"
            "- **Baseline:** the best simple rule (closeness to the specialty's 99th percentile)\n"
            "- **Metrics:** precision on a 1,000-prescriber review list, recall in the top 1%, PR-AUC, calibration; "
            "95% bootstrap intervals (1,000 resamples)")
    with st.container(border=True, key=card_key()):
        card_head("Intended use and limitations", "Model card")
        st.markdown(
            "- **Intended use:** prioritising prescribers for *supportive* review by a payer or public-health program\n"
            "- **Not for:** penalties, network exclusion or any decision without human review; a flag is statistical, "
            "not evidence of inappropriate care\n"
            "- **Limitations:** Medicare Part D only (mostly 65+); prescribers must stay in Medicare for both follow-up years; "
            "peer groups depend on CMS specialty labels; re-validate every year as patterns change\n"
            "- **Privacy:** the app scores profiles; no individual prescriber or NPI is shown or stored\n"
            "- **Code:** [github.com/Isaac-Agyapong/Opioid_Prescriber_Risk_Model](https://github.com/Isaac-Agyapong/Opioid_Prescriber_Risk_Model)")


# --------------------------------------------------------------------------------------------- navigation
st.logo(str(ASSETS / "logo.png"), size="large")
nav = st.navigation([st.Page(overview, title="Overview", icon=":material/home:", default=True),
                     st.Page(assessment, title="Prescriber assessment", icon=":material/person_search:"),
                     st.Page(evidence, title="Model evidence", icon=":material/monitoring:"),
                     st.Page(methods, title="Data & methods", icon=":material/description:")])
st.sidebar.markdown(f"""<div class="side-footer">{ICON['user']}<div><b>Isaac Agyapong</b><br>
    <span>Clinical Data Science</span></div></div>""", unsafe_allow_html=True)
nav.run()

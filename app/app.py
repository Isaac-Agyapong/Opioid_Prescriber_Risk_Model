"""
Prescribing Early Warning: machine learning model for high-risk opioid prescribing (Streamlit app).

    streamlit run app/app.py

Pages: Home (review-list simulator), Try the model (example prescribers, live score, plain-language reasons),
Evidence, About the data. Runs from the files committed in models/ and Image/, so it deploys to Streamlit Community
Cloud without a database. Works on prescriber *profiles*; no real prescriber is shown.
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
PLUM, PLUM_2, PLUM_SOFT = "#4B1D63", "#7A2E6E", "#F3E8F6"
AMBER, AMBER_SOFT, RED, RED_SOFT, GREEN, GREEN_SOFT = "#E8930C", "#FDF1DC", "#B42318", "#FDE8E6", "#1F7A4D", "#E3F3EA"
INK, INK_2, MUTED, IVORY, LINE = "#241A2B", "#5E5566", "#9A92A0", "#FBF7F1", "#EDE5DA"

st.set_page_config(page_title="Prescribing Early Warning | ML model", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------------------------- data
@st.cache_resource
def load():
    art = joblib.load(ROOT / "models" / "early_warning_xgb.joblib")
    ref = json.loads((ROOT / "models" / "app_reference.json").read_text())
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return art, ref, metrics, shap.TreeExplainer(art["booster"])


art, REF, METRICS, explainer = load()
TEST = pd.DataFrame(METRICS["results"]["test"]).set_index("model")
XGB, RULE = TEST.loc["XGBoost"], TEST.loc["Rule: closeness to peer 99th pct"]
CI = METRICS.get("test_bootstrap_95ci", {})
BASE = REF["base_rate"]
NICE = {"rate_vs_peer_p99": "Rate vs. specialty 99th percentile", "peer_percentile": "Percentile within specialty",
        "rate_vs_peer_median": "Rate vs. specialty median", "opioid_rate": "Opioid share of prescriptions",
        "was_outlier_prev_year": "Was an outlier last year", "percentile_change_1y": "1-year change in percentile",
        "rate_change_1y": "1-year change in opioid rate", "opioid_claims": "Opioid claims",
        "total_claims": "Total claims", "long_acting_share": "Long-acting opioid share",
        "days_per_opioid_claim": "Days per opioid claim", "opioid_patient_share": "Share of patients on opioids",
        "bene_avg_risk_score": "Patient risk score", "bene_avg_age": "Patient average age",
        "opioid_claims_growth_1y": "1-year growth in opioid claims", "peer_count": "Specialty size",
        "total_beneficiaries": "Patients", "present_prev_year": "Present last year", "rural": "Rural practice"}
PERSONAS = [
    ("Steady family doctor", "Primary care · prescribes like most peers", "🩺",
     dict(group="Primary Care", rate=2.5, prev=2.5, region="Midwest", rural=False)),
    ("Surgeon near the limit", "Close to the top 1% of surgeons, stable", "🔪",
     dict(group="Surgery & Procedural", rate=70.0, prev=66.0, region="South", rural=False)),
    ("Nurse practitioner, rising", "Opioid share up from 30% to 45% in a year", "📈",
     dict(group="Nurse Practitioner / PA", rate=45.0, prev=30.0, region="South", rural=True)),
    ("Family doctor, climbing fast", "Rural, opioid share up from 11% to 17%", "⚠️",
     dict(group="Primary Care", rate=17.0, prev=11.0, region="South", rural=True)),
]

# --------------------------------------------------------------------------------------------- style
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&family=Manrope:wght@400;500;600;700;800&display=swap');
.stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{ font-family: 'Manrope', system-ui, sans-serif; }}
.stApp {{ background: {IVORY}; }}
.block-container {{ padding-top: 4.6rem; padding-bottom: 3rem; max-width: 1200px; }}
footer, [data-testid="stDecoration"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: {PLUM}; height: 3.6rem; box-shadow: 0 6px 20px -10px rgba(75,29,99,.6); }}
header[data-testid="stHeader"] a, header[data-testid="stHeader"] span, header[data-testid="stHeader"] p {{ color: #F1E6F5 !important; }}
header[data-testid="stHeader"] [aria-current="page"] {{ background: rgba(255,255,255,.14) !important; border-radius: 999px; }}
header[data-testid="stHeader"] [aria-current="page"] span {{ color: #FFFFFF !important; font-weight: 700; }}
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stStatusWidget"] {{ display: none !important; }}
h1, h2, h3, .serif, .h, .kpi .v, .counter .n, .plain, .plain * {{ font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -.5px; }}
.hero {{ border-radius: 26px; padding: 40px 44px; color: #fff; position: relative; overflow: hidden;
         background: radial-gradient(circle at 85% 20%, rgba(232,147,12,.35), transparent 45%),
                     linear-gradient(135deg, {PLUM} 0%, {PLUM_2} 100%); }}
.hero:after {{ content: ""; position: absolute; right: -40px; top: -40px; width: 520px; height: 420px; opacity: .22;
               background-image: radial-gradient(#fff 1.6px, transparent 1.7px); background-size: 16px 16px; }}
.hero .tag {{ display: inline-block; background: rgba(255,255,255,.14); padding: 6px 14px; border-radius: 999px;
              font-size: 12.5px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; }}
.hero h1 {{ color: #fff; font-size: 48px; line-height: 1.08; margin: 16px 0 12px 0; max-width: 700px; position: relative; z-index: 1; }}
.hero p {{ font-size: 17px; line-height: 1.6; color: #EBDDF0; max-width: 640px; margin: 0; position: relative; z-index: 1; }}
.hero .by {{ position: absolute; top: 34px; right: 40px; z-index: 1; background: {AMBER}; color: {INK}; font-weight: 800;
             font-size: 13px; padding: 8px 16px; border-radius: 999px; box-shadow: 0 8px 20px -8px rgba(0,0,0,.4); }}
.kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 18px 0 6px 0; }}
.kpi {{ background: #fff; border: 1px solid {LINE}; border-radius: 20px; padding: 20px 22px; }}
.kpi .v {{ font-family: 'Fraunces', serif; font-size: 42px; font-weight: 800; color: {PLUM}; line-height: 1; }}
.kpi .l {{ font-weight: 700; color: {INK}; margin-top: 8px; font-size: 15px; }}
.kpi .d {{ color: {INK_2}; font-size: 13px; margin-top: 3px; }}
[class*="st-key-card-"] {{ background: #fff; border: 1px solid {LINE} !important; border-radius: 22px !important;
                           padding: 24px 26px; box-shadow: 0 14px 30px -24px rgba(36,26,43,.35); }}
.k {{ color: {PLUM_2}; font-size: 12px; font-weight: 800; letter-spacing: 1.4px; text-transform: uppercase; }}
.h {{ font-family: 'Fraunces', serif; font-size: 27px; font-weight: 700; color: {INK}; line-height: 1.2; margin: 6px 0 6px 0; }}
.p {{ color: {INK_2}; font-size: 15px; line-height: 1.6; }}
.grid1000 {{ display: grid; grid-template-columns: repeat(50, 1fr); gap: 3px; margin: 14px 0 10px 0; }}
.grid1000 i {{ aspect-ratio: 1; border-radius: 3px; background: #EFE8F1; display: block; }}
.grid1000 i.hit {{ background: {AMBER}; box-shadow: 0 0 0 1px rgba(232,147,12,.25); }}
.counter {{ display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }}
.counter .n {{ font-family: 'Fraunces', serif; font-size: 64px; font-weight: 800; color: {AMBER}; line-height: 1; }}
.counter .t {{ font-size: 16px; color: {INK}; font-weight: 600; max-width: 520px; }}
.stButton button {{ border-radius: 16px; border: 1px solid {LINE}; background: #fff; text-align: left; padding: 14px 16px;
                    min-height: 86px; width: 100%; white-space: normal; box-shadow: 0 8px 18px -16px rgba(36,26,43,.5); }}
.stButton button p {{ font-size: 14px; line-height: 1.35; color: {INK}; }}
.stButton button:hover {{ border-color: {PLUM_2}; background: {PLUM_SOFT}; }}
.reason {{ display: flex; gap: 12px; align-items: flex-start; padding: 12px 14px; border-radius: 14px; margin: 8px 0; font-size: 14.5px; color: {INK}; }}
.reason .a {{ width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-weight: 800; flex: none; }}
.tier {{ display: inline-block; padding: 7px 16px; border-radius: 999px; font-weight: 800; font-size: 14px; }}
.plain {{ font-family: 'Fraunces', serif; font-size: 22px; color: {INK}; line-height: 1.35; margin: 4px 0 8px 0; }}
[data-testid="stSegmentedControl"] button {{ border-radius: 999px !important; font-weight: 700; }}
.foot {{ text-align: center; color: {MUTED}; font-size: 13px; margin-top: 30px; }}
.foot a {{ color: {PLUM_2}; font-weight: 700; text-decoration: none; }}
</style>""", unsafe_allow_html=True)

_cards = [0]


def card():
    _cards[0] += 1
    return st.container(border=True, key=f"card-{_cards[0]}")


def head(k, h, p=None):
    st.markdown(f'<div class="k">{k}</div><div class="h">{h}</div>' + (f'<div class="p">{p}</div>' if p else ""),
                unsafe_allow_html=True)


def foot():
    st.markdown('<div class="foot">Built by Isaac Agyapong · Portfolio demonstration, not a clinical tool · '
                '<a href="https://github.com/Isaac-Agyapong/Opioid_Prescriber_Risk_Model">Code and write-up</a></div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------- scoring
def score(group, rate, prev, region, rural, d, was_outlier=False):
    g = REF["groups"][group]
    q = np.array(g["rate_quantiles"])
    pct = lambda r: float(np.searchsorted(q, r, side="right") - 1) / 100
    claims = d["total_claims"]
    opioid_claims, prev_claims = claims * rate / 100, claims * prev / 100
    row = pd.DataFrame([{
        "specialty_group": group, "census_region": region, "rural": float(rural), "peer_count": g["peer_count"],
        "total_claims": claims, "opioid_claims": opioid_claims, "opioid_rate": rate, "peer_percentile": pct(rate),
        "rate_vs_peer_median": rate / g["peer_median"] if g["peer_median"] else np.nan,
        "rate_vs_peer_p99": rate / g["peer_p99"], "long_acting_share": d["long_acting_share"],
        "days_per_opioid_claim": d["days_per_opioid_claim"], "total_beneficiaries": d["total_beneficiaries"],
        "opioid_patient_share": d["opioid_patient_share"], "bene_avg_age": d["bene_avg_age"],
        "bene_avg_risk_score": d["bene_avg_risk_score"], "present_prev_year": 1.0, "rate_change_1y": rate - prev,
        "percentile_change_1y": pct(rate) - pct(prev),
        "opioid_claims_growth_1y": opioid_claims / prev_claims - 1 if prev_claims else 0.0,
        "was_outlier_prev_year": float(was_outlier)}])
    X = pd.DataFrame(art["prep"].transform(row[art["categorical"] + art["numeric"]]), columns=art["features"])
    prob = float(art["calibrator"].predict([float(art["booster"].predict_proba(X)[:, 1][0])])[0])
    at_outlier = rate / g["peer_p99"] >= 1 and (not g["peer_median"] or rate >= 3 * g["peer_median"])
    return prob, X, pct(rate), pct(prev), at_outlier


def plain_reasons(X, group, rate, prev, pct_now, pct_prev):
    sv = pd.Series(explainer.shap_values(X)[0], index=X.columns)
    grouped = sv.groupby(lambda c: "specialty" if c.startswith("specialty_group") else
                         "region" if c.startswith("census_region") else c).sum()
    level = grouped.reindex(["peer_percentile", "rate_vs_peer_p99", "rate_vs_peer_median", "opioid_rate"]).fillna(0).sum()
    trend = grouped.reindex(["rate_change_1y", "percentile_change_1y", "opioid_claims_growth_1y"]).fillna(0).sum()
    rest = grouped.drop([c for c in ["peer_percentile", "rate_vs_peer_p99", "rate_vs_peer_median", "opioid_rate",
                                     "rate_change_1y", "percentile_change_1y", "opioid_claims_growth_1y"] if c in grouped])
    items = [(level, f"Prescribes opioids more often than {pct_now:.0%} of other {group} prescribers"
              if pct_now >= .5 else f"Prescribes opioids less often than most {group} prescribers"),
             (trend, f"Their opioid share went {'up' if rate > prev else 'down'} by {abs(rate - prev):.1f} points in the last year"
                     if rate != prev else "Their opioid share did not change in the last year")]
    PLAIN = {"specialty": "Their type of practice", "region": "The region they work in", "rural": "Working in a rural area",
             "total_claims": "How many prescriptions they write in total", "opioid_claims": "How many opioid prescriptions they write",
             "long_acting_share": "How often they prescribe long-acting opioids", "days_per_opioid_claim": "How many days each opioid prescription lasts",
             "total_beneficiaries": "How many Medicare patients they see", "opioid_patient_share": "The share of their patients on opioids",
             "bene_avg_age": "The average age of their patients", "bene_avg_risk_score": "How sick their patients are",
             "peer_count": "How many prescribers are in their field", "was_outlier_prev_year": "Whether they were an extreme prescriber last year",
             "present_prev_year": "Whether they were active last year"}
    for name, v in rest.items():
        label = PLAIN.get(name, NICE.get(name, name))
        items.append((v, label))
    items.sort(key=lambda t: -abs(t[0]))
    return items[:3], grouped


# --------------------------------------------------------------------------------------------- pages
def home():
    st.markdown(f"""<div class="hero"><div class="by">Built by Isaac Agyapong</div>
      <span class="tag">Machine learning · 7.8 million Medicare records</span>
      <h1>Which prescribers will start prescribing far more opioids than their peers?</h1>
      <p>Health plans can only review a few prescribers each year. This model looks at public Medicare data and
      picks the prescribers most likely to become extreme opioid prescribers in the next two years, so reviewers
      know where to look first.</p></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="kpis">
      <div class="kpi"><div class="v">{XGB['precision_top_1000']:.0%}</div><div class="l">of the model's top 1,000 picks were right</div>
        <div class="d">a simple rule of thumb gets {RULE['precision_top_1000']:.0%} · random picks get under 1%</div></div>
      <div class="kpi"><div class="v">{XGB['lift_top_1000']:.0f}×</div><div class="l">better than picking at random</div>
        <div class="d">tested on newer data the model had never seen</div></div>
      <div class="kpi"><div class="v">{XGB['recall_top_1pct']:.0%}</div><div class="l">of future high prescribers caught</div>
        <div class="d">by reviewing just the top 1% of the list</div></div></div>""", unsafe_allow_html=True)

    with card():
        head("Review list simulator", "A reviewer has time to check 1,000 prescribers. How many are real problems?",
             "Each square is one prescriber on the review list. Orange squares went on to prescribe far more opioids "
             "than others in their field within two years. Pick how the list was made:")
        how = st.segmented_control("How the list was chosen", ["Picked at random", "Simple rule of thumb", "This model"],
                                   default="This model", label_visibility="collapsed")
        hits = {"Picked at random": round(XGB["base_rate"] * 1000), "Simple rule of thumb": round(RULE["precision_top_1000"] * 1000),
                "This model": round(XGB["precision_top_1000"] * 1000)}[how or "This model"]
        order = np.random.default_rng(7).permutation(1000)
        hit_set = set(order[:hits])
        cells = "".join('<i class="hit"></i>' if i in hit_set else "<i></i>" for i in range(1000))
        note = {"Picked at random": "Almost every review is wasted.",
                "Simple rule of thumb": "Better, but most reviews still find nothing.",
                "This model": f"{hits / round(RULE['precision_top_1000'] * 1000):.1f} times as many as the rule of thumb, from the same number of reviews."}[how or "This model"]
        st.markdown(f'<div class="counter"><div class="n">{hits}</div><div class="t">of 1,000 reviewed prescribers were real '
                    f'future high prescribers. {note}</div></div><div class="grid1000">{cells}</div>',
                    unsafe_allow_html=True)
        st.caption("Real results from the 2022 test year: the model never saw these prescribers or their outcomes while it was being built.")

    l, r = st.columns(2, gap="medium")
    with l, card():
        head("Why it matters", "Catch the problem before it grows",
             "About 1 in 100 Medicare prescribers prescribes far more opioids than others in their field, and most "
             "of them keep doing it year after year. Spotting them after the fact is easy. This model looks for the "
             "prescribers who are <b>not</b> there yet but are heading that way, so help can come early.")
    with r, card():
        head("How to use it", "Try it yourself",
             "Open <b>Try the model</b> at the top of the page, pick one of the example prescribers, and move the "
             "sliders. The score and the reasons behind it update instantly. A flag means a friendly review, "
             "not proof that anyone did anything wrong.")
    foot()


def try_model():
    groups = sorted(REF["groups"])
    ss = st.session_state
    ss.setdefault("group", "Primary Care")
    ss.setdefault("rate", 2.5)
    ss.setdefault("prev", 2.5)
    ss.setdefault("region", "Midwest")
    ss.setdefault("rural", False)

    def load_persona(p):
        for k, v in p.items():
            ss[k] = v

    st.markdown(f'<div class="k" style="margin-top:6px">Try the model</div><div class="h" style="font-size:36px">'
                'Meet the prescribers</div><div class="p">Pick an example to load it, then change anything you like. '
                'These are made-up profiles; no real prescriber is shown.</div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="small")
    for c, (name, desc, icon, p) in zip(cols, PERSONAS):
        c.button(f"{icon}  **{name}**  \n{desc}", key=f"p_{name}", on_click=load_persona, args=(p,), use_container_width=True)
    st.write("")

    left, right = st.columns([1, 1.15], gap="medium")
    with left, card():
        head("The prescriber", "Describe who you want to check")
        group = st.selectbox("Type of prescriber", groups, key="group")
        g = REF["groups"][group]
        d = dict(g["defaults"])
        st.slider("Share of their prescriptions that are opioids", 0.0, 100.0, step=0.5, format="%.1f%%", key="rate")
        st.slider("The same share one year earlier", 0.0, 100.0, step=0.5, format="%.1f%%", key="prev")
        c1, c2 = st.columns([1.4, 1])
        c1.selectbox("Region", ["South", "West", "Midwest", "Northeast"], key="region")
        c2.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        c2.toggle("Rural practice", key="rural")
        med, p99 = g["peer_median"], g["peer_p99"]
        st.markdown(f'<div class="p" style="font-size:13.5px;margin-top:6px">For {group}: a typical prescriber is at '
                    f'<b>{med:.1f}%</b>, the top 1% start at <b>{p99:.1f}%</b>.</div>', unsafe_allow_html=True)

    prob, X, pct_now, pct_prev, at_outlier = score(group, ss.rate, ss.prev, ss.region, ss.rural, d)
    mult = prob / BASE
    if at_outlier:
        tier, fg, bg = "Already at the extreme level", "#475467", "#EEF0F3"
    elif mult >= 10:
        tier, fg, bg = "High priority for review", RED, RED_SOFT
    elif mult >= 2:
        tier, fg, bg = "Worth a closer look", "#9A5B00", AMBER_SOFT
    else:
        tier, fg, bg = "Low priority", GREEN, GREEN_SOFT
    per1000 = prob * 1000

    with right, card():
        head("Model result", "")
        st.markdown(f'<span class="tier" style="background:{bg};color:{fg}">● {tier}</span>', unsafe_allow_html=True)
        pos = lambda m: float(np.clip((np.log10(max(m, 0.1)) + 1) / (np.log10(50) + 1), 0, 1)) * 100
        p2, p10, here = pos(2), pos(10), pos(mult)
        st.markdown(f"""<div style="margin:26px 0 8px 0;position:relative">
          <div style="position:absolute;left:calc({here:.1f}% - 9px);top:-22px;font-size:18px;color:{PLUM}">▼</div>
          <div style="display:flex;height:18px;border-radius:999px;overflow:hidden">
            <div style="width:{p2:.1f}%;background:#9ED6B6"></div><div style="width:{p10 - p2:.1f}%;background:#F4C66E"></div>
            <div style="flex:1;background:#EE8E84"></div></div>
          <div style="position:relative;height:34px;font-size:12px;color:{INK_2};margin-top:6px">
            <span style="position:absolute;left:{p2 / 2:.1f}%;transform:translateX(-50%);color:{GREEN};font-weight:700">Low</span>
            <span style="position:absolute;left:{(p2 + p10) / 2:.1f}%;transform:translateX(-50%);color:#9A5B00;font-weight:700">Worth a look</span>
            <span style="position:absolute;left:{(p10 + 100) / 2:.1f}%;transform:translateX(-50%);color:{RED};font-weight:700">High</span></div></div>""",
                    unsafe_allow_html=True)
        times = ("far below the average prescriber" if mult < 0.1 else
                 f"{mult:.1f} times the average prescriber ({BASE * 1000:.1f} in 1,000)")
        st.markdown(f'<div class="plain">About <b style="color:{PLUM}">{per1000:.0f} in 1,000</b> prescribers like this one '
                    f'will prescribe far more opioids than others in their field within two years.</div>'
                    f'<div class="p">That is {times}. Model score {prob:.2%}.</div>' if per1000 >= 1 else
                    f'<div class="plain">Fewer than <b style="color:{PLUM}">1 in 1,000</b> prescribers like this one '
                    f'will prescribe far more opioids than others in their field within two years.</div>'
                    f'<div class="p">That is {times}. Model score {prob:.2%}.</div>', unsafe_allow_html=True)

    with card():
        items, grouped = plain_reasons(X, group, ss.rate, ss.prev, pct_now, pct_prev)
        head("Why", "The main reasons behind this result")
        for v, text in items:
            up = v > 0
            col, soft, arrow, word = (RED, RED_SOFT, "↑", "raises the risk") if up else (GREEN, GREEN_SOFT, "↓", "lowers the risk")
            st.markdown(f'<div class="reason" style="background:{soft}"><div class="a" style="background:#fff;color:{col}">{arrow}</div>'
                        f'<div>{text} <span style="color:{col};font-weight:700">· {word}</span></div></div>',
                        unsafe_allow_html=True)
        with st.expander("See every factor the model used"):
            show = grouped.rename(lambda c: "Specialty" if c == "specialty" else "Region" if c == "region" else NICE.get(c, c))
            top = show.reindex(show.abs().sort_values(ascending=False).index[:10])[::-1]
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                   marker=dict(color=[RED if v > 0 else GREEN for v in top.values], cornerradius=6)))
            fig.update_layout(height=340, margin=dict(l=10, r=20, t=10, b=30), paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope", color=INK, size=13),
                              xaxis=dict(title="Pushes the risk down  ←  →  pushes it up", zerolinecolor=MUTED, gridcolor=LINE))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div class="reason" style="background:{PLUM_SOFT};margin-top:14px"><div class="a" style="background:#fff;color:{PLUM}">i</div>'
                '<div>A high result is a reason for a <b>supportive review</b> (education, a check of the prescription '
                'monitoring database). It is not proof of inappropriate care.</div></div>', unsafe_allow_html=True)
    foot()


def evidence():
    st.markdown('<div class="k" style="margin-top:6px">Evidence</div><div class="h" style="font-size:36px">'
                'How the model was tested</div><div class="p">Built on 2020 data, then tested once on 2022 data it had never '
                'seen, exactly as it would be used in real life. Ranges show 95% confidence from 1,000 resamples.</div>',
                unsafe_allow_html=True)
    lo, hi = CI.get("precision_top", [None, None])
    st.markdown(f"""<div class="kpis">
      <div class="kpi"><div class="v">{XGB['precision_top_1000']:.0%}</div><div class="l">correct picks in the top 1,000</div>
        <div class="d">likely between {lo:.0%} and {hi:.0%}</div></div>
      <div class="kpi"><div class="v">+{(XGB['precision_top_1000'] - RULE['precision_top_1000']) * 100:.0f} pts</div>
        <div class="l">better than the simple rule</div><div class="d">{XGB['precision_top_1000']:.0%} vs {RULE['precision_top_1000']:.0%}</div></div>
      <div class="kpi"><div class="v">{METRICS['calibration']['mean_predicted']:.2%}</div><div class="l">average predicted risk</div>
        <div class="d">vs {METRICS['calibration']['observed_rate']:.2%} that really happened: the scores are honest</div></div></div>""",
                unsafe_allow_html=True)
    with card():
        head("Four approaches, same test", "The machine learning model finds the most real cases")
        names = {"Rule: closeness to peer 99th pct": "Simple rule of thumb", "Logistic regression": "Logistic regression",
                 "Random forest": "Random forest", "XGBoost": "XGBoost (chosen)"}
        t = TEST.rename(index=names).loc[list(names.values())]
        fig = go.Figure(go.Bar(x=t.precision_top_1000 * 1000, y=t.index, orientation="h",
                               marker=dict(color=[MUTED, "#C9B8D1", "#A98BB6", PLUM], cornerradius=8),
                               text=[f"{v * 1000:.0f} of 1,000" for v in t.precision_top_1000], textposition="outside"))
        fig.update_layout(height=260, margin=dict(l=10, r=40, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope", color=INK, size=14),
                          xaxis=dict(visible=False, range=[0, 470]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with st.expander("Full results table"):
            tbl = TEST[["precision_top_1000", "lift_top_1000", "recall_top_1pct", "pr_auc", "roc_auc"]].rename(index=names)
            tbl.columns = ["Precision, top 1,000", "Lift", "Recall, top 1%", "PR-AUC", "ROC-AUC"]
            st.dataframe(tbl.style.format({"Precision, top 1,000": "{:.1%}", "Lift": "{:.0f}x", "Recall, top 1%": "{:.1%}",
                                           "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}"}), use_container_width=True)
    img = ROOT / "Image"
    a, b = st.columns(2, gap="medium")
    with a, card():
        st.image(str(img / "02_capture_curve.png"), use_container_width=True)
    with b, card():
        st.image(str(img / "03_calibration.png"), use_container_width=True)
    foot()


def about():
    st.markdown('<div class="k" style="margin-top:6px">About the data</div><div class="h" style="font-size:36px">'
                'What the model sees, and what it must not be used for</div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="medium")
    with a, card():
        head("Data", "Real, public Medicare data")
        st.markdown(
            "- **Source:** CMS Medicare Part D prescriber data, 2019-2024 (7.8 million prescriber-years), stored in PostgreSQL\n"
            "- **Who is scored:** prescribers with 100+ Medicare prescriptions a year who are **not** extreme prescribers yet\n"
            "- **What is predicted:** whether they reach the top 1% of their field (and 3 times its typical level) within two years\n"
            "- **What the model looks at:** how they compare with their peers, their one-year trend, volume, patient mix, "
            "specialty, region and rural practice")
    with b, card():
        head("Testing", "No peeking at the answers")
        st.markdown(
            "- Built on **2020** data, checked on **2021**, and tested once on **2022** (outcomes in 2023-2024)\n"
            "- Compared with a simple rule of thumb and two other machine learning methods\n"
            "- Every result comes with a 95% range from 1,000 resamples\n"
            "- Each score is explained with SHAP, a standard method for showing why a model decided what it did")
    with card():
        head("Use with care", "What this tool is and is not")
        st.markdown(
            "- **For:** helping a health plan or state program decide who to offer a supportive review first\n"
            "- **Not for:** penalties or any decision without a person reviewing it; a flag is not evidence of wrongdoing\n"
            "- **Limits:** Medicare Part D only (mostly people 65+); prescribers must stay in Medicare for two years; "
            "should be re-checked every year\n"
            "- **Privacy:** the app scores made-up profiles; no real prescriber or ID number is shown or stored")
    foot()


nav = st.navigation([st.Page(home, title="Home", icon=":material/home:", default=True),
                     st.Page(try_model, title="Try the model", icon=":material/tune:"),
                     st.Page(evidence, title="Evidence", icon=":material/verified:"),
                     st.Page(about, title="About the data", icon=":material/info:")], position="top")
nav.run()

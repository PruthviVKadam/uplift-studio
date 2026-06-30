"""Uplift Studio — interactive targeting-policy explorer (Streamlit)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklift.metrics import uplift_curve

import data as data_mod
import metrics as M
from brand import apply_brand
from models import MODEL_NAMES, make_models

st.set_page_config(page_title="Uplift Studio", page_icon="🎯", layout="wide")
apply_brand()


@st.cache_data(show_spinner="Loading Hillstrom…")
def load_data(target: str):
    return data_mod.load(target_col=target, seed=42)


@st.cache_resource(show_spinner="Fitting uplift models…")
def fit_all(target: str):
    d = load_data(target)
    return {name: m.fit(d.X_train, d.y_train, d.t_train).predict(d.X_test)
            for name, m in make_models(42).items()}


st.title("🎯 Uplift Studio")
st.caption("*Who* to target, not just *whether* it works — individual treatment effect on the "
           "Hillstrom e-mail campaign (64k customers, e-mail vs. no-e-mail).")

with st.sidebar:
    st.header("Controls")
    target = st.selectbox("Outcome", ["visit", "conversion", "spend"], index=0,
                          help="spend is modeled as 'spent anything'")
    model_name = st.selectbox("Uplift model", list(MODEL_NAMES))
    k_pct = st.slider("Budget — target the top k% by predicted uplift", 5, 100, 30, 5)
    k = k_pct / 100

d = load_data(target)
up = fit_all(target)[model_name]
y, t = d.y_test.to_numpy(), d.t_test.to_numpy()

s = M.score(y, up, t)
p = M.targeting_policy(y, up, t, k=k)
base_ate = M.ate(y, t)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ATE (treated − control)", f"{base_ate*100:.2f} pp")
c2.metric("Qini AUC", f"{s['qini_auc']:.4f}")
c3.metric(f"Incremental captured @ top {k_pct}%", f"{p['frac_captured_model']*100:.1f}%")
c4.metric("Lift vs. random", f"{p['lift_vs_random']:.2f}×")

# --- Targeting curve: cumulative incremental outcomes as we target by predicted uplift ---
x, yc = uplift_curve(y, up, t)
n = x[-1]
fig = go.Figure()
fig.add_trace(go.Scatter(x=x / n, y=yc, mode="lines", name=model_name,
                         line=dict(color="#4f46e5", width=3)))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, yc[-1]], mode="lines", name="Random targeting",
                         line=dict(color="gray", dash="dash")))
fig.add_vline(x=k, line_dash="dot", line_color="#16a34a",
              annotation_text=f"budget {k_pct}%", annotation_position="top left")
fig.update_layout(title="Targeting curve — incremental conversions captured vs. budget",
                  xaxis_title="Fraction of customers targeted (by predicted uplift)",
                  yaxis_title=f"Cumulative incremental {target}s",
                  legend=dict(orientation="h", y=-0.2), height=460, margin=dict(t=50))
st.plotly_chart(fig, use_container_width=True)

# --- Heterogeneity / sleeping dogs ---
left, right = st.columns([3, 2])
with left:
    hist = go.Figure(go.Histogram(x=up, nbinsx=40, marker_color="#4f46e5"))
    hist.add_vline(x=0, line_color="red", annotation_text="zero uplift")
    hist.update_layout(title="Distribution of predicted uplift (heterogeneity)",
                       xaxis_title="predicted uplift", yaxis_title="customers",
                       height=340, margin=dict(t=50))
    st.plotly_chart(hist, use_container_width=True)
with right:
    neg = float((up < 0).mean())
    st.subheader("Read-out")
    st.markdown(
        f"- The campaign lifts **{target}** by **{base_ate*100:.2f} pp** on average.\n"
        f"- Targeting the **top {k_pct}%** by predicted uplift captures "
        f"**{p['frac_captured_model']*100:.1f}%** of all incremental {target}s — "
        f"**{p['lift_vs_random']:.2f}×** a random policy.\n"
        f"- **{neg*100:.0f}%** of customers have *negative* predicted uplift "
        f"(potential **sleeping dogs** — contacting them may hurt).")
    st.caption("Educational demo on a public dataset. Uplift is evaluated by Qini/AUUC on a "
               "held-out split, not accuracy — see the repo README.")

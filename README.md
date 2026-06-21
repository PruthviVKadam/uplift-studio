---
title: Uplift Studio
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
python_version: "3.12"
pinned: false
---

# 🎯 Uplift Studio — *who* to target, not just *whether* it works

An interactive uplift-modeling app that estimates **individual treatment effect (CATE)** and turns
it into a **targeting policy**: instead of "the campaign lifted visits 5.5% on average," it answers
"*which* customers should we contact, and how much more incremental response do we capture by
targeting the top k%?" The honest sequel to a classic A/B test.

> Built on the A/B foundation in [Experimentation Lab](https://github.com/PruthviVKadam/ab-lab).
> Every number below is copied verbatim from `eval/results.md` (reproducible via `python eval.py`).

## Problem → Approach → Result

- **Problem:** an A/B test gives the *average* treatment effect. But effects are heterogeneous —
  some users convert *because* you targeted them, some would have anyway, and some are **"sleeping
  dogs"** who respond *negatively* to contact. Spraying everyone wastes budget. You need the
  *individual* effect.
- **Approach:** train and compare three uplift meta-learners — **S-learner**, **T-learner**, and a
  **DDR T-learner** (sklift, RandomForest base) — on the Hillstrom e-mail campaign, then **evaluate
  with uplift-specific metrics** (Qini AUC, Uplift-AUC, Uplift@k) — never accuracy. A Streamlit app
  drives a **targeting-policy simulator**: pick a budget (top k%), see incremental response captured
  vs. random.
- **Result:** on the held-out test (19,200 customers), the campaign lifts visits **+5.50 pp** on
  average; the best model (**S-learner**) concentrates that effect — **targeting the top 30% by
  predicted uplift captures 34.2% of all incremental visits, 1.14× a random policy**.

| Model | Qini AUC | Uplift@30% | % incremental captured @ top-30% | Lift vs. random |
| --- | --- | --- | --- | --- |
| **S-learner** | **0.01601** | 6.27 pp | **34.2%** | **1.14×** |
| T-learner | 0.01034 | 5.48 pp | 29.9% | 1.00× |
| T-learner (DDR) | 0.00942 | 5.85 pp | 31.9% | 1.06× |

_ATE on `visit` = 5.50 pp (treated − control) · seed 42 · a random policy captures 30% by definition._

## Dataset

**Hillstrom "MineThatData" e-mail campaign** — 64,000 customers; treatment = an e-mail was sent
(Mens/Womens) vs. the No-E-Mail control; outcomes `visit` / `conversion` / `spend`. Public, no auth
(fetched by `sklift.datasets.fetch_hillstrom`).

## Stack

Python **3.14** · **scikit-uplift 0.5.1** (verified on 3.14) · **scikit-learn 1.9** (pinned `<1.10`,
since sklift uses a function removed in 1.10) · pandas · Plotly · Streamlit.

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python eval.py            # fit S / T / DDR learners -> eval/results.md (the table above)
python -m pytest          # 5 tests: wrappers == sklift, golden Qini, policy sanity
streamlit run app.py      # targeting-policy simulator (model picker + budget slider)
```

## Files

```text
data.py     load Hillstrom, binarize treatment, encode, stratified split
models.py   S-learner / T-learner / DDR T-learner behind one .fit/.predict interface
metrics.py  Qini/Uplift-AUC/Uplift@k wrappers + ATE + targeting-policy simulation
eval.py     run all models -> eval/results.md
app.py      Streamlit targeting-policy explorer (Qini/uplift curve + budget slider)
tests/      pytest: metric correctness, golden Qini, policy sanity
```

## Honesty guardrail

Uplift is easy to fool yourself with: the eval ranks users by **predicted** uplift and measures
**actual** incremental response on a held-out split. Metrics are modest and **positive but small**
(typical for this dataset) — reported as-is, never inflated. A golden test pins the S-learner's Qini
to `0.01601` so the headline can't silently drift.

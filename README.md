# 🎯 Uplift Studio — *who* to target, not just *whether* it works

An interactive uplift-modeling app that estimates **individual treatment effect (CATE)** and turns
it into a **targeting policy**: instead of "the campaign lifted conversion 1.2% on average," it
answers "*which* customers should we contact, and what's the incremental ROI of targeting the top
k%?" The honest sequel to a classic A/B test.

> **Status:** scaffolding (plan + schedule below). Nothing is fabricated — the Results table is a
> template filled only from real `eval.py` output once built. See [Experimentation Lab](https://github.com/PruthviVKadam/ab-lab)
> for the A/B foundation this builds on.

## Problem → Approach → Result

- **Problem:** an A/B test gives the *average* treatment effect. But effects are heterogeneous —
  some users convert *because* you targeted them, some would have converted anyway, and some are
  **"sleeping dogs"** who churn *because* you contacted them. Spraying everyone wastes budget and can
  hurt. You need the *individual* effect.
- **Approach:** train and compare uplift meta-learners (S-, T-, X-learner) and an uplift random
  forest on a real campaign dataset, then **evaluate with uplift-specific metrics** (Qini curve,
  AUUC, uplift@k) — never accuracy. A Streamlit app drives a **targeting-policy simulator**: pick a
  budget (top k%), see captured incremental conversions vs. random and vs. treat-all.
- **Result:** _(filled from `eval.py` output — no numbers until measured)_

| Metric | Value |
| --- | --- |
| AUUC (area under uplift curve) | _TBD_ |
| Qini coefficient | _TBD_ |
| Uplift @ top 30% | _TBD_ |
| Incremental conversions captured by targeting top 30% vs. random | _TBD_ |

## Dataset

**Hillstrom "MineThatData" email campaign** (~64k customers; `segment` = mens/womens/no-email
treatment, `visit`/`conversion`/`spend` outcomes) — small, public, no auth. Optional scale-up:
**Criteo Uplift Modeling** (~13M rows) for a large-data story (download note in `ManualSteps.md`).

## Stack

Python 3.14 · **scikit-uplift (sklift)** · scikit-learn · pandas · Plotly · Streamlit.
(Meta-learners fall back to plain sklearn estimators if a dependency is heavy on 3.14 — see `plan.md`.)

## Reproduce (once built)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python data.py        # load + treatment/control split
python train.py       # fit S/T/X-learners + uplift forest
python eval.py        # Qini / AUUC / uplift@k -> eval/results.md
streamlit run app.py  # targeting-policy simulator
```

## Honesty guardrail

Every figure in this README is copied verbatim from `eval/results.md` (reproducible). Uplift is
easy to fool yourself with — the eval ranks users by **predicted** uplift and measures **actual**
incremental response, with a held-out split. No metric is quoted that `eval.py` didn't produce.

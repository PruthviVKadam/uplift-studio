# Build Schedule — Uplift Studio (~12 days, 2–3 h/day)

| Day | Focus | Done when |
| --- | --- | --- |
| 1 | Repo, venv, **verify sklift installs on Py3.14**, load Hillstrom | one model `.fit` runs end-to-end |
| 2 | Treatment/control split + covariate-balance check; held-out test set | balance table looks random; split frozen |
| 3 | S-learner + T-learner behind common interface | `predict_uplift` returns scores; unit test passes |
| 4 | X-learner + UpliftRandomForest | all four models train; shapes tested |
| 5 | Buffer / EDA: uplift-by-segment exploration | clear which segments respond |
| 6 | Qini curve implementation | Qini matches `sklift.metrics` on a slice |
| 7 | AUUC + uplift@k | metrics written to `eval/results.md` |
| 8 | Targeting-policy simulator (top-k → incremental, ROI) | numbers reproduce; treat-all sanity holds |
| 9 | Streamlit app: model picker + Qini chart + budget slider | app runs locally |
| 10 | Polish UI; sleeping-dogs demo callout | negative-uplift segment visible |
| 11 | pytest (metrics + golden values) + README with **real** numbers | tests green; README filled from eval |
| 12 | Deploy (Streamlit Cloud / HF) + 2-min walkthrough | live URL; recording done |

**Lead-with-the-number (resume line, once real):**
"Targeting the top 30% of customers by predicted uplift captured __% of incremental conversions
(AUUC __, Qini __) — vs __% for random targeting."

**Cut scope if behind:** drop the Criteo scale-up and UpliftRandomForest; S/T/X-learners + Qini +
the policy simulator are a complete, defensible project on their own.

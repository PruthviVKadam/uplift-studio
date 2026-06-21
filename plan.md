# Build Plan — Uplift Studio

## Interview thesis
"Most candidates stop at *does the treatment work on average*. I built the layer that decides *who
to treat* — with the uplift-specific evaluation (Qini/AUUC) that proves the targeting policy
actually captures incremental response, not just correlation." Senior product-DS signal.

## Architecture
```
data.py     load Hillstrom (or Criteo), define treatment/control, train/holdout split
features.py minimal feature prep (no leakage: nothing post-treatment)
models.py   S-learner, T-learner, X-learner, UpliftRandomForest — common .fit/.predict_uplift
eval.py     Qini curve, AUUC, uplift@k, targeting-policy simulation -> eval/results.md
app.py      Streamlit: model picker, Qini chart, top-k budget slider -> incremental ROI
```

## Key decisions (the things you defend in interviews)
- **Why meta-learners:** S/T/X-learners turn any regressor/classifier into an uplift model — shows
  you understand the framework, not just a library call. X-learner shines on imbalanced treatment.
- **Why NOT accuracy:** you never observe the counterfactual per user, so classification accuracy is
  meaningless. Rank users by predicted uplift, then measure realized treatment−control gap by decile
  → **Qini/AUUC**. This is the whole point of the project.
- **Leakage:** features must be pre-treatment only. Eval split must keep treatment/control balance.
- **Sleeping dogs:** show the policy can have *negative* uplift segments and that targeting them is
  worse than doing nothing — a memorable demo moment.

## Datasets
- **Primary — Hillstrom** (~64k rows): tiny, no auth, fast iteration. Treatment = email sent.
- **Optional — Criteo Uplift** (~13M rows, ~300MB): for a "works at scale" story. Sampled load.

## Phases (≈12 days, 2–3 h/day)
1. **Data + treatment/control sanity** — load, verify randomization (covariate balance), holdout.
2. **Meta-learners** — S/T/X + uplift forest behind one interface; unit-test predict_uplift shape.
3. **Uplift evaluation** *(the differentiator)* — Qini curve, AUUC, uplift@k; verify against sklift.
4. **Targeting-policy simulator** — top-k budget → incremental conversions vs random/treat-all; ROI.
5. **App + tests + README** — Streamlit UI, pytest on metrics, copy real numbers into README.

## Py3.14 de-risk
- `pip install scikit-uplift` first and run one model end-to-end **before** building UI. If sklift
  or a transitive dep won't install on 3.14, implement the three meta-learners directly on sklearn
  (≈40 lines) and hand-roll the Qini/AUUC functions — same as P2 avoided the `shap` package.
- Keep it CPU-only; no torch.

## Deploy
Streamlit Community Cloud (like P1) or HF Spaces (Streamlit SDK). Bundle a sampled dataset so the
demo runs without a download. Steps in `ManualSteps.md`.

## Tests
- `predict_uplift` returns one score per row, finite.
- Qini/AUUC match `sklift.metrics` on a fixed slice (golden values).
- Policy simulator: targeting top-k never reports more incremental than treat-all.

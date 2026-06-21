"""Run every uplift model and write a reproducible leaderboard to eval/results.md.

All numbers in the README are copied from this file's output — never typed by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import data as data_mod
import metrics as M
from models import make_models

SEED = 42
TARGET = "visit"
OUT = Path(__file__).resolve().parent / "eval" / "results.md"


def run(target_col: str = TARGET, seed: int = SEED) -> dict:
    d = data_mod.load(target_col=target_col, seed=seed)
    base_ate = M.ate(d.y_test, d.t_test)

    rows = {}
    for name, model in make_models(seed).items():
        model.fit(d.X_train, d.y_train, d.t_train)
        up = model.predict(d.X_test)
        s = M.score(d.y_test, up, d.t_test)
        p = M.targeting_policy(d.y_test, up, d.t_test, k=0.3)
        rows[name] = {**s, **p}
    return {"target": target_col, "ate": base_ate, "n_test": len(d.y_test), "rows": rows}


def to_markdown(res: dict) -> str:
    best = max(res["rows"], key=lambda k: res["rows"][k]["qini_auc"])
    lines = [
        "# Uplift evaluation",
        "",
        f"_Hillstrom e-mail dataset · target = `{res['target']}` · {res['n_test']:,} test rows · "
        f"treatment = e-mail sent vs. no e-mail · seed 42._",
        "",
        f"**Average treatment effect (ATE) on `{res['target']}`:** "
        f"{res['ate']*100:.2f} pp (treated − control).",
        "",
        "| Model | Qini AUC | Uplift AUC | Uplift@10% | Uplift@30% | "
        "% incremental captured @ top-30% | Lift vs random |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in res["rows"].items():
        star = " ⭐" if name == best else ""
        lines.append(
            f"| {name}{star} | {r['qini_auc']:.5f} | {r['uplift_auc']:.5f} | "
            f"{r['uplift_at_10']*100:.2f} pp | {r['uplift_at_30']*100:.2f} pp | "
            f"{r['frac_captured_model']*100:.1f}% | {r['lift_vs_random']:.2f}× |"
        )
    lines += [
        "",
        f"Best by Qini AUC: **{best}**. A random policy captures 30% of incremental response by "
        "definition; values above that mean the model concentrates the campaign's effect on the "
        "users who actually respond to it.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console is cp1252; our report has Unicode
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
    print(f"\nWrote {OUT}")

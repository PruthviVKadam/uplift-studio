# Uplift evaluation

_Hillstrom e-mail dataset · target = `visit` · 19,200 test rows · treatment = e-mail sent vs. no e-mail · seed 42._

**Average treatment effect (ATE) on `visit`:** 5.50 pp (treated − control).

| Model | Qini AUC | Uplift AUC | Uplift@10% | Uplift@30% | % incremental captured @ top-30% | Lift vs random |
| --- | --- | --- | --- | --- | --- | --- |
| S-learner ⭐ | 0.01601 | 0.00935 | 6.72 pp | 6.27 pp | 34.2% | 1.14× |
| T-learner | 0.01034 | 0.00590 | 6.25 pp | 5.48 pp | 29.9% | 1.00× |
| T-learner (DDR) | 0.00942 | 0.00530 | 5.10 pp | 5.85 pp | 31.9% | 1.06× |

Best by Qini AUC: **S-learner**. A random policy captures 30% of incremental response by definition; values above that mean the model concentrates the campaign's effect on the users who actually respond to it.

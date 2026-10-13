import numpy as np
import pandas as pd

NUMERIC = ["income", "age", "tenure", "balance"]
CATEGORICAL = ["segment"]


def _batch(n, rng, week, drift_start):
    """One week of data. After drift_start the income distribution shifts and the segment mix changes;
    from drift_start + 4 the relation between balance and the target also changes (concept drift)."""
    shift = max(0, week - drift_start)
    income = rng.lognormal(10.6 + 0.04 * shift, 0.5, n)
    age = np.clip(rng.normal(42 - 0.3 * shift, 11, n), 18, 80)
    tenure = np.clip(rng.gamma(2.0, 3.0, n), 0, 30)
    balance = rng.normal(2000, 900, n)
    probs = np.array([0.5, 0.35, 0.15]) + shift * np.array([-0.02, 0.0, 0.02])
    probs = np.clip(probs, 0.02, None)
    probs = probs / probs.sum()
    segment = rng.choice(["retail", "smb", "premium"], n, p=probs)

    concept = max(0, week - (drift_start + 4))
    logit = (
        -0.3
        - 0.00035 * (balance - 2000) * (1 - 0.18 * concept)
        - 0.00001 * (income - 40000)
        + 0.03 * (age - 42)
        - 0.05 * tenure
        + np.where(segment == "premium", -0.5, 0.0)
    )
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {"income": income, "age": age, "tenure": tenure, "balance": balance, "segment": segment, "target": y.astype(int), "week": week}
    )


def generate_stream(weeks=24, per_week=3000, drift_start=10, seed=91):
    rng = np.random.default_rng(seed)
    return pd.concat([_batch(per_week, rng, w, drift_start) for w in range(weeks)], ignore_index=True)

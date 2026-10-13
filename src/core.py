import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .data import CATEGORICAL, NUMERIC


def psi_numeric(reference, current, bins=10):
    """Population stability index with bins taken from reference quantiles."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_categorical(reference, current):
    cats = sorted(set(reference) | set(current))
    ref = reference.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4)
    cur = current.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_drift(reference, current):
    rows = {}
    for col in NUMERIC:
        ks = stats.ks_2samp(reference[col], current[col])
        rows[col] = {"psi": psi_numeric(reference[col], current[col]), "ks_stat": ks.statistic, "p_value": ks.pvalue}
    for col in CATEGORICAL:
        table = pd.concat([reference[col].value_counts(), current[col].value_counts()], axis=1).fillna(0)
        chi = stats.chi2_contingency(table.values)
        rows[col] = {"psi": psi_categorical(reference[col], current[col]), "ks_stat": np.nan, "p_value": chi[1]}
    return pd.DataFrame(rows).T


def psi_status(value):
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "warning"
    return "alert"


def train_reference_model(reference, seed=0):
    cols = NUMERIC + ["segment_code"]
    x = _encode(reference)
    model = HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=0.08, random_state=seed)
    model.fit(x[cols], reference["target"])
    return model, cols


def _encode(df):
    out = df.copy()
    out["segment_code"] = out["segment"].map({"retail": 0, "smb": 1, "premium": 2})
    return out


def monitor(stream, reference_weeks=6, train_weeks=3, seed=0):
    """Fit on the first weeks, use the next reference weeks as an out-of-sample baseline,
    then score every week for input drift and model quality."""
    ref_data = stream[stream["week"] < reference_weeks]
    model, cols = train_reference_model(stream[stream["week"] < train_weeks], seed)
    rows = []
    for week, batch in stream.groupby("week"):
        drift = feature_drift(ref_data, batch)
        auc = roc_auc_score(batch["target"], model.predict_proba(_encode(batch)[cols])[:, 1])
        rows.append(
            {
                "week": week,
                "max_psi": drift["psi"].max(),
                "worst_feature": drift["psi"].idxmax(),
                "features_in_alert": int((drift["psi"] >= 0.25).sum()),
                "auc": auc,
            }
        )
    report = pd.DataFrame(rows).set_index("week")
    holdout = (report.index >= train_weeks) & (report.index < reference_weeks)
    base_auc = float(report.loc[holdout, "auc"].mean())
    report["auc_drop"] = base_auc - report["auc"]
    report["status"] = report["max_psi"].map(psi_status)
    return report, base_auc


def first_alert_week(report, column, threshold, after=None):
    """First week (after a given week) where a column crosses the threshold."""
    sub = report if after is None else report[report.index >= after]
    hit = sub[sub[column] >= threshold]
    return None if hit.empty else int(hit.index[0])

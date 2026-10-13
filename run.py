import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import core, data

OUT = Path("reports")
DRIFT_START = 10


def main():
    OUT.mkdir(exist_ok=True)
    stream = data.generate_stream(drift_start=DRIFT_START)
    report, base_auc = core.monitor(stream)

    psi_alert = core.first_alert_week(report, "max_psi", 0.25, after=6)
    auc_alert = core.first_alert_week(report, "auc_drop", 0.03, after=6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(report.index, report["max_psi"], marker="o", color="black")
    axes[0].axhline(0.1, linestyle=":", color="grey")
    axes[0].axhline(0.25, linestyle="--", color="grey")
    axes[0].axvline(DRIFT_START, color="lightgrey")
    axes[0].set_xlabel("week")
    axes[0].set_ylabel("largest PSI over features")
    axes[0].set_title("Input drift")
    axes[1].plot(report.index, report["auc"], marker="o", color="black")
    axes[1].axhline(base_auc, linestyle="--", color="grey")
    axes[1].axvline(DRIFT_START, color="lightgrey")
    axes[1].set_xlabel("week")
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_title("Model quality")
    fig.tight_layout()
    fig.savefig(OUT / "drift.png", dpi=130)

    report.round(4).to_csv(OUT / "weekly_report.csv")
    last = report.iloc[-1]
    summary = {
        "weeks": int(len(report)),
        "drift_injected_at_week": DRIFT_START,
        "baseline_auc": round(base_auc, 4),
        "first_psi_alert_week": psi_alert,
        "first_auc_drop_alert_week": auc_alert,
        "last_week": {
            "max_psi": round(float(last["max_psi"]), 3),
            "worst_feature": last["worst_feature"],
            "auc": round(float(last["auc"]), 4),
        },
        "weeks_by_status": {k: int(v) for k, v in report["status"].value_counts().items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

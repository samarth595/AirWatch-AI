
"""AirWatch AI: MLOps monitoring, evaluation, retraining and quality gate.

Run from the project root:
    python monitoring.py

The original application trains models in memory. This script creates a
persisted baseline champion on its first run using the original simulator.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from airwatch.anomaly import neighbour_median
from airwatch.forecast import PARAMS, _X
from airwatch.simulator import make_network


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

HORIZON = 6

FEATURE_NAMES = [
    "pm25_lag_0",
    "pm25_lag_1",
    "pm25_lag_2",
    "pm25_lag_3",
    "pm25_lag_6",
    "pm25_lag_12",
    "pm25_lag_24",
    "pm25_mean_6h",
    "neighbour_pm25",
    "log_pm25_scale",
    "temperature",
    "humidity",
    "hour_sin",
    "hour_cos"
]


# --------------------------------------------------
# TASK A: GENERATE AND VALIDATE DATA
# --------------------------------------------------

def make_examples(seed, start, stop):
    """Generate labelled six-hour-ahead PM2.5 examples."""

    net = make_network(seed=seed)

    neighbour = neighbour_median(net, net.pm25)

    times = np.arange(start, stop)

    X, base = _X(
        net.pm25,
        neighbour,
        net.temp,
        net.hum,
        times
    )

    target = (
        net.pm25[:, times + HORIZON].reshape(-1)
        / base
    )

    actual_target = target * base

    return X, target, base, actual_target


def validate_and_clean(name, X, y, base, actual_target):
    """Validate dataset and remove invalid and duplicate rows."""

    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"{name}: Expected {len(FEATURE_NAMES)} feature columns"
        )

    raw_rows = len(X)

    if raw_rows == 0 or not (
        len(X) == len(y) == len(base) == len(actual_target)
    ):
        raise ValueError(f"{name}: Empty or inconsistent dataset")

    # Check missing and infinite values
    valid = (
        np.isfinite(X).all(axis=1)
        & np.isfinite(y)
        & np.isfinite(base)
        & np.isfinite(actual_target)
        & (base > 0)
    )

    invalid_rows = int(raw_rows - valid.sum())

    X = X[valid]
    y = y[valid]
    base = base[valid]
    actual_target = actual_target[valid]

    if len(X) == 0:
        raise ValueError(f"{name}: No valid records found")

    # Remove duplicate feature records
    _, indices = np.unique(
        X,
        axis=0,
        return_index=True
    )

    indices.sort()

    duplicate_rows = len(X) - len(indices)

    X = X[indices]
    y = y[indices]
    base = base[indices]
    actual_target = actual_target[indices]

    stats = {
        "rows_received": int(raw_rows),
        "rows_after_cleaning": int(len(X)),
        "feature_columns": int(X.shape[1]),
        "invalid_or_missing_rows_removed": invalid_rows,
        "duplicate_feature_rows_removed": int(duplicate_rows),
        "missing_values_after_cleaning": 0,
        "target_distribution": {
            "mean": round(float(np.mean(actual_target)), 4),
            "std": round(float(np.std(actual_target)), 4),
            "min": round(float(np.min(actual_target)), 4),
            "max": round(float(np.max(actual_target)), 4)
        }
    }

    print(f"\n{name} VALIDATION")
    print(json.dumps(stats, indent=4))

    return X, y, base, actual_target, stats


# --------------------------------------------------
# TASK B: MODEL EVALUATION
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    return {
        "MAE": round(
            float(mean_absolute_error(actual, predicted)), 4
        ),

        "RMSE": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 4
        ),

        "R2": round(
            float(r2_score(actual, predicted)), 4
        )
    }


# --------------------------------------------------
# TASK C: MODEL TRAINING
# --------------------------------------------------

def train_model(X, y):

    model = HistGradientBoostingRegressor(**PARAMS)

    model.fit(X, y)

    return model


# --------------------------------------------------
# SAVE JSON REPORTS
# --------------------------------------------------

def save_json(path, content):

    path.write_text(
        json.dumps(content, indent=4),
        encoding="utf-8"
    )


# --------------------------------------------------
# MAIN MLOPS WORKFLOW
# --------------------------------------------------

def main():

    print("=" * 60)
    print("AIRWATCH AI - MLOPS MONITORING ASSESSMENT")
    print("=" * 60)

    print("Forecast target: PM2.5 after 6 hours")
    print("Data source: Synthetic AirWatch simulator")

    # Existing training data
    old_raw = make_examples(
        seed=7,
        start=24,
        stop=402
    )

    # New incoming labelled data
    new_raw = make_examples(
        seed=42,
        start=24,
        stop=456
    )

    # Separate evaluation data
    test_raw = make_examples(
        seed=99,
        start=24,
        stop=456
    )

    # Validate all datasets
    old_X, old_y, old_base, old_actual, old_stats = (
        validate_and_clean("EXISTING TRAINING DATA", *old_raw)
    )

    new_X, new_y, new_base, new_actual, new_stats = (
        validate_and_clean("NEW INCOMING DATA", *new_raw)
    )

    test_X, test_y, test_base, test_actual, test_stats = (
        validate_and_clean("EVALUATION DATA", *test_raw)
    )

    # --------------------------------------------------
    # B: LOAD OR CREATE CHAMPION MODEL
    # --------------------------------------------------

    champion_path = ARTIFACTS / "champion_model.joblib"

    if not champion_path.exists():

        champion_model = train_model(old_X, old_y)

        joblib.dump(
            {
                "model": champion_model,
                "feature_names": FEATURE_NAMES,
                "horizon_hours": HORIZON,
                "created_from": "Original AirWatch simulator training history"
            },
            champion_path
        )

        print("\nInitial champion baseline created.")

    champion_bundle = joblib.load(champion_path)

    if (
        champion_bundle.get("feature_names") != FEATURE_NAMES
        or champion_bundle.get("horizon_hours") != HORIZON
    ):
        raise ValueError("Champion model schema does not match.")

    champion = champion_bundle["model"]

    champion_predictions = (
        champion.predict(test_X) * test_base
    )

    champion_metrics = calculate_metrics(
        test_actual,
        champion_predictions
    )

    # --------------------------------------------------
    # C: COMBINE DATA AND TRAIN CANDIDATE MODEL
    # --------------------------------------------------

    combined_X = np.vstack((old_X, new_X))

    combined_y = np.concatenate((old_y, new_y))

    print("\nCombined training records:", len(combined_X))

    candidate = train_model(
        combined_X,
        combined_y
    )

    candidate_path = ARTIFACTS / "candidate_model.joblib"

    joblib.dump(
        {
            "model": candidate,
            "feature_names": FEATURE_NAMES,
            "horizon_hours": HORIZON,
            "training_rows": int(len(combined_X))
        },
        candidate_path
    )

    candidate_predictions = (
        candidate.predict(test_X) * test_base
    )

    candidate_metrics = calculate_metrics(
        test_actual,
        candidate_predictions
    )

    # --------------------------------------------------
    # D: QUALITY GATE
    # --------------------------------------------------

    eligible = (
        candidate_metrics["MAE"]
        <= champion_metrics["MAE"] * 1.02
    )

    decision_name = (
        "ELIGIBLE_FOR_PROMOTION"
        if eligible
        else "RETAIN_EXISTING_CHAMPION"
    )

    decision = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_metric": "MAE",
        "quality_gate_rule": (
            "Candidate MAE <= Champion MAE * 1.02"
        ),
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "decision": decision_name,
        "champion_overwritten": False,
        "candidate_artifact": "artifacts/candidate_model.joblib"
    }

    # --------------------------------------------------
    # E: SAVE REPORTS AND DECISION
    # --------------------------------------------------

    report = {
        "project": "AirWatch AI",
        "task": "Six-hour PM2.5 regression monitoring",

        "data_source": (
            "Synthetic simulator, not live CPCB data"
        ),

        "validation": {
            "existing_training_data": old_stats,
            "new_incoming_data": new_stats,
            "evaluation_data": test_stats
        },

        "combined_training_rows": int(len(combined_X)),
        "evaluation_rows": int(len(test_actual)),

        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,

        "quality_gate": decision
    }

    save_json(
        ARTIFACTS / "evaluation_report.json",
        report
    )

    save_json(
        ARTIFACTS / "quality_gate_latest.json",
        decision
    )

    # Maintain decision history
    history_path = ARTIFACTS / "quality_gate_decisions.jsonl"

    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(decision) + "\n")

    # --------------------------------------------------
    # DISPLAY FINAL RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print("\nCHAMPION MODEL:")
    print(json.dumps(champion_metrics, indent=4))

    print("\nCANDIDATE MODEL:")
    print(json.dumps(candidate_metrics, indent=4))

    print("\nQUALITY GATE:", decision_name)

    print("Champion overwritten: False")

    print("\nGENERATED ARTIFACTS:")

    for filename in [
        "champion_model.joblib",
        "candidate_model.joblib",
        "evaluation_report.json",
        "quality_gate_latest.json",
        "quality_gate_decisions.jsonl"
    ]:
        print(" -", ARTIFACTS / filename)

    # Save execution evidence
    evidence = [
        "AIRWATCH AI MLOPS ASSESSMENT",
        f"Execution UTC: {decision['timestamp_utc']}",
        f"Incoming rows: {new_stats['rows_received']}",
        f"Valid incoming rows: {new_stats['rows_after_cleaning']}",
        f"Invalid rows removed: {new_stats['invalid_or_missing_rows_removed']}",
        f"Duplicate rows removed: {new_stats['duplicate_feature_rows_removed']}",
        f"Champion metrics: {champion_metrics}",
        f"Candidate metrics: {candidate_metrics}",
        f"Quality gate: {decision_name}",
        "Champion overwritten: False",
        "Data source: Synthetic simulator"
    ]

    (ARTIFACTS / "execution_evidence.txt").write_text(
        "\n".join(evidence) + "\n",
        encoding="utf-8"
    )

    print("\nASSESSMENT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()

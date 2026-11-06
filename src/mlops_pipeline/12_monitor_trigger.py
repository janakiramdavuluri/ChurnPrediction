import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DRIFT_SCRIPT = Path("src/mlops_pipeline/12_drift_analysis.py")
DRIFT_REPORT = Path("outputs/drift/drift_analysis_report.csv")
NEW_DATA_PATH = Path("data/raw/new_churn.csv")
PIPELINE_SCRIPT = Path("src/mlops_pipeline/churn_prediction_pipeline.py")
TRIGGER_REPORT = Path("outputs/monitoring/monitor_trigger_report.json")
DATA_LOAD_REPORT = Path("artifacts/data_load_report.json")


def run_drift_analysis():
    print("[INFO] Running drift analysis...")
    result = subprocess.run([sys.executable, str(DRIFT_SCRIPT)])
    return result.returncode == 0


def detect_drift():
    if not DRIFT_REPORT.exists():
        raise FileNotFoundError(f"Drift report not found: {DRIFT_REPORT}")

    df = pd.read_csv(DRIFT_REPORT)

    for column in df.columns:
        values = df[column].astype(str).str.strip().str.upper()
        if values.eq("DRIFT").any():
            return True

    return False


def get_data_load_status():
    if not DATA_LOAD_REPORT.exists():
        return None

    with DATA_LOAD_REPORT.open("r", encoding="utf-8") as f:
        report = json.load(f)

    return report.get("status")


def save_trigger_report(report):
    TRIGGER_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with TRIGGER_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=str)


def run_monitor_trigger():
    print("=" * 70)
    print("LAB 13 MONITORING + RETRAINING TRIGGER")
    print("=" * 70)

    drift_analysis_passed = run_drift_analysis()

    if not drift_analysis_passed:
        report = {
            "status": "drift_analysis_failed",
            "drift_detected": False,
            "new_data_present": NEW_DATA_PATH.exists(),
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        save_trigger_report(report)
        print("[ERROR] Drift analysis failed.")
        return False

    drift_detected = detect_drift()
    new_data_present = NEW_DATA_PATH.exists()

    print(f"[INFO] Drift detected: {drift_detected}")
    print(f"[INFO] New dataset present: {new_data_present}")

    if not drift_detected:
        report = {
            "status": "no_drift",
            "drift_detected": False,
            "new_data_present": new_data_present,
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        save_trigger_report(report)
        print("[INFO] No drift detected. Waiting for future monitoring cycles.")
        return True

    if not new_data_present:
        report = {
            "status": "drift_waiting_for_data",
            "drift_detected": True,
            "new_data_present": False,
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        save_trigger_report(report)
        print("[INFO] Drift detected, but no new dataset is available.")
        print("[INFO] Waiting for new_churn.csv.")
        return True

    print("[INFO] Drift detected and new dataset is available.")
    print("[INFO] Full pipeline execution requires approval.")

    report = {
        "status": "retraining_pending_approval",
        "drift_detected": True,
        "new_data_present": True,
        "pipeline_called": False,
        "new_data_ingested": False,
        "retraining_executed": False,
        "production_model_changed": False,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    save_trigger_report(report)

    print("[SUCCESS] Monitoring report saved.")
    print("[INFO] Retraining was not executed.")

    return True


if __name__ == "__main__":
    passed = run_monitor_trigger()
    sys.exit(0 if passed else 1)

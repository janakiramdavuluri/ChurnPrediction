import os
import sys
import json
import pandas as pd
import yaml


def load_data():
    print("[INFO] Starting Data Loading...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {df.shape[0]} rows, {df.shape[1]} columns from {data_path}")

    report = {
        "stage": "ML Pipeline - Data Loading",
        "source_file": data_path,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/data_load_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("[SUCCESS] Data loaded successfully. Report saved to artifacts/data_load_report.json")
    return True


if __name__ == "__main__":
    passed = load_data()
    sys.exit(0 if passed else 1)

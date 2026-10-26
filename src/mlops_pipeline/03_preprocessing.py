import os
import sys
import json
import joblib
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def clean_dataframe(df, config):
    """Apply deterministic dataset cleaning shared by training and inference."""
    df = df.copy()
    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if target_column in df.columns:
        df[target_column] = (
            df[target_column]
            .apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
            .astype(int)
        )

    return df


def split_raw_data(df, config):
    """Clean the raw dataset and create the reproducible train/test split."""
    target_column = config["data"]["target_column"]
    cleaned = clean_dataframe(df, config)

    X = cleaned.drop(columns=[target_column])
    y = cleaned[target_column]

    return train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )


def build_preprocessor(X_train):
    """Create the single canonical feature-preprocessing pipeline used by ML Pipeline."""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    ), cat_cols, num_cols


def run_preprocessing():
    print("[INFO] Starting Preprocessing...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    target_column = config["data"]["target_column"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    raw_df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = split_raw_data(raw_df, config)

    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)

    print("[INFO] Fitting preprocessing pipeline on training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Persist transformed matrices for the validation stage and reproducibility checks.
    np.save(os.path.join(processed_dir, "X_train_final.npy"), X_train_final)
    np.save(os.path.join(processed_dir, "X_test_final.npy"), X_test_final)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train.to_numpy(dtype=np.int64))
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.to_numpy(dtype=np.int64))

    # Persist the raw split used by the training/evaluation stages so the registered
    # model can own preprocessing + model in one deployable Pipeline artifact.
    X_train.to_csv(os.path.join(processed_dir, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test_raw.csv"), index=False)

    joblib.dump(preprocessor, config["paths"]["preprocessor"])

    metadata = {
        "dataset_name": "Telco Customer Churn",
        "target_column": target_column,
        "train_shape_raw": list(X_train.shape),
        "test_shape_raw": list(X_test.shape),
        "train_shape_transformed": list(X_train_final.shape),
        "test_shape_transformed": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "split_test_size": config["split"]["test_size"],
        "split_random_state": config["split"]["random_state"],
        "raw_train_path": os.path.join(processed_dir, "X_train_raw.csv"),
        "raw_test_path": os.path.join(processed_dir, "X_test_raw.csv"),
    }

    with open(os.path.join(processed_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Preprocessing complete. Preprocessor saved to {config['paths']['preprocessor']}")
    return True


if __name__ == "__main__":
    passed = run_preprocessing()
    sys.exit(0 if passed else 1)

import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

def get_telco_schema():
    """Defines the strict schema specification for the Telco Churn dataset."""
    return pa.DataFrameSchema({
        "customerID": Column(pa.String),
        "gender": Column(pa.String, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(pa.Int, Check.isin([0, 1])),
        "Partner": Column(pa.String, Check.isin(["Yes", "No"])),
        "Dependents": Column(pa.String, Check.isin(["Yes", "No"])),
        "tenure": Column(pa.Int, Check.ge(0)),
        "PhoneService": Column(pa.String, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(pa.String, Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(pa.String, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "OnlineBackup": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "DeviceProtection": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "TechSupport": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "StreamingTV": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "StreamingMovies": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "Contract": Column(pa.String, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(pa.String, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(pa.String, Check.isin([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])),
        "MonthlyCharges": Column(pa.Float, Check.ge(0.0)),
        "TotalCharges": Column(pa.String, nullable=True),
        "Churn": Column(pa.String, Check.isin(["Yes", "No"]))
    }, strict=True)

def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_telco_schema()
    
    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        
        failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
        print(failures.to_string())
        
        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False

if __name__ == "__main__":
    # When run directly, it only validates the clean production data
    data_path = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn (1).csv'
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        validate_schema(raw_df, output_report_name="baseline_validation.csv")
    else:
        print(f"[ERROR] Target file not found at: {data_path}")
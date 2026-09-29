import os
import json
import mlflow
from mlflow.tracking import MlflowClient

def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")
    
    client = MlflowClient()
    model_name = "Telco_Churn_Production_Model"
    
    try:
        # Search for the model specifically in Production
        production_models = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in production_models if mv.current_stage == "Production"), None)
        
        if not champion:
            print("[ERROR] No model found in Production stage!")
            return
            
        # Get the run details to extract metrics and parameters
        run = client.get_run(champion.run_id)
        
        # Build the deployment-ready artifact dictionary
        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source
            },
            "performance_metrics": {
                "recall": run.data.metrics.get("recall"),
                "f1_score": run.data.metrics.get("f1_score"),
                "roc_auc": run.data.metrics.get("roc_auc"),
                "accuracy": run.data.metrics.get("accuracy"),
                "precision": run.data.metrics.get("precision")
            },
            "hyperparameters": run.data.params,
            "preprocessing_dependency": "models/preprocessor.pkl"
        }
        
        # Save the report
        os.makedirs("artifacts", exist_ok=True)
        report_path = "artifacts/production_model_report.json"
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        print(f"[SUCCESS] Report successfully generated for Version {champion.version}")
        print(f"[INFO] Saved to: {report_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")

if __name__ == "__main__":
    generate_registry_report()
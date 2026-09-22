import subprocess
import sys

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 4: MLflow Experiment Tracking")
    print("[INFO] =========================================")
    
    scripts = [
        "src/preprocess.py",
        "src/train_mlflow.py",
        "src/validate_reproducibility.py"
    ]
    
    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run([sys.executable, script])
        
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed.")
            sys.exit(1)
            
    print("\n[SUCCESS] Lab 4 Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()
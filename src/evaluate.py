import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_evaluation():
    print("Starting Model Evaluation...")
    
    # 1. Load test data and the trained model
    X_test_final = np.load('data/processed/X_test_final.npy')
    y_test = np.load('data/processed/y_test.npy')
    model = joblib.load('models/random_forest_baseline.pkl')
    
    # 2. Make predictions
    y_pred = model.predict(X_test_final)
    
    # 3. Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-------------------------------\n")
    
    # 4. Error Analysis Export
    # We load the raw dataset just to get the original customer rows back
    df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn (1).csv')
    _, X_test_raw = train_test_split(df.drop('Churn', axis=1), test_size=0.2, random_state=42, stratify=df['Churn'])
    
    errors_df = X_test_raw.copy()
    errors_df['Actual_Churn'] = y_test
    errors_df['Predicted_Churn'] = y_pred
    
    false_negatives = errors_df[(errors_df['Actual_Churn'] == 1) & (errors_df['Predicted_Churn'] == 0)]
    false_positives = errors_df[(errors_df['Actual_Churn'] == 0) & (errors_df['Predicted_Churn'] == 1)]
    
    false_negatives.to_csv('outputs/false_negatives.csv', index=False)
    false_positives.to_csv('outputs/false_positives.csv', index=False)
    
    print("Evaluation complete and error analysis files saved!")

if __name__ == "__main__":
    run_evaluation()
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def run_preprocessing():
    print("Starting Preprocessing Pipeline...")
    
    # 1. Load the raw data
    data_path = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn (1).csv'
    df = pd.read_csv(data_path)
    
    # 2. Data Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # Unconditional string to binary integer conversion
    df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0).astype(int)
        
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Separate Column Types
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'str']).columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # 5. Scale & Encode
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])
    
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])
    
    # Combine Features
    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))
    
    # 6. Save Artifacts with explicit integer type casting
    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(ohe, 'models/ohe.pkl')
    
    # Save Metadata
    metadata = {
        "dataset_name": "Telco Customer Churn",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols)
    }
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    run_preprocessing()
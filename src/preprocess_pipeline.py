import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")
    
    # 1. Load the raw data
    data_path = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn (1).csv'
    df = pd.read_csv(data_path)
    
    # 2. Basic Data Formatting
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # Coerce TotalCharges to numeric (blanks become NaN, which the pipeline will handle)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Target string to binary integer conversion
    df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0).astype(int)
        
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 3. Train-Test Split (Done before transformations to prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Separate Column Types
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 5. Build Modular Scikit-Learn Pipelines
    # Numerical pipeline: Fill missing (NaN) with 0, then scale
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: One-hot encode
    cat_pipeline = Pipeline(steps=[
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    
    # 6. Execute Pipeline
    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    
    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)
    
    # 7. Save Artifacts
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))
    
    # Save the single preprocessor artifact instead of separate scaler/ohe
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    # Save Metadata
    metadata = {
        "dataset_name": "Telco Customer Churn",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols
    }
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")

if __name__ == "__main__":
    run_preprocessing()
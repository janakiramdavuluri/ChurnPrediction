import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def run_training():
    print("Starting Model Training...")
    
    # 1. Load processed training data
    X_train_final = np.load('data/processed/X_train_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    
    # 2. Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced'
    )
    
    model.fit(X_train_final, y_train)
    
    # 3. Save the trained model
    joblib.dump(model, 'models/random_forest_baseline.pkl')
    print("Model training complete and saved to disk!")

if __name__ == "__main__":
    run_training()
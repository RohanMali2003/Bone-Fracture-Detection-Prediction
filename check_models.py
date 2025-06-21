import os
import numpy as np
import joblib
import sys

def check_model_compatibility():
    print("Checking model compatibility...")
    
    # Check if model_weights directory exists
    if not os.path.exists('model_weights'):
        print("ERROR: model_weights directory doesn't exist!")
        return False
    
    # Check if all necessary model files exist
    required_files = ['svc_model.pkl', 'scaler.pkl', 'anomaly_model.pkl']
    missing_files = []
    
    for file in required_files:
        path = os.path.join('model_weights', file)
        if not os.path.exists(path):
            missing_files.append(file)
    
    if missing_files:
        print(f"ERROR: Missing model files: {', '.join(missing_files)}")
        return False
    
    # Load SVC model and check dimensions
    try:
        svc_model = joblib.load(os.path.join('model_weights', 'svc_model.pkl'))
        print(f"SVC model loaded successfully")
        
        # Check if model has n_features_in_ attribute (indicating it's been fitted)
        if hasattr(svc_model, 'n_features_in_'):
            print(f"SVC model expects {svc_model.n_features_in_} input features")
        else:
            print("WARNING: SVC model doesn't have n_features_in_ attribute (not fitted?)")
    except Exception as e:
        print(f"ERROR loading SVC model: {e}")
        return False
    
    # Create a dummy input vector with the correct dimensions to test prediction
    if hasattr(svc_model, 'n_features_in_'):
        try:
            dummy_input = np.random.rand(1, svc_model.n_features_in_)
            prediction = svc_model.predict(dummy_input)
            print(f"SVC model prediction test passed. Result: {prediction}")
        except Exception as e:
            print(f"ERROR predicting with SVC model: {e}")
            return False
    
    # Load and check scaler
    try:
        scaler = joblib.load(os.path.join('model_weights', 'scaler.pkl'))
        print("Scaler loaded successfully")
    except Exception as e:
        print(f"ERROR loading scaler: {e}")
        return False
        
    # Load and check anomaly model
    try:
        anomaly_model = joblib.load(os.path.join('model_weights', 'anomaly_model.pkl'))
        print("Anomaly detection model loaded successfully")
    except Exception as e:
        print(f"ERROR loading anomaly detection model: {e}")
        return False
    
    print("All models loaded and verified successfully!")
    return True

if __name__ == "__main__":
    success = check_model_compatibility()
    if not success:
        print("\nModel compatibility check FAILED. You may need to retrain the models.")
        sys.exit(1)
    else:
        print("\nModel compatibility check PASSED. Models should work with the application.")
        sys.exit(0)

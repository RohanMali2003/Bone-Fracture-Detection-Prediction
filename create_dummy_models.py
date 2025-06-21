import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import os

# Create directory if it doesn't exist
os.makedirs('model_weights', exist_ok=True)

# Create a minimal SVC model and fit it with dummy data
X = np.random.rand(10, 10)  # 10 samples, 10 features
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Binary labels

# Create the models
svc_model = SVC()
svc_model.fit(X, y)

# Create StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create OneClassSVM
anomaly_model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
anomaly_model.fit(X_scaled)

# Save the models
joblib.dump(svc_model, 'model_weights/svc_model.pkl')
joblib.dump(scaler, 'model_weights/scaler.pkl')
joblib.dump(anomaly_model, 'model_weights/anomaly_model.pkl')

print("Created minimal models in model_weights directory")

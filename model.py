import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC
import joblib
import matplotlib.pyplot as plt
import io
from PIL import Image

class BoneFractureModel:
    def __init__(self, model_path=None):
        # Initialize basic ML model
        self.svc_model = SVC()
        
        # Initialize feature extractor
        self.base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.feature_extractor = Model(inputs=self.base_model.input, outputs=self.base_model.output)
        
        # Initialize anomaly detection model
        self.anomaly_model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
        self.scaler = StandardScaler()
        
        # For Grad-CAM
        self.full_model = MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
        
        # If model path is provided, load the saved models
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_features(self, img):
        """Extract features from an image using MobileNetV2"""
        if isinstance(img, str):  # If img is a file path
            img = load_img(img, target_size=(224, 224))
            img_array = img_to_array(img)
        else:  # If img is already a numpy array
            img_array = cv2.resize(img, (224, 224))
            if len(img_array.shape) == 2:  # Grayscale to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        features = self.feature_extractor.predict(img_array)  # Extract features
        return features.flatten()  # Convert to 1D vector
    
    def preprocess_for_svc(self, img):
        """Preprocess image for SVC model"""
        if isinstance(img, str):  # If img is a file path
            img = cv2.imread(img, 0)  # Read as grayscale
        
        # Resize and reshape
        img = cv2.resize(img, (200, 200))
        img = img.reshape(1, -1) / 255.0
        return img
    
    def predict_fracture(self, img):
        """Predict if a bone is fractured using SVC model"""
        img_processed = self.preprocess_for_svc(img)
        prediction = self.svc_model.predict(img_processed)
        return "fractured" if prediction[0] == 0 else "not_fractured"
    
    def detect_anomaly(self, img):
        """Detect anomalies (potential fractures) using One-Class SVM"""
        feature_vector = self.extract_features(img)
        feature_vector_scaled = self.scaler.transform([feature_vector])
        prediction = self.anomaly_model.predict(feature_vector_scaled)
        
        if prediction[0] == -1:
            return "Potential Fracture Risk"
        else:
            return "Normal"
    
    def get_gradcam_heatmap(self, img, class_index=None):
        """Generate Grad-CAM heatmap to visualize important regions"""
        # Preprocess the image
        if isinstance(img, str):
            img = load_img(img, target_size=(224, 224))
            img_array = img_to_array(img)
        else:
            img_array = cv2.resize(img, (224, 224))
            if len(img_array.shape) == 2:  # Grayscale to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Create a grad model
        grad_model = Model(
            inputs=self.full_model.input,
            outputs=[self.full_model.get_layer("Conv_1").output, self.full_model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_index is None:
                class_index = np.argmax(predictions[0])
            loss = predictions[:, class_index]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = np.zeros(shape=conv_outputs.shape[:2])
        
        for i in range(conv_outputs.shape[-1]):
            heatmap += pooled_grads[i] * conv_outputs[:, :, i]
        
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        return heatmap
    
    def generate_gradcam_visualization(self, img):
        """Generate Grad-CAM visualization for an image"""
        # Get the original image
        if isinstance(img, str):
            original_img = cv2.imread(img)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        else:
            if len(img.shape) == 2:  # Grayscale to RGB
                original_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                original_img = img.copy()
        
        original_img = cv2.resize(original_img, (224, 224))
        
        # Get heatmap
        heatmap = self.get_gradcam_heatmap(img)
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        # Generate and return the figure with all images
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].imshow(original_img)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
        
        ax[1].imshow(heatmap)
        ax[1].set_title("Grad-CAM Heatmap")
        ax[1].axis("off")
        
        ax[2].imshow(superimposed_img)
        ax[2].set_title("Overlayed Image")
        ax[2].axis("off")
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return Image.open(buf)
    
    def train_models(self, train_data_path):
        """Train SVC and Anomaly Detection models"""
        # Load and preprocess data
        X = []
        Y = []
        classes = {'fractured': 0, 'not_fractured': 1}
        
        # For SVC model
        X_svc = []
        Y_svc = []
        
        # Prepare data for SVC
        for cls in classes:
            pth = os.path.join(train_data_path, cls)
            for j in os.listdir(pth):
                img_path = os.path.join(pth, j)
                # For SVC
                img = cv2.imread(img_path, 0)
                img = cv2.resize(img, (200, 200))
                X_svc.append(img)
                Y_svc.append(classes[cls])
                
                # For feature extraction (limiting to 200 per class)
                if len([x for x, y in zip(X, Y) if y == classes[cls]]) < 200:
                    feature_vector = self.extract_features(img_path)
                    X.append(feature_vector)
                    Y.append(classes[cls])
        
        # Convert to numpy arrays
        X_svc = np.array(X_svc)
        Y_svc = np.array(Y_svc)
        X = np.array(X)
        Y = np.array(Y)
        
        # Reshape for SVC
        X_svc_reshaped = X_svc.reshape(len(X_svc), -1)
        
        # Split data
        from sklearn.model_selection import train_test_split
        xtrain, xtest, ytrain, ytest = train_test_split(X_svc_reshaped, Y_svc, random_state=10, test_size=.20)
        
        # Normalize
        xtrain = xtrain / 255
        xtest = xtest / 255
        
        # Train SVC model
        self.svc_model = SVC()
        self.svc_model.fit(xtrain, ytrain)
        
        # Train One-Class SVM for anomaly detection
        X_non_fractured = X[Y == 1]  # Y==1 means non-fractured bones
        self.scaler = StandardScaler()
        X_non_fractured_scaled = self.scaler.fit_transform(X_non_fractured)
        
        self.anomaly_model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
        self.anomaly_model.fit(X_non_fractured_scaled)
        
        # Evaluate models
        svc_accuracy = self.svc_model.score(xtest, ytest)
        return svc_accuracy
    
    def save_model(self, model_path):
        """Save the trained models to disk"""
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(self.svc_model, os.path.join(model_path, 'svc_model.pkl'))
        joblib.dump(self.anomaly_model, os.path.join(model_path, 'anomaly_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.pkl'))
    
    def load_model(self, model_path):
        """Load trained models from disk"""
        svc_path = os.path.join(model_path, 'svc_model.pkl')
        anomaly_path = os.path.join(model_path, 'anomaly_model.pkl')
        scaler_path = os.path.join(model_path, 'scaler.pkl')
        
        if os.path.exists(svc_path):
            self.svc_model = joblib.load(svc_path)
        
        if os.path.exists(anomaly_path):
            self.anomaly_model = joblib.load(anomaly_path)
            
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
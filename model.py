"""
Bone Fracture Detection & Prediction Model

This module implements a machine learning model that can detect bone fractures
from X-ray images and predict fracture risk using a combination of deep learning
feature extraction and traditional machine learning techniques.

Author: Rohan Mali
"""

import os
import cv2
import numpy as np
import tensorflow as tf
# Configure TensorFlow to use CPU and avoid errors with incompatible GPU versions
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC
import joblib
import matplotlib.pyplot as plt
import io
from PIL import Image

# Use TensorFlow compatibility mode and silence warnings
import logging
tf.get_logger().setLevel(logging.ERROR)

class BoneFractureModel:
    def __init__(self, model_path=None):
        """
        Initialize the Bone Fracture Detection model.
        
        Args:
            model_path (str, optional): Path to pre-trained model weights.
                If provided, the model will load weights from this path.
        """
        # Initialize Support Vector Classifier for binary classification
        self.svc_model = SVC()
        
        # Initialize feature extractor based on MobileNetV2
        # We use a pre-trained model without the top layers for feature extraction
        self.base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.feature_extractor = Model(inputs=self.base_model.input, outputs=self.base_model.output)
        
        # Initialize anomaly detection model for identifying unusual patterns
        self.anomaly_model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
        self.scaler = StandardScaler()  # For scaling features before anomaly detection
        
        # For Grad-CAM visualization (uses the complete MobileNetV2 model)
        self.full_model = MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
        
        # Track if models are properly loaded
        self.models_loaded = False
        
        # If model path is provided, try to load the saved models
        if model_path:
            if os.path.exists(model_path):
                try:
                    self.load_model(model_path)
                    self.models_loaded = True
                    print(f"Models loaded successfully from {model_path}")
                except Exception as e:
                    print(f"Error loading models: {e}")
            else:
                print(f"Warning: Model path {model_path} does not exist. Models will need to be trained.")
    
    def extract_features(self, img):
        """
        Extract features from an image using MobileNetV2 pre-trained model.
        
        Args:
            img: Either a file path string or a numpy array containing the image
            
        Returns:
            numpy.ndarray: Flattened feature vector
        """
        if isinstance(img, str):  # If img is a file path
            # Use PIL for image loading instead of keras function
            pil_img = Image.open(img).resize((224, 224))
            img_array = np.array(pil_img.convert('RGB'))
        else:  # If img is already a numpy array
            img_array = cv2.resize(img, (224, 224))
            if len(img_array.shape) == 2:  # Grayscale to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        features = self.feature_extractor.predict(img_array)  # Extract features
        return features.reshape(features.shape[0], -1)  # Flatten the features
    
    def predict_fracture(self, img):
        """
        Predict whether an image shows a bone fracture using SVC.
        
        Args:
            img: Either a file path string or a numpy array containing the image
            
        Returns:
            str: "Fracture" or "No Fracture" based on model prediction
        """
        features = self.extract_features(img)
        # Check if model is fitted before prediction
        from sklearn.utils.validation import check_is_fitted
        try:
            check_is_fitted(self.svc_model)
            prediction = self.svc_model.predict(features)
            # Return a consistent format: "Fracture" or "No Fracture"
            # Based on train_model labeling: 0 = normal, 1 = fracture
            result = "Fracture" if prediction[0] == 1 else "No Fracture"
            print(f"Prediction value: {prediction[0]}, Result: {result}")
            return result
        except Exception as e:
            # If not fitted or feature mismatch occurs, return error message
            print(f"SVC model error: {e}")
            # Don't attempt to train a model on the fly as it would need 2+ classes
            return "Model not properly trained. Please train the model with proper training data first."
            
    def detect_anomaly(self, img):
        """
        Detect anomalies in bone images using One-Class SVM.
        
        Args:
            img: Either a file path string or a numpy array containing the image
            
        Returns:
            str: "Potential Fracture Risk" or "Normal" based on anomaly detection
        """
        features = self.extract_features(img)
        
        # Check if models are fitted
        from sklearn.utils.validation import check_is_fitted
        try:
            # Check if scaler is fitted
            check_is_fitted(self.scaler)
            # Scale the features
            features_scaled = self.scaler.transform(features)
            
            # Check if anomaly model is fitted
            check_is_fitted(self.anomaly_model)
            # Predict using anomaly detection model
            prediction = self.anomaly_model.predict(features_scaled)
            
            if prediction[0] == -1:
                return "Potential Fracture Risk"
            else:
                return "Normal"
        except:
            # If models aren't fitted, return a message
            return "Anomaly detection models not properly trained. Please train the model first."
    
    def get_gradcam_heatmap(self, img, class_index=None):
        """
        Generate Grad-CAM heatmap to visualize important regions in the image.
        
        Args:
            img: Either a file path string or a numpy array containing the image
            class_index (int, optional): Class index to generate heatmap for. Defaults to predicted class.
            
        Returns:
            tuple: (cam, superimposed_img) containing the heatmap and superimposed image
        """
        # Preprocess the image
        if isinstance(img, str):
            # Use PIL for image loading instead of keras function
            pil_img = Image.open(img).resize((224, 224))
            img_array = np.array(pil_img.convert('RGB'))
        else:
            img_array = cv2.resize(img, (224, 224))
            if len(img_array.shape) == 2:  # Grayscale to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Use the last convolutional layer for Grad-CAM
        try:
            last_conv_layer = next(l for l in reversed(self.full_model.layers) if isinstance(l, tf.keras.layers.Conv2D))
            last_conv_layer_name = last_conv_layer.name
        except (StopIteration, AttributeError):
            # Fallback to a known layer name in MobileNetV2
            last_conv_layer_name = "Conv_1"
        
        # Create a model that outputs both the last convolutional layer and predictions
        grad_model = tf.keras.models.Model(
            inputs=[self.full_model.inputs], 
            outputs=[self.full_model.get_layer(last_conv_layer_name).output, self.full_model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]
        
        # Extract gradients
        grads = tape.gradient(loss, conv_output)
        
        # Compute guided gradients
        guided_grads = tf.cast(conv_output > 0, "float32") * tf.cast(grads > 0, "float32") * grads
        
        # Average gradients spatially
        weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
        
        # Create a weighted combination of filters
        cam = tf.reduce_sum(tf.multiply(weights, conv_output), axis=-1)
        
        # Process CAM
        cam = cam.numpy()[0]
        cam = np.maximum(cam, 0)  # ReLU
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)  # Normalize
        cam = cv2.resize(cam, (224, 224))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert input image back to uint8
        input_img = (img_array[0] * 255).astype(np.uint8)
        
        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(input_img, 0.6, heatmap, 0.4, 0)
        
        # Return both the heatmap and superimposed image
        return cam, superimposed_img
    
    def visualize_gradcam(self, img, class_index=None):
        """
        Visualize Grad-CAM and return as PIL Image.
        
        Args:
            img: Either a file path string or a numpy array containing the image
            class_index (int, optional): Class index to generate heatmap for
            
        Returns:
            BytesIO: Buffer containing the visualization as a PNG image
        """
        _, superimposed_img = self.get_gradcam_heatmap(img, class_index)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(superimposed_img)
        
        # Create a buffer to save the image
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        
        return buf
    
    def save_model(self, path="model_weights"):
        """
        Save the trained models to disk.
        
        Args:
            path (str): Directory to save the models
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save SVC model
        joblib.dump(self.svc_model, os.path.join(path, "svc_model.pkl"))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        
        # Save anomaly detection model
        joblib.dump(self.anomaly_model, os.path.join(path, "anomaly_model.pkl"))
    
    def load_model(self, path="model_weights"):
        """
        Load models from disk.
        
        Args:
            path (str): Directory containing saved models
        """
        # Load SVC model
        svc_path = os.path.join(path, "svc_model.pkl")
        if os.path.exists(svc_path):
            self.svc_model = joblib.load(svc_path)
        
        # Load scaler
        scaler_path = os.path.join(path, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load anomaly detection model
        anomaly_path = os.path.join(path, "anomaly_model.pkl")
        if os.path.exists(anomaly_path):
            self.anomaly_model = joblib.load(anomaly_path)
    
    def train_model(self, normal_image_dir, fracture_image_dir):
        """
        Train both SVC and anomaly detection models.
        
        Args:
            normal_image_dir (str): Directory containing normal bone X-ray images
            fracture_image_dir (str): Directory containing fractured bone X-ray images
            
        Returns:
            float: Test accuracy of the trained SVC model
        """
        X_svc = []  # Features for SVC
        Y_svc = []  # Labels for SVC
        X = []      # Features for anomaly detection (only normal samples)
        Y = []      # Dummy labels for anomaly detection
        
        # Load normal images
        for filename in os.listdir(normal_image_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(normal_image_dir, filename)
                features = self.extract_features(filepath)
                X_svc.append(features[0])
                Y_svc.append(0)  # Label 0 for normal (no fracture)
                X.append(features[0])
                Y.append(1)  # Dummy label for anomaly detection
        
        # Load fracture images
        for filename in os.listdir(fracture_image_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(fracture_image_dir, filename)
                features = self.extract_features(filepath)
                X_svc.append(features[0])
                Y_svc.append(1)  # Label 1 for fracture
        
        # Convert to numpy arrays
        X_svc = np.array(X_svc)
        Y_svc = np.array(Y_svc)
        X = np.array(X)
        Y = np.array(Y)
        
        # Split data for SVC
        train_ratio = 0.8
        indices = np.random.permutation(len(X_svc))
        train_count = int(len(X_svc) * train_ratio)
        train_indices = indices[:train_count]
        test_indices = indices[train_count:]
        
        xtrain = X_svc[train_indices]
        ytrain = Y_svc[train_indices]
        xtest = X_svc[test_indices]
        ytest = Y_svc[test_indices]
        
        # Train SVC model
        self.svc_model = SVC()
        self.svc_model.fit(xtrain, ytrain)
        
        # Train anomaly detection model on normal samples only
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.anomaly_model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
        self.anomaly_model.fit(X_scaled)
        
        # Save trained models
        self.save_model()
        
        # Return test accuracy
        return self.svc_model.score(xtest, ytest)
    
    def generate_gradcam_visualization(self, img):
        """
        Generate Grad-CAM visualization for an image.
        
        Args:
            img: Either a file path string or a numpy array containing the image
            
        Returns:
            PIL.Image: Image containing original, heatmap, and overlayed visualizations
        """
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
        heatmap, _ = self.get_gradcam_heatmap(img)
        
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
        """
        Train SVC and Anomaly Detection models.
        
        Args:
            train_data_path (str): Path to training data directory
            
        Returns:
            float: Accuracy of the trained SVC model
        """
        # Load and preprocess data
        X = []
        Y = []
        # IMPORTANT: This class mapping is REVERSED from what's used in train_model!
        # Here: fractured = 0, not_fractured = 1
        # In train_model: normal/not_fractured = 0, fractured = 1
        # We'll follow train_model's convention in the predict_fracture method
        classes = {'fractured': 1, 'not_fractured': 0}
        
        # For SVC model
        X_svc = []
        Y_svc = []
        
        # Prepare data for SVC
        for cls in classes:
            pth = os.path.join(train_data_path, cls)
            if os.path.exists(pth):
                for j in os.listdir(pth):
                    img_path = os.path.join(pth, j)
                    # For SVC
                    img = cv2.imread(img_path, 0)
                    if img is not None:
                        img = cv2.resize(img, (200, 200))
                        X_svc.append(img)
                        Y_svc.append(classes[cls])
                        
                        # For feature extraction (limiting to 200 per class)
                        if len([x for x, y in zip(X, Y) if y == classes[cls]]) < 200:
                            try:
                                feature_vector = self.extract_features(img_path)
                                X.append(feature_vector[0])
                                Y.append(classes[cls])
                            except Exception as e:
                                print(f"Error extracting features from {img_path}: {e}")
        
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
        if len(X) > 0:
            # Filter for non-fractured bones (Y==0)
            mask = np.array(Y) == 0
            if np.any(mask):
                X_non_fractured = X[mask]
                self.scaler = StandardScaler()
                X_non_fractured_scaled = self.scaler.fit_transform(X_non_fractured)
                
                self.anomaly_model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
                self.anomaly_model.fit(X_non_fractured_scaled)
        
        # Evaluate models
        svc_accuracy = self.svc_model.score(xtest, ytest)
        
        # Save models
        self.save_model("model_weights")
        
        return svc_accuracy

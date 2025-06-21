import streamlit as st
import os
import numpy as np
import cv2
# Try to import from model_new first, then model_fix, and finally fallback to model
try:
    from model_new import BoneFractureModel
    print("Using new model implementation")
except ImportError:
    try:
        from model_fix import BoneFractureModel
        print("Using fixed model implementation")
    except ImportError:
        from model import BoneFractureModel
        print("Using original model implementation")
from PIL import Image
import matplotlib.pyplot as plt
import io
import tempfile

# Page configuration
st.set_page_config(
    page_title="Bone Fracture Detection & Prediction",
    page_icon="🦴",
    layout="wide"
)

# Initialize model
@st.cache_resource
def load_model():
    """Load the model once and cache it"""
    model_dir = "model_weights"
    model = BoneFractureModel(model_path=model_dir if os.path.exists(model_dir) else None)
    return model

model = load_model()

# App title and description
st.title("Bone Fracture Detection & Prediction")
st.markdown("""
This application uses machine learning to detect bone fractures from X-ray images and predict fracture risk.
Upload an X-ray image to get started!
""")

# Sidebar
with st.sidebar:
    st.header("Options")
    detection_mode = st.radio(
        "Choose Detection Mode:",
        ["SVC Fracture Detection", "Anomaly Detection", "Grad-CAM Visualization"]
    )
    
    st.markdown("---")
    st.header("About")
    st.info("""
    This application is built using:
    - Streamlit for the interface
    - TensorFlow/MobileNetV2 for feature extraction
    - Support Vector Classifier for fracture detection
    - One-Class SVM for anomaly detection
    - Grad-CAM for visualization
    """)

# Upload image
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

# Main content
if uploaded_file is not None:
    # Save uploaded file to temp directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
    
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Uploaded X-ray Image")
        image = Image.open(uploaded_file)
        st.image(image, width=300)
    
    with col2:
        st.subheader("Analysis Results")
        
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process based on selected mode
        if detection_mode == "SVC Fracture Detection":
            with st.spinner("Analyzing with SVC model..."):
                result = model.predict_fracture(img_cv)
                # Display result
                st.markdown(f"### Diagnosis: **{result}**")
                
                if result == "Fracture":
                    st.error("⚠️ Fracture detected! Seek medical attention.")
                else:
                    st.success("✅ No fracture detected.")
                
                st.markdown("""
                *Note: This result is based on a Support Vector Classifier trained on X-ray images.
                This is not a medical diagnosis and should not replace professional medical advice.*
                """)
        
        elif detection_mode == "Anomaly Detection":
            with st.spinner("Running anomaly detection..."):
                result = model.detect_anomaly(img_cv)
                
                # Display result
                st.markdown(f"### Assessment: **{result}**")
                
                if result == "Potential Fracture Risk":
                    st.warning("⚠️ Potential fracture risk detected. Further examination recommended.")
                else:
                    st.success("✅ Normal - No anomalies detected.")
                
                st.markdown("""
                *Note: Anomaly detection identifies unusual patterns that may indicate potential fractures
                or other abnormalities. This is not a medical diagnosis.*
                """)
        
        elif detection_mode == "Grad-CAM Visualization":
            with st.spinner("Generating Grad-CAM visualization..."):
                # Generate visualization
                gradcam_image = model.generate_gradcam_visualization(img_cv)
                
                # Display explanation
                st.markdown("""
                ### Grad-CAM Visualization
                
                Gradient-weighted Class Activation Mapping (Grad-CAM) highlights regions of the image 
                that are important for the model's classification decision:
                
                - **Left**: Original image
                - **Middle**: Heatmap showing areas of interest (red = high attention)
                - **Right**: Overlay of heatmap on original image
                """)
                
                # Display visualization
                st.image(gradcam_image, use_column_width=True)

# Training section (for initial setup)
st.markdown("---")
with st.expander("Model Training (Admin Only)"):
    st.write("This section allows retraining of the models with new data.")
    
    train_data_path = st.text_input("Training Data Directory:", "training_data")
    
    if st.button("Train Models"):
        if os.path.exists(train_data_path):
            with st.spinner("Training models... This may take several minutes."):
                accuracy = model.train_models(train_data_path)
                st.success(f"Training complete! Model accuracy: {accuracy:.2f}")
                
                # Save model
                model_dir = "model_weights"
                model.save_model(model_dir)
                st.info(f"Model saved to {model_dir}")
        else:
            st.error(f"Directory {train_data_path} not found. Please provide a valid path.")

# Remove the temp file
if 'temp_path' in locals():
    os.unlink(temp_path)

# Footer
st.markdown("---")
st.caption("Bone Fracture Detection & Prediction | Made with ❤️ using Streamlit")
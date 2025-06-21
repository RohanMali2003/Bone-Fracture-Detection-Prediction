# Bone Fracture Detection & Prediction

A machine learning application that detects bone fractures from X-ray images and predicts fracture risk. This project includes a Jupyter notebook for model development and a Streamlit web application for interactive use.

## Features

- **Bone Fracture Detection**: Uses a Support Vector Classifier (SVC) to classify X-ray images as fractured or not fractured
- **Anomaly Detection**: Employs One-Class SVM to identify potential fracture risks
- **Visualization**: Implements Grad-CAM to visualize regions of interest in X-ray images
- **User-friendly Interface**: Streamlit web application for easy interaction with the models
- **Pre-trained Models**: Includes saved models ready for inference

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/RohanMali2003/Bone-Fracture-Detection-Prediction.git
   cd Bone-Fracture-Detection-Prediction
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up directory structure:

   ```bash
   python setup_directories.py
   ```

## Quick Start

1. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Open your browser and go to http://localhost:8501

3. Upload an X-ray image and select the detection mode

## Usage

1. **Upload an X-ray image** through the web interface
2. **Choose a detection mode**:
   - SVC Fracture Detection: Classifies the image as fractured or not fractured
   - Anomaly Detection: Identifies potential fracture risks
   - Grad-CAM Visualization: Shows which regions of the image influenced the model's decision
3. **View the results and analysis**

## Model Training

If you want to retrain the models with your own dataset:

1. Organize your dataset with the following structure:

   ```
   data/
   ├── training_data/
   │   ├── fractured/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   └── not_fractured/
   │       ├── image1.jpg
   │       ├── image2.jpg
   │       └── ...
   └── testing_data/
       ├── fractured/
       │   ├── image1.jpg
       │   └── ...
       └── not_fractured/
           ├── image1.jpg
           └── ...
   ```

2. Use the training section in the application (expand "Model Training")
3. Enter the path to your training data directory
4. Click "Train Models"

## Technical Details

- **Feature Extraction**: MobileNetV2 pre-trained on ImageNet
- **Classification**: Support Vector Classifier
- **Anomaly Detection**: One-Class SVM
- **Visualization**: Gradient-weighted Class Activation Mapping (Grad-CAM)

## Project Structure

```
bone-fracture-detection/
├── app.py                # Streamlit application
├── model.py              # Model functions
├── data_preparation.py   # Script for dataset organization
├── setup_directories.py  # Script to set up directory structure
├── requirements.txt      # Dependencies
├── model_weights/        # Saved models
├── data/                 # Dataset (sample images)
└── Bone_Fracture_Prediction_&_Detection.ipynb  # Original notebook
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original notebook was developed in Google Colab
- Dataset source: [provide source of your dataset]

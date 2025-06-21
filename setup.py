# Setup directory structure for data
import os

# Create directories if they don't exist
os.makedirs('model_weights', exist_ok=True)
os.makedirs('training_data/fractured', exist_ok=True)
os.makedirs('training_data/not_fractured', exist_ok=True)
os.makedirs('testing_data/fractured', exist_ok=True)
os.makedirs('testing_data/not_fractured', exist_ok=True)

print("Directory structure created successfully!")
print("\nTo properly set up this project, please:")
print("1. Move fractured bone X-ray images to 'training_data/fractured/' and 'testing_data/fractured/'")
print("2. Move non-fractured bone X-ray images to 'training_data/not_fractured/' and 'testing_data/not_fractured/'")
print("3. Run the Streamlit app using: streamlit run app.py")
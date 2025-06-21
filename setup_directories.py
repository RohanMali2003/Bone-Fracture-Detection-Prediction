import os

# Create directory structure
dirs = [
    'data/training_data/fractured',
    'data/training_data/not_fractured',
    'data/testing_data/fractured',
    'data/testing_data/not_fractured',
    'model_weights'
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    
print("Directory structure created successfully!")
print("\nPlease place a few sample X-ray images in the appropriate directories:")
print("- data/training_data/fractured")
print("- data/training_data/not_fractured")
print("- data/testing_data/fractured")
print("- data/testing_data/not_fractured")
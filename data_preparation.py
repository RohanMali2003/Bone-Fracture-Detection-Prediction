# This script helps you move images from an existing dataset
# to the required training and testing directories

import os
import shutil
import argparse
import random

def split_dataset(source_dir, target_dir, train_ratio=0.8, limit_per_class=None):
    """
    Split images from source_dir into training and testing datasets
    
    Parameters:
        source_dir: Path to the source directory containing fractured and not_fractured subdirectories
        target_dir: Path to the target directory where training_data and testing_data will be created
        train_ratio: Ratio of images to use for training (default: 0.8)
        limit_per_class: Maximum number of images to use per class (default: None, use all)
    """
    # Create target directories
    train_fractured_dir = os.path.join(target_dir, 'training_data', 'fractured')
    train_not_fractured_dir = os.path.join(target_dir, 'training_data', 'not_fractured')
    test_fractured_dir = os.path.join(target_dir, 'testing_data', 'fractured')
    test_not_fractured_dir = os.path.join(target_dir, 'testing_data', 'not_fractured')
    
    os.makedirs(train_fractured_dir, exist_ok=True)
    os.makedirs(train_not_fractured_dir, exist_ok=True)
    os.makedirs(test_fractured_dir, exist_ok=True)
    os.makedirs(test_not_fractured_dir, exist_ok=True)
    
    # Process fractured images
    fractured_dir = os.path.join(source_dir, 'fractured')
    if os.path.exists(fractured_dir):
        images = [f for f in os.listdir(fractured_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit number of images if specified
        if limit_per_class and len(images) > limit_per_class:
            images = random.sample(images, limit_per_class)
        
        # Split into train and test
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Copy to train directory
        for img in train_images:
            src = os.path.join(fractured_dir, img)
            dst = os.path.join(train_fractured_dir, img)
            shutil.copy2(src, dst)
        
        # Copy to test directory
        for img in test_images:
            src = os.path.join(fractured_dir, img)
            dst = os.path.join(test_fractured_dir, img)
            shutil.copy2(src, dst)
        
        print(f"Processed {len(train_images)} fractured images for training")
        print(f"Processed {len(test_images)} fractured images for testing")
    
    # Process not_fractured images
    not_fractured_dir = os.path.join(source_dir, 'not_fractured')
    if os.path.exists(not_fractured_dir):
        images = [f for f in os.listdir(not_fractured_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit number of images if specified
        if limit_per_class and len(images) > limit_per_class:
            images = random.sample(images, limit_per_class)
        
        # Split into train and test
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Copy to train directory
        for img in train_images:
            src = os.path.join(not_fractured_dir, img)
            dst = os.path.join(train_not_fractured_dir, img)
            shutil.copy2(src, dst)
        
        # Copy to test directory
        for img in test_images:
            src = os.path.join(not_fractured_dir, img)
            dst = os.path.join(test_not_fractured_dir, img)
            shutil.copy2(src, dst)
        
        print(f"Processed {len(train_images)} not_fractured images for training")
        print(f"Processed {len(test_images)} not_fractured images for testing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into training and testing sets")
    parser.add_argument("--source", type=str, required=True, 
                        help="Source directory containing fractured and not_fractured subdirectories")
    parser.add_argument("--target", type=str, default=".", 
                        help="Target directory where training_data and testing_data will be created")
    parser.add_argument("--train-ratio", type=float, default=0.8, 
                        help="Ratio of images to use for training (default: 0.8)")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Maximum number of images to use per class (default: None, use all)")
    
    args = parser.parse_args()
    
    split_dataset(args.source, args.target, args.train_ratio, args.limit)
    print("Dataset splitting completed successfully!")
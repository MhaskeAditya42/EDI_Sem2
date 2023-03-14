import os
import shutil
import random

# Define paths
original_dataset_dir = "C:/Users/Asus/Desktop/dataset-resized - Copy"
train_dir = 'C:/Users/Asus/Desktop/train_dir'
val_dir = 'C:/Users/Asus/Desktop/val_dir'

# Define categories
categories = os.listdir(original_dataset_dir)

# Create directories for train and validation
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Copy files to train and validation directories
for category in categories:
    category_path = os.path.join(original_dataset_dir, category)
    images = os.listdir(category_path)
    random.shuffle(images)
    train_size = int(0.8 * len(images))
    train_images = images[:train_size]
    val_images = images[train_size:]
    train_category_path = os.path.join(train_dir, category)
    val_category_path = os.path.join(val_dir, category)
    os.makedirs(train_category_path, exist_ok=True)
    os.makedirs(val_category_path, exist_ok=True)
    for image in train_images:
        src = os.path.join(category_path, image)
        dst = os.path.join(train_category_path, image)
        shutil.copyfile(src, dst)
    for image in val_images:
        src = os.path.join(category_path, image)
        dst = os.path.join(val_category_path, image)
        shutil.copyfile(src, dst)

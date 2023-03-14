import os
import random
import shutil

# Set the path to the directory containing the images
data_dir = 'C:/Users/Asus/Desktop/dataset-resized/Combined'

# Set the proportion of images to use for validation
val_ratio = 0.2

# Get the list of all image filenames
filenames = os.listdir(data_dir)

# Shuffle the list of filenames
random.shuffle(filenames)

# Calculate the number of images to use for validation
num_val = int(len(filenames) * val_ratio)

# Split the filenames into training and validation sets
train_filenames = filenames[num_val:]
val_filenames = filenames[:num_val]

# Create directories for the training and validation sets
train_dir = 'C:/Users/Asus/Desktop/dataset-resized/train'
val_dir = 'C:/Users/Asus/Desktop/dataset-resized/val'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Copy the training images to the training directory
for filename in train_filenames:
    src_path = os.path.join(data_dir, filename)
    dst_path = os.path.join(train_dir, filename)
    shutil.copyfile(src_path, dst_path)

# Copy the validation images to the validation directory
for filename in val_filenames:
    src_path = os.path.join(data_dir, filename)
    dst_path = os.path.join(val_dir, filename)
    shutil.copyfile(src_path, dst_path)

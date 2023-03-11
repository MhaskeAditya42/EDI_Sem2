import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

# Set up the training and testing directories
train_dir = './garbage/train/'
test_dir = './garbage/test/'

# Define the image size and batch size
img_size = (224, 224)
batch_size = 32

# Use VGG16 as the base model for transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Define the model architecture
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the data generators for image preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
          
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Generate the training and testing data batches
train_data = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# Train the model
history = model.fit(train_data, epochs=10, validation_data=test_data)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)



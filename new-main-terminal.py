import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('Resources/Model/keras_model.h5')

# Create a list of class labels
class_labels = ['Dry', 'Wet', 'Not Detected']

# Load an image
img = cv2.imread("C:/Users/Asus/Desktop/Datasets/Dataset - 3/paper/cardboard63.jpg")
# img = cv2.imread("C:/Users/Asus/Desktop/Datasets/Dataset - 2/main/Nothing - White BG/images.jfif")

# Preprocess the image
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict the class label of the image
predictions = model.predict(img)
class_index = np.argmax(predictions[0])
class_label = class_labels[class_index]

confidence = predictions[0][class_index] * 100

# Print the predicted class label and confidence
print(f"Class: {class_label} ({confidence:.2f}%)")

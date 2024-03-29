import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('Resources/Model/New/keras_model.h5')

# Create a list of class labels
class_labels = ['Paper and Cardboard Waste', 'Metallic Waste', 'Plastic Waste', 'Nothing', 'Used Clothes', 'E-Waste',
                'Organic Waste']

# Load an image
img = cv2.imread("C:/Users/Asus/Desktop/Datasets/Dataset - 2/wet-2.jfif")
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

if class_index == 0 or class_index == 1 or class_index == 6 or class_index == 5 or class_index == 4:
    print("Dry Waste")
elif class_index == 2:
    print("Wet Waste")
else:
    print("Not Detected")

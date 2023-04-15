import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('Resources/Model/keras_model.h5')

# Create a list of class labels
class_labels = ['Dry', 'Wet', 'Not Detected']

# Open the webcam
cap = cv2.VideoCapture(0)

# Define the font and text color for the bounding box
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (0, 0, 255)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Preprocess the image
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict the class label of the image
    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])
    class_label = class_labels[class_index]
    confidence = predictions[0][class_index] * 100

    # Draw a bounding box around the classified object
    cv2.rectangle(frame, (0, 0), (150, 40), (255, 255, 255), -1)
    cv2.putText(frame, f"{class_label} ({confidence:.2f}%)", (10, 25), font, font_scale, font_color, 2)

    # Display the frame with the bounding box
    cv2.imshow('Waste Classification', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

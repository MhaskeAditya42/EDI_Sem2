import cv2
import tensorflow as tf
import numpy as np

# Load the pre-trained MobileNetV2 model
model = tf.keras.models.load_model('model.h5')

# Create a list of class labels
class_labels = ['Kitchen Waste', 'Glass Waste', 'Metal Waste', 'Paper Waste', 'Plastic Waste']

# Set the video capture device (webcam)
cap = cv2.VideoCapture(0)

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

    # Draw a rectangle over the detected object
    cv2.rectangle(frame, (100, 100), (500, 500), (0, 255, 0), 2)

    # Display the predicted class label and confidence
    cv2.putText(frame, f"{class_label} ({confidence:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Waste Classification', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and destroy the windows
cap.release()
cv2.destroyAllWindows()

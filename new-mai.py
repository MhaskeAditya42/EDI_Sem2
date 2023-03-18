import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('Resources/Model/keras_model.h5')

# Create a list of class labels
class_labels = ['Dry', 'Wet', 'Not Detected']

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

    # Set the frame color and text color based on the predicted class
    if class_label == 'Dry':
        frame_color = (0, 255, 0)  # Green
        text_color = (0, 255, 0)
    elif class_label == 'Wet':
        frame_color = (0, 0, 255)  # Red
        text_color = (0, 0, 255)
    else:
        frame_color = (255, 0, 0)  # Blue
        text_color = (255, 0, 0)

    # Draw a rectangle over the detected object
    if class_label != 'Not Detected':
        cv2.rectangle(frame, (100, 100), (500, 500), frame_color, 2)

    # Display the predicted class label and confidence
    cv2.putText(frame, f"{class_label} ({confidence:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # Display the frame
    cv2.imshow('Waste Classification', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and destroy the windows
cap.release()
cv2.destroyAllWindows()

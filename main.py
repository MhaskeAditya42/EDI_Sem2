import cv2
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
Classifier = Classifier('Resources/Model/keras_model_2.h5', 'Resources/Model/labels.txt')

while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))

    imgBackground = cv2.imread('Resources/background.png')


    predection = Classifier.getPrediction(img)
    print(predection)

    imgBackground[148:148+340, 159:159+454] = imgResize

    # Display
    # cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)


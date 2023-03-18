import os
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
Classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
imgArrow = cv2.imread("Resources/arrow.png", cv2.IMREAD_UNCHANGED)
classIDBin = 0

# Importing the waste images
imgWasteList = []
pathFolderWaste = "Resources/Waste"
pathList = os.listdir(pathFolderWaste)

for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# Importing the bin images
imgBinsList = []
pathFolderBins = "Resources/Bins"
pathList = os.listdir(pathFolderBins)

for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

classDic = {1:0,
            2:2,
            3:3,
            4:3,
            5:1,
            6:1}

while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))
    
    imgBackground = cv2.imread('Resources/TrashIntelAI.png')


    predection = Classifier.getPrediction(img)
    print(predection)
    classID = predection[1]
    if classID!=0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID-1], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

        classIDBin = classDic[classID]

    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))

    imgBackground[148:148+340, 159:159+454] = imgResize

    # Display
    # cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)

import os
import pickle
import numpy as np
import cv2
import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 520)  # Set width
cap.set(4, 385)  # Set height

imgBackground = cv2.imread('test-image/Realtimebkg.png')

#importing the mode images into list
folderModePath = 'test-image/mode'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

#Load the encoding file
print("Loading Encode File...")
file = open('EncodeFile.p','rb')
encodeListKnownWithCNames = pickle.load(file)
file.close()
encodeListKnown, CriminalName = encodeListKnownWithCNames
#print(CriminalName)
print("Encode File Loaded...")


while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) # Scale
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RBG FROM BGR

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    resized_img = cv2.resize(img, (520, 385))
    imgBackground[95:95+385, 18:18+520] = resized_img  # Move image to the right by 18 pixels
    imgBackground[40:40 + 397, 580:580+398] = cv2.resize(imgModeList[0], (398, 397))

    for encodeFace , faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        # Lower the distance better the match
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("matches", matches)
        print("faceDis", faceDis)

        matchIndex = np.argmin(faceDis)
        print("Match Index", matchIndex)

        if matches[matchIndex]:
            print("CRIMINAL DETECTED!!")
            print(CriminalName[matchIndex])



    cv2.imshow("RealTime Face Recog", imgBackground)
    cv2.waitKey(1)
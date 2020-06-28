# Attendance System



import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pyttsx3

engine = pyttsx3.init()
voice = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
engine.setProperty("voice", voice)
engine.setProperty("rate", 150) 

path = 'Pictures/ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M::%S')
            f.writelines(f'\n{name}, {dtString}')
            
encodeListKnow = findEncodings(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0),None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
            markAttendance(name)
            #cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
            cv2.putText(img,"Attendance Marked",(10,200),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,255),1)
            engine.say("Successfully Marked Your Attendance")
            engine.runAndWait()

    cv2.imshow("Web cam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



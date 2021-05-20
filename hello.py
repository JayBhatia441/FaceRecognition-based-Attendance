import cv2
import numpy as np
import face_recognition
import os

path = 'Attendance'
images=[]
classNames=[]
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodedList=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList

encodeListKnown = findEncodings(images)
print('Encoding Complete')
print(len(encodeListKnown))

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCur = face_recognition.face_locations(imgS)
    encodeCur = face_recognition.face_encodings(imgS,faceCur)

    for encodeFace,faceLoc in zip(encodeCur,faceCur):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('w',img)
    cv2.waitKey(1)











""" img = face_recognition.load_image_file('Resorces/download.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Resorces/trump.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img)[0]
encodedImg = face_recognition.face_encodings(img)[0]
cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodedImgTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodedImg], encodedImgTest)
face_dis = face_recognition.face_distance([encodedImg], encodedImgTest)
print(results,face_dis)

cv2.imshow('i',img)
cv2.imshow('img',imgTest)
cv2.waitKey(0) """
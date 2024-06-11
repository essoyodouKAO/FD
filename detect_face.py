import cv2
import numpy as np
import os
import sqlite3

def create_table():
    conn = sqlite3.connect('database.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS STUDENTS (
            ID INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            Age INTEGER NOT NULL
        )
    ''')
    conn.close()

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

recognizer =cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/Users/PC/Documents/Myproject/face regognize/pythonProject/.venv/recognizer/trainingData.yml")

def getprofile(Id):
    conn = sqlite3.connect('database.db')
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE ID = ?", (Id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile
while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        profile =getprofile(id)
        print(profile)
        if profile ==None:
            cv2.putText(img,"Name"+str(profile[1]), (x,y+h+20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            cv2.putText(img,"Name"+str(profile[1]), (x,y+h+45), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)

    cv2.imshow('FACE', img)

    # APPUYER SUR Q POUR ARRETER
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

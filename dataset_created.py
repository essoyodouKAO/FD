import cv2
import numpy as np
import sqlite3

# Initialisation
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Activer la webcam
cam = cv2.VideoCapture(0)


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


def insertorupdate(Id, Name, age):
    conn = sqlite3.connect('database.db')
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE ID = ?", (Id,))

    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1

    if isRecordExist == 1:
        conn.execute("UPDATE STUDENTS SET Name = ?, Age = ? WHERE ID = ?", (Name, age, Id))
    else:
        conn.execute("INSERT INTO STUDENTS (ID, Name, Age) VALUES (?, ?, ?)", (Id, Name, age))

    conn.commit()
    conn.close()


# Creer une table STUDENTS  si ça n'existe pas
create_table()

# collect d'info
Id = input('Enter ID: ')
Name = input('Enter Name: ')
age = input('Enter Age: ')

# Insert ou  update
insertorupdate(Id, Name, age)

# detection de visage et creation du dataset
sampleNum = 0
while True:
    ret, img = cam.read()  # Capture frame-by-frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        sampleNum += 1  # incrementation
        # Save les image capturé
        cv2.imwrite("dataset/user." + str(Id) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
        # Dessiner un rectangle autour du visage
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.waitKey(100)  # delais d'attente


    cv2.imshow("Face", img)
    # Arreter plus de 20 captures
    if sampleNum > 20:
        break

    # Appuyer sur Q pour arreter la webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()

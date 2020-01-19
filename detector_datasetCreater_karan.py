# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 14:46:38 2019

@author: Karan_Desktop
"""
import os
import numpy as np
import cv2
import sqlite3

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def insertDb(id,name):
    conn = sqlite3.connect("karan.db")
    c = conn.cursor()
    c.execute("INSERT INTO users VALUES (:id, :name)", {'id': id, 'name': name})
    conn.commit()
    c.close()

assure_path_exists("dataSet/")

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://192.168.29.101:8080/video')

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
#smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

id = int(input("Enter ID => "))
name = input("Enter Name => ")
insertDb(id,name)

sampleNum = 0;
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    eyes = eye_detector.detectMultiScale(gray, 1.3, 5)
    #smiles = smile_detector.detectMultiScale(frame, 1.3, 5)
    
    # Loops for each faces
    for (x,y,w,h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        # Crop the image frame into rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)  
        # Display the video frame, with bounded rectangle on the person's face
        #cv2.imshow('frame', frame)
        print(sampleNum)
    
    for (x,y,w,h) in eyes:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        #cv2.imshow('frame', frame)
    #for (x,y,w,h) in smiles:
     #   cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
      #  cv2.imshow('frame', frame)

    # Display the resulting frame
    cv2.imshow('Camera Feed (Karan)',frame)
    if ( cv2.waitKey(100) & 0xFF ) == ord('q'):
        break
    
    elif sampleNum >= 50 :        
        break
print("Done")
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
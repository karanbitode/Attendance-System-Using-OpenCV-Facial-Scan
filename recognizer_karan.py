# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:37:20 2019

@author: Karan_Desktop
"""

import cv2
import numpy as np
import os 
import sqlite3

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def checkDb(id):
    conn = sqlite3.connect("karan.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=:id", {'id': id})
    ans = c.fetchone()
    c.close()
    return ans
        
assure_path_exists("trainer/")
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Load the trained mode
recognizer.read('trainer/trainer.yml')
# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://192.168.29.101:8080/video')
while True:    
    ret, im =cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        # Recognize the face belongs to which ID
        id, loss = recognizer.predict(gray[y:y+h,x:x+w])        
        
        if loss > 30 :
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,0,255), -1)
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,0,255), 4)
            #cv2.putText(im, "Unknown" + " ( {0:.2f}% )".format(round(100 - loss, 2)) , (x,y-40), font, 1, (255,255,255), 3)
            cv2.putText(im, "Unknown",  (x,y-40), font, 1, (255,255,255), 3 )
        else:
            ans = checkDb(id)
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
            cv2.putText(im, str(ans[0]) + " - " + str(ans[1]) + " ( {0:.2f}% )".format(round(100 - loss, 2)) , (x,y-40), font, 1, (255,255,255), 3)
       

    # Display the video frame with the bounded rectangle
    cv2.imshow('Recognizer (Karan)',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
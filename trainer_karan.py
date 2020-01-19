# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:07:45 2019

@author: Karan_Desktop
"""

import cv2, os
import numpy as np
from PIL import Image

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Create method to get the images and label data
def getImagesAndLabels(path):
    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    faceSamples=[]
    ids = []
    c = 0
    for imagePath in imagePaths:
        c = c + 1

        # Get the image and convert it to Python image lib & grayscale
        PIL_img = Image.open(imagePath).convert('L')

        # PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')

        # Get the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        faceSamples.append(img_numpy)
        ids.append(id)
        print(c)
        cv2.imshow('Training (Karan)',img_numpy)
        cv2.waitKey(1)
    
    return faceSamples,ids

# Get the faces and IDs
faces,ids = getImagesAndLabels('dataset')
cv2.destroyAllWindows()

print("Training...")
# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
assure_path_exists('trainer/')

recognizer.save('trainer/trainer.yml')
print("Done")
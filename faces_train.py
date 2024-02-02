import os
import cv2 as cv
import numpy as np

# People's names
# people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'C:\Users/Administrator/Desktop/100dayscode/face detection/images/Faces/train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = []
for i in os.listdir(r'C:\Users\Administrator\Desktop\100dayscode\face detection\images\Faces\train'):
    people.append(i)

print(people)

# features
features = []
labels = []

def train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BG2BGR)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray [y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)












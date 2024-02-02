import os
import cv2 as cv
import numpy as np

# People's names
# people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'C:\Users/Administrator/Desktop/100dayscode/face detection/images/Faces/train'

people = []
for i in os.listdir(r'C:\Users\Administrator\Desktop\100dayscode\face detection\images\Faces\train'):
    people.append(i)

print(people)

# features
features = []
lables = []

def train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BG2BGR)

            








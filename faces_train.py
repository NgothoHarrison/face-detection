import os
import cv2
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
    




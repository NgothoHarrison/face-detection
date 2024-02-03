import cv2 as cv
import numpy as np


haar_cascade = cv.CascadeClassifier('haar_face.xml')


people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
features = np.load('features.npy', allow_pickle=True) 
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users/Administrator/Desktop/100dayscode/face detection/images/Faces/train')
import cv2 as cv
import numpy as np

img = cv.imread('images/group.jpg') 
# cv.imshow('Iron Man', img) 

# convert to grayscale

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person Gray', gray)

# Haar Cascade
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# detect faces
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

# print the number of faces found
print(f'Number of faces found = {len(faces_rect)}')

# draw rectangle around the faces
for (x,y,w,h) in faces_rect:
    cv.rectangle(img , (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)


cv.waitKey(0)
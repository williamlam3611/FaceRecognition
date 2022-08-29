# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 00:43:02 2022

@author: weish
"""

import cv2
from sys import exit 

def face_detection(image_to_detect):
    image_to_detect_gray = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2GRAY)
    
    # for Eigenface and FisherFace
    face_detection_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    # for LBPH
    #face_detection_classifier = cv2.CascadeClassifier('models/lbpcascade_frontalface.xml')
    
    all_face_locations = face_detection_classifier.detectMultiScale(image_to_detect_gray)
    
    
    if (len(all_face_locations) == 0):
        return None, None
    
    x,y,width, height = all_face_locations[0]
    
    #face_coordinates = image_to_detect_gray[y:y+width, x:x+ height]
    face_coordinates = image_to_detect_gray[y:y+height, x:x+ width]
    
    face_coordinates = cv2.resize(face_coordinates, (500, 500))
    
    return face_coordinates, all_face_locations[0]


names = []

names.append('Narendra Modi')
names.append('Joe Biden')


#face_classifier = cv2.face.EigenFaceRecognizer_create()
face_classifier = cv2.face.EigenFaceRecognizer_create()
#face_classifier = cv2.face.FisherFaceRecognizer_create()
#face_classifier = cv2.face.LBPHFaceRecognizer_create()


face_classifier.read('models/modi_biden_model.yml')

image_to_classify = cv2.imread('dataset/testing/joe1.jpg')

image_to_classify_copy = image_to_classify.copy()

face_coordinates_classify, box_locations = face_detection(image_to_classify_copy)

print(box_locations)

if face_coordinates_classify is None:
    print('There are no faces in the image to classify')
    exit()
    
name_index, distance = face_classifier.predict(face_coordinates_classify)
name = names[name_index]
distance = abs(distance)


(x,y,w,h) = box_locations
cv2.rectangle(image_to_classify_copy, (x,y), (x+w, y+h), (0, 255, 0), 2)
cv2.putText(image_to_classify_copy, name, (x,y-5), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,255,0), 2)

cv2.imshow('Prediction' + name, cv2.resize(image_to_classify_copy, (500, 500) ))
cv2.waitKey(0)
cv2.destroyAllWindows()






















        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
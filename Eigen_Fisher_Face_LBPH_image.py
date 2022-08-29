# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 00:43:02 2022

@author: weish
"""

import cv2
import numpy as np
import os
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


def prepare_training_data(images_dir, label_index):
    
    faces_coordinates = []
    labels_index = []
    
    images = os.listdir(images_dir)
    
    for image in images:
        image_path = images_dir + '/' + image 
        training_image = cv2.imread(image_path)
        
        cv2.imshow('Training in progress for' + names[label_index], cv2.resize(training_image, (500, 500) ))
        cv2.waitKey(100)
        
        
        face_coordinates, box_coordinates = face_detection(training_image)
        
        if face_coordinates is not None: 
            faces_coordinates.append(face_coordinates)
            labels_index.append(label_index)
            
    return faces_coordinates, labels_index


names = []

names.append('Narendra Modi')
face_coordinates_modi, labels_index_modi = prepare_training_data('dataset/training/modi', 0)

names.append('Joe Biden')
face_coordinates_biden, labels_index_biden = prepare_training_data('dataset/training/biden', 1)


face_coordinates = face_coordinates_modi + face_coordinates_biden
labels_index = labels_index_modi + labels_index_biden

print('Total faces: ', len(face_coordinates))
print('Total names: ', len(names))

#face_classifier = cv2.face.EigenFaceRecognizer_create()
face_classifier = cv2.face.EigenFaceRecognizer_create()
#face_classifier = cv2.face.FisherFaceRecognizer_create()
#face_classifier = cv2.face.LBPHFaceRecognizer_create()

face_classifier.train(face_coordinates, np.array(labels_index))

face_classifier.save('models/modi_biden_model.yml')



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






















        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
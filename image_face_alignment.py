# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:38:15 2022

@author: weish
"""

import cv2
import dlib

image_to_detect = cv2.imread('images/testing/elon.jpg')

face_detection_classifier = dlib.get_frontal_face_detector()


all_face_locations = face_detection_classifier(image_to_detect, 1)

#print(all_face_locations)

print('There are {} number of faces in this image'. format(len(all_face_locations)))


face_landmarks = dlib.full_object_detections()

face_shape_predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

for index, current_face_location in enumerate(all_face_locations):
    face_landmarks.append(face_shape_predictor(image_to_detect, current_face_location))
    

all_face_chips = dlib.get_face_chips(image_to_detect, face_landmarks)
    

for index, current_face_chip in enumerate(all_face_chips):
    
    cv2.imshow('Face no' + str(index + 1), current_face_chip)

cv2.waitKey(0)

cv2.destroyAllWindows()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
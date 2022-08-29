# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:38:15 2022

@author: weish
"""

import cv2
import dlib

image_to_detect = cv2.imread('images/testing/trump-modi.jpg')



face_detection_classifier = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')


all_face_locations = face_detection_classifier(image_to_detect, 1)

print(all_face_locations)

print('There are {} number of faces in this image'. format(len(all_face_locations)))

for index, current_face_location in enumerate(all_face_locations):
    
    left_x, left_y, right_x, right_y = current_face_location.rect.left(), current_face_location.rect.top(), current_face_location.rect.right(), current_face_location.rect.bottom()
    
    print('Found face {} at left_x: {},left_y: {}, right_x: {}, right_y: {}'.format(index +1, left_x, left_y, right_x, right_y))
    current_face_image = image_to_detect[left_y:right_y, left_x:right_x]
    cv2.imshow('Face no ' + str(index +1 ), current_face_image)
    
    cv2.rectangle(image_to_detect, (left_x, left_y), (right_x, right_y), (0, 255, 0), 2)
    
cv2.imshow('faces in image', image_to_detect)

cv2.waitKey(0)

cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
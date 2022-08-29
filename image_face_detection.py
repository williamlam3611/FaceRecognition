# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:38:15 2022

@author: weish
"""

import cv2
import face_recognition

image_to_detect = cv2.imread('images/testing/trump-modi.jpg')

#cv2.imshow('test', image_to_detect)
#all_face_locations = []
all_face_locations = face_recognition.face_locations(image_to_detect, model = 'hog',number_of_times_to_upsample =2 )

print('There are {} number of faces in this image'. format(len(all_face_locations)))

for index, current_face_location in enumerate(all_face_locations):
    
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print('Found face {} at top: {},right: {}, bottom: {}, left: {}'.format(index +1, top_pos, right_pos, bottom_pos, left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    cv2.imshow('Face no ' + str(index +1 ), current_face_image)
    
    
    
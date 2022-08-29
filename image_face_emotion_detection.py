# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:38:15 2022

@author: weish
"""

import cv2
import face_recognition
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import model_from_json

image_to_detect = cv2.imread('images/testing/trump-modi.jpg')


face_exp_model = model_from_json(open('dataset/facial_expression_model_structure.json', 'r').read())
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')



#cv2.imshow('test', image_to_detect)
#all_face_locations = []
all_face_locations = face_recognition.face_locations(image_to_detect, model = 'hog',number_of_times_to_upsample =2 )

print('There are {} number of faces in this image'. format(len(all_face_locations)))

for index, current_face_location in enumerate(all_face_locations):
    
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print('Found face {} at top: {},right: {}, bottom: {}, left: {}'.format(index +1, top_pos, right_pos, bottom_pos, left_pos))


    cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bottom_pos), (0,0, 255), 2)
    current_face_image = image_to_detect[top_pos: bottom_pos, left_pos: right_pos]
    
    
    current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
    current_face_image = cv2.resize(current_face_image, (48,48))
    img_pixels = image.img_to_array(current_face_image)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255
    
    exp_predictions = face_exp_model.predict(img_pixels)
    max_index = np.argmax(exp_predictions[0])
    emotion_label = emotions_label[max_index]
    
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
    
    



cv2.imshow('Image Face Emotions', image_to_detect)
    
    
    
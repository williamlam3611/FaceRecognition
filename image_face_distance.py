# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:38:15 2022

@author: weish
"""

import cv2
import face_recognition

image_to_recognize_path = 'images/testing/trump.jpg'

original_image = cv2.imread(image_to_recognize_path)

modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]


known_face_encodings = [modi_face_encodings, trump_face_encodings]
known_face_names = ['Narendra Modi', 'Donald Trump']

image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)
image_to_recognize_encodings = face_recognition.face_encodings(image_to_recognize)[0]


face_distances = face_recognition.face_distance(known_face_encodings, image_to_recognize_encodings)


for i, face_distance in enumerate(face_distances):
    print('The calculated face distance is {:.2} against the sample {}'.format(face_distance, known_face_names[i]))
    print('The matching percentage is {} against the sample {}'.format( round((1- face_distance) *100, 2) , known_face_names[i]))















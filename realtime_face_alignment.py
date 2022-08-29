# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 19:57:13 2022

@author: weish
"""

import cv2
import dlib 

webcam_video_stream = cv2.VideoCapture(0)

face_detection_classifier = dlib.get_frontal_face_detector()

face_shape_predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

all_face_locations = []

while True: 
    face_landmarks = dlib.full_object_detections()
    
    ret, current_frame = webcam_video_stream.read()
    
    all_face_locations = face_detection_classifier(current_frame, 1)

    if(len(all_face_locations) >= 1):
        for index, current_face_location in enumerate(all_face_locations):
            face_landmarks.append(face_shape_predictor(current_frame, current_face_location))    
        
        all_face_chips = dlib.get_face_chips(current_frame, face_landmarks)
    
        for index, current_face_chip in enumerate(all_face_chips):
            
            cv2.imshow('Face no' + str(index + 1), current_face_chip)       
        
    cv2.imshow('Webcam Video', current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
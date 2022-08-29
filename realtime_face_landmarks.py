# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 18:36:30 2022

@author: weish
"""

import face_recognition
from PIL import Image, ImageDraw
import cv2
import numpy as np

#webcam_video_stream = cv2.VideoCapture(0)

webcam_video_stream = cv2.VideoCapture('images/testing/modi.mp4')



all_face_locations = []


while True: 
    ret, current_frame = webcam_video_stream.read()
    
    #current_frame_small = cv2.resize(current_frame, (0,0), fx = 0.25, fy = 0.25)
    
    
    face_landmarks_list = face_recognition.face_landmarks(current_frame)
    
    print(len(face_landmarks_list))
    
    pil_image = Image.fromarray(current_frame)
    d= ImageDraw.Draw(pil_image)
    
    index = 0
    while index  < len(face_landmarks_list):
    
        for face_landmarks in face_landmarks_list:
            
    
            
            d.line(face_landmarks['chin'], fill = (255, 255, 255), width = 2)
            d.line(face_landmarks['left_eyebrow'], fill = (255, 255, 255), width = 2)
            d.line(face_landmarks['right_eyebrow'], fill = (255, 255, 255), width = 2)
            d.line(face_landmarks['nose_bridge'], fill = (255, 255, 255), width = 2)
            d.line(face_landmarks['nose_tip'], fill = (255, 255, 255), width = 2)
            d.line(face_landmarks['left_eye'], fill = (255, 255, 255), width = 2)
            d.line(face_landmarks['right_eye'], fill = (255, 255, 255), width = 2)
            d.line(face_landmarks['top_lip'], fill = (255, 255, 255), width = 2)
            d.line(face_landmarks['bottom_lip'], fill = (255, 255, 255), width = 2)
        
        index += 1
    
    #rgb_image = pil_image.convert('RGB')
    rgb_open_cv_image = np.array(pil_image)
    
    bgr_open_cv_image = cv2.cvtColor(rgb_open_cv_image, cv2.COLOR_RGB2BGR)
    bgr_open_cv_image = bgr_open_cv_image[:, :, ::-1].copy()
    
    cv2.imshow('Webcam Video', bgr_open_cv_image)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    
webcam_video_stream.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 19:57:13 2022

@author: weish
"""

import cv2
import dlib 

webcam_video_stream = cv2.VideoCapture('images/testing/modi.mp4')

face_detection_classifier = dlib.get_frontal_face_detector()

all_face_locations = []


while True: 
    ret, current_frame = webcam_video_stream.read()
    
    current_frame_to_detect_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    current_frame_small = cv2.resize(current_frame_to_detect_gray , (0,0), fx = 0.25, fy = 0.25)
    all_face_locations = face_detection_classifier(current_frame_small, 1)

    for index, current_face_location in enumerate(all_face_locations):
        left_pos, top_pos, right_pos, bottom_pos = current_face_location.left(), current_face_location.top(), current_face_location.right(), current_face_location.bottom()

        
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        
        print('Found face {} at top:{}, right:{}, bottom: {}, left:{}'.format(index+1, top_pos, right_pos, bottom_pos, left_pos))
        
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0,0, 255), 2)
    cv2.imshow('Webcam Video', current_frame)
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
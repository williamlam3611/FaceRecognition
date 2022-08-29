# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 19:57:13 2022

@author: weish
"""

import cv2
import numpy as np 

webcam_video_stream = cv2.VideoCapture('images/testing/modi.mp4')

#https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
face_detection_classifier = cv2.dnn.readNetFromCaffe('models/deploy.prototxt.txt', 'models/res10_300x300_ssd_iter_140000.caffemodel')

all_face_locations = []


while True: 
    ret, current_frame = webcam_video_stream.read()
    
    image_height = current_frame.shape[0]
    image_width = current_frame.shape[1]
    
    resized_image = cv2.resize(current_frame, (300,300))
    
    image_to_detect_blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104, 177, 123))
    
    face_detection_classifier.setInput(image_to_detect_blob)
    
    all_face_locations = face_detection_classifier.forward()
    
    #print(all_face_locations)
    
    no_of_detections = all_face_locations.shape[2]
    #print('There are {} number of faces in this image'. format(len(all_face_locations)))
    
    for index in range(no_of_detections):
        
        detection_confidence = all_face_locations[0,0, index, 2]
        
        if (detection_confidence > 0.5) :
            print(all_face_locations[0,0, index, 3:7])
            current_face_location = all_face_locations[0,0, index, 3:7] * np.array([ image_width,image_height, image_width, image_height])
        
        
            left_x, left_y, right_x, right_y = current_face_location.astype('int')
            
            print('Found face {} at left_x: {},left_y: {}, right_x: {}, right_y: {}'.format(index +1, left_x, left_y, right_x, right_y))
            current_face_image = current_frame[left_y:right_y, left_x:right_x]
            cv2.imshow('Face no ' + str(index +1 ), current_face_image)
            
            cv2.rectangle(current_frame, (left_x, left_y), (right_x, right_y), (0, 255, 0), 2)
        
    cv2.imshow('Saved Video', current_frame)
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
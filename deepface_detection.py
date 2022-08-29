# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 17:07:21 2022

@author: weish
"""

from deepface import DeepFace
import cv2

# detector_backend = 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface'

face_detected = DeepFace.detectFace(img_path = 'dataset/testing/modi1.jpg', 
                                    detector_backend = 'opencv')


face_detected = cv2.cvtColor(face_detected, cv2.COLOR_BGR2RGB)
cv2.imshow('face_detected', face_detected)































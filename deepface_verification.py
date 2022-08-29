# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 17:07:21 2022

@author: weish
"""

from deepface import DeepFace

# detector_backend = 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface'
# model_name = 'VGG-Face', 'Facenet', 'Facenet512', OpenFace', 'DeepFace'
# distance_metric = 'cosine', 'euclidean', 'euclidean_12'




face_verified = DeepFace.verify(img1_path = 'dataset/testing/modi1.jpg', 
                            img2_path = 'dataset/testing/modi2.jpg', 
                            detector_backend = 'opencv', 
                            model_name = 'VGG-Face', 
                            distance_metric = 'cosine')


print(face_verified)































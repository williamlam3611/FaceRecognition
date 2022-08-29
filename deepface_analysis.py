# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 17:07:21 2022

@author: weish
"""

from deepface import DeepFace



face_analysis = DeepFace.analyze(img_path = 'dataset/testing/modi1.jpg', 
                            actions = ['emotion', 'age', 'gender', 'race'])


print(face_analysis)































 # -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 18:36:30 2022

@author: weish
"""

import face_recognition
from PIL import Image, ImageDraw

face_image = face_recognition.load_image_file('images/samples/abhi.jpg')

face_landmarks_list = face_recognition.face_landmarks(face_image)

print(face_landmarks_list)


for face_landmarks in face_landmarks_list:
    
    pil_image = Image.fromarray(face_image)
    d= ImageDraw.Draw(pil_image)
    
    d.line(face_landmarks['chin'], fill = (255, 255, 2), width = 2)
    d.line(face_landmarks['left_eyebrow'], fill = (255, 255, 2), width = 2)
    d.line(face_landmarks['right_eyebrow'], fill = (255, 255, 2), width = 2)
    d.line(face_landmarks['nose_bridge'], fill = (255, 255, 2), width = 2)
    d.line(face_landmarks['nose_tip'], fill = (255, 255, 2), width = 2)
    d.line(face_landmarks['left_eye'], fill = (255, 255, 2), width = 2)
    d.line(face_landmarks['right_eye'], fill = (255, 255, 2), width = 2)
    d.line(face_landmarks['top_lip'], fill = (255, 255, 2), width = 2)
    d.line(face_landmarks['bottom_lip'], fill = (255, 255, 2), width = 2)
    
pil_image.show()


pil_image.save('images/samples/abhi_landmark.jpg')
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
    d= ImageDraw.Draw(pil_image, 'RGBA')
    
    d.polygon(face_landmarks['left_eyebrow'], fill= (68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill = (68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill = (68, 54, 39, 150), width = 5)
    d.line(face_landmarks['right_eyebrow'], fill = (68, 54, 39, 150), width = 5)
    
    d.polygon(face_landmarks['top_lip'], fill = (150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill = (150, 0,0, 128))
    d.line(face_landmarks['top_lip'], fill = (150, 0, 0,64), width = 8)
    d.line(face_landmarks['bottom_lip'], fill = (150,0,0,64), width = 8)
    
    #d.polygon(face_landmarks['left_eye'], fill = (255, 0, 0, 100))
    #d.polygon(face_landmarks['right_eye'], fill =  (255, 0, 0, 100))
    #d.polygon(face_landmarks['left_eye'], fill = (0, 0, 0, 100))
    #d.polygon(face_landmarks['right_eye'], fill =  (0, 0, 0, 100))
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill = (0,0,0, 110), width = 6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill = (0,0,0,110), width = 6)
    
    
pil_image.show()


pil_image.save('images/samples/abhi_makeup.jpg')


















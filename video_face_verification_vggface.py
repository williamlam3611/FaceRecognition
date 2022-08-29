# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:38:15 2022

@author: weish
"""
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras_vggface.utils import preprocess_input 
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine, euclidean
import cv2

def detect_extract_face(image_to_detect):

        
    #image_to_detect = plt.imread(image_path_to_detect)
    
    
    mtcnn_detector = MTCNN()
    
    
    all_face_locations = mtcnn_detector.detect_faces(image_to_detect)
    
    
    
    #print('There are {} number of faces in the image {}'. format(len(all_face_locations), image_path_to_detect))
    
    #print(all_face_locations)
    
    
    for index, current_face_location in enumerate(all_face_locations):
        
        x, y, width, height = current_face_location['box']
        
        left_x, left_y = x,y
        
        right_x, right_y = x + width, y + height 
        
        #print('Found face {} at left_x: {},left_y: {}, right_x: {}, right_y: {}'.format(index +1, left_x, left_y, right_x, right_y))
        current_face_image = image_to_detect[left_y:right_y, left_x:right_x]


        current_face_image = Image.fromarray(current_face_image)
        
        current_face_image = current_face_image.resize((224, 224))
        
        current_face_image_np_array = np.asarray(current_face_image)
        
        return current_face_image_np_array





#video_stream = cv2.VideoCapture('images/testing/modi.mp4')
video_stream = cv2.VideoCapture(0)

while True:
    ret, image_to_classify = video_stream.read()

    sample_faces = [detect_extract_face(plt.imread('dataset/training/modi/1.jpg')), 
                    detect_extract_face(image_to_classify)]

    if sample_faces[1] is not None:
        
        sample_faces = np.asarray(sample_faces, 'float32')
        
        sample_faces = preprocess_input(sample_faces, version = 2)
        
        #vggface_model = VGGFace(include_top = False, model = 'vgg16', input_shape = (224, 224, 3), pooling = 'avg')
        vggface_model = VGGFace(include_top = False, model = 'resnet50', input_shape = (224, 224, 3), pooling = 'avg')
        
        sample_faces_embeddings = vggface_model.predict(sample_faces)
        
        #the face to be verified
        image_to_be_verified = sample_faces_embeddings[0]
        
        # the face to be verfied against 
        image_to_verify = sample_faces_embeddings[1]

        face_distance = cosine(image_to_be_verified, image_to_verify)
        #face_distance = euclidean(image_to_be_verified, image_to_verify)
      
        cv2.putText(image_to_classify, str(face_distance), (100,100), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,255,0), 2)
        
        cv2.imshow('Distance', cv2.resize(image_to_classify, (224, 224)))
        cv2.waitKey(5)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
video_stream.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
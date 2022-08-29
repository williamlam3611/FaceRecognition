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

def detect_extract_face(image_path_to_detect):

        
    image_to_detect = plt.imread(image_path_to_detect)
    
    
    mtcnn_detector = MTCNN()
    
    
    all_face_locations = mtcnn_detector.detect_faces(image_to_detect)
    
    
    
    print('There are {} number of faces in the image {}'. format(len(all_face_locations), image_path_to_detect))
    
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
    

sample_faces = [detect_extract_face('dataset/training/modi/1.jpg'), 
                detect_extract_face('dataset/training/modi/2.jpg'), 
                detect_extract_face('dataset/training/modi/3.jpg'), 
                detect_extract_face('dataset/training/biden/1.jpg')]
    
sample_faces = np.asarray(sample_faces, 'float32')

sample_faces = preprocess_input(sample_faces, version = 2)

#vggface_model = VGGFace(include_top = False, model = 'vgg16', input_shape = (224, 224, 3), pooling = 'avg')

vggface_model = VGGFace(include_top = False, model = 'resnet50', input_shape = (224, 224, 3), pooling = 'avg')


print('input_shape_of_the_model')
print(vggface_model.inputs)
    

sample_faces_embeddings = vggface_model.predict(sample_faces)

#the face to be verified
modi_face_1 = sample_faces_embeddings[0]

# the face to be verfied against 
modi_face_2 = sample_faces_embeddings[1]
modi_face_3 = sample_faces_embeddings[2]
biden_face_1 = sample_faces_embeddings[3]

print('*********************** consider 0.5 as a threshold.')
print(cosine(modi_face_1, modi_face_2))
print(cosine(modi_face_1, modi_face_3))
print(cosine(modi_face_1, biden_face_1))
    
print('*********************** consider 0.5 as a threshold.')
print(euclidean(modi_face_1, modi_face_2))
print(euclidean(modi_face_1, modi_face_3))
print(euclidean(modi_face_1, biden_face_1))
      

    
    
    
    
    
    
    
    
    
    
    
    
# base tools
import os, sys
sys.path.append(os.path.join(".."))
import pandas as pd
# tools from tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (VGG16,
                                                 decode_predictions,
                                                 preprocess_input)
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)

# Sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report


# for plotting
import numpy as np
import matplotlib.pyplot as plt

# python framework for working with images
import cv2
from cv2 import imshow



# load data 
data = pd.read_csv("in/MovieGenre.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False, encoding="ISO-8859-1")



# Add labels
test_list = [0]*len(data)

for i in range(0,len(data)):
    s = data["IMDB Score"][i]
    if 0 <= s < 2:
        test_list[i] = "Very bad"
    if 2 <= s < 4:
        test_list[i] = "Bad"
    if 4 <= s < 6:
        test_list[i] = "Average"
    if 6 <= s < 8:
        test_list[i] = "Good"
    if 8 <= s:
        test_list[i] = "Very good"
    
data["Score"] = test_list

n = 65701
data = data.drop(data.tail(n).index)
data



# create subfolders
subfolder_names = ["Very bad", "Bad", "Average", "Good", "Very good"]
for subfolder_name in subfolder_names:
    os.makedirs(os.path.join("..", "..", "Assignments", "final_project", 'in', subfolder_name))
    
    
# Put the posters in classification folders

import os
import shutil

path_to_files = '../../Assignments/final_project/in'
move_to_path_vg = '../../Assignments/final_project/in/Very good'
move_to_path_g = '../../Assignments/final_project/in/Good'
move_to_path_a = '../../Assignments/final_project/in/Average'
move_to_path_b = '../../Assignments/final_project/in/Bad'
move_to_path_vb = '../../Assignments/final_project/in/Very bad'

files_list = sorted(os.listdir(path_to_files))
file_names= data["Title"]

for i in range(0,9999):
    curr_file = file_names[i]
    #print(curr_file)
    if data.at[i,"Score"] == "Very good":
        #print(data.at[i,"Score"])
        curr_file = curr_file+".jpg"
        try:
            shutil.move(os.path.join(path_to_files, curr_file),
                   os.path.join(move_to_path_vg, curr_file))
        except FileNotFoundError:
            continue
            
    if data.at[i,"Score"] == "Good":
        #print(data.at[i,"Score"])
        curr_file = curr_file+".jpg"
        try:
            shutil.move(os.path.join(path_to_files, curr_file),
                   os.path.join(move_to_path_g, curr_file))
        except FileNotFoundError:
            continue
            
    if data.at[i,"Score"] == "Average":
        #print(data.at[i,"Score"])
        curr_file = curr_file+".jpg"
        try:
            shutil.move(os.path.join(path_to_files, curr_file),
                   os.path.join(move_to_path_a, curr_file)) 
        except FileNotFoundError:
            continue
            
    if data.at[i,"Score"] == "Bad":
        #print(data.at[i,"Score"])
        curr_file = curr_file+".jpg"
        try:
            shutil.move(os.path.join(path_to_files, curr_file),
                   os.path.join(move_to_path_b, curr_file))
        except FileNotFoundError:
            continue
            
    if data.at[i,"Score"] == "Very bad":
        print(data.at[i,"Score"])
        curr_file = curr_file+".jpg"
        try:
            shutil.move(os.path.join(path_to_files, curr_file),
                   os.path.join(move_to_path_vb, curr_file))
        except FileNotFoundError:
            continue
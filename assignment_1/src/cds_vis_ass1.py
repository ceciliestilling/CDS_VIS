import os
import sys
sys.path.append(os.path.join("..","in"))
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from utils.imutils import jimshow
from utils.imutils import jimshow_channel
import matplotlib.pyplot as plt
from PIL import Image



# filepath
#flower_dir = os.path.join("..", "..", "CDS-VIS", "flowers")
flower_dir = os.path.join("in", "flowers")
    
results_dir = os.path.join("..", "Assignment", "assignment_1", "out")

# specify target image
target_name = "image_0001.jpg"

# empty dataframe to save data
data = pd.DataFrame(columns=["target","filename", "distance", "path"])
    
# read target image
target_image = cv2.imread(os.path.join("in", "flowers", "image_0001.jpg"))
# create histogram for all 3 color channels
target_hist = cv2.calcHist([target_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
# normalise the histogram
target_hist_norm = cv2.normalize(target_hist, target_hist, 0,255, cv2.NORM_MINMAX)
    
# for each image (ending with .jpg) in the directory
for image_path in Path(flower_dir).glob("*.jpg"):
    # only get the image name by splitting the image_path (using dummy variable _)
    _, image = os.path.split(image_path)
    # if the image is not the target image
    if image != target_name:
        # read the image and save as comparison image
        comparison_image = cv2.imread(os.path.join("in", "flowers", image))
        comp_img_path = image_path
        # create histogram for comparison image
        comparison_hist = cv2.calcHist([comparison_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
        # normalise the comparison image histogram
        comparison_hist_norm = cv2.normalize(comparison_hist, comparison_hist, 0,255, cv2.NORM_MINMAX)    
        # calculate the chi-square distance
        distance = round(cv2.compareHist(target_hist_norm, comparison_hist_norm, cv2.HISTCMP_CHISQR), 2)
        # append info to dataframe
        data = data.append({"target": target_name,
                                "filename": image, 
                                "distance": distance, 
                                "path": comp_img_path}, ignore_index = True)
    
    
    # sort values based on distance to target image
    data = data.sort_values("distance", ignore_index = True)
    # Only keep top 3
    data = data.iloc[:3]
    # save as csv in current directory
    data.to_csv(f"out/{target_name}_comparison.csv") 

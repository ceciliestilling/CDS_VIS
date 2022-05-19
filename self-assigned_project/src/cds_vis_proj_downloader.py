# Imports
import os
import numpy as np
import pandas as pd 

from PIL import Image
import requests

import urllib.request
from bs4 import *


# load data 
data = pd.read_csv("MovieGenre.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False, encoding="ISO-8859-1")


# only take 10.000 images
n = 65701
samp_data = data.drop(data.tail(n).index)
samp_data

# make a list of all the links in the dataframe
poster_list = samp_data['Poster'].to_list()

title_list = samp_data["Title"].to_list()


# check poster length 
poster_length = len(poster_list)
poster_length

# CREATE FOLDER
def folder_create(title_list):
    try:
        folder_title = "in" #call it "in"
        # folder creation
        os.mkdir(folder_title)
    # if folder exists with that name, ask another name
    except:
        print("Folder Exist with that name!")
        folder_create()
 
    # image downloading start
    #download_images(poster_list, title_list)
        

def download_images(poster_list, title_list):
    for i in range(0,10000):
        try:
            urllib.request.urlretrieve(poster_list[i], "in/"+title_list[i]+".jpg")
        except:
            pass

        
folder_create(title_list)

download_images(poster_list, title_list)
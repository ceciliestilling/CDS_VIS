# Assignment 1 - Image search
This repository contains the code related to Assignment 1 for Visual Analytics. 

## Assignment description
Take a user-defined image from the folder.
Calculate the "distance" between the colour histogram of that image and all of the others.
Find which 3 image are most "similar" to the target image.
Save an image which shows the target image, the three most similar, and the calculated distance score.
Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order

## Methods
The script takes a path to an image directory and the filename of a target image within the directory as inputs. 
The output of the script is a csv file that holds the chi squared distance between target image and the three images in the directory that has the shortest distance to the target image.

## Usage (reproducing results)
Download data from here https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ and add to 'in' folder
To run the script through terminal write: python3 src/cds_vis_proj_analysis.py
Also see requirements.txt

## Results
Results show that the three most similar images to the target image (image_0001.jpg) are:
- image_0597.jpg with a distance score of 1241.81
- image_0594.jpg with a distance score of 1283.24	
- image_0614.jpg with a distance score of 1416.44

I did not manage to save an image of the comparison images and target image together.

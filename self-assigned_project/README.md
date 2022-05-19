# Can you judge a movie by its poster?
Self-assigned Project


## Assignment description
With this project, I trained a classifier on a dataset of movie posters to see if the model could predict IMDB score from its poster. 
In other words, I wanted to see if we can judge a movie by its poster. 

## Methods
The project consists of 3 scripts which should be run in the following order:
 * Downloader: cds_vis_proj_downloader.py
 * Subfolders: cds_vis_proj_subfolders_div.py
 * Analysis: cds_vis_proj_analysis.py

'cds_vis_proj_downloader.py' is used for downloading the images. It loads the data from a csv, and creates a subset, where only the first 10.000 are used. 
It then defines a folder_create function and a download_images function. The download_images function downloads the images from their url in the dataframe.

'cds_vis_proj_subfolders_div.py' is used for adding labels to the data frame and creating the subfolders "Very bad", "Bad", "Average", "Good", and "Very Good". It then puts each poster image into the folder corresponding to its IMDB Score.

'cds_vis_proj_analysis.py' is used for the analysis. The data is split in test and train and the labels are then created from the folders made in 'cds_vis_proj_subfolders_div.py'. The model is loaded an initialized and trained. Outputs are saved in 'out' folder


## Usage (reproducing results)
To run the downloader script through terminal write: python3 src/cds_vis_proj_downloader.py

To run the subfolder script through terminal write: python3 src/cds_vis_proj_subfolders_div.py

To run the analysis script through terminal write: python3 src/cds_vis_proj_analysis.py

Also see requirements.txt


## Results
The data is not very balanced which might be messing with the results. One might improve the model by balancing the dataset (e.g.duplicating some of the data). 
The results are really bad. This might be because the dataset is not balanced. It could also be that one should not judge a movie by its poster!


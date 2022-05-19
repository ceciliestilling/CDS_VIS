# Portfolio Assignment 2


## Assignment description

The first script does the following:
Load either the MNIST_784 data or the CIFAR_10 data
Train a Logistic Regression model using scikit-learn
Print the classification report to the terminal and save the classification report to out/lr_report.txt
The second script should do the following:
Load either the MNIST_784 data or the CIFAR_10 data
Train a Neural Network model using the premade module in neuralnetwork.py
Print output to the terminal during training showing epochs and loss
Print the classification report to the terminal and save the classification report to out/nn_report.txt


## Methods
The script called logistic_regression.py loads the CIFAR_10 dataset and trains a Logistic Regression Model. 

The classification report is saved under 'out' as lr_report.txt

The script called nn_classifier.py also loads the CIFAR_10 dataset but then trains a neural network model. 

The classification report is saved under 'out' and is called nn_report.txt.

## Usage (reproducing results)
To run the logistic regression script through terminal write: python3 src/logistic_regression.py

To run the NN script through terminal write: python3 src/nn_classifier.py


Also see requirements.txt


## Results
Both models performs best when it comes to images of machines. 

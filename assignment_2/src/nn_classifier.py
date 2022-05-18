# load modules
# path tools
import sys,os
sys.path.append(os.path.join(".."))

# image processing 
import cv2 

# neural networks with numpy
import numpy as np
from tensorflow.keras.datasets import cifar10
from utils.neuralnetwork import NeuralNetwork

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Get data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Check shape
X_train.shape

# Create labels
labels = ["airplane", 
          "automobile", 
          "bird", 
          "cat", 
          "deer", 
          "dog", 
          "frog", 
          "horse", 
          "ship", 
          "truck"]

# Convert all the data to greyscale
X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

# Normalize the values
def minmax(data):
    X_norm = (data - data.min())/(data.max() - data.min())
    return X_norm
X_train_scaled = minmax(X_train_grey)
X_test_scaled = minmax(X_test_grey)

#Reshaping the data
nsamples, nx, ny = X_train_scaled.shape
X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))
nsamples, nx, ny = X_test_scaled.shape
X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

# Neural network classifier
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

print("[INFO] training network...")
input_shape = X_train_dataset.shape[1]
nn = NeuralNetwork([input_shape, 64, 10])
print(f"[INFO] {nn}")
nn.fit(X_train_dataset, y_train, epochs=10, displayUpdate=1)

predictions = nn.predict(X_test_dataset)

y_pred = predictions.argmax(axis=1)
cl_report = classification_report(y_test.argmax(axis=1), y_pred, target_names=labels)

# Save the classification report
with open ("../assignment_2/out/nn_report.txt", "w", encoding = "UTF8") as f:
    f.write(cl_report)
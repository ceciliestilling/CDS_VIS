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






fig = plt.gcf()
fig.set_size_inches(15,10)
data['Score'].value_counts().plot.bar(fig)

data['Score'].value_counts() 

# The data is not very balanced. This will mess with the results. 
# One might improve the model by duplicating some of the data





# Loading the already trained model

tf.keras.backend.clear_session()

model = VGG16(include_top = False,
             pooling = "avg",
             input_shape = (224,224,3))


train = keras.utils.image_dataset_from_directory(
    directory= "../../Assignments/final_project/in",
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    subset="training",
    seed = 42,
    batch_size=128,
    image_size=(224, 224))

test = keras.utils.image_dataset_from_directory(
    directory= "../../Assignments/final_project/in",
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    subset="validation",
    seed = 42,
    batch_size=128,
    image_size=(224, 224))


# Create labels
labels = train.class_names
#print(labels)

labels.remove(".ipynb_checkpoints")
print(labels)



X_train = np.concatenate([x for x, y in train], axis=0)
X_test = np.concatenate([x for x, y in test], axis=0)
y_train = np.concatenate([y for x, y in train], axis=0)
y_test = np.concatenate([y for x, y in test], axis=0)

# 
X_train = X_train/255
X_test = X_test/255

X_train.shape

# label binarizer
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

#initalise model
model = Sequential()

# define CONV => ReLU
model.add(Conv2D(32, 
                (3,3),
                padding = "same",
                input_shape = (224, 224, 3)))
model.add(Activation("relu"))
          
# FC classifier
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(6))
model.add(Activation("softmax"))

# define the gradient descent
sgd = SGD(0.01)
# compile model
model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=["accuracy"])


H = model.fit(X_train, y_train, 
              validation_data=(X_test, y_test), 
              batch_size=128,
              epochs=10,
              verbose=1)

def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, 10), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 10), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, 10), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 10), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig("../../Assignments/final_project/out/hist_plt.png")
    plt.show()
    
#predictions
predictions = model.predict(X_test, batch_size=128)

print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labels))

cl_report = classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labels)


# This was the issue I was worried about.  

# plot history
plot_history(H, 10)


# Save the classification report
with open ("../../Assignments/final_project/out/class_report.txt", "w", encoding = "UTF8") as f:
    f.write(cl_report)
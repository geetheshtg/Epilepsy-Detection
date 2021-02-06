import numpy as np
import pandas as pd
import os,cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
#from google.colab.patches import cv2_imshow
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

# Any results you write to the current directory are saved as output

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def create_model():
    model = Sequential()
    model.add(Conv2D(3, kernel_size=(3, 3), input_shape=(48,48,3))) 
    model.add(LeakyReLU(alpha=0.1)) 
    model.add(BatchNormalization())
    model.add(Conv2D(6,kernel_size= (5, 5)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))  
    model.add(BatchNormalization()) 
    model.add(Dropout(0.2)) 
#2nd convolution layer  
    model.add(Conv2D(9, (3, 3)))  
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(12, (3, 3)))  
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())  
    model.add(Dropout(0.2))  
    model.add(Flatten()) 

#fully connected neural networks  
    model.add(Dense(10))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  
    model.add(Dense(7, activation='softmax'))  
    Nadam=tf.keras.optimizers.Nadam(learning_rate=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
    model.compile(loss='categorical_crossentropy',  
              optimizer=Nadam,
              metrics=['accuracy'])  

    
    return model


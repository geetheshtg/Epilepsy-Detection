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

data_path = '/home/geethesh/Documents/venv/Neurological-Disorders-Classification/Emotion-CK+/ckplus'
data_dir_list = os.listdir(data_path)

num_epoch=10

img_data_list=[]


for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Dataset :-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        try:
            input_img_resize=cv2.resize(input_img,(48,48))
        except:
            break
        img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

num_classes = 7

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:177]=0 #177 disgust
labels[177:312]=1 #135 anger
labels[312:387]=2 #75 fear
labels[387:636]=3 #249 surprise 
labels[636:843]=4 #207 happy
labels[843:897]=5 #54 contempt
labels[897:980]=6 #84 sadness

Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 3)  

X_test = X_test.reshape(X_test.shape[0], 48, 48, 3)  


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

from keras.preprocessing.image import ImageDataGenerator
model_custom = create_model()
model_custom.summary()
def generator(X_data, y_data, batch_size):

  samples_per_epoch = X_data.shape[0]
  number_of_batches = samples_per_epoch/batch_size
  counter=0

  while 1:

    X_batch = np.array(X_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    y_batch = np.array(y_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    counter += 1
    yield X_batch,y_batch

    #restart counter to yeild data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0

gg=model_custom.fit_generator(generator(X_train, y_train, batch_size = 15), steps_per_epoch = 30, epochs=150, validation_data=(X_test, y_test), shuffle=1)
# visualizing losses and accuracy

train_loss=gg.history['loss']
val_loss=gg.history['val_loss']
train_acc=gg.history['accuracy']
val_acc=gg.history['val_accuracy']


epochs = range(len(train_acc))

plt.plot(epochs,train_loss,'r', label='train_loss')
plt.plot(epochs,val_loss,'b', label='val_loss')
plt.title('train_loss vs val_loss')
plt.legend()
plt.figure()

plt.plot(epochs,train_acc,'r', label='train_acc')
plt.plot(epochs,val_acc,'b', label='val_acc')
plt.title('train_acc vs val_acc')
plt.legend()
plt.figure()


target_dir = './models'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model_json = model_custom.to_json()
with open("./models/model.json", "w") as json_file:
    json_file.write(model_json)
model_custom.save('./models/model.h5')
model_custom.save_weights('./models/weights.h5')
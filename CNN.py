#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[187]:


data = []
labels = []
CATEGORIES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 
              'International', 'Pop', 'Rock']
NUM_CATEGORIES = 8


# In[188]:


# create dictionary for genre class
dictionary = {'Electronic' : 0, 'Experimental' : 1, 'Folk' : 2, 'Hip-Hop' : 3, 'Instrumental' : 4, 
              'International' : 5, 'Pop' : 6, 'Rock' : 7}
#dictionary['Electronic']


# In[189]:


# read train images from folder
# reshape image to (32,32)
for i in CATEGORIES:
    path = 'C:/Users/BEST/Desktop/MusicGenre/wav_file/' + str(i)
    print(path)
    images = os.listdir(path)
    
    counter = 0
    for a in images:
       
        """image = cv2.imread(path + '/' + a)
        image = cv2.resize(img,(32,32))"""
        
        image = Image.open(path + '/' + a).convert("RGB")
        image = image.resize((64, 64))

        image = np.array(image)
        data.append(image)
        
        label_index = dictionary[i]
        labels.append(label_index)

        
data = np.array(data)      
labels = np.array(labels) 

print('data shape ', data.shape)
print('labels shape ', labels.shape)


# In[190]:


# split data to train and validation
X_train, X_test, Y_train, Y_test = train_test_split(data,labels,test_size=0.2,random_state=42, shuffle=True)
X_train = X_train/255 
X_test = X_test/255


# In[191]:


print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[192]:


#Y_train


# In[193]:


# apply one hot encode to y_train and y_test 
# total 43 classes and new shape of y_train is (31367, 43)

Y_train_one_hot=keras.utils.to_categorical(Y_train,NUM_CATEGORIES)
Y_test_one_hot= keras.utils.to_categorical(Y_test,NUM_CATEGORIES)

print(Y_train_one_hot.shape)
print(Y_test_one_hot.shape)


# In[200]:


# Building model
"""#One-Layer CNN
model = Sequential()

# 1st conv layer
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.BatchNormalization())

# flatten output
model.add(Flatten())
model.add(Dense(100, activation='relu'))

model.add(Dropout(0.5))
# output layer
model.add(Dense(8, activation='softmax'))

model.summary()"""


# Three-Layer CNN
model = keras.Sequential()

# First Conv2D
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,3)))
model.add(keras.layers.MaxPooling2D((3, 3)))
model.add(keras.layers.BatchNormalization())

# Second Conv2D
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((3, 3)))
model.add(keras.layers.BatchNormalization())

# Third Conv2D
model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.BatchNormalization())

# flatten output
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.3))

# output layer
model.add(keras.layers.Dense(8, activation='softmax'))
#model.summary()


# In[211]:


# compiling the sequential model
from keras.optimizers import Adam
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

# training the model for 300 epochs
backup = model.fit(X_train, Y_train_one_hot, batch_size=64, epochs=50, validation_data=(X_test, Y_test_one_hot))


# In[210]:


plt.figure(0)
plt.plot(backup.history['accuracy'], label='training accuracy')
plt.plot(backup.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(backup.history['loss'], label='training loss')
plt.plot(backup.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


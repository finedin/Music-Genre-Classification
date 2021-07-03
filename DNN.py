#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools
from keras.models import Sequential
from sklearn import preprocessing
from keras import layers
import keras


# In[2]:


features = pd.read_csv('fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])


# In[45]:


tracks["track", "genre_top"]


# In[16]:


small = tracks['set', 'subset'] == 'small'
training = tracks['set', 'split'] == 'training'
validation = tracks['set', 'split'] == 'validation'
testing = tracks['set', 'split'] == 'test'

# use MFCC features
MFCC = 'mfcc'
feature_array = [MFCC]

X_train_temp = features.loc[small & training , feature_array]
X_test_temp = features.loc[small & testing, feature_array]
y_train_temp = tracks.loc[small & training, ('track', 'genre_top')]
y_test_temp = tracks.loc[small & testing, ('track', 'genre_top')]
y_train = y_train_temp.dropna()
y_test = y_test_temp.dropna()
X_train = X_train_temp.drop(y_train_temp.drop(y_train.index).index)
X_test = X_test_temp.drop(y_test_temp.drop(y_test.index).index)
FOLK = tracks['track', 'genre_top'] == "Folk"
X_train = X_train.drop(X_train.loc[FOLK].index)
y_train = y_train.drop(y_train.loc[FOLK].index)
X_test = X_test.drop(X_test.loc[FOLK].index)
y_test = y_test.drop(y_test.loc[FOLK].index)


# In[17]:


print('X_train shape : ', X_train.shape)
print('y_train shape : ', y_train.shape)
print('X_test shape : ', X_test.shape)
print('y_test shape : ', y_test.shape)


# In[18]:


scaler = skl.preprocessing.StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.fit_transform(X_test)


# In[19]:


# convert categorical label to numeric
encoder = preprocessing.LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)


# In[46]:


# DNN model
model = Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(140,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


classifier = model.fit(X_train, y_train, epochs=50, batch_size=32)


# In[38]:


_, test_acc  = model.evaluate(X_test, y_test, batch_size=32)
score = round(test_acc,2)
print(score)


# In[33]:


_, test_acc  = model.evaluate(X_train, y_train, batch_size=32)
score = round(test_acc,2)
print(score)


# In[16]:


from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plot_model(model, show_shapes=True, to_file='dnn.jpg')


# In[13]:


#model.summary()


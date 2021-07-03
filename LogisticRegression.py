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
import sklearn.metrics, sklearn.utils, sklearn.multiclass, sklearn.linear_model


# In[2]:


features = pd.read_csv('fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])


# In[3]:


small = tracks['set', 'subset'] == 'small'
training = tracks['set', 'split'] == 'training'
testing = tracks['set', 'split'] == 'test'

# use MFCC features
MFCC = 'mfcc'
feature_array = [MFCC]

# extract train test samples from csv

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


# In[4]:


print('X_train shape : ', X_train.shape)
print('y_train shape : ', y_train.shape)
print('X_test shape : ', X_test.shape)
print('y_test shape : ', y_test.shape)


# In[5]:


# Normalize Data
scaler = skl.preprocessing.StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.fit_transform(X_test)


# In[6]:


class LogisticRegression():
    
     def train(self, X_train, y_train, X_test, y_test, feature_array=feature_array):
        
        clf = sklearn.linear_model.LogisticRegression(random_state=0).fit(X_train, y_train)
        predictedLabel = clf.predict(X_test)
        print('Train Accuracy : ', clf.score(X_train, y_train))
        print(skl.metrics.classification_report(y_train, clf.predict(X_train)))
        print('Test Accuracy : ', clf.score(X_test, y_test))
        print(skl.metrics.classification_report(y_test, clf.predict(X_test)))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, clf.predict(X_test))
        plt.figure(figsize=(12, 8))
        ax =sns.heatmap(cm, square=True, annot=True, cbar=False)
        ax.xaxis.set_ticklabels(('Electronic', 'Experimental', 'Hip-Hop', 'Instrumental', 'International','Pop', 'Rock'), fontsize = 12)
        ax.yaxis.set_ticklabels(('Electronic', 'Experimental', 'Hip-Hop', 'Instrumental', 'International','Pop', 'Rock'), fontsize = 12, rotation=0)
        ax.set_xlabel('Predicted Labels',fontsize = 15)
        ax.set_ylabel('True Labels',fontsize = 15)

        plt.show()
        


# In[7]:


lr = LogisticRegression()
lr.train(X_train, y_train, X_test, y_test)


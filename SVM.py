#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import svm
import itertools


# In[2]:


features = pd.read_csv('fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])


# In[5]:


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


# In[6]:


print('X_train shape : ', X_train.shape)
print('y_train shape : ', y_train.shape)
print('X_test shape : ', X_test.shape)
print('y_test shape : ', y_test.shape)


# In[7]:


scaler = skl.preprocessing.StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.fit_transform(X_test)


# ### SVM

# In[8]:


class SVM(object):
    def train(self, feature_array=[MFCC]):

        self.scaler = skl.preprocessing.StandardScaler(copy=False)
        self.scaler.fit_transform(X_train)
        self.scaler.transform(X_test)
        
        self.classifier = svm.SVC()
        self.classifier.fit(X_train, y_train)
        
        print("Training Report")
        print("Train Accuracy: ", self.classifier.score(X_train, y_train))
        y_pred = self.classifier.predict(X_train)
        print(sklearn.metrics.classification_report(y_train, y_pred))
        self.confusionmatrix(y_train, y_pred, "train")
        
        print()
        print("Test Report")
        print("Test Accuracy: ", self.classifier.score(X_test, y_test))
        y_pred = self.classifier.predict(X_test)
        print(sklearn.metrics.classification_report(y_test, y_pred))
        self.confusionmatrix(y_test, y_pred, "test")        
            
    def confusionmatrix(self, y_test, y_pred, title):
        
        class_names = y_test.unique()

        from sklearn.metrics import confusion_matrix

        plt.figure(figsize=(12, 9))
        cm = confusion_matrix(y_test, y_pred)
        ax = sns.heatmap(cm, square=True, annot=True, cbar=False)
        ax.xaxis.set_ticklabels(class_names, fontsize = 12)
        ax.yaxis.set_ticklabels(class_names, fontsize = 12, rotation=0)
        ax.set_xlabel('Predicted Labels',fontsize = 15)
        ax.set_ylabel('True Labels',fontsize = 15)
        plt.show()


# In[10]:


modelSVM = SVM()


# In[11]:


modelSVM.train(feature_array=feature_array)


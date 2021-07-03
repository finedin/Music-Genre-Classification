#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import itertools


# In[2]:


features = pd.read_csv('fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])


# In[3]:


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


# In[4]:


print('X_train shape : ', X_train.shape)
print('y_train shape : ', y_train.shape)
print('X_test shape : ', X_test.shape)
print('y_test shape : ', y_test.shape)
#X_train


# In[5]:


scaler = skl.preprocessing.StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.fit_transform(X_test)


# In[6]:


class kNN(object):
    
    def train(self, X_train, y_train, X_test, y_test, feature_array=[MFCC], k=7):

        self.classifier = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(X_train,y_train)
        print(self.classifier)
        train_acc = self.classifier.score(X_train,y_train)
        print("Training Report: ", train_acc)
        print(skl.metrics.classification_report(y_train, self.classifier.predict(X_train)))
        print()
        test_acc = self.classifier.score(X_test,y_test)
        print("Test Report: ", test_acc)
        print(skl.metrics.classification_report(y_test, self.classifier.predict(X_test)))
        predictArray = self.classifier.predict(X_test)
        
        print('\n')
        # return train and test accuracy
        return train_acc, test_acc, predictArray
    
        
        


# In[7]:


knn = kNN()
tr, ts, predictArray = knn.train(X_train, y_train, X_test, y_test, feature_array=[MFCC], k=11)



train_accuracy = []
test_accuracy = []
for kValue in range(1,25):
    train_acc, test_acc,_ = knn.train(X_train, y_train, X_test, y_test,k=kValue)
    train_accuracy.append(train_acc*100)
    test_accuracy.append(test_acc*100)


# In[ ]:


kvalues = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]


# In[ ]:


plt.plot(kvalues, train_accuracy, label = "train")

plt.plot(kvalues, test_accuracy, label = "test")
  
plt.xlabel('K value')
plt.ylabel('Accuracy')

plt.title('Accuracy with 8 Genres')
  
plt.legend()
plt.savefig('graph.jpg')
  
plt.show()


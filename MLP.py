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


# In[3]:


features


# In[18]:


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


# In[19]:


print('X_train shape : ', X_train.shape)
print('y_train shape : ', y_train.shape)
print('X_test shape : ', X_test.shape)
print('y_test shape : ', y_test.shape)


# In[20]:


scaler = skl.preprocessing.StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.fit_transform(X_test)


# In[21]:


#label encoder for the y_train and y_test
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)


# ### NN model

# In[27]:


import sklearn as skl
import sklearn.neural_network, sklearn.preprocessing, sklearn.metrics, sklearn.utils, sklearn.multiclass

class multiLayerNN():
        
    def fit(self, feature_array=feature_array, layer_tuple=(300,32), activation="relu",solver="adam"):

        self.classifier = skl.neural_network.MLPClassifier(hidden_layer_sizes=layer_tuple, activation=activation)
        train_acc = []
        test_acc = []
        loss = []
        for i in range(30):
            self.classifier.partial_fit(X_train, y_train, np.unique(y_test))
            tr_acc = self.classifier.score(X_train,y_train) *100
            ts_acc = self.classifier.score(X_test,y_test) * 100
            train_acc.append(tr_acc)
            test_acc.append(ts_acc)
            loss.append(self.classifier.loss_)
        
        plt.plot(train_acc, color="r",label="Train")
        plt.plot(test_acc, color="b",label="Test")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig('acc.jpg')
        plt.show()
        print("Train Accuracy  ", train_acc[test_acc.index(max(test_acc))])
        print("Test Accuracy  ", max(test_acc))
        print("Best Test Accuracy in Epoch ", test_acc.index(max(test_acc)))
        plt.plot(loss)
        plt.title("Loss Value")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig('loss.png')
        plt.show()
        print("Loss = ", loss[test_acc.index(max(test_acc))])        
        return test_acc


# In[28]:


multiNN = multiLayerNN()
acc = multiNN.fit(activation="relu",layer_tuple=(300,32))


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:09:12 2022

Project: Credit Card highrisk customer identifier using Self Organizing Map + ANN for probabilty of fraud

@author: Dinesh
"""

import numpy as np
import pandas as pd
#import matplotlib as plt

#importing dataset

dataset = pd.read_csv("D:/Clemson/COURSE/SEM-2/Deeplearning_selfStudy/P16-Self-Organizing-Maps/Self_Organizing_Maps/Credit_Card_Applications.csv")
dataset.head(5)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
X.shape

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#Training Self Org Map
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#Visualizing the results

from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']

for i, j in enumerate(X):
    w = som.winner(j)
    plot(
        w[0]+0.5,
        w[1]+0.5,
        markers[Y[i]],
        markeredgecolor = colors[Y[i]],
        markerfacecolor = 'None',
        markersize = 10,
        markeredgewidth = 2
        )
show()

#Finding the fraudulent clients

mapping =  som.win_map(X)
fraud_clients = np.concatenate((mapping [(1,3)], mapping [(1,7)]),axis=0)
fraud_clients = sc.inverse_transform(fraud_clients)

print('Fraud Customer IDs:')
for i in fraud_clients[:, 0]:
    print(int(i))

#ANN

import tensorflow as tf

ann = tf.keras.models.Sequential()

#Adding 1st Hidden Layer

ann.add(tf.keras.layers.Dense(units=2,activation='relu'))

#Adding O/P Layer

ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#Compiling ANN
ann.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])

#Creating Feature Matrix for trainning

customer = dataset.iloc[:,1:].values
is_fraud = np.zeros(len(dataset))

#Creating Dependent Var

for i in range (len(dataset)):
    if dataset.iloc[i,0] in fraud_clients:
        is_fraud[i] = 1
        
#Traing ANN on training set

ann.fit(customer, is_fraud, batch_size = 1, epochs = 15)

# Predicting

Pred = ann.predict(customer)

Pred = np.concatenate((dataset.iloc[:, 0:1].values, Pred), axis = 1)
#Pred = np.concatenate(((dataset.iloc[:,0:1].values).astype(int),np.round(Pred,2)),axis = 1)

Pred = Pred[Pred[:,1].argsort()]

print(Pred)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
#from pandas import get_dummies
#import matplotlib as mpl
#import seaborn as sns
import pandas
import numpy as np
import matplotlib
import warnings
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)


filename = 'train.csv'
train = pandas.read_csv(filename)


X = train.iloc[:,2:5]
Y = train.iloc[:,5]
print(train.shape)
# develop a Convolutional Neural Network model or CNN for ELO

#from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# define dataset
# reshape from [samples, timesteps] into [samples, timesteps, features]
#X = X.reshape((X.shape[0], X.shape[1], 1))
#print(X)
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 201916)))
model.add(MaxPooling1D(pool_size=(2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, Y, epochs=1000, verbose=0)

filename1 = 'test.csv'
test = pandas.read_csv(filename1)
print(test.shape)

test = test.iloc[:,2:4]

yhat = model.predict(test, verbose=0)
print(yhat)


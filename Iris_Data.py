import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data=pd.read_csv('Iris_Data.csv')

 

X=data.iloc[:,0:4].values
y=data.iloc[:,4:5].values
 
from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
y[:,0]=labelencoder_Y.fit_transform(y[:,0])
#Test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(  6, kernel_initializer = 'uniform', activation = 'relu', input_dim =4 ))

# Adding the second hidden layer
classifier.add(Dense(  6,kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense( 1,kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
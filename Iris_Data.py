import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('Iris_Data.csv')
X=data.iloc[:,0:4].values
y=data.iloc[:,4:5].values
 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_y=LabelEncoder()
y[:,0]=labelencoder_y.fit_transform(y[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
y=onehotencoder.fit_transform(y).toarray() 
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
classifier.add(Dense( 3,kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model
y_pred = classifier.predict(X_test)
y_new=np.zeros(30) 
for i in range(0,30):
        k=0
        for j in range(1,3):
            pivot=y_pred[i][k]
            if y_pred[i][j] > pivot:
                pivot=y_pred[i][j]
                k=j
        y_new[i]=k
            
y_pred1=np.zeros(30) 
for i in range(0,30):
        k=0
        for j in range(1,3):
            pivot=y_test[i][k]
            if y_test[i][j] > pivot:
                pivot=y_test[i][j]
                k=j
        y_pred1[i]=k
            
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred1, y_new)

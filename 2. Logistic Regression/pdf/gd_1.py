# Jesus is my Saviour!
# https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931
# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

data = pd.read_csv("C:/Users/Dr Vinod/Desktop/gd linR/gd_lr.csv")

# Preprocessing Input data

X = data.iloc[:, 0]
Y = data.iloc[:, 1]

plt.scatter(X, Y)
plt.show()

# Building the model
b = 0.75
a = 0.45
L = 0.01  # The learning Rate
epochs = 150 # The number of iterations to perform gradient descent

for i in range(epochs): 
    Y_pred = b*X + a  # The current predicted value of Y
    D_b = -sum(X * (Y - Y_pred))  # Derivative wrt b
    D_a = -sum(Y - Y_pred)  # Derivative wrt a
    b = b - L * D_b  # Update b
    a = a - L * D_a  # Update a
    
print (b, a)





































#____________________________________dont do below

from sklearn.linear_model import LinearRegression
LinearRegression.fit(x, y)



x = X.to_numpy(dtype = 'float32')
y = Y.to_numpy(dtype = 'float32')

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter = 1000, penalty = None, eta0=0.1)
sgd_reg.fit(x, y) 

SGDRegressor












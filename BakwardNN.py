
#lab 2 - part 1.5
import numpy as np
import timeit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#import pandas as pd

def onehot(X):
    T = np.zeros((X.shape[0],np.max(X)+1))
    T[np.arange(len(X)),X] = 1 #Set T[i,X[i]] to 1
    return T

#Train Data
xtrain = np.loadtxt('xtrain.txt' , delimiter=',')
xtrain /= 255
ytrain = np.loadtxt('ytrain.txt' , delimiter=',').astype(int)
ytrain = onehot(ytrain)

#Test Data
xtest = np.loadtxt('xtest.txt' , delimiter=',')
xtest /= 255
ytest = np.loadtxt('ytest.txt' , delimiter=',').astype(int)
ytest_onehot = onehot(ytest)

#initialize weights and biases
s = 0.2
w = (np.random.rand(784,10)-0.5)*s
b = (np.random.rand(10,)-0.5)*s

start = timeit.default_timer()

y_plot = []
for i in range(25000):
    
    k = np.random.randint(60000, size = 1)
    xtrain_rand = xtrain[k,:]
    ytrain_rand = ytrain[k,]

    p = (xtrain_rand .dot(w)) + b
    l = 0.001
    w = w - l * (np.transpose(xtrain_rand).dot(p - ytrain_rand))
    b = b - l * (p - ytrain_rand)
    
    # error in each iteration
    #out = np.argmax((xtest .dot(w) + b) , axis=1 )
    out = xtest .dot(w) + b
    #error = 0.5 * np.sum((ytest_onehot - out) ** 2)
    error = mean_squared_error(ytest_onehot, out)
    y_plot.append(error) 

stop = timeit.default_timer()
time = print('running time:', round((stop - start ),2), 's')    


output = np.argmax((xtest .dot(w) + b) , axis=1 )

k = 0
for i in range(10000):
    if output[i,]==ytest[i,]:
        k = k + 1
accuracy = (k/10000)*100        
print('accuracy:', round(accuracy,2), '%')  
        
                 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix        
cm = confusion_matrix(output, ytest)
print(cm)

# plot error vs iterations
plt.plot( y_plot, 'r')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()

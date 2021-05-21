import numpy as np
import matplotlib.pyplot as plt
from dataloader import *


#Feature extraction
Xtrain = np.reshape(Xtrain, (60000, 784))
Xtest = np.reshape(Xtest, (10000, 784))

#Logistic Regression
Xtrain1 = Xtrain/255.0
Xtest1 = Xtest/255.0
Ytrain1 = np.where(Ytrain == 0, 1, 0)
Ytest1 = np.where(Ytest == 0, 1, 0)
lr = 0.1/60000
w = np.zeros(784)

log_likelihood,accuracy = [], []

for times in range(100):
    step = np.zeros(784)
    for x,y in zip(Xtrain1, Ytrain1):
        step = step + x*( y - np.exp(np.dot(w,x))/(1+np.exp(np.dot(w,x))) )
    w = w + step*lr
    likelihood = 0.0
    acc = 0
    for x,y in zip(Xtrain1, Ytrain1):
        likelihood += y*np.dot(x,w) - np.log(1+np.exp(np.dot(w,x)))
    log_likelihood.append(likelihood)
    for x,y in zip(Xtest1, Ytest1):
        pred = np.exp(np.dot(w,x))/(1+np.exp(np.dot(w,x)))
        if (((pred >= 0.5) and (y == 1)) or ((pred < 0.5) and (y == 0))):
            acc += 1
    accuracy.append(acc/10000)
    

plt.plot([i for i in range(1,101)],log_likelihood)
plt.xlabel('number of iterations')
plt.ylabel('log likelihood')
plt.show()

plt.plot([i for i in range(1,101)],accuracy)
plt.xlabel('number of iterations')
plt.ylabel('accuracy')
plt.show()





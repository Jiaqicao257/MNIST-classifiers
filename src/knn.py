import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from scipy import spatial

#Feature extraction
Xtrain = np.reshape(Xtrain, (60000, 784))
Xtest = np.reshape(Xtest, (10000, 784))
Xtrain1 = Xtrain/255.0
Xtest1 = Xtest/255.0

#knn algo
kdtree = spatial.KDTree(Xtrain1)

training = {i:[] for i in range(10)}
for i in range(len(Xtrain1)):
    training[Ytrain[i]].append(i)

euclidean_acc, manhattan_acc = [], []

for k in [1,3,5,7,9]:
    euc_acc = 0
    man_acc = 0
    for x,y in zip(Xtest1, Ytest):
        nearest_1norm = kdtree.query(x, k, p=1)[1]
        nearest_2norm = kdtree.query(x, k, p=2)[1]

        count1 = {i:0 for i in range(10)}
        count2 = {i:0 for i in range(10)}
      
        if k==1:
            nearest_1norm = np.array([nearest_1norm])
            nearest_2norm = np.array([nearest_2norm])

        for neighbor1, neighbor2 in zip(nearest_1norm, nearest_2norm):
            for ytrain, xtrain in training.items():
                if neighbor1 in xtrain:
                    count1[ytrain] += 1
                if neighbor2 in xtrain:
                    count2[ytrain] += 1
        
        pred1 = max(count1, key = count1.get)
        pred2 = max(count2, key = count2.get)

        euc_acc += (pred1 == y)
        man_acc += (pred2 == y)
    
    euclidean_acc.append(euc_acc/10000)
    manhattan_acc.append(man_acc/10000)
    print('k: ', euclidean_acc, manhattan_acc)

plt.plot([i for i in [1,3,5,7,9]],euclidean_acc)
plt.xlabel('number of k')
plt.ylabel('accuracy using euclidean distance')
plt.show()

plt.plot([i for i in [1,3,5,7,9]],manhattan_acc)
plt.xlabel('number of k')
plt.ylabel('accuracy using manhattan distance')
plt.show()

#euc = [0.9631, 0.9633, 0.9618, 0.9615, 0.9597]; man = [0.9691, 0.9705, 0.9688, 0.9694, 0.9659]




        
                    
        
                    
        



import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader import *
import numpy as np

#mlp model
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        #784 --> hidden layer
        self.fc1 = nn.Linear(28*28, 32)
        #hidden --> 10
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # flatten image input
        x = x.view(x.size(0),-1)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add output layer
        x = self.fc2(x)
        return x

#process the data
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

test_batch_size = 1000
train_batch_size = 256

Xtrain = torch.Tensor(Xtrain/255.0)
Ytrain = torch.Tensor(Ytrain)
Xtest = torch.Tensor(Xtest/255.0)
Ytest = torch.Tensor(Ytest)
train_dataset = torch.utils.data.TensorDataset(Xtrain, Ytrain)
test_dataset = torch.utils.data.TensorDataset(Xtest, Ytest)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)



trainingLoss, testingLoss, trainingAcc, testingAcc = [], [], [], []

for iteraton in range(3):
    model = MLP()

    #loss function and optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
    criterion = torch.nn.CrossEntropyLoss()


    epoch = 40
    training_loss, testing_loss, training_acc, testing_acc = [], [], [], []
    for i in range(epoch):
        #train
        # monitor losses
        train_loss = 0
        acc = 0
        model.train() 
        
        for features,labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(features)

            loss = criterion(output,labels.long())
            loss.backward()
            optimizer.step()

            _, prediction = torch.max(output, 1)
            acc += (prediction == labels).sum().item()
            train_loss += loss.item()
        
        training_loss.append(train_loss)   
        training_acc.append(acc/60000) 

        #test
        test_loss = 0.0
        acc = 0
    
        model.eval() 
        for features, label in test_loader:
            output = model(features)
            loss = criterion(output, label.long()) 
            test_loss += loss.item()

            _, prediction = torch.max(output, 1)
            acc += (prediction == label).sum().item()
            

        testing_loss.append(test_loss)
        testing_acc.append(acc/10000) 
       
    

    trainingLoss.append(training_loss) 
    testingLoss.append(testing_loss)
    trainingAcc.append(training_acc) 
    testingAcc.append(testing_acc)
    print('done')

training_loss = np.mean(trainingLoss, axis = 0)
std1 = np.std(trainingLoss, axis = 0)
testing_loss = np.mean(testingLoss, axis = 0)
std2 = np.std(testingLoss, axis = 0)
training_acc = np.mean(trainingAcc, axis = 0)
std3 = np.std(trainingAcc, axis = 0)
testing_acc = np.mean(testingAcc, axis = 0)
std4 = np.std(testingAcc, axis = 0)
print(training_loss)
#plt.plot([i for i in range(40)],training_loss, label='Training loss',color = 'lightcoral')
plt.errorbar([i for i in range(40)],training_loss, yerr=std1, color='lightcoral', ecolor='lightcoral',label='training loss')
#plt.plot([i for i in range(40)],testing_loss, label='Testing loss', color = 'darkcyan')#
plt.errorbar([i for i in range(40)],testing_loss, yerr=std2, color='darkcyan', ecolor='darkcyan',label='testing loss')
plt.xlabel('number of iteratons')
plt.ylabel('loss')
plt.legend(loc='lower right')
plt.show()

#plt.plot([i for i in range(40)],training_acc,label = 'Traning accuracy',color = 'lightcoral')
plt.errorbar([i for i in range(40)],training_acc, yerr=std3, color='lightcoral', ecolor='lightcoral',label='training accuracy')
#plt.plot([i for i in range(40)],testing_acc,label = 'Testing accuracy', color = 'darkcyan')
plt.errorbar([i for i in range(40)],testing_acc, yerr=std4, color='darkcyan', ecolor='darkcyan',label='testing accuracy')
plt.xlabel('number of iteratons')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()


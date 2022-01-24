import numpy as np
import pandas as pd

alpha=0.01
dataset=pd.read_csv('admissionsScores.txt', header=None)
train_data = dataset[:int(len(dataset) * .80)]
test_data = dataset[int(len(dataset) * .80):]
outputTrain=(train_data.values[0:,[2]])
outputTest=(test_data.values[0:,[2]])

train_data=train_data.values[:,0:2]
test_data = test_data.values[:,0:2]

x1=train_data.min(axis=0)
x2=train_data.max(axis=0)
train_data = (train_data- x1)/ (x2-x1)
train_data= np.hstack((np.matrix(np.ones(train_data.shape[0])).T,train_data))
theta = (np.matrix(np.zeros((train_data.shape[1]))).T)

def sigmoid(thetaX):
    sigmoid= 1/(1+np.exp(-thetaX))
    return sigmoid

def GradientDescent(train_data,theta):
   for x in range(1500):
      yPredict = np.dot(train_data, theta)
      h=sigmoid(yPredict)
      cost=np.multiply(-outputTrain,np.log10(h))-np.multiply((1-outputTrain),np.log10(1-h))
      thetaa= theta.T-np.multiply(alpha,(np.multiply((1/(len(train_data))),np.dot((h-outputTrain).T,train_data))))
      theta = thetaa.T
      print("Error in Iteration =",x,cost)
   return theta

def testPrediction(test_data,theta,x1,x2):
    test_data = (test_data - x1) / (x2 - x1)
    test_data = np.hstack((np.matrix(np.ones(test_data.shape[0])).T, test_data))
    yPredict = np.dot(test_data, theta)
    h = sigmoid(yPredict)
    counter=0
    for i in range(len(h)):
        if h[i]>=0.5 and outputTest[i]==1:
           counter=counter+1
        elif h[i]<0.5 and outputTest[i]==0:
            counter=counter+1
    print("Accuracy On Test Data=",(counter/len(h))*100 ,"%")


theta=GradientDescent(train_data,theta)
finalTheta=theta.T
print("Final Optimized theta",finalTheta)
testPrediction(test_data,theta,x1,x2)

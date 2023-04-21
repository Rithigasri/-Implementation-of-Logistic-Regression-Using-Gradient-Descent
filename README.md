# EXPERIMENT 05:IMPLEMENTATION OF LOGISTIC REGRESSION USING GRADIENT DESCENT

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## EQUIPMENT'S REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1.Import the packages required.
2.Read the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value.

## PROGRAM:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: RITHIGA SRI.B
RegisterNumber:212221230083  
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt('ex2data1.txt',delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

#Visualizing the data

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

#Sigmoid Function

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)

print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train=np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## OUTPUT:
### 1. ARRAY VALUE OF X:
![image](https://user-images.githubusercontent.com/93427256/233660114-61bb29f8-0546-4fe5-9f8f-e3c909848c5b.png)

### 2.ARRAY VALUE OF Y:
![image](https://user-images.githubusercontent.com/93427256/233660182-a8f82de6-a7ef-4e3e-8e09-ccd5e6f41796.png)

### 3.EXAM 1-SCORE GRAPH:
![image](https://user-images.githubusercontent.com/93427256/233660431-1b2b708f-3c0e-4f98-b348-6ce5fdf870ad.png)

### 4.SIGMOID FUNCTION GRAPH:
![image](https://user-images.githubusercontent.com/93427256/233660531-89aa707b-f59d-4382-bcf9-a65efbb649d1.png)

### 5.X_TRAIN_GRAD VALUE:
![image](https://user-images.githubusercontent.com/93427256/233661861-f2764b87-d016-49c3-ac24-6d8bd9fe438d.png)

### 6.Y_TRAIN_GRAD_VALUE:
![image](https://user-images.githubusercontent.com/93427256/233661925-7f7e5481-577f-46fb-a524-92573f81f23e.png)

### 7.PRINT RES.X:
![image](https://user-images.githubusercontent.com/93427256/233661322-fb8cf37b-5096-4dad-a6a8-120c6d679f4b.png)

### 8.DECISION BOUNDARY - GRAPH FOR EXAM SCORE:
![image](https://user-images.githubusercontent.com/93427256/233661371-cc0d3140-b023-4cb2-a1ec-334e2554a806.png)

### 9.PROBABILITY VALUE:
![image](https://user-images.githubusercontent.com/93427256/233661409-2c8cb6cd-97aa-438d-9d7e-511334b8f527.png)

### 10.PREDICTION VALUE OF MEAN:
![image](https://user-images.githubusercontent.com/93427256/233661443-57baaa7e-4505-4752-8867-f78f6786ebc0.png)

## RESULT:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


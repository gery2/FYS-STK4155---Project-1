#project1d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

np.random.seed(120)
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#adding normalized noise to the Franke function
sigma2 = 0.5
z = (FrankeFunction(x, y) + np.random.normal(0,sigma2, len(x))).reshape(-1,1)

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def create_X(x, y, n ,intercept=True):
        if len(x.shape) > 1:
                x = np.ravel(x)
                y = np.ravel(y)

        N = len(x)
        l = int((n+1)*(n+2)/2)          # Number of elements in beta
        X = np.ones((N,l))
        idx = 0
        for i in range(2-intercept,n+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                        X[:,idx] = (x**(i-k))*(y**k)
                        idx +=1


        return X

X = create_X(x, y, n=9, intercept=False)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X.shape)
I = np.eye(X.shape[1],X.shape[1])
nlambdas = 20
MSEPredict = np.zeros(nlambdas)
MSETrain = np.zeros(nlambdas)
lambdas = np.logspace(-10, 1, nlambdas)

for i in range(nlambdas):
    lmb = lambdas[i]
    Ridgebeta = np.linalg.pinv(X_train.T @ X_train+lmb*I) @ X_train.T @ z_train
    B0 = np.mean(z_train)
    # and then make the prediction
    ztildeRidge = X_train @ Ridgebeta + B0 #adding the intercept
    zpredictRidge = X_test @ Ridgebeta + B0 #+intercept
    MSEPredict[i] = MSE(z_test,zpredictRidge)
    MSETrain[i] = MSE(z_train,ztildeRidge)

plt.figure()
plt.plot(np.log10(lambdas), MSETrain, label = 'MSE Ridge train')
plt.plot(np.log10(lambdas), MSEPredict, 'r--', label = 'MSE Ridge Test')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()

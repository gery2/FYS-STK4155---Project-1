#project1b
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

seed(123)
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

#legger til normalfordelt støy til funksjonen
sigma2 = 0.3
z = (FrankeFunction(x, y) + np.random.normal(0,sigma2, len(x))).reshape(-1,1)

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def create_X(x, y, n ):
        if len(x.shape) > 1:
                x = np.ravel(x)
                y = np.ravel(y)

        N = len(x)
        l = int((n+1)*(n+2)/2)          # Number of elements in beta
        X = np.ones((N,l))
        idx = 0
        for i in range(1,n+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                        X[:,idx] = (x**(i-k))*(y**k)
                        idx +=1

        return X

degrees = [i for i in range(12)]
MSE_train = np.zeros(len(degrees)); MSE_test = np.zeros(len(degrees))
testsize = 0.2
bootstraps = 100
bias = []
var = []
ztildeOLS = np.empty((int(z.shape[0]*testsize), bootstraps))

for j in degrees:
    X = create_X(x, y, n=j)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=testsize)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train[:,0] = 1
    X_test[:,0] = 1


    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    ztilde = X_train @ beta
    MSE_train[j] = MSE(z_train,ztilde)

    zpredict = X_test @ beta
    MSE_test[j] = MSE(z_test,zpredict)


    for i in range(bootstraps):
        X_, z_ = resample(X_train, z_train)
        OLSbeta = np.linalg.pinv(X_.T @ X_) @ X_.T @ z_
        # Evaluate the new model on the same test data each time.
        print(X_train.shape, z_train.shape)
        ztildeOLS[:, i] = (X_test @ OLSbeta).ravel()
        #ymod = X_test @ OLSbeta

    #error = np.mean( np.mean((z_train - ztildeOLS)**2, axis=1, keepdims=True) ) #MSE
    bias.append(np.mean( (z_test - np.mean(ztildeOLS, axis=1, keepdims=True))**2 ))
    var.append(np.mean( np.var(ztildeOLS, axis=1, keepdims=True) ))

fig = plt.figure()
plt.xlabel('Model Complexity')
plt.ylabel('Prediction Error')
plt.plot(degrees, bias, label='bias')
plt.plot(degrees, var, label='variance')
plt.legend()
plt.show()

fig = plt.figure()
plt.xlabel('Model Complexity')
plt.ylabel('Prediction Error')
plt.plot(degrees, MSE_test, label='MSE_test')
plt.plot(degrees, MSE_train, label='MSE_train')
plt.legend()
plt.show()

#seaborn

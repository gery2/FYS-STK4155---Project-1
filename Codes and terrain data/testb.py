#testb
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
from imageio import imread

#seed(130)
# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')

def create_X(x, y, n ):
    print(n)
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

# just fixing a set of points
N = 100
m = 5 # polynomial order
terrain1 = terrain1[:N,:N]

# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain1)[0])
y = np.linspace(0,1, np.shape(terrain1)[1])
x_mesh, y_mesh = np.meshgrid(x,y)
# Note the use of meshgrid
z = terrain1.ravel() #height

X = create_X(x_mesh, y_mesh,m)

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

degrees = [i for i in range(20)]
MSE_train = np.zeros(len(degrees)); MSE_test = np.zeros(len(degrees))
testsize = 0.2
bootstraps = 200
bias = []
var = []
ztildeOLS = np.empty((int(z.shape[0]*testsize), bootstraps))

for j in degrees:
    X = create_X(x_mesh, y_mesh, n=j)
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
        #print(X_train.shape, z_train.shape)
        X_, z_ = resample(X_train, z_train)
        OLSbeta = np.linalg.pinv(X_.T @ X_) @ X_.T @ z_
        # Evaluate the new model on the same test data each time.

        ztildeOLS[:, i] = (X_test @ OLSbeta).ravel()

    bias.append(np.mean( (z_test - np.mean(ztildeOLS, axis=1, keepdims=True))**2 ))
    var.append(np.mean( np.var(ztildeOLS, axis=1, keepdims=True) ))

fig = plt.figure()
plt.title('Bias-variance trade-off')
plt.xlabel('Model Complexity')
plt.ylabel('Prediction Error (log10)')
plt.plot(degrees, np.log10(bias), label='bias')
plt.plot(degrees, np.log10(var), label='variance')
plt.legend()
plt.show()

fig = plt.figure()
plt.title('Test and training MSEs')
plt.xlabel('Model Complexity')
plt.ylabel('Prediction Error (log10)')
plt.plot(degrees, np.log(MSE_test), label='MSE_test')
plt.plot(degrees, np.log10(MSE_train), label='MSE_train')
plt.legend()
plt.show()

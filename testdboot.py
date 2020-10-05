#testd_boot
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
from sklearn.linear_model import LinearRegression, Ridge
from imageio import imread
seed(123)
# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')

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

# just fixing a set of points
N = 15
m = 5 # polynomial order
terrain1 = terrain1[:N,:N]

# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain1)[0])
y = np.linspace(0,1, np.shape(terrain1)[1])
x_mesh, y_mesh = np.meshgrid(x,y)
# Note the use of meshgrid
z = terrain1.ravel() #height


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


X = create_X(x_mesh, y_mesh,m, intercept=False) #use same polynomials as in b)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#ikke med i ridge
#X_train[:,0] = 1
#X_test[:,0] = 1


I = np.eye(X.shape[1],X.shape[1]) #same row and column dimensions
nlambdas = 20
MSEPredict = np.zeros(nlambdas)
MSETrain = np.zeros(nlambdas)
MSEPredict2 = np.zeros(nlambdas)
MSETrain2 = np.zeros(nlambdas)
lambdas = np.logspace(-10, 0, nlambdas)
testsize = 0.2
bootstraps = 1000
bias = []
var = []
z_Ridge_boot = np.empty((int(z.shape[0]*testsize), bootstraps))
clf = Ridge(fit_intercept=True)

for i in range(nlambdas):
    lmb = lambdas[i]
    Ridgebeta = np.linalg.pinv(X_train.T @ X_train+lmb*I) @ X_train.T @ z_train
    B0 = np.mean(z_train)
    # and then make the prediction
    ztildeRidge = X_train @ Ridgebeta + B0 #adding the intercept
    zpredictRidge = X_test @ Ridgebeta + B0 #+intercept
    MSEPredict[i] = MSE(z_test,zpredictRidge)
    MSETrain[i] = MSE(z_train,ztildeRidge)

    clf.set_params(alpha=lmb)
    clf.fit(X_train, z_train)
    z_fit = clf.predict(X_train)
    z_pred = clf.predict(X_test)
    MSEPredict2[i] = MSE(z_test,z_pred)
    MSETrain2[i] = MSE(z_train,z_fit)



    for j in range(bootstraps):

        X_, z_ = resample(X_train, z_train)
        R_beta = np.linalg.pinv(X_.T @ X_+lmb*I) @ X_.T @ z_
        R_beta0 = np.mean(z_)
        # Evaluate the new model on the same test data each time.
        z_Ridge_boot[:, j] = (X_test @ R_beta + R_beta0).ravel()
        #ymod = X_test @ Ridgebeta

    #error = np.mean( np.mean((z_train - ztildeOLS)**2, axis=1, keepdims=True) ) #MSE
    bias.append(np.mean( (z_test - np.mean(z_Ridge_boot, axis=1, keepdims=True))**2 ))
    var.append(np.mean( np.var(z_Ridge_boot, axis=1, keepdims=True) ))


# Now plot the results
plt.figure()
plt.plot(np.log10(lambdas), MSETrain, label = 'MSE Ridge train')
plt.plot(np.log10(lambdas), MSEPredict, 'r--', label = 'MSE Ridge Test')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.log10(lambdas), MSETrain2, label = 'MSE Ridge train')
plt.plot(np.log10(lambdas), MSEPredict2, 'r--', label = 'MSE Ridge Test')
plt.xlabel('log10(lambda)')
plt.ylabel('MSEsk')
plt.legend()
plt.show()

fig = plt.figure()
plt.xlabel('log10(lambda)')
plt.ylabel('Prediction Error')
plt.plot(np.log10(lambdas), bias, label='bias')
plt.plot(np.log10(lambdas), var, label='variance')
plt.legend()
plt.show()

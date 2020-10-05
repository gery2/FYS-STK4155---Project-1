#testdcross
#project1dcrossval
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
from imageio import imread
np.random.seed(140)

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

#legger til normalfordelt støy til funksjonen
sigma2 = 0.3
#z = (z + sigma2*np.random.normal(0,1, len(x_mesh))).reshape(-1,1)
#z = (FrankeFunction(x, y) + sigma2*np.random.normal(0,1, len(x))).reshape(-1,1)
X = create_X(x_mesh, y_mesh,m)



def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled[:,0] = 1
X_test_scaled[:,0] = 1

print(X.shape)
I = np.eye(X.shape[1],X.shape[1])
nlambdas = 20
lambdas = np.logspace(-10, 1, nlambdas)

k = 5
kfold = KFold(n_splits = k)
estimated_MSE_KFold = np.zeros(nlambdas)

MSE_test_CV = np.zeros(k)


#ikke med i ridge
#X_train[:,0] = 1
#X_test[:,0] = 1

for i in range(nlambdas):
    lmb = lambdas[i]

    j = 0
    for train_inds, test_inds in kfold.split(X_train_scaled):
        x_cv_train = X_train_scaled[train_inds]
        z_cv_train = z_train[train_inds]

        x_val = X_train_scaled[test_inds]
        z_val = z_train[test_inds]

        B0 = np.mean(z_cv_train)
        Ridgebeta = np.linalg.pinv(x_cv_train.T @ x_cv_train+lmb*I) @ x_cv_train.T @ z_cv_train
        ztilde = x_cv_train @ Ridgebeta + B0
        MSE_train_CV = MSE(z_cv_train,ztilde)
        zpredict = x_val @ Ridgebeta + B0
        MSE_test_CV[j] = MSE(z_val,zpredict)
        j += 1

    estimated_MSE_KFold[i] = np.mean(MSE_test_CV)


fig = plt.figure()
plt.xlabel('log10(lambda)')
plt.ylabel('Prediction Error')
plt.plot(np.log10(lambdas), estimated_MSE_KFold, label='estimated_MSE_KFold')
plt.legend()
plt.show()

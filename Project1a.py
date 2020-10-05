#project1a
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
#import os
seed(101)
fig = plt.figure()
ax = fig.gca(projection='3d')

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
sigma2 = 0.01
z = FrankeFunction(x, y) + np.random.normal(0,sigma2, len(x))

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

x = np.ravel(x)
y = np.ravel(y)
z = np.ravel(z)

#  The design matrix now as function of a given polynomial
X = np.zeros((len(x),21))
X[:,0] = 1; X[:,1] = x; X[:,2] = y
X[:,3] = x*x
X[:,4] = y*y
X[:,5] = x*y
X[:,6] = x**3
X[:,7] = y**3
X[:,8] = x*x*y
X[:,9] = x*y*y
X[:,10] = x**4
X[:,11] = y**4
X[:,12] = x*x*x*y
X[:,13] = x*x*y*y
X[:,14] = x*y*y*y
X[:,15] = x**5
X[:,16] = y**5
X[:,17] = x*x*x*x*y
X[:,18] = x*x*x*y*y
X[:,19] = x*x*y*y*y
X[:,20] = x*y*y*y*y

# We split the data in test and training data
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train[:,0] = 1
X_test[:,0] = 1

# matrix inversion to find beta
OLSbeta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train
print(OLSbeta)
# and then make the prediction
ytildeOLS = X_train @ OLSbeta
print("Training R2 for OLS")
print(R2(z_train,ytildeOLS))
print("Training MSE for OLS")
print(MSE(z_train,ytildeOLS))
ypredictOLS = X_test @ OLSbeta
print("Test R2 for OLS")
print(R2(z_test,ypredictOLS))
print("Test MSE OLS")
print(MSE(z_test,ypredictOLS))

#testeboot
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
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression
from imageio import imread
np.random.seed(777)


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
N = 20
m = 7 # polynomial order
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


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n



X = create_X(x_mesh, y_mesh, m, intercept=False) #use same polynomials as in b)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
print(X_test)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


print(X.shape)
I = np.eye(X.shape[1],X.shape[1]) #same row and column dimensions
nlambdas = 20
MSEPredictLasso = np.zeros(nlambdas)
MSETrainLasso = np.zeros(nlambdas)
lambdas = np.logspace(-3, 0, nlambdas)
testsize = 0.2
bootstraps = 100

bias = []
var = []
z_Lasso_boot = np.zeros((int(z.shape[0]*testsize), bootstraps))
clf = Lasso(fit_intercept=True, max_iter=100000)

for i in range(nlambdas):
    lmb = lambdas[i]

    clf.set_params(alpha=lmb)
    clf.fit(X_train, z_train)
    z_fit = clf.predict(X_train)
    z_pred = clf.predict(X_test)

    #print(clf.score(X_train,z_train))

    MSEPredictLasso[i] = MSE(z_test,z_pred)
    MSETrainLasso[i] = MSE(z_train,z_fit)
    print(z_test.shape, z_pred.shape)

    for j in range(bootstraps):

        X_, z_ = resample(X_train, z_train)


        clf_lasso = Lasso(alpha=lmb, max_iter=1000*2, tol=0.01, fit_intercept=True).fit(X_, z_)
        beta_Lasso = clf_lasso.coef_
        intercept_Lasso = clf_lasso.intercept_

        # Evaluate the new model on the same test data each time.
        z_Lasso_boot[:, j] = (X_test @ beta_Lasso + intercept_Lasso)#.ravel()
        #z_Lasso_boot[:, j] = clf_lasso.predict(X_test)

    bias.append(np.mean( (z_test - np.mean(z_Lasso_boot, axis=1, keepdims=True))**2 ))
    var.append(np.mean( np.var(z_Lasso_boot, axis=1, keepdims=True) ))


plt.figure()
plt.plot(np.log10(lambdas), MSETrainLasso, label = 'MSE Lasso train')
plt.plot(np.log10(lambdas), MSEPredictLasso, 'r--', label = 'MSE Lasso Test')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()


fig = plt.figure()
plt.xlabel('log10(lambda)')
plt.ylabel('Prediction Error')
plt.loglog(lambdas, bias, label='bias')
plt.loglog(lambdas, var, label='variance')
plt.legend()
plt.show()

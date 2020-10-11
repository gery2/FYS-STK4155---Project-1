#project1c
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

np.random.seed(200)
# Make data.
x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#adding normalized noise to the Franke function
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


k = 5
kfold = KFold(n_splits = k)

degrees = [i for i in range(21)]
estimated_MSE_KFold = np.zeros(len(degrees))

for j in degrees:

    MSE_test_CV = np.zeros(k)
    X = create_X(x, y, n=j)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled[:,0] = 1
    X_test_scaled[:,0] = 1


    i = 0
    for train_inds, test_inds in kfold.split(X_train_scaled):
        x_cv_train = X_train_scaled[train_inds]
        z_cv_train = z_train[train_inds]

        x_val = X_train_scaled[test_inds]
        z_val = z_train[test_inds]

        beta = np.linalg.pinv(x_cv_train.T @ x_cv_train) @ x_cv_train.T @ z_cv_train
        ztilde = x_cv_train @ beta
        MSE_train_CV = MSE(z_cv_train,ztilde)
        zpredict = x_val @ beta
        MSE_test_CV[i] = MSE(z_val,zpredict)
        i += 1

    estimated_MSE_KFold[j] = np.mean(MSE_test_CV)

fig = plt.figure()
plt.title('K-fold cross-validation')
plt.xlabel('Model Complexity')
plt.ylabel('Prediction Error')
#plt.plot(degrees, scores_mean, label='sklearn')
plt.plot(degrees, np.log10(estimated_MSE_KFold), label='estimated_MSE_KFold')
plt.legend()
plt.show()

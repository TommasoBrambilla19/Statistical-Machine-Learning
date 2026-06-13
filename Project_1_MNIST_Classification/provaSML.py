import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, ExpSineSquared
from sklearn.preprocessing import StandardScaler

#choose a seed
seed = 0
np.random.seed(seed)

# import some data to play with
digits = datasets.load_digits()

X = digits.data
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)
N,d = X.shape

#change to int32 otherwise doesn't work
N = np.int32(500) 
Ntrain = np.int32(400)
Ntest = np.int32(100)


Xtrain = X[0:Ntrain,:]
ytrain = y[0:Ntrain]
Xtest = X[Ntrain:N,:]
ytest = y[Ntrain:N]

scaler = StandardScaler().fit(Xtrain)
X_train_s = scaler.transform(Xtrain)
X_test_s  = scaler.transform(Xtest)


#kernel = 1.0 * RBF([1.0]) # isotropic kernel
#kernel = DotProduct(1.0) 
#kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
#kernel = RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-3, 1e3), alpha_bounds=(1e-3, 1e9))
#kernel = ExpSineSquared(length_scale=1, periodicity=1) DOESN'T WORK FOR THIS TYPE OF DATA

kernels = [
    1.0 * RBF(length_scale=1.0),
    DotProduct(sigma_0=1.0),
    1.0 * Matern(length_scale=1.0, nu=1.5),
    RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-3, 1e3), alpha_bounds=(1e-3, 1e9))
]

print('Number of samples',N )
for kernel in kernels:
    print("\nTesting kernel:", kernel)
    gpc = GaussianProcessClassifier(kernel=kernel).fit(Xtrain, ytrain)
    yp_train = gpc.predict(Xtrain)
    train_error_rate = np.mean(np.not_equal(yp_train,ytrain))
    yp_test = gpc.predict(Xtest)
    test_error_rate = np.mean(np.not_equal(yp_test,ytest))
    print('Train error rate', train_error_rate)
    print('Test error rate')
    print(test_error_rate)

#plt.matshow(digits.images[1])
#print(digits.target[1])

#GaussianProcessClassifier performs hyperparameter estimation, this means that the the value specified above may not be the final hyperparameters
#If you dont want it to do hyperparameter optimization set optimizer = None like this GaussianProcessClassifier(kernel=kernel,optimizer = None).fit(Xtrain, ytrain)
#You can check the final hyperparameters with gpc_rbf.kernel_.get_params()['kernels']
#gpc_rbf = GaussianProcessClassifier(kernel=kernel).fit(Xtrain, ytrain)
#yp_train = gpc_rbf.predict(Xtrain)
#train_error_rate = np.mean(np.not_equal(yp_train,ytrain))
#yp_test = gpc_rbf.predict(Xtest)
#test_error_rate = np.mean(np.not_equal(yp_test,ytest))
#print('Training error rate')
#print(train_error_rate)
#print('Test error rate')
#print(test_error_rate)



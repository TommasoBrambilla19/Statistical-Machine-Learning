import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, ExpSineSquared
from sklearn.preprocessing import StandardScaler

seed = 0
np.random.seed(seed)

digits = datasets.load_digits()

X = digits.data
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)
N,d = X.shape

kernels = [
    1.0 * RBF(length_scale=1.0),
    DotProduct(sigma_0=1.0),
    1.0 * Matern(length_scale=1.0, nu=1.5),
    RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-3, 1e3), alpha_bounds=(1e-3, 1e9))
]

# Sizes of the traning data sample
train_sizes = [20, 50, 100, 200, 500, 1000, 1500] 
test_errors = np.zeros((4,len(train_sizes)))


for j, Ntrain in enumerate(train_sizes):
    Ntest = np.int32(100)
    N= Ntest + Ntrain

    Xtrain = X[0:Ntrain,:]
    ytrain = y[0:Ntrain]
    Xtest = X[Ntrain:N,:]
    ytest = y[Ntrain:N]

    scaler = StandardScaler().fit(Xtrain)
    X_train_s = scaler.transform(Xtrain)
    X_test_s  = scaler.transform(Xtest)
    print('Number of samples',N )
    kern = 0
    for kernel in kernels:
        print("\nTesting kernel:", kernel)
        gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=3).fit(Xtrain, ytrain)
        if kernel == kernels[0]:
            # Accedo a k2 perché il kernel è del tipo (Constant * RBF)
            print('length scale RBF:', gpc.kernel_.kernels[0].k2.length_scale)
        yp_train = gpc.predict(Xtrain)
        train_error_rate = np.mean(np.not_equal(yp_train,ytrain))
        yp_test = gpc.predict(Xtest)
        # Error rate
        test_error_rate = np.mean(np.not_equal(yp_test,ytest)) 
        print('Train error rate', train_error_rate)
        print('Test error rate')
        print(test_error_rate)
        test_errors[kern,j] = test_error_rate
        kern += 1
print(test_errors)

# Plotting the results
plt.figure(figsize=(10, 6))

for i, kernel in enumerate(kernels):
    label_name = str(kernel).split('(')[0].replace("1**2 * ", "")
    
    plt.plot(train_sizes, test_errors[i], marker='o', label=label_name)

plt.xlabel('Amount of Training Data (Ntrain)')
plt.ylabel('Test Error Rate')
plt.title('Gaussian Process Classifier Error Rates by Kernel')
plt.legend()
plt.grid(True)
plt.show()


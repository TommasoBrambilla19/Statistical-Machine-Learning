#Exercise 2 part b)
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, ExpSineSquared

seed = 0
np.random.seed(seed)

digits = datasets.load_digits()

X = digits.data
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)
N,d = X.shape

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

    kernels = {
        "RBF": 1.0 * RBF(length_scale=1.0),
        "DotProduct": DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5)),
        "Matern": 1.0 * Matern(length_scale=1.0, nu=1.5),
        "RationalQuadratic": RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-3,1e3), alpha_bounds=(1e-3,1e9))
    }

    #build one-hot vectors
    encoder = OneHotEncoder(categories=[range(10)], sparse_output=False, handle_unknown='ignore')
    y_train_onehot = encoder.fit_transform(ytrain.reshape(-1,1))
    print(y_train_onehot.shape)
    print('Number of samples',N )
    def GP_regression_classifier(kernel, kernel_name):
        """
        Build a multi-class classifier using GP regression.
        - Trains 10 regressors (one per class)
        - Predicts 10 continuous outputs
        - Class = argmax(output vector)
        """

        regressors = []

        for k in range(10):
            gpr = GaussianProcessRegressor(kernel=kernel)
            gpr.fit(X_train_s, y_train_onehot[:, k])
            if k == 0:
                if kernel_name == "RBF":
                    # RBF è composto (1.0 * RBF), quindi usiamo .k2
                    print(f"   -> [RBF Params] Length Scale: {gpr.kernel_.k2.length_scale:.4f}")
                
                elif kernel_name == "RationalQuadratic":
                    # RQ è semplice nel tuo dizionario, quindi accesso diretto
                    print(f"   -> [RQ Params] Length Scale: {gpr.kernel_.length_scale:.4f}, Alpha: {gpr.kernel_.alpha:.4f}")
            regressors.append(gpr)

        predictions = []
        for reg in regressors:
            preds = reg.predict(X_test_s)
            predictions.append(preds)
        preds_matrix = np.vstack(predictions).T

        y_pred = np.argmax(preds_matrix, axis=1)

        # Error rate
        test_error = np.mean(y_pred != ytest)

        return test_error, preds_matrix

    kern = 0
    for name, kernel in kernels.items():
        print(f"Testing kernel: {name}")
        error, _ = GP_regression_classifier(kernel, name)
        test_errors[kern,j] = error
        kern += 1
        print(f"Test error rate = {error:.4f}\n")
print(test_errors)

# Plotting the results
plt.figure(figsize=(10, 6))

kernel_names = list(kernels.keys())

for i, name in enumerate(kernel_names):
    plt.plot(train_sizes, test_errors[i], marker='o', label=name)

plt.xlabel('Amount of Training Data (Ntrain)')
plt.ylabel('Test Error Rate')
plt.title('GP Regression Classifier Error Rates by Kernel (One-Hot Encoded)')
plt.legend()
plt.grid(True)
plt.show()
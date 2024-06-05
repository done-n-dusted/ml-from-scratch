import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            ypred = np.dot(X, self.weights) + self.bias

            dw = (2 / n_samples)*np.dot(X.T, (ypred - y))
            db = (2 / n_samples)*np.sum(ypred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
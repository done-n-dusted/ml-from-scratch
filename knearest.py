import numpy as np
from collections import Counter
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
class KNN:

    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        # take x and compute distances to all points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # sort labels based on these distances
        best_indices = np.argsort(distances)[:self.k]
        labels = [self.y_train[i] for i in best_indices]
        # poll the most common label
        return Counter(labels).most_common()[0][0]
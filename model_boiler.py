import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse

from knearest import KNN

iris = datasets.load_iris()

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Name of the model")
args = parser.parse_args()

if args.model == "knn":
    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(predictions)
    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse

from knearest import KNN
from linear_regression import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Name of the model")
parser.add_argument("--type", choices=['regression', 'classification'], help="Type of model", required=True)
args = parser.parse_args()

if args.type == 'classification':
    iris = datasets.load_iris()

    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    plt.figure()
    plt.title("Iris Dataset, classification example")
    plt.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.show()

if args.type == 'regression':

    X, y = datasets.make_regression(n_samples=10000, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    plt.figure()
    plt.title("Random Dataset, regression example")
    plt.scatter(X, y, color="black")
    plt.show()

if args.model == "knn":
    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(predictions)
    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)

if args.model == "linear-regression":
    reg = LinearRegression(n_iters=100000, learning_rate=0.001)
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    print(np.sqrt(np.mean((predictions - y_test) ** 2)))

    plt.figure()
    ypredline = reg.predict(X)
    plt.title("Linear Regression")
    plt.scatter(X_test, y_test, color="black", label="Test data")
    plt.plot(X, ypredline, color="blue", linewidth=3, label="Prediction")
    plt.show()

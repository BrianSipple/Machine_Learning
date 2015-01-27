import numpy as np
from sklearn import tree


def classify(features_train, labels_train):

    X = np.array(features_train)
    Y = np.array(labels_train)
    clf = tree.DecisionTreeClassifier(min_samples_split=50)
    clf.fit(X, Y)
    return clf

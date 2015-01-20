import numpy as np
from sklearn.naive_bayes import GaussianNB


def classify(features_train, labels_train):

    X = np.array(features_train)
    Y = np.array(labels_train)
    clf = GaussianNB()
    clf.fit(X, Y)
    return clf
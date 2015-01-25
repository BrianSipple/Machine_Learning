import numpy as np
from sklearn import svm


def classify(features_train, labels_train):

    X = np.array(features_train)
    Y = np.array(labels_train)
    clf = SVC(kernel="rbf", C=1)
    clf.fit(X, Y)
    return clf

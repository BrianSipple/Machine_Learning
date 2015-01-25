#!/usr/bin/python

"""
    using an SVM to identify emails from the Enron corpus by their authors

    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

X = np.array(features_train)
Y = np.array(labels_train)

clf = svm.SVC(kernel="linear")
clf.fit(X, Y)

predictions = clf.predict(features_test)


accuracy = accuracy_score(predictions, labels_test)

print "Accuracy: {}".format(accuracy)

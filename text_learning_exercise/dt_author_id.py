#!/usr/bin/python

"""
    using an Decision Tree to identify emails from the
    Enron corpus by their authors

    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
import numpy as np
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


X = np.array(features_train)
Y = np.array(labels_train)

print "Number of features: {}".format(len(X[0]))

clf = tree.DecisionTreeClassifier(
    min_samples_split=40
)
clf.fit(X, Y)

predictions = clf.predict(features_test)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(predictions, labels_test)

print "Accuracy: {}".format(accuracy)

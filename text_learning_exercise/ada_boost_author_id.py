#!/usr/bin/python

"""
    using Ada Boosting to identify emails from the
    Enron corpus by their authors

    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
import numpy as np
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

features_train, features_test, labels_train, labels_test = preprocess()

n_estimators = 400
learning_rate = 1.0


X = np.array(features_train)
Y = np.array(labels_train)

### Start with a Decision Tree Stump
dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X, Y)
dt_stump_error = 1.0 - dt_stump.score(features_test, labels_test)


ada_discrete = AdaBoostClassifier(
    n_estimators=n_estimators,
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    algorithm="SAMME"
)
ada_discrete.fit(X, Y)

# Compute error for each of the estimators
ada_discrete_error = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(features_train)):
    ada_discrete_error[i] = zero_one_loss(y_pred, labels_train)

# Compute error over the training set
ada_discrete_error_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(features_test)):
    ada_discrete_error_train[i] = zero_one_loss(y_pred, labels_test)




#scores = cross_val_score(clf, )
clf.fit(X, Y)

predictions = clf.predict(features_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(predictions, labels_test)

print "AdaBoost Accuracy -- \n" + \
      "Number of classifiers: {}\n" + \
      "Score: {}".format(
        clf.n_estimators,
        accuracy
      )

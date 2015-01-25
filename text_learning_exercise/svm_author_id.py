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

# (Optionally) Slim down the training set to speed things up
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

X = np.array(features_train)
Y = np.array(labels_train)

clf = svm.SVC(kernel="rbf", C=10000)
clf.fit(X, Y)

pred_time_0 = time()
predictions = clf.predict(features_test)
pred_time_1 = time()

print "Computed predictions in time {}".format(pred_time_1 - pred_time_0)

acc_time_0 = time()
accuracy = accuracy_score(predictions, labels_test)
acc_time_1 = time()

print "Computed accuracy in time {}".format(acc_time_1 - acc_time_0)

print "Accuracy: {}".format(accuracy)

sara_emails = 0
chris_emails = 0
for i in range(len(predictions)):
    if predictions[i] == 0:
        sara_emails += 1
    elif predictions[i] == 1:
        chris_emails += 1

print "Number of emails predicted for Chris: {}".format(chris_emails)
print "Number of emails predicted for Sara: {}".format(sara_emails)

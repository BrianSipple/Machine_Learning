#!/usr/bin/python

"""
    using a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
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




from sklearn.naive_bayes import GaussianNB


clf = GaussianNB()

t0_classifier = time()
clf.fit(features_train, labels_train)
t1_classifier = time()

total_time_classifier = t1_classifier - t0_classifier
print "Total time for classification: {}".format(total_time_classifier)



t0_predict = time()
predictions = clf.predict(features_test)
t1_predict = time()

total_time_predict = t1_predict - t0_predict
print "Total time for prediction: {}".format(total_time_predict)


from sklearn.metrics import accuracy_score
score = accuracy_score(predictions, labels_test)

print "Accuracy score: {}".format(score)

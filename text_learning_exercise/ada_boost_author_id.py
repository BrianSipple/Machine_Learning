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
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from email_preprocess import preprocess
from class_viz import plot_results, prettyPicture
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

features_train, features_test, labels_train, labels_test = preprocess()


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# #plt.show()


n_estimators = 400
learning_rate = 1.0


X = np.array(features_train)
Y = np.array(labels_train)

### Start with a Decision Tree Stump
dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X, Y)
dt_stump_error = 1.0 - dt_stump.score(features_test, labels_test)

dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X, Y)
dt_err = 1.0 - dt.score(features_test, labels_test)

print "Error computed for dt_stump (depth 1): {}".format(dt_stump_error)
print "Error computed for dt with depth 9: {}".format(dt_err)

ada_discrete = AdaBoostClassifier(
    n_estimators=n_estimators,
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    algorithm="SAMME"
)
ada_discrete.fit(X, Y)

print "Ada Discrete created and fit: {}".format(ada_discrete)

#plot_results(ada_discrete, features_test, labels_test, "res/ada_boost_discrete.png")


ada_real = AdaBoostClassifier(
    n_estimators=n_estimators,
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    algorithm="SAMME.R"
)
ada_real.fit(X, Y)

print "Ada Real created and fit: {}".format(ada_real)

#plot_results(ada_real, features_test, labels_test, "res/ada_boost_real.png")



# Compute training error for each of the estimators
ada_discrete_error_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(features_train)):
    ada_discrete_error_train[i] = zero_one_loss(y_pred, labels_train)

print "Final errors computed for estimators" + \
      " in AdaBoost Discrete: {}".format(ada_discrete_error_train)

print "Final Training accuracy: {}".format(1 - ada_discrete_error_train[-1])


# Compute test error for each of the estimators
ada_discrete_error_test = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(features_test)):
    ada_discrete_error_test[i] = zero_one_loss(y_pred, labels_test)

print "Testing errors computed for estimators" + \
      " in AdaBoost Discrete: {}".format(ada_discrete_error_test)

print "Final Test accuracy: {}".format(1 - ada_discrete_error_test[-1])


ada_real_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(features_train)):
    ada_real_err_train[i] = zero_one_loss(y_pred, labels_train)

print "Training errors computed for estimators" + \
      " in AdaBoost Real: {}".format(ada_real_err_train)

print "Final Training accuracy for Ada Real: {}".format(1 - ada_real_err_train[-1])


ada_real_err_test = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(features_test)):
    ada_real_err_test[i] = zero_one_loss(y_pred, labels_test)

print "Testing errors computed for estimators" + \
      " in AdaBoost Real: {}".format(ada_real_err_test)

print "Final Test accuracy for Ada Real: {}".format(1 - ada_real_err_test[-1])


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_estimators], [dt_stump_error] * 2, 'k-',
        label='Decision Stump Error')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
        label='Decision Tree Error')


ax.plot(np.arange(n_estimators) + 1, ada_discrete_error_test,
        label='Discrete AdaBoost Test Error',
        color='red')
ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train,
        label='Discrete AdaBoost Train Error',
        color='blue')
ax.plot(np.arange(n_estimators) + 1, ada_real_err_test,
        label='Real AdaBoost Test Error',
        color='orange')
ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
        label='Real AdaBoost Train Error',
        color='green')

ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')

leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)

plt.show()






# #scores = cross_val_score(clf, )
# clf.fit(X, Y)
#
# predictions = clf.predict(features_test)
#
# from sklearn.metrics import accuracy_score
#
# accuracy = accuracy_score(predictions, labels_test)
#
# print "AdaBoost Accuracy -- \n" + \
#       "Number of classifiers: {}\n" + \
#       "Score: {}".format(
#         clf.n_estimators,
#         accuracy
#       )

#!/usr/bin/python

"""
    Loads up/formats a modified version of the dataset
    (modified, as in "outlier-free" for this particular exercise)

    Dcatterplot of the training/testing data
"""

import numpy as np
import math
import sys
import pickle
sys.path.append("../tools/")


def percentile(N, percent, key=lambda x:x):
    """
    Find the percentile of a list of values.

    @parameter N - a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of N.

    @return - the percentile of the values
    """
    if not N:
        return None
    k = (len(N)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c-k)
    d1 = key(N[int(c)]) * (k-f)
    return d0+d1


def create_regression_for_feature_outcome_pair(feature, outcome):

    from feature_format import featureFormat, targetFeatureSplit
    dictionary = pickle.load( open("data/enron_dataset_modified.pkl", "r") )

    ### list the features you want to look at--first item in the
    ### list will be the "target" feature
    data = featureFormat( dictionary, [outcome, feature], remove_any_zeroes=True)#, "long_term_incentive"], remove_any_zeroes=True )
    target, features = targetFeatureSplit( data )

    ### training-testing split needed in regression, just like classification
    from sklearn.cross_validation import train_test_split
    feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)

    return feature_train, feature_test, target_train, target_test


def classify(feature_train, target_train):
    from sklearn import linear_model

    reg = linear_model.LinearRegression()
    reg.fit(feature_train, target_train)

    return reg



def make_plot(feature_train, target_train, feature_test, target_test, classifier, x_label, y_label):
    """
    draw the scatterplot, with color-coded training and testing points
    """
    train_color = "#00fffd"
    test_color = "#6600ff"

    import matplotlib.pyplot as plt
    for feature, target in zip(feature_test, target_test):
        plt.scatter( feature, target, color=test_color )
    for feature, target in zip(feature_train, target_train):
        plt.scatter( feature, target, color=train_color )

    ### labels for the legend
    plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
    plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")


    try:
        plt.plot( feature_test, classifier.predict(feature_test) )
    except NameError:
        pass
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def clean_outliers(feature_train, feature_test, target_train, target_test, classifier):

    x_y_pairs = zip(feature_test, target_test)
    sq_errors = [ (classifier.predict(x[0]) - y)**2 for x, y, in x_y_pairs ]
    mse = np.mean(sq_errors)

    #print "MSE: {}".format(mse)

    residuals = [ (classifier.predict(x) - mse)**2 for x, y in x_y_pairs ]
    residuals = sorted(residuals)

    #print "Residuals: {}".format(residuals)

    indicies_to_keep = []
    thresh = percentile(residuals, .90)

    for i in range(len(residuals)):
        if residuals[i] < thresh:
            indicies_to_keep.append(i)

    #print "Indices to keep: {}".format(indicies_to_keep)

    new_feature_train = []
    new_feature_test = []
    new_target_train = []
    new_target_test = []

    for j in range(len(indicies_to_keep)):
        new_feature_train.append(feature_train[indicies_to_keep[j]])
        new_feature_test.append(feature_test[indicies_to_keep[j]])
        new_target_train.append(target_train[indicies_to_keep[j]])
        new_target_test.append(target_test[indicies_to_keep[j]])

    return new_feature_train, new_feature_test, new_target_train, new_target_test





if __name__ == "__main__":

    ################
    ### Set the feature along with the outcome that it will predict

    #feature = "long_term_incentive"
    feature = "salary"
    outcome = "bonus"

    ################

    feature_train, feature_test, target_train, target_test = create_regression_for_feature_outcome_pair(feature, outcome)

    reg = classify(feature_train, target_train)
    slope = reg.coef_[0]
    intercept = reg.intercept_


    # X_train = np.reshape(np.array(feature_train), (len(feature_train), 1))
    # Y_train = np.reshape(np.array(target_train), (len(target_train), 1))
    #
    # X_test = np.reshape(np.array(feature_test), (len(feature_test), 1))
    # Y_test = np.reshape(np.array(target_test), (len(target_test), 1))


    train_score = reg.score(feature_train, target_train)
    test_score = reg.score(feature_test, target_test)

    print "Slope: {}".format(slope)
    print "Intercept: {}".format(intercept)
    #print "Mean Squared Error: {}".format(mse)
    print "Prediction Score on training data: {}".format(train_score)
    print "Prediction Score on testing data: {}".format(test_score)

    #make_plot(feature_train, target_train, feature_test, target_test, reg, feature, outcome)


    ### Now, to attempt to account for outliters,
    ### we can remove the training items with the top 10% of residual error,
    ### and retrain.

    new_feature_train, new_feature_test, new_target_train, new_target_test = clean_outliers(
        feature_train,
        feature_test,
        target_train,
        target_test,
        reg
    )

    # print new_feature_train
    # print new_feature_test
    # print new_target_train
    # print new_target_test


    reg = classify(new_feature_train, new_target_train)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    train_score = reg.score(new_feature_train, new_target_train)
    test_score = reg.score(new_feature_test, new_target_test)

    print "Slope: {}".format(slope)
    print "Intercept: {}".format(intercept)
    #print "Mean Squared Error: {}".format(mse)
    print "Prediction Score on training data: {}".format(train_score)
    print "Prediction Score on testing data: {}".format(test_score)

    make_plot(new_feature_train, new_target_train, new_feature_test, new_target_test, reg, feature, outcome)

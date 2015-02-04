#!/usr/bin/python

"""
    Loads up/formats a modified version of the dataset
    (modified, as in "outlier-free" for this particular exercise)

    Dcatterplot of the training/testing data
"""

import numpy as np
import sys
import pickle
sys.path.append("../tools/")


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
    mse = np.mean( [ (reg.predict(x) - y)**2 for x,y in zip(feature_test, target_test) ])

    # X_train = np.reshape(np.array(feature_train), (len(feature_train)), 1)
    # Y_train = np.reshape(np.array(target_train), (len(target_train)), 1)
    #
    # X_test = np.reshape(np.array(feature_test), (len(feature_test)), 1)
    # Y_test = np.reshape(np.array(target_test), len(target_test), 1)


    train_score = reg.score(feature_train, target_train)
    test_score = reg.score(feature_test, target_test)

    print "Slope: {}".format(slope)
    print "Intercept: {}".format(intercept)
    print "Mean Squared Error: {}".format(mse)
    print "Prediction Score on training data: {}".format(train_score)
    print "Prediction Score on testing data: {}".format(test_score)


    make_plot(feature_train, target_train, feature_test, target_test, reg, feature, outcome)

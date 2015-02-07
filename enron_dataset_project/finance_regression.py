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

from sklearn.cross_validation import train_test_split

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
    dictionary = pickle.load( open("data/enron_dataset.pkl", "r") )

    ### list the features you want to look at--first item in the
    ### list will be the "target" feature

    #data = featureFormat( dictionary, [outcome, feature], remove_any_zeroes=True)#, "long_term_incentive"], remove_any_zeroes=True )

    data = featureFormat(dictionary, [outcome, feature])
    target, features = targetFeatureSplit( data )

    ### training-testing split needed in regression, just like classification
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


def process_outliers(outliers, feature_attr_name, target_attr_name):
    """
    As we're computing and cleaning outliers,
    we can still gain valuable insights by processing
    those values in various ways
    """
    outlier_names = []

    outlier_features, outlier_targets = zip(*outliers)[0:2]

    dictionary = pickle.load(open('data/enron_dataset.pkl', 'r'))

    for i in range(len(outlier_features)):
        for person_entry in dictionary:
            if dictionary[person_entry].get(feature_attr_name) == outlier_features[i]:
                outlier_names.append(person_entry)

    print outlier_names





def clean_outliers(predictions, feature_values, target_values, feature_attr_name, target_attr_name):

    x_y_pairs = zip(feature_values, target_values)
    pred_outcome_pairs = zip(predictions, target_values)

    errors = abs(predictions - target_values)
    cleaned_data = zip(feature_values, target_values, errors)

    ###sort the uncleaned data by error
    cleaned_data.sort(key=lambda tup: tup[2])

    ## Remove values with top 10% of errors
    cutoff = int(math.floor(len(cleaned_data) * .90))

    outliers = cleaned_data[cutoff:]
    process_outliers(outliers, feature_attr_name, target_attr_name)

    cleaned_data = cleaned_data[:cutoff]

    print len(feature_values)
    print len(cleaned_data)
    #print (cleaned_data)

    return cleaned_data




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

    cleaned_data = clean_outliers(
        reg.predict(feature_train),
        feature_train,
        target_train,
        feature_attr_name=feature,
        target_attr_name=outcome
    )


    if len(cleaned_data) >= 0:
        new_feature_data, new_target_data, errors = zip(*cleaned_data)
        new_feature_data = np.reshape(np.array(new_feature_data), (len(new_feature_data), 1))
        new_target_data = np.reshape(np.array(new_target_data), (len(new_target_data), 1))


    new_feature_train, new_feature_test, new_target_train, new_target_test = train_test_split(
        new_feature_data,
        new_target_data,
        test_size=0.5,
        random_state=42
    )

    reg = classify(new_feature_train, new_target_train)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    train_score = reg.score(new_feature_train, new_target_train)
    test_score = reg.score(new_feature_test, new_target_test)

    print "Slope, after cleaning outliers: {}".format(slope)
    print "Intercept, after cleaning outliers: {}".format(intercept)
    #print "Mean Squared Error: {}".format(mse)
    print "Prediction Score on training data, after cleaning outliers: {}".format(train_score)
    print "Prediction Score on testing data, after cleaning outliers: {}".format(test_score)

    #make_plot(new_feature_train, new_target_train, new_feature_test, new_target_test, reg, feature, outcome)

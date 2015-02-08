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

DICTIONARY = pickle.load(open('data/enron_dataset.pkl', 'r'))
DICTIONARY.pop('TOTAL', 0)



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

    from feature_format import featureFormat, labelFeatureSplit
    #DICTIONARY = pickle.load( open("data/enron_dataset.pkl", "r") )

    ### list the features you want to look at--first item in the
    ### list will be the "label" feature

    #data = featureFormat( DICTIONARY, [outcome, feature], remove_any_zeroes=True)#, "long_term_incentive"], remove_any_zeroes=True )

    data = featureFormat(DICTIONARY, [outcome, feature])
    label, features = labelFeatureSplit( data )

    ### training-testing split needed in regression, just like classification
    feature_train, feature_test, label_train, label_test = train_test_split(features, label, test_size=0.5, random_state=42)

    return feature_train, feature_test, label_train, label_test



def classify(feature_train, label_train):
    from sklearn import linear_model

    reg = linear_model.LinearRegression()
    reg.fit(feature_train, label_train)

    return reg



def make_plot(feature_train, label_train, feature_test, label_test, classifier, x_label, y_label):
    """
    draw the scatterplot, with color-coded training and testing points
    """
    train_color = "#00fffd"
    test_color = "#6600ff"

    import matplotlib.pyplot as plt
    for feature, label in zip(feature_test, label_test):
        plt.scatter( feature, label, color=test_color )
    for feature, label in zip(feature_train, label_train):
        plt.scatter( feature, label, color=train_color )

    ### labels for the legend
    plt.scatter(feature_test[0], label_test[0], color=test_color, label="test")
    plt.scatter(feature_test[0], label_test[0], color=train_color, label="train")


    try:
        plt.plot( feature_test, classifier.predict(feature_test) )
    except NameError:
        pass
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def find_largest_outliers_names(outliers, errors, feature_attr_name, label_attr_name):
    """
    As we're computing and cleaning outliers,
    we can still gain valuable insights by processing
    those values in various ways
    """
    outlier_names = []

    feature_outliers, label_outliers = zip(*outliers)[0:2]

    # Create a list of the outliers' errors and the values that caused them...
    # sorted from largest to smallest
    outlier_label_error_pairs = [(outlier_label, outlier_error) for outlier_label, outlier_error in zip(label_outliers, errors)]
    outlier_label_error_pairs.sort(key=lambda tup: -1 * tup[1])


    for i in range(len(outlier_label_error_pairs)):
        label_val_responsible_for_error = outlier_label_error_pairs[i][0]

        for person_name in DICTIONARY:
            if DICTIONARY[person_name].get(label_attr_name) == label_val_responsible_for_error:
                outlier_names.append(person_name)

    #print outlier_names
    return outlier_names



def find_largest_outlier_entry(outliers, errors, feature_attr_name, label_attr_name):
    """
    Finds the entry corresponding to the outlier with the highest-valued
    label.
    """

    feature_outliers, label_outliers = zip(*outliers)[0:2]

    largest_outlier_label_pos = np.where(errors==max(errors))[0][0]
    larget_outlier_val = label_outliers[largest_outlier_label_pos]

    for i in range(len(label_outliers)):
        for person_name in DICTIONARY:
            if DICTIONARY[person_name].get(label_attr_name) == larget_outlier_val:
                return { person_name: DICTIONARY[person_name] }







def clean_outliers(predictions, feature_values, label_values, feature_attr_name, label_attr_name):

    x_y_pairs = zip(feature_values, label_values)
    pred_outcome_pairs = zip(predictions, label_values)

    errors = abs(predictions - label_values)
    cleaned_data = zip(feature_values, label_values, errors)

    ###sort the uncleaned data by error
    cleaned_data.sort(key=lambda tup: tup[2])
    errors.sort()

    ## Remove values with top 10% of errors
    cutoff = int(math.floor(len(cleaned_data) * .90))

    outliers = cleaned_data[cutoff:]
    outlier_errors = errors[cutoff:]

    outlier_names = find_largest_outliers_names(
        outliers,
        outlier_errors,
        feature_attr_name,
        label_attr_name
    )

    largest_outlier_entry = find_largest_outlier_entry(
        outliers,
        outlier_errors,
        feature_attr_name,
        label_attr_name
    )

    #print largest_outlier_entry

    cleaned_data = cleaned_data[:cutoff]

    #print len(feature_values)
    #print len(cleaned_data)
    #print (cleaned_data)

    return cleaned_data




if __name__ == "__main__":

    ################
    ### Set the feature along with the outcome that it will predict

    #feature = "long_term_incentive"
    feature = "salary"
    outcome = "bonus"

    ################

    feature_train, feature_test, label_train, label_test = create_regression_for_feature_outcome_pair(feature, outcome)

    reg = classify(feature_train, label_train)
    slope = reg.coef_[0]
    intercept = reg.intercept_


    # X_train = np.reshape(np.array(feature_train), (len(feature_train), 1))
    # Y_train = np.reshape(np.array(label_train), (len(label_train), 1))
    #
    # X_test = np.reshape(np.array(feature_test), (len(feature_test), 1))
    # Y_test = np.reshape(np.array(label_test), (len(label_test), 1))


    train_score = reg.score(feature_train, label_train)
    test_score = reg.score(feature_test, label_test)

    print "Slope: {}".format(slope)
    print "Intercept: {}".format(intercept)
    #print "Mean Squared Error: {}".format(mse)
    print "Prediction Score on training data: {}".format(train_score)
    print "Prediction Score on testing data: {}".format(test_score)

    #make_plot(feature_train, label_train, feature_test, label_test, reg, feature, outcome)


    ### Now, to attempt to account for outliters,
    ### we can remove the training items with the top 10% of residual error,
    ### and retrain.

    cleaned_data = clean_outliers(
        reg.predict(feature_train),
        feature_train,
        label_train,
        feature_attr_name=feature,
        label_attr_name=outcome
    )


    if len(cleaned_data) >= 0:
        new_feature_data, new_label_data, errors = zip(*cleaned_data)
        new_feature_data = np.reshape(np.array(new_feature_data), (len(new_feature_data), 1))
        new_label_data = np.reshape(np.array(new_label_data), (len(new_label_data), 1))


    new_feature_train, new_feature_test, new_label_train, new_label_test = train_test_split(
        new_feature_data,
        new_label_data,
        test_size=0.5,
        random_state=42
    )

    reg = classify(new_feature_train, new_label_train)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    train_score = reg.score(new_feature_train, new_label_train)
    test_score = reg.score(new_feature_test, new_label_test)

    print "Slope, after cleaning outliers: {}".format(slope)
    print "Intercept, after cleaning outliers: {}".format(intercept)
    #print "Mean Squared Error: {}".format(mse)
    print "Prediction Score on training data, after cleaning outliers: {}".format(train_score)
    print "Prediction Score on testing data, after cleaning outliers: {}".format(test_score)

    #make_plot(new_feature_train, new_label_train, new_feature_test, new_label_test, reg, feature, outcome)

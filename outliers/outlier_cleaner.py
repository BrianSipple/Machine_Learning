#!/usr/bin/python

import numpy as np
import math

def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error)
    """
    pass




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


def clean_outliers(predictions, feature_values, target_values):

    x_y_pairs = zip(feature_values, target_values)
    pred_outcome_pairs = zip(predictions, target_values)

    errors = abs(predictions - target_values)
    cleaned_data = zip(feature_values, target_values, errors)

    ###sort the uncleaned data by error
    cleaned_data.sort(key=lambda tup: tup[2])

    ## Remove values with top 10% of errors
    cutoff = int(math.floor(len(cleaned_data) * .90))
    cleaned_data = cleaned_data[:cutoff]

    print len(feature_values)
    print len(cleaned_data)

    return cleaned_data

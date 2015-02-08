#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, labelFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../enron_dataset_project/data/enron_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### TODO: Run against outlier cleaning method

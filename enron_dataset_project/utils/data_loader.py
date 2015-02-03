#!/usr/bin/python

"""
    Starting point for exploring the Enron dataset (emails + finances)
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    Here's an example:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
"""
import pickle

enron_data = pickle.load(open("data/enron_dataset.pkl", "r"))

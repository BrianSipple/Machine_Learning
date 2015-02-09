#!/usr/bin/python
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, labelFeatureSplit



def get_min_max_values_for_feature(data_dict, feature):

    #print data_dict
    feature_vals = []

    for key in data_dict:
        val = data_dict[key][feature]

        if val > 0 and val != 'NaN': # ignore all values of "NaN" or 0
            feature_vals.append(val)

    feature_vals.sort()
    return (feature_vals[0], feature_vals[-1])


def scale_features(values):

    res = []

    min_val = np.min(values)
    max_val = np.max(values)
    val_range = max_val - min_val

    if min_val == max_val:  # Prevent division by zero if min and max are equal
        return values

    for i in range(len(values)):
        scaled_val = float(values[i] - min_val)) / float(val_range)
        res.append(scaled_val)

    return res






def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than 4 clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("data/enron_dataset.pkl", "r") )
### there's an outlier--remove it!
data_dict.pop("TOTAL", 0)


### the input features we want to use
### can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = labelFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to
### for f1, f2, _ in finance_features:
### (as it's currently written, line below assumes 2 features)
for f1, f2, _ in finance_features:
    plt.scatter( f1, f2 )

plt.xlabel(finance_features[0])
plt.ylabel(finance_features[1])
plt.show()



from sklearn.cluster import KMeans
features_list = ["poi", feature_1, feature_2, feature_3]
data2 = featureFormat(data_dict, features_list )
poi, finance_features = labelFeatureSplit( data2 )
clf = KMeans(n_clusters=3)
pred = clf.fit_predict( finance_features )
Draw(pred, finance_features, poi, name="clusters_before_scaling.pdf", f1_name=feature_1, f2_name=feature_2)


## cluster here; create predictions of the cluster labels
## for the data and store them to a list called pred

try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"


if __name__ == '__main__':


    print get_min_max_values_for_feature(data_dict, 'exercised_stock_options')
    print get_min_max_values_for_feature(data_dict, 'salary')

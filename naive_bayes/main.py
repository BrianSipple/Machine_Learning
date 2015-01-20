from prep_terrain_data import make_terrain_data
from class_viz import prettyPicture, output_image
from ClassifyNB import classify

import numpy as np
import pylab as pl

def main():
    features_train, labels_train, features_test, labels_test = make_terrain_data()

    ### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
    ### in together--separate them so we can give them different colors in the scatterplot,
    ### and visually identify them
    grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
    bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
    grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
    bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

    clf = classify(features_train, labels_train)

    prettyPicture(clf, features_test, labels_test)

    ### draw the decision boundary with the text points overlaid
    output_image('test', 'png', open('test.png', 'rb').read())


if __name__ == '__main__':

    main()
#!/usr/bin/python

"""
    using an Decision Tree to identify emails from the
    Enron corpus by their authors

    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
import numpy as np
sys.path.append("../tools/")
from email_preprocess import preprocess

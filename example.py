import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import os
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
from kMeansFun import kMeansPlot


dirName = os.getcwd()

## create a random seed ##
np.random.seed(0)


## create random clusters of points using make_blobs class in sklearn package ##
# output: save to variables coords and labs
# coords: the randomly generated data output as an nxn array such that
#         [n_samples, n_features] in this example n_features is the x and y
#         coordinates of the randomly generated data
# labs:   is the cluster labels for each sample [n_samples]
################################################################################

## Define variables to be used in function ##
numSamp = 5000    # Sample size of the random data
clCents = [[4,4], [-2, -1], [2, -3], [1, 1]]  # center points of each blob
stdBlobs = 0.9    # standard deviation of the blobs

coords, labs = make_blobs(n_samples = numSamp, centers = clCents, cluster_std = stdBlobs)

kMeansPlot(coords, labs, saveFile = True)

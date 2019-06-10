import numpy as np
from sklearn.datasets.samples_generator import make_blobs 
from kMeansFun import kMeansPlot

'''
create random clusters of points using make_blobs class in sklearn package
use kMeansPlot to k-means test and group data into clusters.

Parameters:
-----------
numSamp :      Sample size of the random dataset.
clCents:       center points of each blob.
stdBlobs:      standard deviation of the blobs.
coords:        the randomly generated data output as an nxn array such that
               [n_samples, n_features] in this example n_features is the x and y
               coordinates of the randomly generated data.
labs:          is the cluster labels for each sample [n_samples].
'''


## create a random seed ##
np.random.seed(0)

numSamp = 5000 
clCents = [[4,4], [-2, -1], [2, -3], [1, 1]]  
stdBlobs = 0.9    

coords, labs = make_blobs(n_samples = numSamp, centers = clCents, cluster_std = stdBlobs)

kMeansPlot(coords, labs, numClust = 4, tempColorFix=False,saveFile = True)

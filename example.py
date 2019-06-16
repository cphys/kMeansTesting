import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs 
from kMeansFun import kMeansPlot
from kMeansFun import kMeansPlotClust

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

randCoords = {'x values':coords[:,0],'y values':coords[:,1]}
dfRand = pd.DataFrame.from_dict(randCoords)
kMeansPlotClust(dfRand,'x values','y values',saveFile=True,pltTitle='Random Clusters', numClust = 4)



'''
Dataset downloaded from IBM storage as part of customer segmentation lab. Goal
is to partition customer base into groups based on set of collected data.
'''
## Reading in the data and creating a dataframe using pandas ##
# dfCust = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv")

## Save the data set as a csv ##
# df.to_csv("cust_Segmentation.csv", index=False)

dfCust = pd.read_csv("cust_Segmentation.csv")
kMeansPlotClust(dfCust,'Age','Income',size='Edu', dropValues=['Address','Customer Id'],saveFile=True,pltTitle='Customer Segmentation',nInit=1000,scType=False,showCent=False)


'''
kMeansPlot(coords, labs, numClust = 4, tempColorFix=False,saveFile = True)
'''

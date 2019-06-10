import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans 



def kMeansPlot(data, labels, numClust=False, kMehtod = "k-means++", nInit = 12, saveFile = False, dirName = os.getcwd(), scale = 5, ratio = .75, tempColorFix = True):

    """
    Function which uses sklearn machine learning package to perform a
    k-means test. Function will group clustered data and output a plot showing
    both the original data as well as the k-means test data and centroids

    Parameters:
    -----------
    data :         the randomly generated data output as an nxn array such that
                   [n_samples, n_features] in this example n_features is 
                   the x and y coordinates of the randomly generated data.
    labels:        is the cluster labels for each sample [n_samples]
    numClust:      number of clusters in the k-means test. If false default is 
                   the number of unique values in labels.
    kMehtod:       method used to initialize the centroids
    nInit:         number of times to run k-means algorithm with different
                   centroid seeds.
    dirName :      (str) name of the directory to which the file should be saved
    tempColorFix:  Changes the ordering of the colors in the original data to
                   match the k-means test data. This is a temporary fix for the
                   existing data and needs to be changed to handle different 
                   data sets.

    Returns:
    --------
    A plot containing the original data and the kmeans cluster data. Includes
    centroids.
    """
    dataClust = len(set(labels))
    if not numClust:
        numClust = dataClust
    
    kMeansCl = KMeans(init = "k-means++", n_clusters = numClust, n_init = nInit)

    kMeansCl.fit(data) # Fit k-means model to data

    # gives the label of the cluster data based on the k-means model #
    kMeansLabs = kMeansCl.labels_

    # gives the center of the cluster data based on the k-means model #
    kMeansClCents = kMeansCl.cluster_centers_

    # Initialize the plot with the specified dimensions.
    fig = plt.figure(figsize=(scale, ratio * scale))
    # Create plots
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Produce an array of colors based on the number of unique k-means labels
    labSet = set(kMeansLabs) # gives set of unique k-means labels
    colors1 = plt.cm.Spectral(np.linspace(0, 1, dataClust))
    colors2 = plt.cm.Spectral(np.linspace(0, 1, len(labSet)))

    # For loop that plots the data points and centroids.
    # k will range from 0-3, which will match the possible clusters that each
    # data point is in.
    for lab1, color in zip(range(dataClust), colors1):

        # Create a list of all data points, where the data poitns that are 
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        clustMemb1 = (labels == lab1)

        if tempColorFix:
            color = [colors1[i + (-1)**i] for i in range(len(colors1))][lab1]

        # Plots the datapoints with color.
        ax1.scatter(data[clustMemb1, 0], data[clustMemb1, 1], alpha = 0.33, c = [color], edgecolors = [color], s = 5)

    for lab2, color in zip(range(numClust), colors2):

        # Create a list of all data points, where the data poitns that are 
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        clustMemb = (kMeansLabs == lab2)
        
        # Define the centroid, or cluster center.
        kMeansCentroid = kMeansClCents[lab2]

        # Plots the datapoints with color.
        ax2.scatter(data[clustMemb, 0], data[clustMemb, 1], alpha = 0.33, c = [color], edgecolors = [color], s=5)
        
        # Plots the centroids with specified color, but with a darker outline
        ax2.scatter(kMeansCentroid[0], kMeansCentroid[1], alpha = 1, c = [color], edgecolors = 'k', s=30)

    '''
    for k, color in zip(range(numClust), colors):

        # Create a list of all data points, where the data poitns that are 
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        clustMemb1 = (labels == k)
        clustMemb = (kMeansLabs == k)
        
        # Define the centroid, or cluster center.
        kMeansCentroid = kMeansClCents[k]
        print(len(colors))
        if tempColorFix:
            color2 = [colors[i + (-1)**i] for i in range(len(colors))][k]
        else:
            color2 = color
        # Plots the datapoints with color.
        ax1.scatter(data[clustMemb1, 0], data[clustMemb1, 1],alpha = 0.33, c = [color2], edgecolors = [color2], s=5)
        ax2.scatter(data[clustMemb, 0], data[clustMemb, 1],alpha = 0.33, c = [color], edgecolors = [color], s=5)
        
        # Plots the centroids with specified color, but with a darker outline
        ax2.scatter(kMeansCentroid[0], kMeansCentroid[1],alpha = 1, c = [color], edgecolors = 'k', s=30)
    '''
    # Title of the plot
    ax1.set_title('randomly generated data')
    ax2.set_title('k-means data')

    # Show the plot
    # plt.show()
    if saveFile:
        plt.savefig(os.path.join(dirName, 'kMeansClust.png'), bbox_inches='tight')


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans

# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler 



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
    
    kMeansCl = KMeans(init = kMehtod, n_clusters = numClust, n_init = nInit)

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

    # Title of the plot
    ax1.set_title('randomly generated data')
    ax2.set_title('k-means data')

    # Show the plot
    # plt.show()
    if saveFile:
        plt.savefig(os.path.join(dirName, 'kMeansClust.png'), bbox_inches='tight')


def kMeansPlotClust(df,xAxis,yAxis,size=False,sScale=1,dropValues=False, numClust = 3, scType = 'StandardScaler', kMehtod = "k-means++", nInit = 12, saveFile = False, dirName = os.getcwd(), scale = 5, ratio = .75, tempColorFix = True, fontSize = 18, pltTitle=False,showCent=True):

    """
    Function which uses sklearn machine learning package to perform a
    k-means test. Function will group clustered data and output a plot showing
    the k-means test data and centroids

    Parameters:
    -----------
    df :           a pandas dataframe
    xAxis:         dataframe column header to be plotted on the x axis.
    yAxis:         dataframe column header to be plotted on the y axis.
    size:          dataframe column header associated with the size of the dots.
    sScale:        scale for dots in the scatter plot.
    dropValues:    values to be dropped from the dataframe before the kmeans
                   test. Used with non-numerical values
    numClust:      number of clusters in the k-means test. If false default is 
                   the number of unique values in labels.
    scType:        scaling techniques associated with sci-kit learn package.
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
    A plot of scaled data separated by k-means labels. Includes
    centroids.
    """

    # here we want to clean the dataset. Because Address, in this form, is not
    # meaningful in a Euclidean space we drop the function.
    if dropValues != False:
        df = df.drop(dropValues, axis=1)

    # Before defining a Eclidean distance we must first normalize the data
    # This can be done easily with sklearn function StandardScaler

    # create numpy arrays for each row, excluding the first column.
    dfValues = df.values

    # replace nan with 0 and inf(-inf) with large(small) float.
    # in general just replacing nan with 0 is not the appropriate way to 
    # go forward, but this was done in example, should fix later.
    dfValues = np.nan_to_num(dfValues) 


    if scType == False:
        normData = dfValues
    else:
        if scType == 'StandardScaler':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif scType == 'MinMaxScaler':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif scType == 'minmax_scale':
            from sklearn.preprocessing import minmax_scale
            scaler = minmax_scale()
        elif scType == 'MaxAbsScaler':
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler()
        elif scType == 'RobustScaler':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        elif scType == 'Normalizer':
            from sklearn.preprocessing import Normalizer
            scaler = Normalizer()
        elif scType == 'QuantileTransformer':
            from sklearn.preprocessing import QuantileTransformer
            scaler = QuantileTransformer()
        elif scType == 'PowerTransformer':
            from sklearn.preprocessing import PowerTransformer
            scaler = PowerTransformer()
        normData = scaler.fit_transform(dfValues)     

    dfNorm = pd.DataFrame(normData, columns=df.columns, index=df.index)

    # Apply k-means to the dataset and look at the labels
    kMeansDat = KMeans(init = kMehtod, n_clusters = numClust, n_init = nInit)

    kMeansDat.fit(normData)

    # gives the label of the cluster data based on the k-means model #
    kMeansLabs = kMeansDat.labels_

    # gives the center of the cluster data based on the k-means model #
    kMeansClCents = kMeansDat.cluster_centers_

    # Assign labels to each row in the dataframe
    dfNorm['k means clusters'] = kMeansLabs

    # Print the centroid values
    # print(df.groupby('k means clusters').mean())

    # Initialize the plot with the specified dimensions.
    fig = plt.figure(figsize=(scale, ratio * scale))
    # Create plots
    ax1 = fig.add_subplot(111)

    # plot the distribution of customers based on their age and income
    # the size of the circles represents education level
    if not size:
        area = sScale * 30
    else:
        area = sScale * np.pi * (df[size])**2
    normLabs = kMeansLabs.astype(np.float)/np.max(kMeansLabs)
    colors = plt.cm.Spectral(normLabs)
    ax1.scatter(dfNorm[xAxis], dfNorm[yAxis], s=area, c= colors, alpha=0.45, edgecolors = 'k')

    # Plots the datapoints with color.
    if showCent:
        ax1.scatter(kMeansClCents[:,0],kMeansClCents[:,1], alpha = 1, c = 'b', edgecolors = 'r', s = sScale * 45)

    ax1.set_xlabel(xAxis.title().replace("_"," "),fontsize=fontSize)
    ax1.set_ylabel(yAxis.title().replace("_"," "),fontsize=fontSize)

    if pltTitle != False:
        ax1.set_title(pltTitle.title().replace("_"," "),fontsize=fontSize)

    if saveFile:
        plt.savefig(os.path.join(dirName,'{}_{}_vs_{}'.format(pltTitle.replace(" ","_"),xAxis.replace(" ","_"),yAxis.replace(" ","_"))), bbox_inches='tight')

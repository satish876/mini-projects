import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import time


with open('cancer.csv') as f:
    reader = csv.reader(open("./cancer.csv", "r"), delimiter=",")
    x = [x[2:] for x in list(reader)]
    # make a copy of original data
    datapointsOriginal = np.array(x[2:]).astype("float")
    datapoints = np.array(x[2:]).astype("float")

    # normalaziation using z-score
    # getting the attribute wise mean
    means = np.sum(datapoints, axis=0)/datapoints.shape[1]
    # attribute wise subtract from each datapoints the mean of attribute
    datapoints = np.subtract(datapoints,  means)

    # getting the standard deviation
    dataStandardDeviation = np.subtract(datapoints,  means)
    dataStandardDeviation **= 2
    dataStandardDeviation = np.sum(
        dataStandardDeviation, axis=0)/datapoints.shape[0]

    for row in range(datapoints.shape[0]):
        for col in range(datapoints.shape[1]):
            # calculating the z-score for all values
            datapoints[row][col] /= dataStandardDeviation[col]


def distance(src, dest):
    # calculates the eculidian distance
    return np.linalg.norm(src - dest)


def getDistanceVectorMartix(datapoints, epsilon):
    # this is a N*N symmetric matrix which holds the euclidian distance for every pair of datapoints
    N = datapoints.shape[0]
    distanceVectorMartix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if(i == j):
                distanceVectorMartix[i][j] = 0
            else:
                distanceVectorMartix[i][j] = distance(datapoints[i], datapoints[j]) \
                    if i < j else distanceVectorMartix[j][i]
    return distanceVectorMartix


def getFilteredCluster(distanceVectorMartix, minPoints, epsilon):
    # this function will filter the datapoints based on minp=Points and epsilon
    filteredCluster = []
    outerIndex = 0
    for arr in distanceVectorMartix:
        brr = []
        # index = 0
        for index in range(outerIndex + 1, distanceVectorMartix.shape[0]):
            if arr[index] <= epsilon:
                # since outerIndex point is within the radius of outerIndex, addd it to its cluster
                brr.append(index)
            index += 1
        if len(brr) >= minPoints:
            # since this cluster's cardinality is atleast minPoints, we add this to result
            filteredCluster.append(brr)

        outerIndex += 1
    return np.array(filteredCluster, dtype=np.object)


def plot(datapoints, labels, title):
    # this function will plot the clusters
    # for each clusters it will get the respective x-axis and y-axis value and add it to a 2d array
    fig = plt.figure()
    fig.suptitle(title)

    # plotting the data
    # initialize the labels
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")

    K = len(labels)
    # init the array for x and y coordinates for all the medoids
    plotDataX = [[] for x in range(K)]
    plotDataY = [[] for x in range(K)]

    for labelIndex in range(K):
        for data in labels[labelIndex]:
            plotDataX[labelIndex].append(datapoints[data][0])
            plotDataY[labelIndex].append(datapoints[data][1])

    colors = ["red", "blue", "green", "orange", "brown", "yellow", "pink"]
    # colors = ['#67B4B0', '#0978E6', '#50CD10', '#89C191', '#5C8C6D', '#29D6BC', '#688414', '#963724', '#844455', '#86A336', '#CEA461', '#F326DC', '#A19DB1', '#535C3C', '#55D04C', '#CFF8FC', '#46F332', '#BE3738', '#C30513', '#FB585B', '#FF8A88', '#FCFC5D', '#5E3E1B', '#7CD530', '#E444F4', '#70897E', '#5AF7E9', '#A73215', '#58F4C5', '#BF27F0', '#44417D', '#F5BD82', '#1BC5D0', '#D8E5B1', '#A203F5', '#C58BA5', '#6AA663', '#C761D7', '#5ED3E0', '#2F1C7F', '#E5B42C', '#B08E82', '#AB7FD9', '#F764D9', '#A9D2B5', '#FB04A5', '#7B1060', '#4D89BB', '#F4D527', '#13BBDD',
    #           '#7C234A', '#8F689C', '#18D487', '#70DB8D', '#D3729B', '#05F40F', '#ECE0B6', '#FAD492', '#BCF039', '#5535AE', '#C6AACE', '#8F398F', '#4C0356', '#655193', '#9CDB35', '#5DE0FE', '#0CA01C', '#EB8440', '#30F09E', '#611533', '#D08FFC', '#39CF27', '#2F5BB3', '#CB7FEB', '#C9D1B2', '#0B0BA3', '#879A0C', '#1FE229', '#AA9DD3', '#E6556A', '#8555CA', '#B93B79', '#6DE05D', '#8DABD0', '#9EFCD0', '#13ADC0', '#F28F55', '#1A92FD', '#2DDF6E', '#AA9111', '#06FBEF', '#D862B4', '#F70384', '#DCFBA0', '#8C5140', '#1A6EF8', '#9C536C', '#53F4DC', '#D579E6', '#0F9F46']

    # finally plotting the datapoints and graph
    for k in range(K):
        plt.scatter(plotDataX[k], plotDataY[k], color=colors[k], alpha=0.5)


def dbscan(datapoints, minPoints, epsilon):
    starttime = time.time()
    N = datapoints.shape[0]
    # get the distance matrix for every pair of datapoints
    distanceVectorMartix = getDistanceVectorMartix(datapoints, epsilon)
    # get the filtered cluster which satisfy the minPoints and epsilon criteria
    initialClusters = getFilteredCluster(
        distanceVectorMartix, minPoints, epsilon)

    # start merging the unique clusters
    NN = initialClusters.shape[0]
    # labels array will contains the labels of clusters,
    labels = np.full(NN, -1)
    indexx = 0

    # labeling all the clusters
    # for datapoints in same cluster, we assign them same label
    while indexx < NN:
        while indexx < NN and labels[indexx] != -1:
            indexx += 1
        if indexx >= NN:
            break
        if labels[indexx] == -1:
            labels[indexx] = indexx

        nextptr = indexx + 1
        while nextptr < NN:
            while nextptr < NN and labels[nextptr] != -1:
                nextptr += 1

            if nextptr >= NN:
                break

            # if two clusters A,B are not mutually exclusive, they will be given same label
            if bool(set(initialClusters[indexx]) & set(initialClusters[nextptr])):
                labels[nextptr] = indexx
            nextptr += 1
        indexx += 1

    # this will contain the unique points in each cluster
    finalPoints = [set() for i in range(np.unique(labels).shape[0])]

    # for all the cluster, assign numbers 0,1,2 in increasing order
    # this is only needed because I have used 2d array
    assignedSlot = {}
    slotCounter = 0
    for label in np.unique(labels):
        if str(label) not in assignedSlot:
            assignedSlot[str(label)] = slotCounter
            slotCounter += 1

    # for 2 intersecting clusters take union of elemnts and save
    for index, label in enumerate(labels):
        finalPoints[assignedSlot[str(label)]] = finalPoints[assignedSlot[str(
            label)]].union(set(initialClusters[index]))

    # plot title
    title = "DBSCAN minPoints={}, epsilon={}".format(minPoints, epsilon)

    print('time     %0.2fs' % (time.time()-starttime))
    plot(datapointsOriginal, finalPoints, title)


dbscan(datapoints, minPoints=6, epsilon=0.5)
dbscan(datapoints, minPoints=6, epsilon=0.2)
dbscan(datapoints, minPoints=3, epsilon=0.2)
plt.show()

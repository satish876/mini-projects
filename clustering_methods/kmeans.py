import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import time


with open('cancer.csv') as f:
    reader = csv.reader(open("./cancer.csv", "r"), delimiter=",")
    x = [x[2:] for x in list(reader)]
    datapoints = np.array(x[2:]).astype("float")


def distance(src, dest):
    # calculates the eculidian distance
    return np.linalg.norm(src - dest)


def kmeans(datapoints, K, maxIter):
    starttime = time.time()
    # k-random datapoitns as centroids
    centroids = np.array([datapoints[randIndex] for randIndex in np.random.choice(
        range(datapoints.shape[0]), K, replace=False)])

    # store the old centroid to compare if the ireration has converged
    oldCentroidDistance = np.zeros(datapoints.shape[1])
    # run the kmeans algo
    for iterCount in range(maxIter):
        # find which centroid is close to a datapoint
        # initialize the close centroid index for all the data points
        closeCentroidIndex = np.zeros(datapoints.shape[0], dtype=np.int)

        # distance of data-points with its assigned centroid
        distanceVectors = np.zeros(datapoints.shape[0])
        # calculate the distortion for centroids
        distortionValues = np.zeros(K)
        # count datapoints which are close to each of the centroids
        datasetCountForCentroids = np.zeros(K, dtype=np.int)

        #  cluster assignment
        dpIndex = 0
        for values in datapoints:
            # calculate the euclidian distance of point with all centroids
            dist_vector = np.array([distance(center, values)
                                    for center in centroids])
            # find which minm value, will use to find which centroid is nearer
            minm_dist = dist_vector.min()

            distanceVectors[dpIndex] = minm_dist
            # update the index of centorid which is closer to this datapt
            closeCentroidIndex[dpIndex] = 0 if minm_dist == dist_vector[0] else 1

            dpIndex += 1

        # calculate the distortion, by adding distance for all the assigned points to a cluster
        for index in closeCentroidIndex:
            datasetCountForCentroids[index] += 1
            distortionValues[index] += distanceVectors[index]
        # get the average distortion
        for centroidIndex in range(distortionValues.shape[0]):
            distortionValues[centroidIndex] /= datasetCountForCentroids[centroidIndex]

        # check if centroids have converged
        # if yes, we have reached the global optima
        if(np.array_equal(oldCentroidDistance,  distortionValues)):
            break

        # move-cluster
        # take the attribute-wise average for datapoints belonging to a particular centroid
        centroids = np.zeros([K, datapoints.shape[1]])
        for rowIndex in range(len(closeCentroidIndex)):
            centroidIndex = closeCentroidIndex[rowIndex]
            centroids[centroidIndex] = np.add(
                centroids[centroidIndex], datapoints[rowIndex])
        # divide by row count to get the average
        for countIndex in range(len(datasetCountForCentroids)):
            centroids[countIndex] /= datasetCountForCentroids[countIndex]

        # save the oldCentoid distance to compare it with the new value gengerated in next iteration
        oldCentroidDistance = np.array(distortionValues)

    # plotting the data
    # initialize the labels

    fig = plt.figure()
    fig.suptitle("k-means")
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")
    # init the array for x and y coordinates for all the centroids
    plotDataX = [[] for x in range(K)]
    plotDataY = [[] for x in range(K)]

    # for all the datapoints collect the x-axis and y-axis ans append it to plot-data array
    for index in range(datapoints.shape[0]):
        plotDataX[closeCentroidIndex[index]].append(datapoints[index][0])
        plotDataY[closeCentroidIndex[index]].append(datapoints[index][1])

    colors = ["red", "blue", "green"]
    # finally plotting the datapoints and centroids
    for k in range(K):
        plt.scatter(plotDataX[k], plotDataY[k], color=colors[k], alpha=0.5)
        plt.scatter(centroids[k][0], centroids[k]
                    [1], marker="X", color="black")

    print('time     %0.2fs' % (time.time()-starttime))
    plt.show()


# args: datapoints, value of K, max no of iteration
kmeans(datapoints, 2, 100)

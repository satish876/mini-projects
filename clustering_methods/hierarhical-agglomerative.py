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


def getDistanceVectorMartix(datapoints):
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


def mergeClusters(oldClusters, src, dest):
    # merege the 2 most close clusters, based on select algo
    N = len(oldClusters)
    newClusters = oldClusters

    newClusters[src] = newClusters[src] + newClusters[dest]
    newClusters.remove(newClusters[dest])

    return newClusters


def findClosestClusterPairs(clusters, distanceVectorMartix, algo):
    # for every pair of clusters, we will calculate the closeness based on algo(single/complete/average)
    # we will finally get the index of 2 such clusters and call mergeCluster() on them
    N = len(clusters)
    allClusterPairs = []
    for i in range(N):
        for j in range(i+1, N):
            allPairs = []
            # for every cluster paris, get the ordered pair distance of points anmong them
            for m in range(len(clusters[i])):
                for n in range(len(clusters[j])):
                    # record all possible ordered pair between 2 clusters
                    allPairs.append(
                        (i, j, distanceVectorMartix[clusters[i][m]][clusters[j][n]]))

            if algo == "single":
                # if the algorithm is single-link, we will find the most closest /similar point
                # for a cluster pair and uses that to compare similarities between cluster
                allClusterPairs.append(min(allPairs, key=lambda k: k[2]))
            elif algo == "complete":
                # if the algorithm is complete-link, we will find the lest closest/similar point
                # for a cluster pair and uses that to compare similarities between cluster
                allClusterPairs.append(max(allPairs, key=lambda k: k[2]))
            elif algo == "average":
                # if the algorithm is average-link, we will get the average of all possible ordered pairs
                # for a cluster pair and uses that to compare similarities between cluster
                allClusterPairs.append(
                    (i, j, sum(cost for a, b, cost in allPairs) / len(allPairs)))

    # find the closeset cluster pair and merge them in next step
    bestsrc, bestdest, bestcost = min(allClusterPairs, key=lambda k: k[2])

    # merge these two closest clusters
    modifiedCluster = mergeClusters(clusters, bestsrc, bestdest)
    return modifiedCluster


def plot(datapoints, cluster):
    # plotting the data
    # initialize the labels
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")
    K = 2
    # init the array for x and y coordinates for all the medoids
    plotDataX = [[] for x in range(K)]
    plotDataY = [[] for x in range(K)]

    for clusterIndex in range(K):
        for data in cluster[clusterIndex]:
            plotDataX[clusterIndex].append(datapoints[data][0])
            plotDataY[clusterIndex].append(datapoints[data][1])

    colors = ["red", "blue", "green"]

    # finally plotting the datapoints and medoids
    for k in range(K):
        plt.scatter(plotDataX[k], plotDataY[k], color=colors[k], alpha=0.5)

    # plt.show()


def hierarchiacalAgglomerative(datapoints, algo):
    starttime = time.time()
    N = datapoints.shape[0]

    # get the distance matrix for every pair of datapoints
    distanceVectorMartix = getDistanceVectorMartix(datapoints)
    # init the inital N clusters
    clusters = [[i] for i in range(N)]

    # repeatetively merge the clusters until there are 2 clusters left
    modifiedCluster = clusters
    while(len(modifiedCluster) > 2):
        modifiedCluster = findClosestClusterPairs(
            clusters, distanceVectorMartix, algo)

    fig = plt.figure()
    fig.suptitle(algo + ' linkage')
    print('time     %0.2fs' % (time.time()-starttime))
    plot(datapoints, modifiedCluster)


hierarchiacalAgglomerative(datapoints, "single")
hierarchiacalAgglomerative(datapoints, "complete")
hierarchiacalAgglomerative(datapoints, "average")
plt.show()

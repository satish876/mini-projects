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


def getCost(medoids, distanceVectorMartix, src, dest):
    # src and dest are the 2 medoids, for which we have to calculate the total cost
    cost = 0
    M, N = distanceVectorMartix.shape

    # for all other points find the sum of min(distance form src, distance from dest)
    for i in range(M):
        if i not in medoids:
            cost += min(distanceVectorMartix[src]
                        [i], distanceVectorMartix[dest][i])
    return cost


def kmedoids(datapoints, K):
    starttime = time.time()
    # k-random datapoitns as medoids
    M, N = datapoints.shape

    # chose k random points as k-medoids
    # randomMedoids = np.array([372, 484], dtype=np.int)
    randomMedoids = np.random.choice(range(M), K, replace=False)

    # calculate the distance matrix for pair of data-points
    distanceVectorMartix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if(i == j):
                distanceVectorMartix[i][j] = 0
            else:
                distanceVectorMartix[i][j] = distance(datapoints[i], datapoints[j]) \
                    if i < j else distanceVectorMartix[j][i]

    # const vectort matrix will contain the total cost if 2 datapoint pairs are considered to be the medoids
    costVectorMatrix = np.array([[-1 for x in range(M)]
                                 for y in range(M)], dtype=np.float)

    newMedoids = np.array(randomMedoids)
    # this is to make the non-medoid datapoint as random, for the k-medoid algoritm below
    visitedDatapoints = np.array(range(M))
    np.random.shuffle(visitedDatapoints)

    # cost of first k-medoids
    currentCost = getCost(randomMedoids, distanceVectorMartix,
                          randomMedoids[0], randomMedoids[1])
    costVectorMatrix[randomMedoids[0], randomMedoids[1]] = currentCost
    costVectorMatrix[randomMedoids[1], randomMedoids[0]] = currentCost

    newCost = -1.0
    # if for a whole pair of medoid-nonmedoid, there is no change in the chosen k-medoids, mark hasImproved as faslse, true otherwise
    hasImproved = True
    while currentCost != newCost and hasImproved:
        # loop until we get a better total cost, or there is no change in medoid for allmedoid-nonmedoid pairs
        hasImproved = False
        for medoidIndex in range(K):
            for i in range(M):
                dataIndex = visitedDatapoints[i]
                if dataIndex not in randomMedoids:
                    newMedoids = np.array(randomMedoids)
                    newMedoids[medoidIndex] = dataIndex

                    # if this is a new pair of medoids to be considered, get the totoal cost and save it for later use, if repeated
                    if costVectorMatrix[newMedoids[0], newMedoids[1]] == -1:
                        newCost = getCost(
                            newMedoids, distanceVectorMartix, newMedoids[0], newMedoids[1])
                        costVectorMatrix[newMedoids[0],
                                         newMedoids[1]] = newCost
                        costVectorMatrix[newMedoids[1],
                                         newMedoids[0]] = newCost
                    else:
                        # we already have calculated the cost between these medoid, use that old value
                        newCost = costVectorMatrix[newMedoids[0],
                                                   newMedoids[1]]
                    if newCost < currentCost:
                        # better cost found, so we will update the medoids
                        hasImproved = True
                        # update this as new medoid
                        randomMedoids = newMedoids
                        currentCost = newCost

    # plotting the data
    # initialize the labels
    fig = plt.figure()
    fig.suptitle("k-medoids")
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")
    # init the array for x and y coordinates for all the medoids
    plotDataX = [[] for x in range(K)]
    plotDataY = [[] for x in range(K)]

    closeCentroidIndex = np.zeros(M, dtype=np.int)
    for dpIndex in range(M):
        if dpIndex not in newMedoids:
            closeCentroidIndex[dpIndex] = 0 if distanceVectorMartix[dpIndex][newMedoids[0]
                                                                             ] < distanceVectorMartix[dpIndex][newMedoids[1]] else 1
    # for all the datapoints collect the x-axis and y-axis ans append it to plot-data array

    # return
    for index in range(M):
        if index not in newMedoids:
            plotDataX[closeCentroidIndex[index]].append(datapoints[index][0])
            plotDataY[closeCentroidIndex[index]].append(datapoints[index][1])

    colors = ["red", "blue", "green"]
    # finally plotting the datapoints and medoids
    for k in range(K):
        plt.scatter(plotDataX[k], plotDataY[k], color=colors[k], alpha=0.5)
        plt.scatter(datapoints[newMedoids[k]][0], datapoints[newMedoids[k]]
                    [1], marker="X", color="black")

    print('time     %0.2fs' % (time.time()-starttime))
    plt.show()


# args: datapoints, value of K
kmedoids(datapoints, 2)

#The below code is the implementation of Best first search, A* and Hill climbing under two heuristics- Misplaced tiles and Manhattan Distance.
#The goal state is set in the form of 3X3 array and can be configured
#The input file name is considered as "input.txt" and is accesed under "Files" in Google Colab
import numpy as np
import time
import heapq
#goal state array
# goalState = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0]).reshape(3, 3)
#goalState = np.array([1, 2, 3, 8, 0, 4, 7, 6, 5]).reshape(3, 3)


class State():

    def __init__(self, state, parent, heuristicCost):
        self.state = state
        self.parent = parent
        self.hcost = heuristicCost  # considered as sum of h(n)+g(n)
        self.up = None
        self.down = None
        self.left = None
        self.right = None

# left move check
    def checkLeft(self):
        matrix = np.copy(self.state) 
        result = np.where(matrix == 0)  
        row = result[0]
        col = result[0]

        for r in range(0, 3):
            for c in range(0, 3):
                if matrix[r][c] == 0:
                    row = r
                    col = c

        if col == 0:
            return False, 0
        matrix[row][col] = matrix[row][col-1]
        matrix[row][col-1] = 0

        return True, matrix
# right move check
    def checkRight(self):
        matrix = np.copy(self.state)  
        result = np.where(matrix == 1) 
        row = result[0]
        col = result[0]

        for r in range(0, 3):
            for c in range(0, 3):
                if matrix[r][c] == 0:
                    row = r
                    col = c

        if col == 2:
            return False, 0
        matrix[row][col] = matrix[row][col+1]
        matrix[row][col+1] = 0

        return True, matrix
# up move check
    def checkUp(self):
        matrix = np.copy(self.state)
        result = np.where(matrix == 1)  
        row = result[0]
        col = result[0]

        for r in range(0, 3):
            for c in range(0, 3):
                if matrix[r][c] == 0:
                    row = r
                    col = c

        if row == 0:
            return False, 0
        matrix[row][col] = matrix[row-1][col]
        matrix[row-1][col] = 0

        return True, matrix
# Down move check
    def checkDown(self):
        matrix = np.copy(self.state)
        result = np.where(matrix == 1)  
        row = result[0]
        col = result[0]

        for r in range(0, 3):
            for c in range(0, 3):
                if matrix[r][c] == 0:
                    row = r
                    col = c

        if row == 2:
            return False, 0
        matrix[row][col] = matrix[row+1][col]
        matrix[row+1][col] = 0
        return True, matrix

    def h1(self, in_array):  # heuristic1, used to check on the basis of number of misplaced tiles
        misplaced = 0
        val = 1
        for r in range(0, 3):
            for c in range(0, 3):
                if(in_array[r][c] != val % 9 and in_array[r][c]!=0 ): 
                    misplaced += 1
                val += 1
        return misplaced

    def h2(self, in_array, goal_array):  # heuristic2, used to check on the basis of manhattan distance
        man_dis = 0  # manhattan distance
        pos = False
        for i in range(0, 3):
            for j in range(0, 3):
                if(in_array[i][j] != 0):
                    pos = False
                    i1 = 0
                    j1 = 0
                    for i1 in range(0, 3):
                        for j1 in range(0, 3):
                            if(in_array[i][j] == goal_array[i1][j1]):
                                pos = True
                                break
                        if(pos == True):
                            break
                    diffrow = abs(i-i1)
                    diffcol = abs(j-j1)
                    man_dis = man_dis+diffrow+diffcol

        return man_dis
#get the required path, traverse and print
    def getPath(self, isFinal,arrow):
        if self.parent is None:
            if isFinal:
                #print(self.state)
                self.prettyPrint(self.state)
                if arrow == 1:
                  print('   ',u'\u2193')
            return 1
        else:
            op = 1
            op = op + self.parent.getPath(isFinal,1)
            if isFinal:
                #print(self.state)
                self.prettyPrint(self.state)
                if arrow == 1:
                  print('   ',u'\u2193')
            return op

    def start(self, init_state, goal_state,algo,hrN):
        states_explored = 1
        start = time.time()
        pQueue = []
        pQueue.append((self, 0))
        visitedState = set([])

        while pQueue:
            # print(self.state)
            pQueue = sorted(pQueue, key=lambda x: x[1])
            currentState = pQueue.pop(0)[0]
            visitedState.add(
                tuple(np.array(currentState.state).reshape(1, 9)[0]))
            if np.array_equal(currentState.state, goal_state):
                print('Success')
                print('Start state')
                self.prettyPrint(input_arr)
                print('Goal state')
                self.prettyPrint(goalState)
                print('Total number of states explored:- ', states_explored)
                print()
                optimalStates = 0
                optimalStates = currentState.getPath(True,0)
                print('Total number of states to optimal path:- ', optimalStates)
                print('Optimal path cost:- ', optimalStates-1)
                print('Total time     %0.2fs' % (time.time()-start))
                pQueue.clear()
                visitedState.clear()
                return
            else:
                # check for left side
                b, mat = currentState.checkLeft()
                # print(mat)
                if b and tuple(tuple(np.array(mat).reshape(1, 9)[0])) not in visitedState:   # check if move can be done and state is not visited before
                    if hrN == 1:
                      herCost = self.h1(mat)
                    else: # hrN == 2
                      herCost = self.h2(mat,goalState)
                    # print("h1",herCost)
                    if algo == "a*":
                      totalCost = self.getPath(False,0)-1+herCost
                    else:
                      totalCost = herCost
                    currentState.left = State(
                        state=mat, parent=currentState, heuristicCost=totalCost) #sending the h(n)+g(n) as totalcost
                    pQueue.append((currentState.left, totalCost))
                    states_explored += 1
                    visitedState.add(tuple(np.array(mat).reshape(1, 9)[0]))

                # check for right side
                b, mat = currentState.checkRight()
                # print(mat)
                if b and tuple(tuple(np.array(mat).reshape(1, 9)[0])) not in visitedState:
                    if hrN == 1:
                      herCost = self.h1(mat)
                    else: # hrN == 2
                      herCost = self.h2(mat,goalState)
                    # print("h1",herCost)
                    if algo == "a*":
                      totalCost = self.getPath(False,0)-1+herCost
                    else:
                      totalCost = herCost
                    currentState.right = State(
                        state=mat, parent=currentState, heuristicCost=totalCost) #sending the h(n)+g(n) as totalcost
                    pQueue.append((currentState.right, totalCost))
                    states_explored += 1
                    visitedState.add(tuple(np.array(mat).reshape(1, 9)[0]))

                # check for up side
                b, mat = currentState.checkUp()
                # print(mat)
                if b and tuple(tuple(np.array(mat).reshape(1, 9)[0])) not in visitedState:
                    if hrN == 1:
                      herCost = self.h1(mat)
                    else: # hrN == 2
                      herCost = self.h2(mat,goalState)
                    # print("h1",herCost)
                    if algo == "a*":
                      totalCost = self.getPath(False,0)-1+herCost
                    else:
                      totalCost = herCost
                    currentState.up = State(
                        state=mat, parent=currentState, heuristicCost=totalCost)  #sending the h(n)+g(n) as totalcost
                    pQueue.append((currentState.up, totalCost))
                    states_explored += 1
                    visitedState.add(tuple(np.array(mat).reshape(1, 9)[0]))

                # check for down side
                b, mat = currentState.checkDown()
                # print(mat)
                if b and tuple(tuple(np.array(mat).reshape(1, 9)[0])) not in visitedState:
                    if hrN == 1:
                      herCost = self.h1(mat)
                    else: # hrN == 2
                      herCost = self.h2(mat,goalState)
                    # print("h1",herCost)
                    if algo == "a*":
                      totalCost = self.getPath(False,0)-1+herCost
                    else:
                      totalCost = herCost
                    currentState.down = State(
                        state=mat, parent=currentState, heuristicCost=totalCost)  #sending the h(n)+g(n) as totalcost
                    pQueue.append((currentState.down, totalCost))
                    states_explored += 1
                    visitedState.add(tuple(np.array(mat).reshape(1, 9)[0]))
        #failure, exiting from while loop and printing the desired result
        print('Failure')      
        print('Start state:-' )
        self.prettyPrint(input_arr)
        #print(input_arr)
        print('Goal state:- ')
        self.prettyPrint(goalState)
        #print(goalState)
        print('Total number of states explored:- ',states_explored)

    def prettyPrint(self, arr):
        print('{:-<14}'.format(""))
        for x in arr:
            print("| ", end=" ")
            for y in x:
                print(f'{ y }', end=" ")
            print(" |", end=" ")
            print("")
        print('{:-<14}'.format(""))


class HillClimbing:
    startState = []
    goalState = []
    pQueue = []
    visitedStates = {}
    noOfScannedNodes = 0

    goal_indices = []
    hashedMatrices = {}

    def __init__(self, state, parent, heuristicCost):
        self.state = state
        self.parent = parent
        self.hcost = heuristicCost
        self.up = None
        self.down = None
        self.left = None
        self.right = None

        parentNode = HillClimbing.visitedStates[parent] if parent is not None else False
        self.row = parentNode.row if parentNode and parentNode.row is not None else None
        self.col = parentNode.col if parentNode and parentNode.col is not None else None

        # find the position of '0' tile in the current state
        if self.row is None or self.col is None:
            for r in range(0, 3):
                for c in range(0, 3):
                    if self.state[r][c] == 0:
                        self.row = r
                        self.col = c
                        break

    def getNextState(self, dirn, algo, hfn):
        # create new state
        matrix = np.copy(self.state)
        newNode = HillClimbing(state=matrix, parent=id(self), heuristicCost=0)

        # return if the move is not possible
        if ((dirn == "up" and newNode.row == 0)
            or (dirn == "down" and newNode.row == 2)
            or (dirn == "left" and newNode.col == 0)
                or (dirn == "right" and newNode.col == 2)):
            return None

        if dirn == "up":
            newNode.state[newNode.row][newNode.col] = newNode.state[
                newNode.row - 1][newNode.col]
            newNode.state[newNode.row - 1][newNode.col] = 0
            #  update new row,col for blank
            newNode.row -= 1
        elif dirn == "down":
            newNode.state[newNode.row][newNode.col] = newNode.state[
                newNode.row + 1][newNode.col]
            newNode.state[newNode.row + 1][newNode.col] = 0
            #  update new row,col for blank
            newNode.row += 1
        elif dirn == "left":
            newNode.state[newNode.row][newNode.col] = newNode.state[
                newNode.row][newNode.col - 1]
            newNode.state[newNode.row][newNode.col - 1] = 0
            #  update new row,col for blank
            newNode.col -= 1
        elif dirn == "right":
            newNode.state[newNode.row][newNode.col] = newNode.state[
                newNode.row][newNode.col + 1]
            newNode.state[newNode.row][newNode.col + 1] = 0
            #  update new row,col for blank
            newNode.col += 1

        hashedValue = newNode.getHashedMatrixValue()

        if hashedValue not in HillClimbing.hashedMatrices:
            HillClimbing.noOfScannedNodes += 1
        # if the new node is not better than its parent or old state is generated, delete it
        if hashedValue in HillClimbing.hashedMatrices or (algo == "HILL" and newNode.hcost >= self.hcost):
            setattr(self, dirn, None)
            del newNode
            return
        else:
            # calculate the heuristic cost for new state
            newNode.hcost = getattr(HillClimbing, hfn)(newNode)

            # store current martix confi to hash, to prevent reentering same matrix again
            HillClimbing.hashedMatrices[hashedValue] = True

            # link parent
            setattr(self, dirn, id(newNode))
            HillClimbing.visitedStates[id(newNode)] = newNode

    def calculateIndicesForGoal(self):
        # calculatre indices for goal node, once and for all
        HillClimbing.goal_indices = [[0, 0] for x in range(9)]
        for r in range(0, 3):
            for c in range(0, 3):
                HillClimbing.goal_indices[HillClimbing.goalState[r][c]] = [
                    r, c]

    def h1(self):  # heuristic1
        misplaced = 0
        for r in range(0, 3):
            for c in range(0, 3):
                if self.state[r][c] != 0 and HillClimbing.startState[r][
                        c] != self.state[r][c]:
                    misplaced += 1
        return misplaced

    def h2(self):  # heuristic2
        man_dis = 0  # manhattan distance

        ip_indices = [[0, 0] for x in range(9)]
        for r in range(3):
            for c in range(3):
                ip_indices[self.state[r][c]] = [r, c]

        for i in range(1, 9):
            man_dis += abs(ip_indices[i][0] - HillClimbing.goal_indices[i][0]) + \
                abs(ip_indices[i][1] - HillClimbing.goal_indices[i][1])

        return man_dis

    def getPath(self):
        if self.parent not in HillClimbing.visitedStates:
            self.prettyPrint(self.state)
            return 1
        else:
            op = 1
            op = op + HillClimbing.visitedStates[self.parent].getPath()
            self.prettyPrint(self.state)
            return op

    def getHashedMatrixValue(self):
        hash = ""
        for x in self.state:
            for y in x:
                hash += str(y)
        return hash

    def start(self, init_state, goal_state, algo, hfn):
        # store the result here
        result = {'algo': algo, 'hfn': hfn}

        self.calculateIndicesForGoal()
        starttime = time.time()

        # store current martix confi to hash, to prevent reentering same matrix again
        HillClimbing.hashedMatrices[self.getHashedMatrixValue()] = True

        # init the priority queue
        HillClimbing.pQueue = []
        heapq.heapify(HillClimbing.pQueue)

        self.hcost = getattr(self, hfn)()  # self.h2()
        # we are storing the objectId instead of the object itself
        heapq.heappush(HillClimbing.pQueue, (self.hcost, id(self)))
        HillClimbing.visitedStates[id(self)] = self

        while len(HillClimbing.pQueue):
            currentState = HillClimbing.visitedStates[heapq.heappop(HillClimbing.pQueue)[
                1]]

            if np.array_equal(currentState.state, goal_state):
                print("Success")
                print("Start state:- ")
                self.prettyPrint(HillClimbing.startState)
                print("Goal state:- ")
                self.prettyPrint(HillClimbing.goalState)
                print("Total number of states explored:- ",
                      HillClimbing.noOfScannedNodes)
                optimalStates = 0
                optimalStates = currentState.getPath()
                print("Total number of states to optimal path:- ",
                      optimalStates)
                print("Optimal path cost:- ", optimalStates - 1)
                print("Total time     %0.2fs" % (time.time() - starttime))

                result['time'] = time.time() - starttime
                result['exploredCount'] = HillClimbing.noOfScannedNodes
                result['cost'] = optimalStates - 1
                result['status'] = 'Success'

               # ComparisionReport.reports.append(result)

                self.resetAll()
                return
            else:
                currentState.getNextState("left", algo, hfn)
                currentState.getNextState("right", algo, hfn)
                currentState.getNextState("up", algo, hfn)
                currentState.getNextState("down", algo, hfn)

                currentState.reorderAllStates(algo)

        # failure
        print("\n\nFailure")
        print("Start state:- ")
        self.prettyPrint(HillClimbing.startState)
        print("Goal state:- ")
        self.prettyPrint(HillClimbing.goalState)
        print("Total number of states explored:- ",
              HillClimbing.noOfScannedNodes)

        result['time'] = time.time() - starttime
        result['exploredCount'] = HillClimbing.noOfScannedNodes
        result['cost'] = '-'
        result['status'] = 'Failure'

        #ComparisionReport.reports.append(result)
        self.resetAll()
        return

    def reorderAllStates(self, algo):
        if algo == "HILL":
            HillClimbing.pQueue.clear()
        elif algo == "BEST":
            pass

        if (self.left is not None):
            heapq.heappush(
                HillClimbing.pQueue, (HillClimbing.visitedStates[self.left].hcost, self.left))
        if (self.right is not None):
            heapq.heappush(
                HillClimbing.pQueue, (HillClimbing.visitedStates[self.right].hcost, self.right))
        if self.up is not None:
            heapq.heappush(
                HillClimbing.pQueue, (HillClimbing.visitedStates[self.up].hcost, self.up))
        if (self.down is not None):
            heapq.heappush(
                HillClimbing.pQueue, (HillClimbing.visitedStates[self.down].hcost, self.down))

    def resetAll(self):
        HillClimbing.visitedStates = {}
        HillClimbing.pQueue.clear()
        HillClimbing.noOfScannedNodes = 0
        HillClimbing.hashedMatrices = {}
        self.removeNode()

    def removeNode(self):
        if self is None:
            return

        if self.down in HillClimbing.visitedStates:
            HillClimbing.visitedStates[self.down].removeNode()
        if self.up in HillClimbing.visitedStates:
            HillClimbing.visitedStates[self.up].removeNode()
        if self.right in HillClimbing.visitedStates:
            HillClimbing.visitedStates[self.right].removeNode()
        if self.left in HillClimbing.visitedStates:
            HillClimbing.visitedStates[self.left].removeNode()
        del self

    def prettyPrint(self, arr):
        print('{:-<14}'.format(""))
        for x in arr:
            print("| ", end=" ")
            for y in x:
                print(f'T{ y }', end=" ")
            print(" |", end=" ")
            print("")
        print('{:-<14}'.format(""))



#input array and code to fetch input data from file
# ii=[]
# ii1=[]
# file1 = open("input.txt","r")
# lines = file1.readlines() 
# for line in  lines:
#    ii = line.split()
#    ii1.append(int(ii[0]))
#    ii1.append(int(ii[1]))
#    ii1.append(int(ii[2]))
# input_arr = np.array(ii1).reshape(3, 3)

#calling the state class for initializing the start state and call the start function
# print("     BEST FIRST SEARCH   ")
# print('Applying heuristic 1 Misplaced tiles-')
# initialState = HillClimbing(state=input_arr, parent=None,heuristicCost=0)
# HillClimbing.startState = np.copy(input_arr)
# HillClimbing.goalState = np.copy(goalState)
# print("=================================================================")
# initialState.start(input_arr, goalState, "BEST", "h1")
# print('Applying heuristic 2 Manhattan distance-')
# initialState.start(input_arr, goalState, "BEST", "h2")

# input_arr = np.array([2, 8, 3, 1, 6, 4, 7, 0, 5]).reshape(3, 3)
print("******************************************************************")
goalState = np.array([1, 2, 3, 8, 0, 4, 7, 6, 5]).reshape(3, 3)
input_arr = np.array([1, 2, 3,4, 0, 8, 7, 6, 5]).reshape(3, 3)
print("     A* SEARCH   ")
print('Applying heuristic 1 Misplaced tiles-')
initialState3 = State(state=input_arr, parent=None,heuristicCost=0) 
initialState3.start(input_arr,goalState,"a*",1)
print("=================================================================")
print('Applying heuristic 2 Manhattan distance-')
initialState4 = State(state=input_arr, parent=None,heuristicCost=0) 
initialState4.start(input_arr,goalState,"a*",2)

print("******************************************************************")

# print("     HILL CLIMBING SEARCH   ")
# initialState5 = HillClimbing(state=input_arr, parent=None, heuristicCost=0)
# HillClimbing.startState = np.copy(input_arr)
# HillClimbing.goalState = np.copy(goalState)

# initialState5.start(input_arr, goalState, "HILL", "h1")
# print("======================================================\n\n")
# initialState5.start(input_arr, goalState, "HILL", "h2")


"""

print('Applying heuristic 1 Misplaced tiles-')
initialState5 = State(state=input_arr, parent=None,heuristicCost=0) 
initialState5.start(input_arr,goalState,"hc",1)
print("=================================================================")
print('Applying heuristic 2 Manhattan distance-')
initialState6 = State(state=input_arr, parent=None,heuristicCost=0) 
initialState6.start(input_arr,goalState,"hc",2)
"""

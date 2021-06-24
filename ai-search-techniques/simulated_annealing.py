# import functools
# import array
import numpy as np
import time
# import heapq
import os


def coolingFn(T):
    # return 1
    return int(T*0.995)
    # return int(T-1)


def P(curr_hval, next_hval, T):
    # The probability function, SIGMOID
    delE = next_hval - curr_hval
    return 1 / (1 + np.exp(-delE/T))
    # return round(1 / (1 + np.exp(-delE/T)), 2)


def getRandom():
    # get a random value in close interval [0, 1]
    return int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
    # return round(int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1), 2)


def calculateIndicesForGoal():
    # calculatre row, col for tiles in the goal state
    goal_indices = [[0, 0] for x in range(9)]
    for r in range(0, 3):
        for c in range(0, 3):
            goal_indices[goal_state[r][c]] = [r, c]
    return goal_indices


def h1(curr):
    # heuristic1,  counts the number of misplaced tiles WRT goal state
    global heuristic
    heuristic = "h1(n)"

    misplaced = 0
    for r in range(0, 3):
        for c in range(0, 3):
            if curr[r][c] != 0 and curr[r][c] != goal_state[r][c]:
                misplaced += 1
    return misplaced


def h2(curr):
    # heuristic2, calculates the manhatten distance
    global heuristic
    heuristic = "h2(n)"

    man_dis = 0  # manhattan distance

    # get the row, col for all tiles in current state
    ip_indices = [[0, 0] for x in range(9)]
    for r in range(3):
        for c in range(3):
            ip_indices[curr[r][c]] = [r, c]

    # calculate the diffence between tile of current state with goal state
    for i in range(1, 9):
        man_dis += abs(ip_indices[i][0] - goal_indices[i][0]) + \
            abs(ip_indices[i][1] - goal_indices[i][1])

    return man_dis


def h3(curr):
    global heuristic
    res = h1(curr)*h2(curr)
    heuristic = "h3(n)"
    return res


def findPositionOfBlank(curr):
    # find the position of '0' tile for a state
    for r in range(0, 3):
        for c in range(0, 3):
            if curr[r][c] == 0:
                return r, c


def SimulatedAnnealing(curr, hfn, T, coolT):
    start = time.time()
    # initial setup
    global goal_indices, history
    goal_indices = calculateIndicesForGoal()
    history = []

    history.append(curr.flatten())
    row, col = findPositionOfBlank(curr)
    curr_hval = hfn(curr)

    while T != 0:
        count = 0
        before_state = np.copy(curr)
        while count < 10:
            count += 1
            if (curr == goal_state).all():
                return 1, time.time()-start

            nextStates = getNextStates(curr, row, col)

            randomNextStates = list(nextStates.keys())
            np.random.shuffle(randomNextStates)
            countNextMoves = len(randomNextStates)
            choseNextMove = 0

            next = nextStates[randomNextStates[choseNextMove]][0]
            next_hval = hfn(next)

            probability = P(curr_hval, next_hval, T)
            randomNo = getRandom()

            if randomNo <= probability:
                history.append(next.flatten())
                curr = next
                curr_hval = next_hval
                row = nextStates[randomNextStates[choseNextMove]][1]
                col = nextStates[randomNextStates[choseNextMove]][2]
            else:
                choseNextMove += 1
                while choseNextMove < countNextMoves:
                    next = nextStates[randomNextStates[choseNextMove]][0]
                    next_hval = hfn(next)

                    probability = P(curr_hval, next_hval, T)
                    randomNo = getRandom()

                    if randomNo <= probability:
                        history.append(next.flatten())
                        curr = next
                        curr_hval = next_hval
                        row = nextStates[randomNextStates[choseNextMove]][1]
                        col = nextStates[randomNextStates[choseNextMove]][2]
                        break

                    choseNextMove += 1

                if choseNextMove >= countNextMoves:
                    choseNextMove = 0

            if (curr == before_state).all():
                break
        T = coolT(T)
    return 0, time.time()-start


def getNextStates(curr, row, col):
    # returns all possble next state
    next_states = {}

    L, R, U, D = col != 0, col != 2, row != 0, row != 2

    if(L):
        move_left = np.copy(curr)
        move_left[row][col] = move_left[row][col - 1]
        move_left[row][col - 1] = 0

        next_states['left'] = [move_left, row, col - 1]
    if(R):
        dddmove_right = np.copy(curr)
        dddmove_right[row][col] = dddmove_right[row][col + 1]
        dddmove_right[row][col + 1] = 0

        next_states['right'] = [dddmove_right, row, col + 1]
    if(U):
        move_up = np.copy(curr)
        move_up[row][col] = move_up[row - 1][col]
        move_up[row - 1][col] = 0

        next_states['up'] = [move_up, row - 1, col]
    if(D):
        move_down = np.copy(curr)
        move_down[row][col] = move_down[row + 1][col]
        move_down[row + 1][col] = 0

        next_states['down'] = [move_down, row + 1, col]

    return next_states


def getOptimalPath(path):
    arr = np.copy(path)
    n = path.shape[0]
    i = 0

    while i < n:
        j = n-1
        while j > i:
            if (arr[i] == arr[j]).all():
                arr = np.delete(arr, (range(i, j)), axis=0)
                n = arr.shape[0]
                i += 1
                break
            j -= 1
        i += 1
    return arr


# INIT
ii = []
ii1 = []
file1 = open("input.txt", "r")
lines = file1.readlines()
for line in lines:
    ii = line.split()
    ii1.append(int(ii[0]))
    ii1.append(int(ii[1]))
    ii1.append(int(ii[2]))
start_state = np.array(ii1).reshape(3, 3)
file1.close()

history = []
goal_indices = []
heuristic = ""
T = 1000000

# start_state = np.array([2, 8, 3, 1, 6, 4, 7, 0, 5]).reshape(3, 3)
# goal_state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0]).reshape(3, 3)
goal_state = np.array([1, 2, 3, 8, 0, 4, 7, 6, 5]).reshape(3, 3)
result1, time_taken1 = SimulatedAnnealing(start_state, h1, T, coolingFn)
optimal_path1 = getOptimalPath(np.array(history))

# Writing output
file_name = 'output_simulated_annealing.txt'
f = open(os.path.join("./", file_name), 'w')

if result1:
    f.write("Result: Success\n")
else:
    f.write("Result: Failure\n")

f.write("Heuristic: {}\n".format(heuristic))
f.write("Temperature: {}\n".format(T))
f.write("Cooling function: T = T*0.995\n\n")
f.write("Start state: \n{}\n".format(start_state))
f.write("Goal state: \n{}\n\n".format(goal_state))

if result1:
    f.write("(Sub)optimal path:\n")
    for x in optimal_path1:
        f.write("{}\n\n".format(x.reshape(3, 3)))

f.write("Total states explored: {}\n".format(
    len(np.unique(history, axis=0))))
f.write("Time taken: {0:.3f}s\n".format(time_taken1))

f.write("\n====================================================================\n\n")

result2, time_taken2 = SimulatedAnnealing(start_state, h2, T, coolingFn)
optimal_path2 = getOptimalPath(np.array(history))

# Writing output

if result2:
    f.write("Result: Success\n")
else:
    f.write("Result: Failure\n")

f.write("Heuristic: {}\n".format(heuristic))
f.write("Temperature: {}\n".format(T))
f.write("Cooling function: T = T*0.995\n\n")
f.write("Start state: \n{}\n".format(start_state))
f.write("Goal state: \n{}\n\n".format(goal_state))

if result2:
    f.write("(Sub)optimal path:\n")
    for x in optimal_path2:
        f.write("{}\n\n".format(x.reshape(3, 3)))

f.write("Total states explored: {}\n".format(
    len(np.unique(history, axis=0))))
f.write("Time taken: {0:.3f}s\n".format(time_taken2))

# f.write("\n====================================================================\n\n")

# result3, time_taken3 = SimulatedAnnealing(start_state, h3, T, coolingFn)
# optimal_path3 = getOptimalPath(np.array(history))

# # Writing output

# if result3:
#     f.write("Result: Success\n")
# else:
#     f.write("Result: Failure\n")

# f.write("Heuristic: {}\n".format(heuristic))
# f.write("Temperature: {}\n".format(T))
# f.write("Cooling function: T = T*0.995\n\n")
# f.write("Start state: \n{}\n".format(start_state))
# f.write("Goal state: \n{}\n\n".format(goal_state))

# if result3:
#     f.write("(Sub)optimal path:\n")
#     for x in optimal_path3:
#         f.write("{}\n\n".format(x.reshape(3, 3)))

# f.write("Total states explored: {}\n".format(
#     len(np.unique(history, axis=0))))
# f.write("Time taken: {0:.3f}s\n".format(time_taken3))

f.close()

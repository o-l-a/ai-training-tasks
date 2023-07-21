import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
from queue import PriorityQueue
from itertools import count


def canBeASolution(state):
    """
    function to check if a given state might lead to a solution of the n-Queens problem
    """
    # ------eliminate nans------#
    state_copy = np.copy(state[~np.isnan(state)])
    # ------check if every number unique------#
    if len(np.unique(state_copy)) == len(state_copy):
        for i in range(len(state_copy)):
            for j in range(i + 1, len(state_copy)):
                if np.abs(j - i) == np.abs(state_copy[j] - state_copy[i]):
                    return False
        return True
    return False


def manhattan(x1, x2, y1, y2):
    d = np.abs(x1 - x2) + np.abs(y1 - y2)
    return d


class NQueens:
    def __init__(self, n):
        self.solution = None
        if n < 4:
            n = 4
        self.__n = n
        self.__generatedStatesCount = 0
        self.__examinedStatesCount = 0
        self.__searchTime = 0
        self.__solved = False
        # ------a variable to ensure no errors using priority queue------#
        self.__tiebreaker = count()

    @property
    def n(self):
        return self.__n

    @property
    def generatedStatesCount(self):
        return self.__generatedStatesCount

    @property
    def examinedStatesCount(self):
        return self.__examinedStatesCount

    @property
    def searchTime(self):
        return self.__searchTime

    def generateChildren(self, state, isSmart=False):
        """
        function to generate children of given state of size n in the n-Queens problem
        """
        # ------find first free slot------#
        N = self.n - np.sum(np.isnan(state))
        # ------empty queue------#
        listOfChildren = deque()
        # ------if state full do nothing------#
        if N == self.n:
            return listOfChildren
        # ------add a child for every n------#
        for i in range(self.n):
            newChild = np.copy(state)
            newChild[N] = i
            if isSmart:
                if canBeASolution(newChild):
                    listOfChildren.append(newChild)
                else:
                    continue
            else:
                listOfChildren.append(newChild)
            self.__generatedStatesCount += 1
        return listOfChildren

    def isASolution(self, state):
        """
        function to check if a given state of size n is a solution of the n-Queens problem
        """
        self.__examinedStatesCount += 1
        # ------check for nans------#
        if not np.sum(np.isnan(state)):
            # ------check if every number unique------#
            if len(np.unique(state)) == self.n:
                # ------check diagonally------#
                for i in range(len(state)):
                    for j in range(i + 1, len(state)):
                        if np.abs(j - i) == np.abs(state[j] - state[i]):
                            return False
                return True
        return False

    def BFS(self, isSmart=False):
        """
        function to determine first found solution of the n-Queens problem breadth-first search-wise
        """
        startTime = time.time()
        # ------initialize vector------#
        initialState = np.empty((1, self.n)).ravel()
        initialState[:] = None
        self.__generatedStatesCount += 1
        # ------initialize queue------#
        listOfStates = deque()
        # ------add initial state------#
        listOfStates.append(initialState)
        while listOfStates:
            currentState = listOfStates.popleft()
            # ------check if a solution------#
            if self.isASolution(currentState):
                self.solution = currentState
                self.__solved = True
                self.__searchTime = time.time() - startTime
                return
            else:
                # ------add children to queue------#
                listOfChildren = self.generateChildren(currentState, isSmart=isSmart)
                listOfStates.extend(listOfChildren)

    def DFS(self, isSmart=False):
        """
        function to determine first found solution of the n-Queens problem depth-first search-wise
        """
        startTime = time.time()
        # ------initialize vector------#
        initialState = np.empty((1, self.n)).ravel()
        initialState[:] = None
        self.__generatedStatesCount += 1
        # ------initialize queue------#
        listOfStates = deque()
        # ------add initial state------#
        listOfStates.append(initialState)
        while listOfStates:
            currentState = listOfStates.pop()
            # ------check if a solution------#
            if self.isASolution(currentState):
                self.solution = currentState
                self.__solved = True
                self.__searchTime = time.time() - startTime
                return
            else:
                # ------add children to queue------#
                listOfChildren = self.generateChildren(currentState, isSmart=isSmart)
                if listOfChildren:
                    listOfStates.extend(listOfChildren)

    def h1(self, state):
        """
        discourages against outer rows by applying greater weights to them
        """
        i = self.n - np.sum(np.isnan(state))  # how many queens already inserted
        weights = np.copy(state) + 1
        for j in range(len(weights)):
            if weights[j] <= (float(self.n) / 2):
                weights[j] = self.n + 1 - weights[j]
        h = (self.n - i) * np.nansum(weights)
        return h

    def h2(self, state):
        """
        goes after the most free fields posiible
        """
        N = self.n - np.sum(np.isnan(state))  # how many queens already inserted
        fields = np.ones((self.n, self.n))
        for i in range(N):
            fields[:, i] = 0  # it works because we only put a new queen at the end of the vector
            fields[int(state[i]), :] = 0
        for i in range(N):
            for j in range(self.n):
                for k in range(self.n):
                    if np.abs(state[i] - j) == np.abs(i - k):
                        fields[j, k] = 0
        h = np.sum(fields)
        return h

    def h3(self, state):
        """
        intends to maximize sum of hamming distances between every pair of queens
        """
        N = self.n - np.sum(np.isnan(state))  # how many queens already inserted
        d = 0  # variable for Hamming distance
        S = self.n / 2 * (self.n - 1)  # maximum Hamming distance
        for i in range(N - 1):
            for j in range(i + 1, N):
                if state[i] != state[j]:
                    d += 1
        h = S - d
        return h

    def h4(self, state):
        """
        wants Manhattan distances between pairs of queens to aim for 3 (no less)
        """
        optimalDistance = 3
        N = self.n - np.sum(np.isnan(state))  # how many queens already inserted
        h = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                manhattanDistance = manhattan(state[i], state[j], i, j)
                if manhattanDistance < optimalDistance:
                    h += np.inf
                else:
                    h += (manhattanDistance - optimalDistance)
        return h

    def BestFS(self, h=1):
        """
        function to determine first found solution of the n-Queens problem best-first search-wise
        """
        startTime = time.time()
        # ------initialize vector------#
        initialState = np.empty((1, self.n)).ravel()
        initialState[:] = None
        self.__generatedStatesCount += 1
        # ------choose the heuristic------#
        if h < 1 or h > 4:
            h = 1
        h = [self.h1, self.h2, self.h3, self.h4][h - 1]
        # ------initialize queue------#
        listOfStates = PriorityQueue()
        # ------add initial state------#
        listOfStates.put((h(initialState), next(self.__tiebreaker), initialState))
        while listOfStates:
            currentState = listOfStates.get()[2]
            # ------check if a solution------#
            if self.isASolution(currentState):
                self.solution = currentState
                self.__solved = True
                self.__searchTime = time.time() - startTime
                return
            else:
                # ------add children to queue------#
                listOfChildren = self.generateChildren(currentState)
                while listOfChildren:
                    newChild = listOfChildren.pop()
                    listOfStates.put((h(newChild), next(self.__tiebreaker), newChild))

    def printNQueens(self):
        """
        purely decorative function to print a solution of the n-Queens problem
        """
        if not self.__solved:
            print("No solution yet")
            return
        arr = np.zeros((self.n, self.n), dtype=int)
        for i in range(len(self.solution)):
            arr[int(self.solution[i]), i] = 1
        for i in range(self.n):
            if i == 0:
                for j in range(self.n):
                    if j == 0:
                        print("┌───", end='')
                    else:
                        print("┬───", end='')
                print("┐")
            else:
                for j in range(self.n):
                    if j == 0:
                        print("├───", end='')
                    else:
                        print("┼───", end='')
                print("┤")
            for j in range(self.n):
                character = arr[i][j]
                if character == 0:
                    character = " "
                else:
                    character = "Q"
                print("│", character, "", end='')
            print("│")
            if i == self.n - 1:
                for j in range(self.n):
                    if j == 0:
                        print("└───", end='')
                    else:
                        print("┴───", end='')
                print("┘")


def plotNQueensStatistics(x, st1, st2, st3, version):
    """
    function to plot the statistics of the n-Queens problem 
    """
    # plt.figure(figsize=(12, 3.5))
    plt.subplot(1, 3, 1)
    plt.plot(x, st1, '--', label=version)
    plt.xticks(x)
    plt.xlabel("n")
    plt.ylabel("generated states")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(x, st2, '--', label=version)
    plt.xticks(x)
    plt.xlabel("n")
    plt.ylabel("examined states")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(x, st3, '--', label=version)
    plt.xticks(x)
    plt.xlabel("n")
    plt.ylabel("search times [s]")
    plt.legend()
    plt.tight_layout()


def allHailTheQueens(nMin, nMax, isSmart=False, version="BFS", heuristic=1):
    """
    function to compare statistics of n-Queens solutions with n ranging from nMin to nMax
    """
    xAxis = np.arange(nMin, nMax + 1, dtype=int)
    generatedStates = np.zeros(xAxis.shape, dtype=int)
    examinedStates = np.zeros(xAxis.shape, dtype=int)
    searchTimes = np.zeros(xAxis.shape)
    counter = 0
    for i in range(nMin, nMax + 1):
        iQueens = NQueens(i)
        if version == "BFS":
            iQueens.BFS(isSmart=isSmart)
        elif version == "DFS":
            iQueens.DFS(isSmart=isSmart)
        elif version == "BestFS":
            iQueens.BestFS(heuristic)
        else:
            print("This version has not been implementeed")
            return
        iQueens.printNQueens()
        generatedStates[counter] = iQueens.generatedStatesCount
        examinedStates[counter] = iQueens.examinedStatesCount
        searchTimes[counter] = iQueens.searchTime
        counter += 1
    print('# of generated states for every n:', generatedStates)
    print('# of examined states for every n:', examinedStates)
    print('search times for every n:', searchTimes)
    if version == "BestFS":
        version = version + " h" + str(heuristic)
    return xAxis, generatedStates, examinedStates, searchTimes, version


def plotComparison(results):
    plt.figure(figsize=(13, 4))
    for result in results:
        print(result)
        plotNQueensStatistics(*result)


smart = True

nStart = 4
nStop = 6
bfs = allHailTheQueens(nStart, nStop, version="BFS")
dfs = allHailTheQueens(nStart, nStop, version="DFS")
bestfs1 = allHailTheQueens(nStart, nStop, version="BestFS", heuristic=1)
bestfs2 = allHailTheQueens(nStart, nStop, version="BestFS", heuristic=2)
bestfs3 = allHailTheQueens(nStart, nStop, version="BestFS", heuristic=3)
bestfs4 = allHailTheQueens(nStart, nStop, version="BestFS", heuristic=4)

resultsCombined = [bfs, dfs, bestfs1, bestfs2, bestfs3, bestfs4]

plotComparison(resultsCombined)
plt.show()

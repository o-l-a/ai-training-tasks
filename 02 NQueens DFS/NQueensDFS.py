import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt


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

    def generateChildren(self, state, smart=False):
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
            if smart:
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

    def BFS(self, smart=False):
        """
        function to determine first found solution of the n-Queens problem
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
                listOfChildren = self.generateChildren(currentState, smart=smart)
                listOfStates.extend(listOfChildren)

    def DFS(self, smart=False):
        """
        function to determine first found solution of the n-Queens problem
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
                listOfChildren = self.generateChildren(currentState, smart=smart)
                if listOfChildren:
                    listOfStates.extend(listOfChildren)

    def printNQueens(self):
        """
        purely decorative function to print a solution of the n-Queens problem
        """
        if not self.__solved:
            print("No solution yet")
            return
        arr = np.zeros((self.n, self.n), dtype=int)
        for i in range(len(self.solution)):
            arr[i, int(self.solution[i])] = 1
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


def nQueensTask(nMin, nMax, smart=False, version="BFS"):
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
            iQueens.BFS(smart=smart)
        elif version == "DFS":
            iQueens.DFS(smart=smart)
        else:
            print("This version has not been implemented.")
            return
        iQueens.printNQueens()
        generatedStates[counter] = iQueens.generatedStatesCount
        examinedStates[counter] = iQueens.examinedStatesCount
        searchTimes[counter] = iQueens.searchTime
        counter += 1
    print('# of generated states for every n:', generatedStates)
    print('# of examined states for every n:', examinedStates)
    print('search times for every n:', searchTimes)
    return xAxis, generatedStates, examinedStates, searchTimes, version


def plotComparison(results1, results2):
    plt.figure(figsize=(13, 4))
    plotNQueensStatistics(*results1)
    plotNQueensStatistics(*results2)


r1 = nQueensTask(4, 8, smart=True, version="DFS")
r2 = nQueensTask(4, 8, smart=True, version="BFS")
plotComparison(r1, r2)
r3 = nQueensTask(4, 8, smart=True, version="DFS")

plt.figure(figsize=(13, 4))
plotNQueensStatistics(*r3)
plt.show()

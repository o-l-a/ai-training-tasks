import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt


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

    def generateChildren(self, state):
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

    def BFS(self):
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
                listOfChildren = self.generateChildren(currentState)
                while listOfChildren:
                    listOfStates.append(listOfChildren.popleft())

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


def plotNQueensStatistics(x, st1, st2, st3):
    plt.figure(figsize=(12, 3.5))
    plt.subplot(1, 3, 1)
    plt.plot(x, st1, 'r--')
    plt.xticks(x)
    plt.xlabel("n")
    plt.ylabel("generated states")
    plt.subplot(1, 3, 2)
    plt.plot(x, st2, 'g--')
    plt.xticks(x)
    plt.xlabel("n")
    plt.ylabel("examined states")
    plt.subplot(1, 3, 3)
    plt.plot(x, st3, 'b--')
    plt.xticks(x)
    plt.xlabel("n")
    plt.ylabel("search times [s]")
    plt.tight_layout()
    plt.show()


def nQueensTask(nMin, nMax):
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
        iQueens.BFS()
        iQueens.printNQueens()
        generatedStates[counter] = iQueens.generatedStatesCount
        examinedStates[counter] = iQueens.examinedStatesCount
        searchTimes[counter] = iQueens.searchTime
        counter += 1
    plotNQueensStatistics(xAxis, generatedStates, examinedStates, searchTimes)
    print('# of generated states for every n:', generatedStates)
    print('# of examined states for every n:', examinedStates)
    print('search times for every n:', searchTimes)


nQueensTask(4, 6)

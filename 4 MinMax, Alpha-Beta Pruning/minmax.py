import numpy as np
from itertools import count


class minMaxNode(object):
    _ids = count(0)

    def __init__(self, state):
        self.state = state
        self.children = []
        self.id = next(self._ids)
        self.parent = None
        self.v = None
        # ------who we want to win------#
        self.favorite = 'o'
        # ------who played first------#
        self.firstPlayer = 'o'
        # ------who played second------#
        self.secondPlayer = 'x'

    def fillWithChildren(self):
        # ------check if not a final state------#
        if self.isAFinalState():
            return
        # ------determine who plays next------#
        if np.count_nonzero(self.state == 'x') == np.count_nonzero(self.state == 'o'):
            nextPlayer = self.firstPlayer
        else:
            nextPlayer = self.secondPlayer
        # ------determine empty fields------#
        emptyFields = np.argwhere(self.state == 'N')
        # ------generate a child per every empty field------#
        for field in emptyFields:
            newChild = np.copy(self.state)
            newChild[field[0], field[1]] = nextPlayer
            newChild = minMaxNode(newChild)
            newChild.parent = self
            self.children.append(newChild)
        # ------fill every child with children------#
        for child in self.children:
            child.fillWithChildren()

    def printBranch(self, depth=0):
        for i in range(depth):
            if depth != 0:
                if i == depth - 1:
                    print('└──────────────', end='')
                else:
                    print('               ', end='')
        print('(#', self.id, ', v=', self.v, ')', sep='')
        for i in range(len(self.children)):
            self.children[i].printBranch(depth + 1)

    def printTree(self):
        currentNode = self
        while currentNode.parent:
            currentNode = currentNode.parent
        currentNode.printBranch()

    def isAFinalState(self):
        # ------check if 3 same in rows------#
        for row in range(self.state.shape[0]):
            if np.all(self.state[row, :] == self.state[row, :][0]) and self.state[row, :][0] != 'N':
                return 1
        # ------check if 3 same in columns------#
        for col in range(self.state.shape[1]):
            if np.all(self.state[:, col] == self.state[:, col][0]) and self.state[:, col][0] != 'N':
                return 1
        # ------check if 3 same on diagonal------#
        if np.all(self.state.diagonal() == self.state[0, 0]) and self.state[0, 0] != 'N':
            return 1
        # ------check if 3 same on reverse diagonal------#
        if np.all(np.diag(np.fliplr(self.state)) == self.state[0, 2]) and self.state[0, 2] != 'N':
            return 1
        # ------check if board full------#
        chars = np.unique(self.state)
        if len(chars) == 2 and 'N' not in chars:
            return 1
        return 0

    def value(self):
        # ------check if 3 same in rows------#
        for row in range(self.state.shape[0]):
            if np.all(self.state[row, :] == self.state[row, :][0]):
                if self.state[row, :][0] == self.favorite:
                    return 1
                else:
                    return -1
        # ------check if 3 same in columns------#
        for col in range(self.state.shape[1]):
            if np.all(self.state[:, col] == self.state[:, col][0]):
                if self.state[:, col][0] == self.favorite:
                    return 1
                else:
                    return -1
        # ------check if 3 same on diagonal------#
        if np.all(self.state.diagonal() == self.state[0, 0]):
            if self.state[0, 0] == self.favorite:
                return 1
            else:
                return -1
        # ------check if 3 same on reverse diagonal------#
        if np.all(np.diag(np.fliplr(self.state)) == self.state[0, 2]):
            if self.state[0, 2] == self.favorite:
                return 1
            else:
                return -1
        return 0

    def maxValue(self):
        if self.isAFinalState():
            self.v = self.value()
        else:
            self.v = -np.Inf
            for child in self.children:
                self.v = int(np.max([self.v, child.minValue()]))
        print('-' * 60, '\nMAX')
        # print('#', self.id, sep='')
        print(self.state, '\n')
        self.printBranch()
        return self.v

    def minValue(self):
        if self.isAFinalState():
            self.v = self.value()
        else:
            self.v = np.Inf
            for child in self.children:
                self.v = int(np.min([self.v, child.maxValue()]))
        print('-' * 60, '\nMIN')
        # print('#', self.id, sep='')
        print(self.state, '\n')
        self.printBranch()
        return self.v

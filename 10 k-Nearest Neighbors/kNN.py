import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import neighbors
from sklearn import datasets
import pandas as pd
import time


def myPCA(M, nd):
    means = np.mean(M, axis=0)
    M1 = np.apply_along_axis(lambda row: row - means, axis=1, arr=M)
    Mcov = np.cov(M1.T)
    w, v = np.linalg.eig(Mcov)
    max_abs_cols = np.argmax(np.abs(v), axis=0)
    signs = np.sign(v[max_abs_cols, range(v.shape[1])])
    v *= signs
    ind = np.argsort(w)
    w = w[ind][::-1]
    v = np.flip(v[:, ind], axis=1)
    F = v[:, :nd]
    MZPC = np.dot(M1, F)
    return MZPC, v, w, means


def reversePCA(scores, eigv, mean):
    return scores.dot(eigv.T) + mean


class knn:
    def __init__(self, n_neighbors=1, use_KDTree=False, regress=False):
        self.classes = None
        self.values = None
        self.objects = None
        self.n = n_neighbors
        self.KDT = use_KDTree
        self.regress = regress

    def fit(self, X, y):
        if self.KDT:
            self.objects = sk.neighbors.KDTree(X, len(np.unique(y)))
        else:
            self.objects = X
        if self.regress:
            self.values = y
        else:
            self.classes = y

    def predict(self, X):
        if self.regress:
            if self.KDT:
                dist, ind = self.objects.query(np.array(X).reshape(1, -1), k=self.n)
                c = self.values[ind]
            else:
                ans = np.apply_along_axis(lambda row: np.linalg.norm(row - X), 1, self.objects)
                mySorted = np.argsort(ans)
                c = self.values[mySorted][:self.n]
            return np.mean(c)
        if self.KDT:
            _, ind = self.objects.query(np.array(X).reshape(1, -1), k=self.n)
            c = self.classes[ind]
        else:
            ans = np.apply_along_axis(lambda row: np.linalg.norm(row - X), 1, self.objects)
            mySorted = np.argsort(ans)
            c = self.classes[mySorted][:self.n]
        ui, ni = np.unique(c, return_counts=True)
        choice = ui[np.argsort(ni)[::-1]][0]
        return choice

    def score(self, y):
        if self.regress:
            err = sk.metrics.mean_squared_error(self.values, y)
            return err
        else:
            l = np.sum(self.classes != y)
            L = len(self.classes)
            return (L - l) / L


# -------------------------------------------------------------------------------
# RANDOM DATA

X1, y1 = datasets.make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=3
)

xmin = np.min(X1[:, 0])
xmax = np.max(X1[:, 0])
ymin = np.min(X1[:, 1])
ymax = np.max(X1[:, 1])

myModel = knn(use_KDTree=True)
myModel.fit(X1, y1)
myModel.predict([1, 2])

sx, sy = np.meshgrid(np.arange(xmin, xmax, 0.05), np.arange(ymin, ymax, 0.05))
a = sx.shape
helper = np.array([sx.ravel(), sy.ravel()]).T
z = np.apply_along_axis(lambda row: myModel.predict(row), 1, helper).reshape(a)

for i in np.unique(y1):
    plt.scatter(X1[:, 0][y1 == i], X1[:, 1][y1 == i], alpha=0.4, s=12, label=i)
plt.legend()

plt.contour(sx, sy, z, cmap='coolwarm')
plt.title('Visualization of data and separation boundaries')

# -----------------------------------------------------------------------------------
# IRIS
iris = datasets.load_iris()
x, y1 = iris.data, iris.target
iris_prim, myVectors, myValues, myMeans = myPCA(x, 2)
ax2 = plt.figure()
for i in np.unique(y1):
    plt.scatter(iris_prim[:, 0][y1 == i], iris_prim[:, 1][y1 == i], alpha=0.3, s=12)
plt.legend(iris.target_names)
plt.title('Visualization of data and separation boundaries')

myModel1 = knn(use_KDTree=True)
myModel1.fit(x, y1)

xmin = np.min(iris_prim[:, 0])
xmax = np.max(iris_prim[:, 0])
ymin = np.min(iris_prim[:, 1])
ymax = np.max(iris_prim[:, 1])
sx, sy = np.meshgrid(np.arange(xmin, xmax, 0.05), np.arange(ymin, ymax, 0.05))
a = sx.shape
helper = np.array([sx.ravel(), sy.ravel()]).T
myGrid = reversePCA(helper, myVectors[:, :2], myMeans)
z = np.apply_along_axis(lambda row: myModel1.predict(row), 1, myGrid)
plt.contour(sx, sy, z.reshape(a))

# leave one out
df = pd.DataFrame(range(x.shape[0]), columns=['one'])
mylist = [i for i in range(1, x.shape[0])]
DF = pd.DataFrame(mylist, columns=['k'])
DF['accuracy[%]'] = [0.0 for i in range(len(mylist))]

start = time.time()
for n in range(len(DF['k'])):
    y2 = np.zeros(x.shape[0])
    for i in df['one']:
        iris2 = knn(DF['k'][n], use_KDTree=True)
        iris2.fit(np.delete(np.copy(x), i, axis=0), np.delete(np.copy(y1), i, axis=0))
        y2[i] = iris2.predict(x[i, :])
    DF.loc[n, 'accuracy[%]'] = myModel1.score(y2) * 100
stop = time.time()
print('KD Tree ON:\ttime=', stop - start, sep='')

plt.figure()
plt.plot(DF['k'], DF['accuracy[%]'])
plt.xlabel('k')
plt.ylabel('accuracy[%]')

X1, y1 = datasets.make_regression(
    n_samples=100,
    n_features=1,
    n_informative=1,
    random_state=3
)

k1 = 2
xd = knn(k1, use_KDTree=True, regress=True)
xd.fit(X1, y1)

xtrain = X1
myInd = np.argsort(X1, axis=0)
ytrain = np.apply_along_axis(lambda row: xd.predict(row), 1, xtrain)

plt.figure()
plt.scatter(X1, y1, alpha=0.3, s=30, label='points')
plt.plot(xtrain[myInd].ravel(), ytrain[myInd].ravel(), 'C1', label='trend')
plt.title('k=%d' % k1)
plt.legend()
print(xd.score(ytrain))

boston = datasets.load_boston()
x, y1 = boston.data, boston.target
tmp = np.arange(x.shape[0])
ranges = np.array_split(tmp, 10)

boston = knn(use_KDTree=True, regress=True)
boston.fit(x, y1)

mylist = [i for i in range(1, 455)]
DF = pd.DataFrame(mylist, columns=['k'])
DF['MSE'] = [0.0 for i in range(len(mylist))]

for n in range(len(DF['k'])):
    y2 = np.zeros(x.shape[0])
    for i in ranges:
        b2 = knn(DF['k'][n], use_KDTree=True, regress=True)
        b2.fit(np.delete(np.copy(x), i, axis=0), np.delete(np.copy(y1), i, axis=0))
        for j in i:
            y2[j] = b2.predict(x[j, :])
    DF.loc[n, 'MSE'] = boston.score(y2)

print(DF)
plt.figure()
plt.plot(DF['k'], DF['MSE'])
plt.xlabel('k')
plt.ylabel('MSE')
plt.show()

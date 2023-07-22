import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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


test = np.dot(np.random.randn(200, 2), np.random.randn(2, 2))
m = np.min(test[:, 0])
myM = np.max(test[:, 0])

plt.scatter(test[:, 0], test[:, 1], alpha=0.3)
testPrim, myVectors, myValues, myMeans = myPCA(test, 1)
w1 = myVectors[:, 0]
w2 = myVectors[:, 1]
ax = plt.gca()
arrows = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
ax.annotate('', myMeans + w1 * np.sqrt(myValues[0]), myMeans, arrowprops=arrows)
ax.annotate('', myMeans + w2 * np.sqrt(myValues[1]), myMeans, arrowprops=arrows)
plt.axis('equal')

heights = np.zeros((len(testPrim), 1))
theta = np.arctan2(w1[1], w1[0])
P = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
newdata = np.hstack((testPrim, heights))
newdata = np.dot(P, newdata.T)
plt.scatter(newdata[0] + myMeans[0], newdata[1] + myMeans[1], alpha=0.3)
plt.title('Visualization of feature space and eigenvectors')

pca = PCA(n_components=1)
pca.fit(test).transform(test)

# -------------------------------------------------------------------------------------------
iris = datasets.load_iris()
x, y = iris.data, iris.target
irisPrim, myVectors, myValues, myMeans = myPCA(x, 2)
ax2 = plt.figure()
for i in np.unique(y):
    plt.scatter(irisPrim[:, 0][y == i], irisPrim[:, 1][y == i], alpha=0.3)
plt.legend(iris.target_names)
plt.title('Iris dataset visualization')

pca = PCA(n_components=2)
pca.fit(x).transform(x)

# -------------------------------------------------------------------------------------------
digits = datasets.load_digits()
x, y = digits.data, digits.target
digitsPrim, myVectors, myValues, myMeans = myPCA(x, 2)

plt.figure()
plt.plot(np.arange(len(myValues)), np.cumsum(myValues) / np.sum(myValues))
plt.title('Visualization of Principal Components Variance')
plt.xlabel('component number')
plt.ylabel('cumulative variance [%]')

ax2 = plt.figure()
for i in np.unique(y):
    plt.scatter(digitsPrim[:, 0][y == i], digitsPrim[:, 1][y == i], alpha=0.3)
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.title('Digits dataset visualization')
plt.tight_layout()
ndim = 2

fig = plt.figure()
fig.suptitle('original data', fontsize="x-large")
for j in range(len(digits.target_names)):
    plt.subplot(2, 5, j + 1)
    myDigit = x[y == j][21, :].reshape((8, 8))
    plt.imshow(myDigit, cmap='cividis')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()

myComponents = [2, 4, 10, 50]
for ndim in myComponents:
    digitsPrim, myVectors, myValues, myMeans = myPCA(x, ndim)
    myReversed = reversePCA(digitsPrim, myVectors[:, :ndim], myMeans)
    fig = plt.figure()
    fig.suptitle('%d components' % ndim, fontsize="x-large")
    for j in range(len(digits.target_names)):
        plt.subplot(2, 5, j + 1)
        myDigit = myReversed[y == j][21, :].reshape((8, 8))
        plt.imshow(myDigit, cmap='cividis')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()

ERR = np.zeros(x.shape[1])
for i in range(x.shape[1]):
    digitsPrim, myVectors, myValues, myMeans = myPCA(x, i)
    myReversed = reversePCA(digitsPrim, myVectors[:, :i], myMeans)
    ERR[i] = np.linalg.norm(x - myReversed)

plt.figure()
plt.plot(np.arange(x.shape[1]), ERR)
plt.xlim([0, 64])
plt.ylim([0, 1500])
plt.title('Difference between original and reconstructed objects')
plt.xlabel('number of components')
plt.ylabel('distance')
plt.show()

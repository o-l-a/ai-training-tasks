import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def euclideanDistance(A, C):
    return np.sqrt(np.add.outer(np.sum(A * A, axis=1), np.sum(C * C, axis=1)) - 2 * A.dot(C.T))


def mahalanobisDistance(A, C):
    Mah = (-1) * np.ones((A.shape[0], C.shape[0]))
    for i in range(C.shape[0]):
        roznica = A - C[i, :]
        icov = np.linalg.matrix_power(np.cov(A, rowvar=False), -1)
        Mah[:, i] = (np.diag(np.sqrt(roznica.dot(icov).dot(roznica.T))))
    return Mah


def calculateAccuracy(X, C):
    numerator, denominator = 0, 0
    for l in range(C.shape[0]):
        for k in range(C.shape[0]):
            if not (k == l):
                numerator += np.linalg.norm(C[l, :] - C[k, :])
    for k in range(C.shape[0]):
        denominator += np.linalg.norm(X - C[k, :])
    return numerator / (denominator ** 2)


def kmeans(X, k, metrics=0, example=False):
    classes = np.arange(k)
    C = X.take(np.random.choice(X.shape[0], k, replace=False), axis=0)
    if example:
        C = X[[2 * i for i in range(k)], :]
    Cold = np.zeros(C.shape)
    CX = (-1) * np.ones((X.shape[0], 1), dtype=int)
    CXold = np.zeros(CX.shape)
    while not (np.allclose(Cold, C) and np.allclose(CXold, CX)):
        Cold = np.copy(C)
        CXold = np.copy(CX)
        if metrics == 0:
            d = euclideanDistance(X, C)
        else:
            d = mahalanobisDistance(X, C)
        CX = np.argmin(d, axis=1)
        for i in classes:
            C[i] = np.mean(X[CX == i], axis=0)
    return C, CX


autos = pd.read_csv('autos.csv')
test = np.array(autos[['width', 'height']])
C0, CX0 = kmeans(test, 3, example=True)
C1, CX1 = kmeans(test, 3, 1, example=True)
print(test)
plt.subplot(1, 2, 1)
plt.plot(C0[:, 0], C0[:, 1], 'k.')
for i in np.unique(CX0):
    plt.scatter(test[CX0 == i][:, 0], test[CX0 == i][:, 1], alpha=0.3)
plt.title('Euclidean distance')
print('F(C0)=', calculateAccuracy(test, C0))

plt.subplot(1, 2, 2)
plt.plot(C1[:, 0], C1[:, 1], 'k.')
for i in np.unique(CX1):
    plt.scatter(test[CX1 == i][:, 0], test[CX1 == i][:, 1], alpha=0.3)
plt.title('Mahalanobis distance')
print('F(C1)=', calculateAccuracy(test, C1))
plt.show()

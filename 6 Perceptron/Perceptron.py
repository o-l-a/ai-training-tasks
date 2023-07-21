import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True


class Perceptron:
    def __init__(self, eta=0.5, k_max=20000):
        self.w = None
        self.y_pred = None
        self.y = None
        self.X = None
        self.eta = eta
        self.k_max = k_max

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.y[self.y > 0] = 1
        self.y[self.y <= 0] = -1
        self.y_pred = np.zeros(y.shape)
        self.w = np.zeros(X.shape[1] + 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        k = 0
        while True:
            self.y_pred = np.sum(X * self.w, axis=1)
            self.y_pred[self.y_pred > 0] = 1
            self.y_pred[self.y_pred <= 0] = -1
            E = X[self.y_pred != self.y]
            y_e = self.y[self.y_pred != self.y]
            if E.shape[0] == 0:
                break
            _id = np.random.randint(0, E.shape[0])
            self.w = self.w + self.eta * y_e[_id] * E[_id]
            k = k + 1
            if k > self.k_max:
                break
        return self.w, k


def draw_classes(X, y, title, line=False, w=None, ylim=None, xlim=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
    if line:
        plt.axline((0, f2d(0, w)), (1, f2d(1, w)))
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.title(title)


def f2d(x, w):
    """
    Compute the output of a 2D linear classifier.
    """
    return -x * w[1] / w[2] - w[0] / w[2]


def generate_dataset(m):
    """
    Generates a separable dataset.
    """
    separable = False
    while not separable:
        X, y = datasets.make_classification(n_samples=m, n_features=2, n_redundant=0, n_clusters_per_class=1,
                                            class_sep=1)
        red = X[y == 0]
        blue = X[y == 1]
        separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
    return X, y


def task2():
    m = np.arange(10, 1000, 5)
    eta = np.arange(0.001, 1, 0.005)
    K = np.zeros((len(m), len(eta)))

    for i in range(len(m)):
        X, y = generate_dataset(m[i])
        for j in range(len(eta)):
            p = Perceptron(eta=eta[j])
            w, k = p.fit(X, y)
            K[i, j] = k

    M, Eta = np.meshgrid(m, eta, indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    top = K.ravel()
    M, Eta = M.ravel(), Eta.ravel()
    bottom = np.zeros_like(top)
    width = 5
    depth = 0.005
    ax.bar3d(M, Eta, bottom, width, depth, top, shade=True)
    plt.xlabel(r'$m$')
    plt.ylabel(r'$\eta$')
    ax.set_zlabel(r'$k$')

    plt.figure()
    plt.subplot(1, 2, 1)
    draw_classes(X, y, 'original')
    ylim = plt.gca().get_ylim()
    xlim = plt.gca().get_xlim()
    plt.subplot(1, 2, 2)
    draw_classes(X, p.y_pred, 'k=' + str(k), line=True, w=w, ylim=ylim, xlim=xlim)

    plt.show()


task2()

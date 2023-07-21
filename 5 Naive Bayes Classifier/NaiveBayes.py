import numpy as np
from sklearn import preprocessing, model_selection, datasets
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


class NBC(BaseEstimator, ClassifierMixin):
    """
Discrete mode:
    self._frequency_tables         key: (class_number, feature_number), value: array of size of number of feature values
Non-discrete mode:
    self._values                   key: (class_number, feature_number), value: set of values of feature f when class==c
    self._means                    key: as former, value: means of entries in self._values
    self._stds                     key: as former, value: standard deviation of entries in self._values
    """

    def __init__(self, n_classes=None, discrete=False, LaPlace=True, log=True):
        self._stds = None
        self._means = None
        self._values = None
        self._frequency_tables = None
        self._classes_count = None
        self._n_features = None
        self._n_samples = None
        self._y = None
        self._X = None
        self._discrete = discrete
        self._n_classes = n_classes
        self._laplace = LaPlace
        self._log = log
        if log and not LaPlace:
            raise ZeroDivisionError('LaPlace required to apply logarithm')

    def fit(self, X, y):
        self._X = X
        self._y = y
        if not self._n_classes:
            self._n_classes = len(np.unique(self._y))
        self._n_samples = X.shape[0]
        self._n_features = X.shape[1]
        self._classes_count = {c: 0 for c in range(self._n_classes)}
        if self._discrete:
            self._frequency_tables = {(c, f): np.zeros(len(np.unique(self._X[:, f])), dtype=int)
                                      for c in range(self._n_classes)
                                      for f in range(self._n_features)}
            for i in range(self._n_samples):
                row = self._X[i, :]  # select row from X
                self._classes_count[self._y[i]] += 1  # increase count of the row's class
                for j in range(len(row)):
                    self._frequency_tables[(self._y[i], j)][row[j]] += 1
                    # increase count in table of X's class for atribute j, in specific bin
        else:
            self._values = {(c, f): []
                            for c in range(self._n_classes)
                            for f in range(self._n_features)}
            self._means = {(c, f): None
                           for c in range(self._n_classes)
                           for f in range(self._n_features)}
            self._stds = {(c, f): None
                          for c in range(self._n_classes)
                          for f in range(self._n_features)}
            for i in range(self._n_samples):
                row = self._X[i, :]
                self._classes_count[self._y[i]] += 1
                for j in range(len(row)):
                    self._values[(self._y[i], j)].append(row[j])
            for c in range(self._n_classes):
                for f in range(self._n_features):
                    self._means[(c, f)] = np.sum(self._values[(c, f)]) / len(self._values[(c, f)])
                    self._stds[(c, f)] = np.sqrt(1 / (len(self._values[(c, f)]) + 1)
                                                 * np.sum(np.power(self._values[(c, f)] - self._means[(c, f)], 2)))

    def predict(self, Data):
        y_pred = self.predict_proba(Data)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, Data):
        if self._log:
            probabilities = np.zeros((Data.shape[0], self._n_classes))
        else:
            probabilities = np.ones((Data.shape[0], self._n_classes))
        for i in range(Data.shape[0]):
            row = Data[i, :]
            for c in range(self._n_classes):
                if self._discrete:
                    for j in range(len(row)):
                        if self._laplace and self._log:
                            probabilities[i, c] += np.log((self._frequency_tables[(c, j)][row[j]] + 1) / (
                                    self._classes_count[c] + self._n_classes))
                        elif self._laplace and not self._log:
                            probabilities[i, c] *= (self._frequency_tables[(c, j)][row[j]] + 1) / (
                                    self._classes_count[c] + self._n_classes)
                        elif not self._laplace and self._log:
                            probabilities[i, c] += np.log(
                                self._frequency_tables[(c, j)][row[j]] / self._classes_count[c])
                        else:
                            probabilities[i, c] *= self._frequency_tables[(c, j)][row[j]] / self._classes_count[c]

                else:
                    for j in range(len(row)):
                        exponent = - np.power((row[j] - self._means[(c, j)]), 2) / (2 * np.power(self._stds[(c, j)], 2))
                        factor = 1 / (self._stds[(c, j)] * np.sqrt(2 * np.pi))
                        if self._log:
                            probabilities[i, c] += (-np.log(self._stds[(c, j)]) + exponent)
                        else:
                            probabilities[i, c] *= (factor * np.exp(exponent))
                if self._log:
                    probabilities[i, c] += np.log(self._classes_count[c] / self._n_samples)
                else:
                    probabilities[i, c] *= self._classes_count[c] / self._n_samples
        if self._discrete and not self._laplace:
            return probabilities
        elif self._log:
            return np.divide(np.sum(probabilities, axis=1)[:, None] - probabilities,
                             2 * np.sum(probabilities, axis=1)[:, None])
        return np.divide(probabilities, np.sum(probabilities, axis=1)[:, None])

    def score(self, y_test, y_pred):
        return np.sum(np.array(y_test == y_pred, dtype=int)) / len(y_test)


def plotBayesScore(X, y_1, y_2, title, subtitle1, subtitle2, subtitle3, score=None):
    pca = PCA(n_components=2)
    pca.fit(X)
    x = pca.transform(X)
    plt.figure(figsize=(13, 4))
    plt.subplot(1, 3, 1)
    plt.scatter(x[:, 0], x[:, 1], c=y_1, alpha=0.5)
    plt.title(subtitle1)
    plt.subplot(1, 3, 2)
    plt.scatter(x[:, 0], x[:, 1], c=y_2, alpha=0.5)
    plt.title(subtitle2)
    plt.subplot(1, 3, 3)
    categories = np.array(y_1 == y_2, dtype=int)
    colormap = np.array(['r', 'g'])
    plt.scatter(x[:, 0], x[:, 1], c=colormap[categories], alpha=0.5)
    plt.title(subtitle3)
    if score:
        plt.suptitle(title + ', score=' + str(score) + '%')
    else:
        plt.suptitle(title)
    plt.tight_layout()


def task1(dataset=0, discrete=False, bins=7, laplace=True, log=True):
    if dataset == 0:
        X, y = datasets.load_wine(return_X_y=True)
    elif dataset == 1:
        X, y = datasets.load_breast_cancer(return_X_y=True)
    else:
        X, y = datasets.load_iris(return_X_y=True)
    if discrete:
        est = preprocessing.KBinsDiscretizer(n_bins=bins, encode='ordinal')
        est.fit(X)
        X = np.array(est.transform(X), dtype=int)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
    bayes = NBC(discrete=discrete, LaPlace=laplace, log=log)
    bayes.fit(X_train, y_train)
    print(bayes.predict_proba(X_test))
    y_pred_test = bayes.predict(X_test)
    score = np.round(bayes.score(y_test, y_pred_test) * 100, 4)
    plotTitle = 'NBC, discrete=' + str(discrete) + ', '
    if discrete:
        plotTitle += 'LaPlace=' + str(not not laplace) + ', ' + 'bins=' + str(bins) + ', '
    plotBayesScore(X_test, y_test, y_pred_test, plotTitle + 'test set', 'original labels', 'predicted labels',
                   'accuracy', score)
    y_pred_train = bayes.predict(X_train)
    print(bayes.predict_proba(X_train))
    score = np.round(bayes.score(y_train, y_pred_train) * 100, 4)
    plotBayesScore(X_train, y_train, y_pred_train, plotTitle + 'train set', 'original labels', 'predicted labels',
                   'accuracy', score)


def task2(dataset=0, log=True):
    if dataset == 0:
        X, y = datasets.load_wine(return_X_y=True)
    elif dataset == 1:
        X, y = datasets.load_breast_cancer(return_X_y=True)
    else:
        X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
    bayes_lib = GaussianNB()
    bayes_lib.fit(X_train, y_train)
    y_pred_lib_test = bayes_lib.predict(X_test)
    bayes_mine = NBC(log=log)
    bayes_mine.fit(X_train, y_train)
    y_pred_mine_test = bayes_mine.predict(X_test)
    comp = np.round(np.sum(y_pred_lib_test == y_pred_mine_test) / len(y_pred_lib_test) * 100, 4)
    plotTitle = 'Naive Bayes comparison'
    plotBayesScore(X_test, y_pred_lib_test, y_pred_mine_test,
                   plotTitle + ', test set, compatibility=' + str(comp) + '%', 'GaussianNB', 'NBC', 'accuracy')
    y_pred_lib_train = bayes_lib.predict(X_train)
    y_pred_mine_train = bayes_mine.predict(X_train)
    comp = np.round(np.sum(y_pred_lib_train == y_pred_mine_train) / len(y_pred_lib_train) * 100, 4)
    plotBayesScore(X_train, y_pred_lib_train, y_pred_mine_train,
                   plotTitle + ', train set, compatibility=' + str(comp) + '%', 'GaussianNB', 'NBC', 'accuracy')


task1(2, True, 7, log=True, laplace=True)
# task1(2, True, 7, log=False)
# task1(2, False, log=False)
# task1(0, False, log=True)

task2(2, True)
# task2(0, False)
# task2(2)

plt.show()

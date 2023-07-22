import numpy as np
from sklearn import datasets, model_selection, preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, hidden=10, epochs=100, eta=0.1, shuffle=True):
        self._hidden = hidden
        self._epochs = epochs
        self._eta = eta
        self._shuffle = shuffle
        self._w_h = None
        self._w_o = None
        self._b_h = None
        self._b_o = None
        self.results = {'epoch': [], 'cost': [], 'accuracy': []}

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _forward(self, X):
        s1 = X.dot(self._w_h) + self._b_h
        s2 = np.apply_along_axis(self._sigmoid, axis=0, arr=s1)
        s3 = s2.dot(self._w_o) + self._b_o
        s4 = np.apply_along_axis(self._sigmoid, axis=0, arr=s3)
        return s2, s4

    def _compute_cost(self, y, out):
        cost = 0
        for i in range(y.shape[0]):
            cost += np.sum(y[i, :] * np.log(out[i, :]) + (1 - y[i, :]) * np.log(1 - out[i, :]))
        return -cost

    def fit(self, X_train, y_train):
        self._w_h = np.random.normal(loc=0.0, scale=0.1, size=(X_train.shape[1], self._hidden))
        self._b_h = np.zeros((1, self._hidden))
        self._w_o = np.random.normal(loc=0.0, scale=0.1, size=(self._hidden, y_train.shape[1]))
        self._b_o = np.zeros((1, y_train.shape[1]))
        for i in range(self._epochs):
            if self._shuffle:
                shuffled_id = np.random.permutation(X_train.shape[0])
            else:
                shuffled_id = np.arange(X_train.shape[0])
            for k in range(X_train.shape[0]):
                a_h, a_o = self._forward(X_train[shuffled_id[k], :])
                df_o = a_o * (1 - a_o)
                delta_o = (a_o - y_train[shuffled_id[k], :]) * df_o
                df_h = a_h * (1 - a_h)
                delta_h = delta_o.dot(self._w_o.T) * df_h  # długość=hidden
                grad_w_h = X_train[shuffled_id[k], :][:, None].dot(delta_h)
                grad_b_h = delta_h
                grad_w_o = a_h.T.dot(delta_o)
                grad_b_o = delta_o
                self._w_h -= grad_w_h * self._eta
                self._b_h -= grad_b_h * self._eta
                self._w_o -= grad_w_o * self._eta
                self._b_o -= grad_b_o * self._eta
            self.results['epoch'].append(i)
            self.results['cost'].append(
                self._compute_cost(y_train[shuffled_id, :], self._forward(X_train[shuffled_id, :])[1]))
            self.results['accuracy'].append(
                self.accuracy(self.predict(X_train[shuffled_id, :]), y_train[shuffled_id, :]))

    def predict(self, X):
        y_pred_coded = self._forward(X)[1]
        return np.argmax(y_pred_coded, axis=1)

    def accuracy(self, y_pred, y):
        y = np.argmax(y, axis=1)
        return np.sum(y_pred == y) / len(y)


def plot_score(X, y_1, y_2, title, subtitle1, subtitle2, subtitle3, score=None):
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
        plt.suptitle(title + ', score=' + str(100 * np.round(score, 2)) + '%')
    else:
        plt.suptitle(title)
    plt.tight_layout()


def plot_learning(results, title):
    plt.figure(figsize=(13, 4))
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.plot(results['epoch'], results['cost'])
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.subplot(1, 2, 2)
    plt.plot(results['epoch'], results['accuracy'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.tight_layout()


def plot_learning_comparative(r1, r2, l1, l2, title):
    plt.figure(figsize=(13, 4))
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.plot(r1['epoch'], r1['cost'], label=l1)
    plt.plot(r2['epoch'], r2['cost'], label=l2)
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(r1['epoch'], r1['accuracy'], label=l1)
    plt.plot(r2['epoch'], r2['accuracy'], label=l2)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.tight_layout()


def task1(X, y_coded, set_name, shuffle=True):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_coded, random_state=13)

    hidden = 10
    epochs = 100

    bob = MLP(hidden, epochs, shuffle=shuffle)
    bob.fit(X_train, y_train)
    y_pred = bob.predict(X_test)
    print(y_pred)
    print(np.argmax(y_test, axis=1))

    y_pred2 = bob.predict(X_train)
    print(np.argmax(y_train, axis=1))
    print(y_pred2)

    title_learning = 'MLP classification for ' + set_name + '\nhidden=' + str(hidden) + ', epochs=' + str(epochs)
    title_test = 'MLP classification for ' + set_name + ': test set\nhidden=' + str(hidden) + ', epochs=' + str(epochs)
    title_train = 'MLP classification for ' + set_name + ': train set\nhidden=' + str(hidden) + ', epochs=' + str(
        epochs)
    if not shuffle:
        title_learning += ', shuffle=False'
        title_test += ', shuffle=False'
        title_train += ', shuffle=False'
    plot_learning(bob.results, title_learning)
    plot_score(X_test, np.argmax(y_test, axis=1), y_pred, title_test, 'labels', 'predicted labels', 'differences',
               bob.accuracy(y_pred, y_test))
    plot_score(X_train, np.argmax(y_train, axis=1), y_pred2, title_train, 'labels', 'predicted labels', 'differences',
               bob.accuracy(y_pred2, y_train))

    plt.show()


def task2(X, y_coded, set_name):
    hidden = 10
    epochs = 100

    model1 = MLP(hidden, epochs)
    model2 = MLP(hidden, epochs)
    scaler = preprocessing.MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_coded, random_state=13)
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = model_selection.train_test_split(X_norm, y_coded,
                                                                                            random_state=13)

    model1.fit(X_train, y_train)
    r1 = model1.results

    model2.fit(X_train_norm, y_train_norm)
    r2 = model2.results

    plot_learning_comparative(r1, r2, 'original', 'normalized',
                              'MLP classification for ' + set_name + '\nhidden=' + str(hidden) + ', epochs=' + str(
                                  epochs))
    plt.show()


choice = True
if choice:
    X, y = datasets.make_classification(n_samples=200, n_features=5, n_redundant=1, n_clusters_per_class=2,
                                        random_state=13)
    y_coded = np.zeros((len(y), 2))
    y_coded[y == 0, 0] = 1
    y_coded[y == 1, 1] = 1
    set_name = 'artificial set'
else:
    X, y = datasets.load_iris(return_X_y=True)
    y_coded = preprocessing.LabelBinarizer().fit_transform(y)
    set_name = 'Iris'

task1(X, y_coded, set_name, shuffle=True)
task2(X, y_coded, set_name)

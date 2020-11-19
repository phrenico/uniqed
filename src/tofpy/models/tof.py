import numpy as np

from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from scipy.spatial import cKDTree


class TOF(BaseEstimator, ClassifierMixin, OutlierMixin):
    def __init__(self, cutoff_n=1.0, k=None, q=2, centrality_func=np.mean):
        """Constructor for Temporal Outlier Factor detector

        :param int cutoff_n: maximal possible event length in sample points (default: cutoff_n=1)
        :param int k: number of neighbors to use (default: k=None, itt will use k=d+1, where d is the data-dimension)
        :param float q: norm to us (default: q=2 Euclidean norm)
        :param func centrality_func: Centrality-measure to use in the estimation of TOF score
                                     (default: centrality_func=np.mean)
        """
        self.cutoff_n = cutoff_n
        self.k = k
        self.q = q
        self.centrality_func = centrality_func

    def fit(self, X, y=None):
        """Fits the model

        :param X: data (n_dataponts, d) shape
        :param y: None
        :return: self
        """
        self.X_ = X
        self.d_ = X.shape[1]

        self.outlier_score_ = self._compute_outlier_score(X, self.k)
        self.cutoff_ = self._compute_cutoff(self.cutoff_n)
        self.p_value_ = self._compute_p_value(self.outlier_score_)
        return self

    def predict(self, X):
        """Classify the points

        :param numpy.ndarray X: data (n_dataponts, d) shape
        :return: class labels (-1 and 1)
        :rtype: numpy.ndarray
        """
        outliers_bool = self._compute_outlier_score(X, self.k) > self.cutoff_

        if sum(outliers_bool) > (len(outliers_bool) / 2.0):
            outliers_bool = (1 - outliers_bool).astype(bool)

        prediction = np.zeros([X.shape[0], 1])
        prediction[outliers_bool] = 1

        prediction = (2 * prediction.astype(int)) - np.ones([X.shape[0], 1], dtype=int)

        self.y_pred_ = prediction
        self.outliers_inds_ = self._get_outliers_inds(outliers_bool)
        self.outlier_p_values_ = self.p_value_[outliers_bool]
        return prediction

    def _get_outliers_inds(self, outliers_bool):
        """Returns outlier inds from Boolean array

        :param numpy.ndarray of bool outliers_bool: array with truth-values of outlierness
        :return: indices
        :rtype: numpy.ndarray of int
        """
        return np.where(outliers_bool)[0]

    def _find_nearest_neighbors(self, X, k=None):
        """Finds k-nearest neighbor distances and indices by using the cKDTree class

        :param np.ndarray of float X: points
        :param int k: number of neighbors (default: k=None)
        :return: neighbor distances and indices
        :rtype: (numpy.ndarray of float, numpy.ndarray of int)
        """
        if k is None:
            k = self.d_ + 1
        tree = cKDTree(self.X_)
        distances, indicis = tree.query(X, k + 1)
        return distances[:, 1:], indicis[:, 1:]

    def _compute_cutoff(self, cutoff_n):
        """Computes cutoff threshold for the outlier-score

        :param int cutoff_n: the length of
        :return: cutoff threshold value
        :rtype: float
        """
        if (
            type(cutoff_n) == int
            or type(cutoff_n) == np.float64
            or type(cutoff_n) == float
            or type(cutoff_n) == np.int64
        ):
            if self.k is None:
                k = self.d_ + 1
            else:
                k = self.k
            if cutoff_n < k:
                cutoff_n = k

            neighborinds = np.reshape(np.arange(cutoff_n, cutoff_n - k, -1), [1, k])
            cutoff = self._compute_tof(neighborinds, np.zeros([1, 1]))[0]
            cutoff = 1.0 / cutoff
        else:
            print(type(cutoff_n), cutoff_n)
            raise ValueError("Invalid value for cutoff_n")

        return cutoff

    def _compute_p_value(self, outlier_score):
        p_value = (np.argsort(np.argsort(outlier_score)) + 1) / float(
            len(outlier_score)
        )
        return p_value

    def _compute_outlier_score(self, X, k=None):
        """Computes 1/TOF for X

        :param numpy.ndarray X: dataset
        :param int k: number of neighbors
        :return: 1/TOF outlier score
        """
        embedded_length = X.shape[0]

        # get point indicis, find k-nn and compute
        # the average temporal distance
        indicis = np.arange(0, embedded_length).reshape([embedded_length, 1])
        nearest_indicis = self._find_nearest_neighbors(X, k=k)[1]
        outlier_score_ = 1.0 / self._compute_tof(nearest_indicis, indicis)

        return outlier_score_

    def _compute_tof(self, nearest_indicis, indicis):
        """Computes TOF score from nearest neighbor indices and the indice list of the actual moment

        :param nearest_indicis: indices of nearestneighbors
        :param indicis: the actual moment
        :return:
        """
        q = self.q
        tof = self.centrality_func(np.abs(nearest_indicis - indicis) ** q, axis=1) ** (
            1.0 / q
        )

        return tof

    def _compute_perc_cutoff(self, cutoff_n):
        """Computes threshold for cutoff value given in per cent

        :param int cutoff_n: per cent value (between 0 and 100)
        :return: threshold value
        :rtype: float
        """
        cutoff_len = int(self.X_.shape[0] * (cutoff_n / 100.0))
        cutoff = np.sort(self.outlier_score_)[-cutoff_len]
        return cutoff

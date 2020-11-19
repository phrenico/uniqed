import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TimeDelayEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, d=3, tau=1):
        """Time delay.
        Embedd the time series with [len(x) - (d - 1) * tau, d] shape.

        :param int d: embedding dimension
        :param int tau: embedding delay
        """
        self.d = d
        self.tau = tau

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        X = self._embedding(x, self.d, self.tau)
        return X

    def fit_transform(self, x, y=None):
        return self.fit(x).transform(x)

    def _embedding(self, x, d, tau):
        """Time delay embedding

        :param numpy.ndarray x: 1D time series
        :param int d: Embedding dimension
        :param int tau: Embedding delay
        :return: Embedded time series with [len(x) - (d - 1) * tau, d] shape
        :rtype: numpy.ndarray
        """
        embedded_length = len(x) - (d - 1) * tau
        X = np.zeros((embedded_length, d))
        for i in range(d):
            X[:, i] = x[i * tau : embedded_length + i * tau]
        return X


class TransformYTrue(BaseEstimator, TransformerMixin):
    def __init__(self, d=3, tau=1):
        """Transforms the groundtruth classes to the similar size as the prediction

        :param int d: Embeddig dimension
        :param int tau: Embedding delay
        """
        self.d = d
        self.tau = tau

    def fit(self, x, y=None):
        self.length_ = len(x) - (self.d - 1) * self.tau
        return self

    def transform(self, x):
        X = self._transform_y_true(x)
        return X

    def fit_transform(self, x, y=None):
        return self.fit(x).transform(x)

    def _transform_y_true(self, x):
        """Transforms y_true to shorter version aligned with a specific embedding (d, tau)

        :param numpy.ndarray x: array with values
        :return: array truncated symmetrically at the begining and at the end
        :rtype: numpy.ndarray
        """
        return x[self._get_faketime_axis(self.d, self.tau)]

    def _get_faketime_axis(self, d, tau):
        """Computes a new shifted time-axis for embedded time-series


        :param numpy.ndarray embededd_time_series: embedded time series (n x d) array, time instances as rows
        :param int embedding_delay: the embedding delay parameter used in the embedding
        :return: numpy.array with new shifted time-axis
        """

        # factor for the adjustment of time indicis after embedding
        dimension_time_shift = ((d - 1) / 2.0) * tau
        time_x = (
            np.arange(dimension_time_shift, self.length_ + dimension_time_shift)
        ).astype(int)
        return time_x


def invertit(score, doit=False):
    """Inverts score if doit is True

    :param np.ndarray score: score to conditionally invert
    :param bool doit: invert the score or not (default: False)
    :return: inverted or original score
    :rtype: np.ndarray
    """
    if doit:
        x = 1 / score
    else:
        x = score
    return x


def _make_result_df(new_time_axis, outlier_score, y_pred, inv_it, prefix=""):
    """Make result dataFrame for detections

    :param np.ndarray new_time_axis: truncated time axis after embedding
    :param np.ndarray outlier_score: computed outlier scores
    :param np.ndarray y_pred: predicted class labels (contains -1s and 1s for the two classes)
    :param bool inv_it: wheather invert the outlier score or not
    :param str prefix: some prefix to the columns
    :return: DataFrame with results, in the columns are the score, class_label respectively
    :rtype: pandas.DataFrame
    """
    df = pd.DataFrame(
        {
            prefix + "_score": invertit(outlier_score, doit=inv_it).flatten(),
            prefix: ((y_pred + 1) / 2).flatten(),
        },
        index=new_time_axis.flatten(),
    )
    return df

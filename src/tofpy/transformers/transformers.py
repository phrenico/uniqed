import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TimeDelayEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, d=3, tau=1):
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

        # embedded length
        #print len(x), d, tau
        embedded_length = len(x) - (d - 1) * tau

        # initialize embedding matrix
        X = np.zeros((embedded_length, d))

        # fill up initialized matrix column by column
        for i in range(d):
            X[:, i] = x[i * tau: embedded_length + i * tau]
        return X


class TransformYTrue(BaseEstimator, TransformerMixin):
    def __init__(self, d=3, tau=1):
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
        return x[self._get_faketime_axis(self.d, self.tau)]

    def _get_faketime_axis(self, d, tau):
        """Computes a new shifted time-axis for embedded time-series


        :param numpy.ndarray embededd_time_series: embedded time series (n x d) array, time instances as rows
        :param int embedding_delay: the embedding delay parameter used in the embedding
        :return: numpy.array with new shifted time-axis
        """

        # factor for the adjustment of time indicis after embedding
        dimension_time_shift = ((d - 1) / 2.) * tau
        time_x = (np.arange(dimension_time_shift, self.length_ +
                            dimension_time_shift)).astype(int)
        return time_x


def invertit(score, doit=False):
    if doit:
        x = 1/score
    else:
        x = score
    return x

def make_result_df(new_time_axis, outlier_score, y_pred, inv_it, prefix=''):
    """Make result dataFrame from verbose version of the ruunerfuncs.detect_oulier function

    :param new_time_axis:
    :param outlier_score:
    :param y_pred:
    :param inv_it:
    :param prefix:
    :return:
    """
    df = pd.DataFrame({prefix+'_score': invertit(outlier_score, doit=inv_it).flatten(), prefix: ((y_pred + 1) / 2).flatten()},
                       index=new_time_axis.flatten())
    return df
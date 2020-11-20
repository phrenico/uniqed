import pandas as pd

from uniqed.models.tof import TOF
from uniqed.transformers.transformers import (
    TimeDelayEmbedder,
    TransformYTrue,
    _make_result_df,
)


def detect_outlier(
    time_series,
    cutoff_n=1.0,
    k=None,
    in_percent=False,
    embedding_dimension=3,
    embedding_delay=1,
    **other_method_kwargs
):
    """Detects outliers with TOF

    :param pandas.DataFrame time_series: pandas dataframe with the time series
    :param float cutoff_n: the threshold for the detector
                            (max event length, or % of #datapoints)
    :param int k: numbert of neighbors to use (default is embedding)dimension+1)
    :param bool in_percent: if True then the threshold is draw at the given percentage not in event length
    :param int embedding_dimension: embedding dimension value (>=1) [default: 3]
    :param int embedding_delay: embedding delay (>=1) [default: 1]
    :return: result DataFrame
    :rtype: pandas.DataFrame
    """

    # Conversion to numpy array
    np_time_series = time_series.values[:, 0]

    # Time series embedding, and new time axis
    embededd_time_series = TimeDelayEmbedder(
        d=embedding_dimension, tau=embedding_delay
    ).fit_transform(np_time_series)
    new_time_axis = TransformYTrue(
        d=embedding_dimension, tau=embedding_delay
    ).fit_transform(time_series.index)
    new_time_axis = pd.DataFrame(new_time_axis).values

    # initialize method object
    mytof = TOF(cutoff_n=cutoff_n, k=k, **other_method_kwargs)
    mytof = mytof.fit(embededd_time_series)
    if in_percent:
        mytof.cutoff_ = mytof._compute_perc_cutoff(cutoff_n)
    y_pred = mytof.predict(embededd_time_series)

    # locally scoring outlierness for each time series points
    outlier_score = mytof.outlier_score_

    res_df = _make_result_df(
        new_time_axis, outlier_score, y_pred, inv_it=True, prefix="TOF"
    )
    return pd.concat([time_series, res_df], axis=1, sort=False)

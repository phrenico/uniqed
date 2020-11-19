import numpy as np
import pandas as pd


def basic(x_old, r=3.9):
    """Logmap update dynamics (1 step)

    :param float x_old: current value
    :param float r: parameter
    :return: next value
    """
    x_new = r * x_old * (1.0 - x_old)
    return x_new


def switch(x_old):
    """tentmap update dynamics

    :param float x_old: current value
    :return: next value
    """
    x_new = 1.59 - 2.15 * np.abs(x_old - 0.7) - 0.9 * x_old
    return x_new


def generate_logmapdata(N=2000, rseed=112):
    """Generates logistic-map time series with tent-map anomaly segment

    :param int N: length
    :param int rseed: random seed
    :return: DataFrame with time series and anomaly value
    """
    np.random.seed(rseed)
    anomaly_length = np.random.randint(20, 200)
    anomaly_place = np.random.randint(N - anomaly_length)

    x = [np.random.rand()]
    truth = []
    for q in range(N):
        if q < anomaly_place:
            x.append(basic(x[-1]))
            truth.append(0)
        elif (q > anomaly_place) and (q < anomaly_place + anomaly_length):
            x.append(switch(x[-1]))
            truth.append(1)
        else:
            x.append(basic(x[-1]))
            truth.append(0)

    data_dict = {"value": np.array(x)[1:], "is_anomaly": truth}

    df = pd.DataFrame(data_dict)
    return df

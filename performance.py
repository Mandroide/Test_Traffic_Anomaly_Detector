from typing import Union, Sequence

import numpy as np
from sklearn.metrics import mean_squared_error


# TODO: Implement detection time error.
def measure_detection_time_error(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Calculate detection time error of the traffic anomaly detector of a video.

    Parameters
    ----------
    y_true : array_like
      Ground-truth time in seconds.
    y_pred : array_like
      Predicted time in seconds.

    Returns
    -------
    float
      Normalized Root Mean Square Error.

    Notes
    -----
    The detection time error is computed as the RMSE between the ground-truth anomaly start time and predicted start
    time for all true positives. To obtain a normalized evaluation score, the NRMSE is calculated as the normalized
    detection time RMSE using min-max normalization between 0 and 300 frames (for videos of 30 FPS, this corresponds to
    10 seconds), which represents a reasonable range of RMSE values for the anomaly detection task. Specifically,
    :math:`NRMSE^t` is computed as

    .. math:: NRMSE^t = \dfrac{min(RMSE, 300)}{300}

    References
    ----------
    .. [1] Milind Naphade; Shuo Wang; David C. Anastasiu; Ming-Ching Chang; Liang Zheng; Anuj Sharma; Rama Chellappa;
           Pranamesh Chakraborty; Zheng Tang; Xiaodong Yang "The 4th AI City Challenge", page 6, 2020.
    """
    nrmse = min(mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False), 300) / 300
    return nrmse


# TODO: Implement detection performance.
def measure_detection_performance(nrmse: float, anomalies: int) -> Union[float, np.ndarray]:
    """Calculate detection performance of the traffic anomaly detector.

    Parameters
    ----------
    nrmse : float
      Normalized Root Mean Square Error.
    anomalies : int
      Number of anomalies detected in video.

    Returns
    -------
    float, numpy.ndarray
      Score of performance.

    Notes
    -----
    Track 4 performance is measured by combining the detection performance and detection time error. Specifically,
    the Track 4 score (S4), for each participating team, is computed as

    .. math:: S_4 =  F_1 × (1 − NRMSE^t),

    where the :math:`F_1` score is the harmonic mean of the precision and recall of anomaly prediction. For video clips
    containing multiple ground-truth anomalies, credit is given for detecting each anomaly. Conversely, multiple false
    predictions in a single video clip are counted as multiple false alarms. If multiple anomalies are provided within
    the time span of a single ground-truth anomaly, we only consider the one with minimum detection time error and
    ignore the rest. We expect all anomalies to be successfully detected and penalize missed detection and spurious ones
    through the :math:`F_1` component in the :math:`S_4` evaluation score.

    References
    ----------
    .. [1] Milind Naphade; Shuo Wang; David C. Anastasiu; Ming-Ching Chang; Liang Zheng; Anuj Sharma; Rama Chellappa;
           Pranamesh Chakraborty; Zheng Tang; Xiaodong Yang "The 4th AI City Challenge", page 6, 2020.
    """
    return 0

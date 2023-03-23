import math

import numpy as np


def get_estimator(price_data, window=30, trading_periods=252, clean=False):

    log_hc = (price_data['high'] / price_data['close']).apply(np.log)
    log_ho = (price_data['high'] / price_data['open']).apply(np.log)
    log_lc = (price_data['low'] / price_data['close']).apply(np.log)
    log_lo = (price_data['low'] / price_data['open']).apply(np.log)
    rs = log_ho * log_hc + log_lo * log_lc 
    def rs_vol_func(v):
        return v.sum()
    rs_vol = rs.rolling(
        window=window,
        center=False
    ).apply(func=rs_vol_func)
    rs_vol /= window

    def vol_func(v):
        return sum((v - v.mean())**2)

    log_co = (price_data['close'] / price_data['open']).apply(np.log)
    close_vol = log_co.rolling(
        window=window,
        center=False
    ).apply(func=vol_func)
    close_vol /= (window - 1.0)

    log_oc = (price_data['open'] / price_data['close'].shift(1)).apply(np.log)
    open_vol = log_oc.rolling(
        window=window,
        center=False
    ).apply(func=vol_func)
    open_vol /= (window - 1.0)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    result = (open_vol + k * close_vol + (1 - k) * rs_vol).apply(np.sqrt) * math.sqrt(trading_periods)

    if clean:
        return result.dropna()
    else:
        return result

def get_name():
    return 'YZ'

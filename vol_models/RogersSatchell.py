import math

import numpy as np


def get_estimator(price_data, window=30, trading_periods=252, clean=False):
    
    log_ho = (price_data['high'] / price_data['open']).apply(np.log)
    log_hc = (price_data['high'] / price_data['close']).apply(np.log)

    log_lo = (price_data['low'] / price_data['open']).apply(np.log)
    log_lc = (price_data['low'] / price_data['close']).apply(np.log)
    
    rs = log_ho * log_hc + log_lc * log_lo 

    def f(v):
        return (trading_periods * v.sum() / window )**0.5
    
    result = rs.rolling(
        window=window,
        center=False
    ).apply(func=f)
    
    if clean:
        return result.dropna()
    else:
        return result

def get_name():
    return 'RS'

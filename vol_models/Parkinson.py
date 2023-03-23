import math
import numpy as np

def get_estimator(price_data, window, trading_periods=252, clean=False):

    factor = (1.0 / (4.0 * window * math.log(2.0)))
    rs = (np.log(price_data['high'] / price_data['low']))**2.0 # TODO this rounds to 6 decimal points...how change

    def f(v):
        return (trading_periods**0.5) * (factor * v.sum())**0.5

    result = rs.rolling(
        window=window
    ).apply(func=f)

    if clean:
        return result.dropna()
    else:
        return result

def get_name():
    return 'PK'

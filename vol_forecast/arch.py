from arch.__future__ import reindexing
from arch.univariate import *
import numpy as np

import logging
logger = logging.getLogger('VOL')

'''
gives garch forecast from log returns on asset
'''

means = {
    'Constant':ConstantMean,
    'Zero':ZeroMean, 
    'LS':LS,
    'ARX':ARX,
    'HARX':HARX}

distribs = {
    "Normal":Normal,
    "StudentsT":StudentsT,
    "SkewStudent":SkewStudent,
    "GeneralizedError":GeneralizedError
    }

class Forecast():
    def __init__(self, df, model_type, mean='HARX', dist='SkewStudent',
        q=1,o=1,p=1,power=1.0, rescale=True):

        
        assert mean in means.keys(), 'Mean not in keys'
        assert dist in distribs.keys(), 'Mean not in keys'

        self.model_type = model_type
        self.df = df.copy()

        # https://quant.stackexchange.com/questions/50816/forecasting-volatility-using-garch-in-python-arch-package
        # need to rescale cause stackoverflow says to
        self.rescale = 100 
        if rescale:
            self.df *= self.rescale

        self.mean = mean
        self.dist = dist
        self.p = p
        self.q = q
        self.o = o
        self.power = power

        self.fore = None
        self.res = None
        self.am = None

    def get_predicted_vol(self):
        if self.fore is None:
            return
        return ((self.fore.variance / (self.rescale**2)) ** .5) * np.sqrt(252) * 100

    def model(self):
        logger.info('')
        if self.model_type == 'ARCH':
            vol = GARCH(p=1, o=0, q=0, power=2.0)
        elif self.model_type == 'GARCH':
            vol = GARCH(p=1, o=0, q=1, power=2.0)
        elif self.model_type == 'GJR-GARCH':
            vol = GARCH(p=1, o=1, q=1, power=2.0)
        elif self.model_type == 'AVARCH':
            vol = GARCH(p=1, o=0, q=0, power=1.0)
        elif self.model_type == 'AVGARCH':
            vol = GARCH(p=1, o=0, q=1, power=1.0)
        elif self.model_type == 'TARCH' or self.model_type == 'ZARCH':
            vol = GARCH(p=1, o=1, q=1, power=1.0)
        elif self.model_type == 'Power ARCH':
            vol = GARCH(p=1, o=0, q=0, power=self.power)
        elif self.model_type == 'Power GARCH':
            vol = GARCH(p=1, o=0, q=1, power=self.power)
        elif self.model_type == 'Asym Power GARCH':
            vol = GARCH(p=1, o=1, q=1, power=self.power)
        
        elif self.model_type == 'Symm EGARCH':
            vol = EGARCH(p=1,q=1)
        elif self.model_type == 'EGARCH':
            vol = EGARCH(p=1,o=1,q=1)
        elif self.model_type == 'Exp EGARCH':
            vol = EGARCH(p=self.p)
        
        elif self.model_type == 'FIARCH': 
            vol = FIGARCH(q=0, power=2.0)
        elif self.model_type == 'FIGARCH':
            vol = FIGARCH(q=1, power=2.0)
        elif self.model_type == 'FIAVARCH':
             vol = FIGARCH(q=0, power=1.0)
        elif self.model_type == 'FIAVGARCH':
             vol = FIGARCH(q=1, power=1.0)
        elif self.model_type == 'Power FIARCH': 
             vol = FIGARCH(q=0, power=self.power)
        elif self.model_type == 'Power FIGARCH':
             vol = FIGARCH(q=1, power=self.power)

        elif self.model_type == 'HARCH':
             vol = HARCH(lags=[1,5,22])

        else:
            raise Exception("bad model_type")


        am = means[self.mean](self.df)
        am.volatility = vol
        am.distribution = distribs[self.dist]()
        res = am.fit(disp='off')
        try:    fore = res.forecast(horizon=10)
        except: fore = res.forecast(horizon=10, method='bootstrap')

        self.am = am
        self.res = res
        self.fore = fore

'''

am = means['ARX'](day)
vol = GARCH(p=1, o=0, q=1, power=2.0)
am.volatility = vol
am.distribution = distribs['Normal']()
res = am.fit(disp='off')
fore = res.forecast(horizon=5)

res.plot()
plt.show()

========


sym='AAPL'
db = MyDB()
now = datetime.now().date()
start = now.replace(year = now.year - 10)
end   = now
out = db.get_data_from_sym(sym,start,end)
out = filter_by_time(out) # only get 930:4

df = db.update_returns(sym , out) # log returns. not percentage
day = pd.DataFrame()
day['Date'] = df['date']
day['day'] = df['1day']
day.set_index('Date',inplace=True)
day = day.dropna()

fore = Forecast(day, 'GARCH')
fore.model()

'''

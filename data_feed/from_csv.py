from numpy_ext import rolling_apply as rapp
import pandas as pd
import numpy as np
from datetime import datetime, date


# TODO start needs to have the form "month/day/year
def from_csv(sym,start=None,start_form='%Y-%M-%d',tz='-0500'):
    if start == None:
        now = datetime.now().date()
        start = str(now.replace(year = now.year - 5))

    columns = ['date','time','open','high','low','close','volume']
    df = pd.read_csv(f'../data/{sym}.txt', names=columns)
    df = df.round(4)
    
    # delete rows before we apply timestamp logic 
    csv_form = '%M/%d/%Y'
    ts = datetime.strptime(start,start_form)
    df = df[df.date.apply(lambda x: datetime.strptime(x,csv_form) >= ts)]

    times = df.apply(lambda row: datetime.strptime(f'{row.date}-{row.time} {tz}','%m/%d/%Y-%H:%M %z'), axis=1)

    df.drop('date',inplace=True,axis=1)
    df.drop('time',inplace=True,axis=1)

    try:
        df['date'] = times
    except:
        print(f"ERROR FROM_CSV {sym}")
        return None

    return df 


# filters by [930:4)
def filter_by_time(df):
    _df = df[(df.date.dt.hour == 16) | ((df.date.dt.hour == 15) & (df.date.dt.minute == 55))]
    gpby = _df.groupby(by=[_df.date.dt.year , _df.date.dt.month, _df.date.dt.day])
    for i,g in gpby:
        try:
            g1 = g.iloc[0] ; g2 = g.iloc[1]
            df.loc[df.date == g1.date, c] = g2.close
        except:
            pass
    out = df[(df.date.dt.hour >= 9) & (df.date.dt.hour < 16)]
    out = out[(out.date.dt.hour >= 10) | (out.date.dt.minute >= 30)]
    return out 



'''
import pmdarima as pm
from arch_model import arch


model = pm.auto_arima(day,
d=0, # non-seasonal difference order
start_p=1, # initial guess for p
start_q=1, # initial guess for q
max_p=4, # max value of p to test
max_q=4, # max value of q to test                        
                    
seasonal=False, # is the time series seasonal
                    
information_criterion='bic', # used to select best model
trace=True, # print results whilst training
error_action='ignore', # ignore orders that don't work
stepwise=True, # apply intelligent order search
)

_arma_model = sm.tsa.SARIMAX(endog=day,order=model.order)
_model_result = _arma_model.fit()
resid = _model_result.resid

garch_model = arch_model(resid , mean='HARX',p=1,q=1,o=1, vol='EGARCH', dist='skewt')
garch_result = garch_model.fit(disp='off')
fore = garch_result.forecast(horizon=5, method='bootstrap')

fore.variance
fore.mean

'''
    

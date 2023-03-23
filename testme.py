# from data_feed import tiingo, Time
# from vol_models.YangZhang import get_estimator
from helper.print_full_pd import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from os import path
from datetime import date
from timeit import default_timer as timer

# from data_feed.tiingo import Tiingo
# from data_feed.tdameritrade import *
# from data_feed.update_data import *
# from model.vol_surface import *
# from vol_models.RealizedVol import RealizedVol 
# 
from opt_plot.bs_opt_plot import *
from opt_plot.pyqt_graph import *
# 
# from vol_forecast.arch import Forecast
# from vol_forecast.vol_cones import VolCone

# from data_feed.db import MyDB
# from data_feed.from_csv import from_csv, filter_by_time

from PyQt5.QtGui import *

# du = pdate_data()
'''
sym='AAPL'
db = MyDB()
now = datetime.now().date()
start = now.replace(year = now.year - 10)
end   = now

# rv = RealizedVol(sym,db)

out = db.get_data_from_sym(sym,start,end)
out = filter_by_time(out) # only get 930:4
df = db.update_returns(sym , out)

vc = VolCone(df['1day'].dropna())
vc.create()
vc.plot() 

day = pd.DataFrame()
day['Date'] = df['date']
day['day'] = df['1day']
day.set_index('Date',inplace=True)
day = day.dropna()

fore = Forecast(day, 'GARCH')
fore.model()
'''

# fore = Forecast(day, p=1,q=1, mean='constant', vol='GARCH', dist='normal')
# mod = fore.model()

app = QApplication(sys.argv)        
win = Window()
sys.exit(app.exec_())

'''
IDEAS:

be able to put in any amount of positions, see the profit form they take from now to expiration
    abel to perturb IV and price separately (is this where wed use a monte carlo sim ?) 
    for each moment, be able to see delta,vega,gamma,vanna,etc

for trading vol, be able to determine cost of delta hedging. when would it make sense to sell vol wrt 
    cost of delta hedging?

'''

# TODO 
'''
PLAN:

find the historical vol(s) of stock
forecast next couple weeks vol using *GARCH methods
how does this forecasted/realized vol compare with current atm vol's 
how does this forecasted/realized vol compare with vol cones on equity

-----------

vol_time_change fix

    be able to perturb price (like you can dte and vol) ????  
        when you perturb vol, you should allow for sticky strike/delta changes

    question -- is atm vol the closest dte option's atm the iv? 

be able for each strategy, inc/dec the amount of each option to buy/sell
    i.e. -- if you wanna create some delta neutral call spread, you may want to buy more than you sell or vice---------versa

able to toggle on and off some options 

add self.note to each strategy for 'long' type
    need to add new widget -- simple description (fairly sure one already exists but forget for which)

remove earnings from historical data -- this fucks with daily vol 


-----

euan sinclair::
    selling options over the weekend
        sell friday at close, buy back at open 
        sell spy straddle if Vix < 38 (98th percentile)
        vega weighted for sizing 

    christoph dollinger
        calculate fair value of options given the vix



'''

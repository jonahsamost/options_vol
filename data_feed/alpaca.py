import requests
from dataclasses import dataclass,asdict,field
import json
import pandas as pd
import sys
from datetime import datetime,date, timedelta
from dateutil.relativedelta import relativedelta

from data_feed.Time import TimeFrame, sampleFreq, sampleType, remove_holidays

import alpaca_trade_api as alp

CLIENT_ID = ''
CLIENT_SECRET = ''

class Alpaca:
    def __init__(self):
        self.base_url = 'https://api.alpaca.markets'
        self.api = alp.REST(CLIENT_ID, CLIENT_SECRET, self.base_url)

    @property
    def is_inited(self):
        try:
            self.api.get_account()
            return True
        except:
            return False

    def getbars(self, sym, start, end, tframe='15Min'):
        df = self.api.get_bars(sym,timeframe=tframe,start=start,end=end,adjustment='all').df
        if df.empty: return df
        df.index = df.index.tz_convert('US/Eastern')
        df = df.between_time('09:30', '16:00')

        prev = pd.options.mode.chained_assignment 
        pd.options.mode.chained_assignment = None

        # change final ohlc 
        grps = df.groupby(by=pd.Grouper(freq='D')) # groupby day
        for i,g in grps:
            if g.empty or len(g) < 2: continue 
            g.loc[g.iloc[-2].name, 'close'] = g.iloc[-1].open   
        
        pd.options.mode.chained_assignment = prev
        df = df.between_time('09:30', '15:45')
        df.reset_index(inplace=True)
        df = df.rename(columns={'timestamp':'date'})
        df = df.round(4)
        return df 


'''
from data_feed.alpaca import *
a = Alpaca()
a.is_inited
df = a.getbars('NFLX','2021-01-01','2022-01-01')  

# GET ALL OPTIONS
import requests
import string
from bs4 import BeautifulSoup as bs

url = 'https://www.poweropt.com/optionable.asp?fl='

where = [char for char in string.ascii_uppercase]
where += ['ETF']
# where += ['IND']

options = []
for l in where:
    q = url + l
    r = requests.get(q)

    soup = bs(r.text, 'html.parser')
    tbody = soup.find_all('table')[1]
    h6 = tbody.find_all('h6')
    for sym in h6:
        search = 'partnerdetail.asp'
        if str(sym).find(search) != -1:
            opt = sym.a['href'].split('?')[1].split('&')[0].split('=')[1] 
            options.append(opt)
'''

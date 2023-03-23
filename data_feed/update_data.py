import requests
from dataclasses import dataclass,asdict,field
import json
import pandas as pd
import sys
from datetime import datetime,date, timedelta
from sqlalchemy.dialects import mysql as mysql_types

from data_feed.db import MyDB 
from data_feed.db import o,h,l,c,dte,dt,tbl_cols
from data_feed.tiingo import Tiingo
from data_feed.alpaca import Alpaca
from data_feed.Time import sampleFreq, sampleType
from data_feed.from_csv import *

import logging
logger = logging.getLogger('VOL')

alp = Alpaca()
assert alp.is_inited, 'Alpaca not inited'

class DataUpdate():
    def __init__(self, syms):
        self.alp = alp

        self.db = MyDB() 
        assert self.db.isSet, 'MyDB not set'

        if type(syms) == str:
            self.syms = [syms]
        elif type(syms) == list:
            self.syms = syms
        else:
            raise Exception('syms is of wrong type')

        # self.tt = Tiingo()
        # assert self.tt.connected, 'Tiingo cant connect'

        self.min_freq = 5 

    def update_price_data(self):
        sfreq=sampleFreq(stype=sampleType.MIN,freq=self.min_freq)
        for sym in self.syms:
            logger.info(sym)
            # csv only has data up to when i download it...though it does update...hmm
            if not self.db.sym_exists(sym): # get data from wire
                self.db.add_sym_table(sym)

            #    if self.db.lastError: 
            #        #  log an error TODO
            #        continue 
            #
            #    df = from_csv(sym)
            #    if df is None:
            #        print(f'ERROR: update price data {sym}')
            #        return False
            #    df = filter_by_time(df) # only keep market hours data
            #    df = df[tbl_cols]
            #    self.db.add_bulk_data(sym,df,dt)
            #    self.db.commit()

            end = datetime.now().date()
            start = self.db.latest_timestr_by_sym(sym) 
            if start:
                start = datetime.strptime(start,'%Y-%m-%d').date()
                start += timedelta(days=1) # get next day
                if start > end:
                    logger.info(f"No new data to add for {sym}")
                    continue 
                start = start.strftime('%Y-%m-%d')
                end   = end.strftime('%Y-%m-%d')
            else:
                start = str(end.replace(year = end.year - 5))
                end = str(end)

            # df = self.tt.get_data_with_symbol(sym,start,end,sfreq)
            df = self.alp.getbars(sym, start, end)
            if df is not None and not df.empty:
                df = df[tbl_cols]
                self.db.add_bulk_data(sym,df,dt)
                self.db.commit()

        return True


def is_day_over(today, day):
    latest = day.date.iloc[-1]
    if latest.strftime('%Y-%m-%d') != today:
        return True
    if latest.hour == 15 and latest.minute == 55:
        return True
    return False


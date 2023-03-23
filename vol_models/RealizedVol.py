from vol_models import Parkinson
from vol_models import GarmanKlass
from vol_models import HodgesTompkins
from vol_models import CloseClose
from vol_models import RogersSatchell
from vol_models import YangZhang

from data_feed.from_csv import from_csv, filter_by_time

from data_feed.Time import sampleFreq, sampleType
from data_feed.tiingo import *
from model.structures import *

from data_feed.db import o,h,l,c,dte,default_val
import data_feed.update_data as update_data

import sys
from datetime import datetime,date,timedelta
from timeit import default_timer as timer

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pickle, math

from sqlalchemy.dialects import mysql as mysql_types
import logging
logger = logging.getLogger('VOL')

'''
The purpose of this module is to plot realized volatility over 
using different models, sampling frequencies and days

'''


all_models = [
    Parkinson,
    CloseClose,
    GarmanKlass,
    RogersSatchell,
    # HodgesTompkins, 
    YangZhang
]

_1day  = '1day'
_1hour = '1hour'
_2hour = '2hour'
_3hour = '3hour'
_15min = '15min'
_30min = '30min'

all_freqs = [
    _1day  ,
    _1hour ,
    _2hour ,
    _3hour ,
    _15min ,
    _30min 
]

all_samps = {
    _1day  : 1,
    _1hour : 7, 
    _2hour : 3,
    _3hour : 2,
    _15min : 26,
    _30min : 13
}

all_days = [
    # 10,
    20,
    # 30,
    # 60,
    # 90,
    # 360
]

class RealizedVol:
    def __init__(self, symbol, db, start=None, end=None, min_freq=5):
        self.connected = None
        self.df        = None
        self.samplings = {}
        self.plots     = None

        self.min_freq = min_freq
        self.symbol = symbol.upper()
        self.db = db 

        now = datetime.now().date()
        if not start: self.start = now.replace(year = now.year - 1)
        else:         self.start = start

        if not end: self.end = now
        else:       self.end = end
        assert type(self.start)==date and type(self.end)==date, 'start or end wrong type'

        self.day_cnt = (self.end - self.start).days
        self.today = now.strftime('%Y-%m-%d')

        self.cols = [dte,o,h,l,c]

    def run(self):
        self.full_df = self.db.get_data_from_sym(self.symbol,start=self.start, end=self.end)
        self.df = self.full_df[self.cols]

        try:
            self.grp_days = self.df.groupby(by=[self.df.date.dt.year , self.df.date.dt.month, self.df.date.dt.day])
        except:
            print(f"RVOL ERROR {self.symbol}")
            return 

        assert not self.df.empty , 'self.df is None after db load'

        logger.info(self.symbol)
        self.set_samplings(self.df) 
        self.set_estimators()  

    def plot_estimators(self):
        raise Exception('just grab df from db and use those cols')
        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1,len(all_models))]
        fig = plt.figure(1)
        fig.gca().set_prop_cycle('color', colors)
        plt.gcf().autofmt_xdate()

        num_subplots = len(all_freqs) * len(all_days)
        num_cols     = 3
        position     = range(1,num_subplots + 1)
        rows         = num_subplots // num_cols
        rows         += (num_subplots % num_cols)

        k = 0 
        keys = list(plots.keys())
        for day in all_days:
            d = f'{day}day'
            cp = [x for x in keys if d in x] # keys that match day
            for freq in all_freqs:
                f = str(freq)
                vol_types = [x for x in cp if f in x]
                ax = fig.add_subplot(rows,num_cols,position[k])
                k += 1

                cur_min = 100 ; cur_max = 0
                for vt in vol_types:
                    vt_df  = plots[vt]
                    vt_df  = vt_df[vt_df[vt] > 0]
                    vt_dates = vt_df[dte]
                    vt_vols  = vt_df[vt]

                    # get bounds
                    mini = max(round(vt_vols.min(),1) - .1, 0)
                    maxi = round(vt_vols.max(),1) + .1
                    cur_min = mini if mini < cur_min else cur_min
                    cur_max = maxi if maxi > cur_max else cur_max
                                       
                    ax.plot(vt_dates, vt_vols, label=vt.split('_')[0])

                minor = np.arange(cur_min,cur_max,.01)
                major = np.arange(cur_min,cur_max,.05)
                ax.set_yticks(major)
                ax.set_yticks(minor,minor=True)
                ax.grid(which='major', alpha=0.05)
                ax.grid(which='minor', alpha=0.01)

                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
                ax.tick_params(axis='x', labelrotation = 90)

                ax.set_title(f'{f}_{d}')


        for i in range(6):
            ax = fig.add_subplot(rows,num_cols,position[i])
            lines, labels = ax.get_legend_handles_labels() 
            plt.legend(lines, labels, loc='upper right', prop={'size':7}) # , bbox_to_anchor=(1,1, .5,1))

        plt.subplots_adjust(top = 0.95, bottom=0.2, hspace=.5, wspace=0.4)
        plt.grid()
        plt.show()

    def set_estimators(self):
        prec = 8 ; sym = self.symbol
        for k,df in self.samplings.items():
            for day in all_days:
                for model in all_models:
                    col_name = f'{model.get_name()}_{k}_{day}day'
                    
                    samps = all_samps[k]
                    est = model.get_estimator(df , window = day * samps, trading_periods = 252 * samps)
                    df[col_name] = est

                    if not self.db.col_exists(sym,col_name):
                        self.db.add_col_to_tbl(sym, col_name)

                    dtype = {
                            dte:mysql_types.TIMESTAMP,
                            col_name:mysql_types.NUMERIC(19,prec)
                        }

                    _df = df[list(dtype.keys())]
                    _df = _df.dropna()
                    _df = _df.round({col_name:prec})

                    # check against latest value in db
                    if col_name in self.full_df.columns:
                        latest = self.full_df[[dte,col_name]]
                        latest = latest.dropna()
                        if not latest.empty:
                            ldate  = latest.date.iloc[-1]

                            _df = _df[_df.date > ldate]
                            if _df.empty:
                                continue 

                    tmp_name = 'TMP_TBL_' + self.symbol.replace('.','_')
                    _df.to_sql(tmp_name, self.db.engine, if_exists='replace', dtype=dtype, index=False)
                    self.db.exec_query(f'alter table {tmp_name} add primary key({dte})')
                    
                    q = f'''
                        update {sym} sym, {tmp_name} tmp 
                        set sym.{col_name} = tmp.{col_name} 
                        where sym.{dte} = tmp.{dte} and sym.{col_name} is {default_val} ; 
                        '''
                    self.db.exec_query(q)
                    self.db.exec_query(f'drop table {tmp_name}')



    def set_samplings(self, df):

        max_day = max(all_days)
        model = all_models[0].get_name()
        col_name = f'{model}_{_15min}_{max_day}day'
        if not col_name in self.full_df:
            _df = df
        else:
            datas = self.full_df[[dte,col_name]]
            last_hvol = datas.dropna()
            if last_hvol.empty:
                _df = df
            else:
                lastdate = last_hvol.iloc[-1][dte]
                if len(datas[datas.date > lastdate]) <= 3:
                    return 

                cutoff   = lastdate - pd.Timedelta(days= 2 * max_day)
                _df = df[df.date >= cutoff]

        if _df is not None and not _df.empty:
            half_hr, quart_hr = self._minute_filter(_df)
            hr1, hr2, hr3     = self._hour_filter(_df)
            day1              = self._day_filter(_df)

            self.samplings[_1day]  = day1
            self.samplings[_1hour] = hr1
            self.samplings[_2hour] = hr2
            self.samplings[_3hour] = hr3
            self.samplings[_15min] = quart_hr
            self.samplings[_30min] = half_hr


    def _day_filter(self, df):
        rows = []
        def func(day):
            if day is None or not update_data.is_day_over(self.today, day):
                return 

            d = {
                dte      : day.date.iloc[0] ,
                o        : day.iloc[0].open ,
                h        : max(day.high) , 
                l        : min(day.low) ,
                c        : day.iloc[-1].close
                }
            rows.append(d)

        self.grp_days.apply(func)
        return pd.DataFrame(rows, columns=self.cols) 

    def _hour_filter(self,df):
        rows1 = [] ; rows2 = [] ; rows3 = [] ;
        for grp1,day in self.grp_days:
            if not update_data.is_day_over(self.today, day):
                continue
            hours = day.groupby(by=[day.date.dt.hour])
            day_rows = []
            def func(hour):
                d = {
                    dte      : hour.iloc[0].date ,
                    o        : hour.iloc[0].open ,
                    h        : max(hour.high) , 
                    l        : min(hour.low) ,
                    c        : hour.iloc[-1].close
                    }
                day_rows.append(d)

            hours.apply(func)

            rows1 += day_rows 
            rows2 += [day_rows[:3], day_rows[3:5], day_rows[5:]]
            rows3 += [day_rows[:4], day_rows[4:]]

        df_h1 = pd.DataFrame(rows1, columns=self.cols) 

        df_h2 = []
        for hr in rows2:
            if not hr: continue
            d = {
                dte : hr[0][dte] ,
                o   : hr[0][o] ,
                h   : max(x[h] for x in hr)  , 
                l   : min(x[l] for x in hr)  ,
                c   : hr[-1][c]
                }
            df_h2.append(d)
        df_h2 = pd.DataFrame(df_h2, columns=self.cols) 

        df_h3 = []
        for hr in rows3:
            if not hr: continue
            d = {
                dte : hr[0][dte] ,
                o   : hr[0][o] ,
                h   : max(x[h] for x in hr)  , 
                l   : min(x[l] for x in hr)  ,
                c   : hr[-1][c]
                }
            df_h3.append(d)
        df_h3 = pd.DataFrame(df_h3, columns=self.cols) 

        return (df_h1, df_h2, df_h3)
            

    def _minute_filter(self,df):
        rows = []

        half  = df.groupby(pd.Grouper(freq='30T',key=dte))
        quart = df.groupby(pd.Grouper(freq='15T',key=dte))

        rows = []
        def func(grp, cnt):
            if not grp.empty and len(grp) == cnt:
                d = {
                    dte : grp.iloc[0].date ,
                    o   : grp.iloc[0].open ,
                    h   : max(grp.high) , 
                    l   : min(grp.low) ,
                    c   : grp.iloc[-1].close
                    }
                rows.append(d)
        half.apply(func, 6)
        half_hr = rows

        rows = []
        quart.apply(func, 3)
        quart_hr = rows

        half_hr  = pd.DataFrame(half_hr, columns=self.cols)
        quart_hr = pd.DataFrame(quart_hr, columns=self.cols)
        return (half_hr, quart_hr)

    
    def get_sym_rvol_with_date(self, date=None):
        if date == None:
            date = datetime.now().date() - timedelta(days=45)
        data = self.db.get_data_from_sym(self.symbol,start=date)

        cols = data.columns
        
        day1  = [x for x in cols if x.find(_1day)  != -1 or x == 'date']
        hour1 = [x for x in cols if x.find(_1hour) != -1 or x == 'date']
        hour2 = [x for x in cols if x.find(_2hour) != -1 or x == 'date']
        hour3 = [x for x in cols if x.find(_3hour) != -1 or x == 'date']
        min15 = [x for x in cols if x.find(_15min) != -1 or x == 'date']
        min30 = [x for x in cols if x.find(_30min) != -1 or x == 'date']

        out = {'Symbol':self.symbol}
        out[_1day]  = data[day1].dropna()
        out[_1hour] = data[hour1].dropna()
        out[_2hour] = data[hour2].dropna()
        out[_3hour] = data[hour3].dropna()
        out[_15min] = data[min15].dropna()
        out[_30min] = data[min30].dropna()
        return out 

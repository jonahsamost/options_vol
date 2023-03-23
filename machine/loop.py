import datetime, pytz, time
from data_feed import tdameritrade as tda
import logging
logger = logging.getLogger('VOL')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

from machine import helper_procs 
import os, sys

from timeit import default_timer as timer
from helper.run_with_timer import run_with_timer
from data_feed.update_data import DataUpdate
from vol_models.RealizedVol import RealizedVol 
from vol_forecast.vol_cones import VolCone
from vol_forecast.arch import Forecast
import pandas as pd
import multiprocessing as mp
import threading
import numpy as np
import math

from multiprocessing import Process, Queue
from multiprocessing.connection import Client

from data_feed.db import log_to_int, DAYRET,dte

from helper.print_full_pd import pfull

# only top 50 stocks for now from kibot
pacific = pytz.timezone('US/Pacific')                                                                                                                                                                     
eastern = pytz.timezone('US/Eastern')
last_run = (0,0)

manager = mp.Manager()
ret_dict = manager.dict()

def _process_sym(sym):
    cur = process_sym(sym)
    ret_dict[sym] = cur

def process_sym(sym):
    now = datetime.datetime.now().date()
    start = now.replace(year = now.year - 5)

    datas = DataUpdate(sym)
    if not datas.update_price_data():
        return 

    df = datas.db.get_data_from_sym(sym,start=start)
    datas.db.update_daily_returns(sym , df) 
    
    day = datas.db.get_data_from_sym(sym,cols=[dte,DAYRET])
    day = day.dropna()
    day.set_index(dte,inplace=True)

    # volcone
    vc = VolCone(day)
    vc.create()

    # garch
    fore = Forecast(day, 'GARCH')
    fore.model()

    # rvol
    rv = RealizedVol(sym,datas.db)
    rv.run()
    rvs = rv.get_sym_rvol_with_date()

    # opts
    td = tda.TD(sym)
    td.run()
    td.calc_atm_vols()

    start = now.replace(year = now.year - 1)
    df = datas.db.get_data_from_sym(sym,start=start)

    # vvol
    vv = df[[dte,DAYRET]].dropna()
    def func(r): return math.sqrt(sum(((r - r.mean()) ** 2) / (len(r) - 1)))
    df['vvol'] = vv.day_ret.rolling(window=10,center=False).apply(func)

    # cleanup mysql connections
    datas.db.terminate()
    del datas.db.engine
    rv.db.terminate()
    rv.db = None

    cur = {}
    cur['df'] = df
    cur['volcone'] = vc
    cur['garch'] = fore
    cur['rv'] = rvs
    cur['td'] = td
    return cur 


def find_iv_rv_diffs(dic):
    pass
    

# update td chains
from multiprocessing.pool import ThreadPool
def update_td_chains(dic):

    def func(cur):
        cur['td'].run()
        cur['td'].calc_atm_vols()

    pool = ThreadPool(processes=15)
    pool.map(func, dic.items())


sym_dict = manager.dict()
def main():

    return 

    now = datetime.datetime.now()
    pac_time = pacific.localize(now)
    est_time = pac_time.astimezone(eastern)

    # 500 most liq options
    fname = 'data_feed/liq_options.txt'
    f = open(fname,'r')
    opts = f.read().split('\n')
    opts = [opt.strip() for opt in opts]
    opts = opts[:-1] if not opts[-1] else opts
    f.close()

    # update_q_out = Queue() ; 
    # p = Process(target=helper_procs._update_data, args=(opts[:2] , update_q_out))
    # p.start()

    # forecast_q_out = Queue()
    # t = threading.Thread(target=helper_procs._do_forecast, args=(update_q_out, sym_dict, forecast_q_out))
    # t.start()

    # opt_q_in = Queue() ; opt_q_out = Queue() 
    # p = Process(target=helper_procs._find_options, args=(forecast_q_out, sym_dict, opt_q_out))
    # p.start()
    # return 

    # update db for all stocks in universe
    for sym in opts:
        ret_dict[sym] = None

    cpus = mp.cpu_count() - 1
    with mp.Pool(cpus) as p:
        p.map(_process_sym, opts)

if __name__ == '__main__':
    main()


# TODO
'''
interesting idea -- vega neutral strats to play kurtosis 
    option vol and pricing page 522

option vol and pricing page 350
    tells how to normalize vega/theta

implement this:
    https://twitter.com/pat_hennessy/status/1471540560796151808/photo/1

implement a correlation strategy

say you buy a straddle
    by how much per day does the underlying need to move 
    to overcome decay
    -- opposite for selling -- 

kelly betting guide and derivation
positional option trading chapter 9

should i be squaring log returns, then taking average to get vol with root-mean-squared not std? 
    check against vol trading sinclair
    exploiting earnings vol page 30 in pdf

advice:
    For ATM IV use the original VIX (now VXO) 
    calculation as set out in Whaley's 1993 paper. 
    For skew (mentioned in your post title) use the 
    25 delta risk reversal expressed in vol terms. 
    For curvature (smile, not mentioned in your 
    title but useful anyway) use the 25 delta fly. 
    For dispersion (implied correlation, also useful) 
    divide the weighted average ATM IV of your top 
    twenty index components by your ATM index IV and 
    you'll come close enough.


idea:
    for each opt, return list of where atm vol is trading
    relative to volcone's historical vol 

'''

from multiprocessing.connection import Listener
import multiprocessing as mp
import datetime, pytz, time
import threading

from data_feed.update_data import DataUpdate
from data_feed.db import log_to_int, DAYRET,dte
from data_feed.update_data import DataUpdate
from vol_models.RealizedVol import RealizedVol 
from vol_forecast.vol_cones import VolCone
from vol_forecast.arch import Forecast

import logging
logger = logging.getLogger('VOL')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)


UPDATE_DATA_PORT   = 6000
DO_FORECAST_PORT   = 6001
FIND_OPTIONS_PORT  = 6002

now = datetime.datetime.now().date()
start = now.replace(year = now.year - 5)


def _thread_update_data(sym):
    datas = DataUpdate(sym)
    if not datas.update_price_data():
        return 

    df = datas.db.get_data_from_sym(sym,start=start)
    datas.db.update_daily_returns(sym , df) 

def _update_data(syms, q_out):
    # pool = mp.pool.ThreadPool(processes=10)
    while 1:
        for sym in syms:
            # pool.apply_async(_thread_update_data, [sym])
            _thread_update_data(sym)
            q_out.put(sym)
        break

def _async_result_func(result):
    print(result)

def _async_forecast(sym, dic, q_out):
    datas = DataUpdate(sym)
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
 
    # vvol
    start = now.replace(year = now.year - 1)
    vv = day[day.index >= str(start)]
    def func(r): return math.sqrt(sum(((r - r.mean()) ** 2) / (len(r) - 1)))
    vvol = vv.day_ret.rolling(window=10,center=False).apply(func).dropna()
 
    df = datas.db.get_data_from_sym(sym,start=start)
    df['vvol'] = vvol
 
    datas.db.terminate()
    datas.db.engine = None
    rv.db = None
 
    cur = {}
    cur['df'] = df
    cur['volcone'] = vc
    cur['garch'] = fore
    cur['rv'] = rvs
    cur['sym'] = sym 
 
    if sym in dic:
        dic[sym] = cur # replace if already exists
    q_out.put(sym)
    return sym 

def _do_forecast(q_in, dic , q_out):
    pool = mp.Pool(4)
    while 1:
        if q_in.empty():
            continue 

        sym = q_in.get()
        print('forecast ', sym)
        pool.apply_async(_async_forecast, args=(sym, dic , q_out), callback=_async_result_func)

def _find_options(q_in, sym_dict, q_out):
    while 1:

        while q_in.empty():
            continue 

        sym = q_in.get()
    
        # opts
        td = tda.TD(sym)
        td.run()
        td.calc_atm_vols()
        sym_dict[sym]['td'] = td.chains

        q_out.put(sym)

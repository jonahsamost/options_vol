import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.dialects import mysql as mysql_types
from datetime import date, timedelta
import pandas as pd
import numpy as np
from numpy_ext import rolling_apply as rapp
import data_feed.update_data as update_data
import math

import logging
logger = logging.getLogger('VOL')

o='open'
h='high'
l='low'
c='close'
dte='date'
dec_to_int = 1000.
vol_to_int = 1000000.
log_to_int = 1000000000.
default_val = 'null'
prim='id'
TMP_TABLE='TMP'
TMP_STEP = 'step'
TMP_ROLL = 'roll'
TIMESTAMP = 'timestamp'
DAYRET='day_ret'

tbl_cols = [o,h,l,c,dte]
dt = {
    o:mysql_types.NUMERIC(19,4),
    h:mysql_types.NUMERIC(19,4),
    l:mysql_types.NUMERIC(19,4),
    c:mysql_types.NUMERIC(19,4),
    dte:mysql_types.TIMESTAMP
    }

class MyDB:

    def __init__(self):
        self.isSet = False
        self.lastError = ''
        try:
            self.u = 'options'
            self.p = 'password'
            self.host = '127.0.0.1'
            self.db_name = 'historical_prices'
            self.conn = mysql.connector.connect(user=self.u,password=self.p,host=self.host,database=self.db_name)
            self.cursor = self.conn.cursor()
            self.engine = create_engine(f'mysql+pymysql://{self.u}:{self.p}@{self.host}/{self.db_name}')                                                                                                                    
            self.isSet = True
        except Exception as e:
            self.lastError = e
            return

    def terminate(self):
        self.conn.disconnect()
        self.cursor.close()

    def exec_query(self,query,multi=False):
        try:
            if not multi:
                self.cursor.execute(query)
                r = self.cursor.fetchall()
                self.lastError = 0 
                return r
            else:
                for res in self.cursor.execute(query,multi=True):
                    r = self.cursor.fetchall()
                self.lastError = 0 
                return r
        except Exception as e:
            self.lastError = e 
            return False

    def commit(self):
        self.conn.commit()
    
    def tbl_process(self, tbl):
        return tbl.replace('.','_')

    def sym_exists(self,sym):
        sym = self.tbl_process(sym)
        query = f"show tables like '{sym}'"
        return self.exec_query(query)

    def add_sym_table(self,tbl):
        tbl = self.tbl_process(tbl)
        create_ohlc_data = f'''
            create table if not exists {tbl} (
                {o} numeric(19,4) NOT NULL,
                {h} numeric(19,4) NOT NULL,
                {l} numeric(19,4) NOT NULL,
                {c} numeric(19,4) NOT NULL, 
                {dte} TIMESTAMP NOT NULL, 
                PRIMARY KEY ({dte})
            ) ENGINE = InnoDB
        '''
        self.exec_query(create_ohlc_data)
        return self.lastError

    def add_log_sym_table(self):
        tbl_create = f'''
        create table if not exists {TMP_TABLE} (
            {TMP_STEP} INT NOT NULL , 
            {TMP_ROLL} INT NOT NULL , 
            {dte} INT NOT NULL
            PRIMARY KEY ({dte})
        ) ENGINE = InnoDB
        '''
        self.exec_query(tbl_create)
        return self.lastError

    def add_log_single_data(self, date, step, roll):
        s = int(step * log_to_int) if not math.isnan(step) else -1
        r = int(roll * log_to_int) if not math.isnan(roll) else -1 
        q = f'insert into {TMP_TABLE} (step,roll,{dte}) values ({s},{r},{date})'
        self.exec_query(q)

    def join_log_returns_with_sym_tbl(self, sym, col_to, col_frm):
        sym = self.tbl_process(sym)
        q = f'''
        update {sym.upper()} s
        inner join {TMP_TABLE} t on s.date = t.date
        set s.{col_to} = t.{col_frm}
        '''
        self.exec_query(q)

    def del_sym_table(self, sym):
        sym = self.tbl_process(sym)
        del_tbl = f'drop table if exists {sym.upper()}'
        return self.exec_query(del_tbl)

    def add_vol_table(self, name):
        create_vol_data = f'''
            create table if not exists {name} (
                {prim} INT NOT NULL AUTO_INCREMENT,
                {dte} INT NOT NULL, 
                PRIMARY KEY ({prim})
            ) ENGINE = InnoDB
        '''
        self.exec_query(create_vol_data)
        return self.lastError

    def add_vol_timestamp(self, tbl, d):
        tbl = self.tbl_process(tbl)
        q = f'insert into {tbl} ({dte}) values ({d})'
        return self.exec_query(q)

    def add_vol_by_timestamp(self, tbl, col, vol, d):
        tbl = self.tbl_process(tbl)
        v = int(vol * vol_to_int)
        q = f'update {tbl} set {col} = {v} where {dte} = {d}'
        return self.exec_query(q)

    def get_logreturn_query(self, tbl, col, log, d):
        tbl = self.tbl_process(tbl)
        l = int(log * log_to_int)
        q = f'update {tbl} set {col} = {l} where {dte} = {d} and {col} is {default_val} ;'
        # return self.exec_query(q)
        return q 

    def bulk_run_query(self, q):
        return self.exec_query(q)

    def get_vol_by_col(self, tbl, col):
        tbl = self.tbl_process(tbl)
        q = f'select {dte}, {col} from {tbl}'
        out = self.exec_query(q)

        if not out:
            return None

        arr = []
        for ts,vol in out: 
            stamp = pd.Timestamp(ts,unit='s',tz='US/Eastern').date() 
            if vol == default_val: 
                vol = 0 
            v = vol/vol_to_int 
            arr.append([stamp,v]) 

        return pd.DataFrame(arr, columns=[dte,col])


    def col_exists(self, tbl, col):
        tbl = self.tbl_process(tbl)
        q = f"select column_name from information_schema.columns where table_name='{tbl}'"
        out = self.exec_query(q)
        if not out: return False
        return col in [x[0] for x in out]

    def add_col_to_tbl(self, tbl, col):
        tbl = self.tbl_process(tbl)
        q = f'alter table `{tbl}` add {col} NUMERIC(19,8) default {default_val}'
        return self.exec_query(q)

    def add_col_to_tbl_with_type(self, tbl, col, typ):
        tbl = self.tbl_process(tbl)
        q = f'alter table {tbl} add {col} {typ} default {default_val}'
        return self.exec_query(q)

    def latest_timestamp_by_sym(self,sym):
        sym = self.tbl_process(sym)
        query = f'select {dte} from `{sym.upper()}` order by {dte} desc limit 1'
        out = self.exec_query(query)
        if out:
            return pd.Timestamp(out[0][0],unit='s',tz='US/Eastern')
        else: 
            return None

    def latest_timestr_by_sym(self, sym):
        sym = self.tbl_process(sym)
        ts = self.latest_timestamp_by_sym(sym)
        if not ts: return None
        return ts.strftime('%Y-%m-%d')

    # make sure no duplicate dates and make sure dates are ascendingly ordered
    def uniqify_timestamps(self, sym):
        sym = self.tbl_process(sym)
        q = f'''delete s1 from {sym} s1, {sym} s2 where s1.id > s2.id and s1.date = s2.date'''
        self.exec_query(q)

        q = f'''create table _{sym} like {sym} '''
        self.exec_query(q)

        q = f'''INSERT INTO _{sym} ({o},{h},{l},{c},{dte})
        SELECT {o},{h},{l},{c},{dte} from {sym} order by {dte} asc'''
        self.exec_query(q)

        q = f'''drop table {sym}'''
        self.exec_query(q)

        q = f'''rename table _{sym} to {sym}'''
        self.exec_query(q)

    def get_timestamp_from_data(self, date):
        if type(date) == np.float64 or type(date) == np.int64:
            return date
        elif type(date) == date or type(date) == pd.Timestamp:
            return int(date.timestamp())
        else:
            raise Exception(f'bad date type (type(date))')


    def add_single_data(self,tbl,date,op,hi,lo,cl):
        tbl = self.tbl_process(tbl)
        add_data_pt = f""" 
        INSERT INTO 
          `{tbl}` ({o},{h},{l},{c},{dte})
        VALUES 
          ({op},{hi},{lo},{cl},{str(date)})
        """         
        return self.exec_query(add_data_pt)

    def add_bulk_data(self,tbl,df,dt):
        tbl = self.tbl_process(tbl)
        df.to_sql(tbl, self.engine, if_exists='append',index=False, dtype=dt)

    def get_data_from_sym(self, tbl, cols=None, start=None, end=None):
        tbl = self.tbl_process(tbl)
        front = back = None
        if start:
            assert type(start)==date, 'start wrong type'
            front = f'{dte} >= "{str(start)} 00:00:00"'

        if end:
            assert type(end)==date, 'end wrong type'
            end = end + timedelta(days=1) # add day to get 'end' day's full data
            back = f'{dte} <= "{str(end)} 23:59:59"'

        col = '*' if cols is None else ','.join(cols)
        base = f'select {col} from `{tbl}`'
        if not front and not back:
            q = base
        else:
            if front and back:
                q = base + f' where {front} and {back}'
            elif front and not back:
                q = base + f' where {front}'
            elif back and not front:
                q = base + f' where {back}'

        return pd.read_sql_query(q,self.conn)

    def update_daily_returns(self, sym , df, roll_type='log'):
        '''
        gets log returns of certain sized periods and adds them to symbol's db 
        '''
        logger.info(sym)
        sym = self.tbl_process(sym)
        assert roll_type=='log' or roll_type=='perc'

        day_dt = {
            dte  :mysql_types.TIMESTAMP,
            DAYRET :mysql_types.NUMERIC(19,8)
            }
        try:
            now = str(pd.Timestamp.now()).split()[0] # today as string
            laststamp = df[[dte,DAYRET]].dropna().iloc[-1].date
            lastdate  = df.date.iloc[-1]
            if now == str(laststamp).split()[0] or now == str(lastdate).split()[0]:
                return True

            _df = df[df.date >= laststamp] # include end of day for Close to Close 
        except: # DAYRET not in columns
            _df = df


        out = _df.groupby(by=[_df.date.dt.year,_df.date.dt.month,_df.date.dt.day]).apply(lambda x: x.iloc[[-1]].close)
        for i in range(len(out.index.names) - 1):
            out = out.droplevel(0)
        # should we increase index by 1 ? 
        # out.index += 1

        log_rets = np.log(out / out.shift(1))
        if len(log_rets) == 1: return True
        log_rets = log_rets.round(8)

        drs = {dte: _df.date, DAYRET: log_rets}
        drs = pd.DataFrame(drs)
        drs = drs.dropna()

        tmp_name = 'TMP_TBL_' + sym
        drs.to_sql(tmp_name, self.engine, if_exists='replace', dtype=day_dt, index=False)
        self.exec_query(f'alter table {tmp_name} add primary key({dte})')
        
        if not self.col_exists(sym,DAYRET):
            self.add_col_to_tbl(sym, DAYRET)

        q = f'''
            update `{sym}` sym, {tmp_name} tmp 
            set sym.{DAYRET} = tmp.{DAYRET} 
            where sym.{dte} = tmp.{dte} and sym.{DAYRET} is {default_val} ; 
            '''
        self.exec_query(q)
        self.exec_query(f'drop table {tmp_name}')

        return True

''' DEAD CODE ''' 
#        pd.options.mode.chained_assignment = None 
#
#        arr = [None] * len(df.columns)
#        arr[df.columns.get_loc('close')] = df.iloc[0].open
#        df.loc[-1] = arr
#        df.index += 1
#        df.sort_index(inplace=True)
#
#        def inner_log(grp):
#            return np.log(grp.iloc[-1] / grp.iloc[0])
#
#        def get_last(df):
#            return df.iloc[-1]
#
#        stamp = 'Timestamp'; rets = 'Returns'
#        dt_tmp = {
#            stamp:mysql_types.TIMESTAMP,
#            rets :mysql_types.NUMERIC(19,4)
#            }
#
#         day_rets = grp_days.rolling(window=2, center=False).apply(inner_log)
# 
#         return 
# 
#         def inner_perc(grp):
#             cur = grp.iloc[-1]
#             prev = grp.iloc[0]
#             return 100. * (cur - prev) / prev
# 
#         def inner_log(grp):
#             return np.log(grp.iloc[-1] / grp.iloc[0])
# 
#         if roll_type=='perc':
#             f = inner_perc
#         elif roll_type=='log':
#             f = inner_log
# 
#         for k,step in freq_dic.items():
#             col_name_k  = k  
#             col_name_ol = olap_pre + k 
# 
#             idx = 0 # which index to look at 
#             if col_name_k not in df.columns:
#                 # self.add_col_to_tbl(f'{sym}', f'{col_name_k}')
#                 # self.add_col_to_tbl(f'{sym}', f'{col_name_ol}')
#                 cur_df = df 
#                 col_added = 1
#             else:
#                 bools = df[df[col_name_k].notnull() & (df[col_name_k] != -1)]
#                 idx = bools.index[-1]
#                 cur_df = df[(idx - (step + 1)) : ]
# 
#             # include +1 because were rolling close to close 
# 
#             r = cur_df.close.rolling(window=step + 1, center=False).apply(func=f) # rolling, overlapping
#             nr = r[ :: step] # non-rolling, non-overlapping
# 
#             r.index += 1 ; nr.index += 1 # increase index because we increased index above
# 
#             # cur_df has view over df, get warning for trying to set on copy of dataframe
#             cur_df[col_name_ol] = r.dropna()
#             cur_df[col_name_k]  = nr.dropna()
# 
#             if col_added:
#                 continue 
# 
#             tmp_name = 'TMP_TBL'
# 
#             # make new table cause its faster
#             def typecheck(val):
#                 nat = pd._libs.tslibs.nattype.NaTType
#                 return type(val) != nat and not math.isnan(val)
# 
#         
#             print(col_name_k)
#             print("ol_df shit")  
#             ol_df =  cur_df[[dte,col_name_ol]]
#             ol_df.to_sql(tmp_name, self.engine, if_exists='replace', dtype=dt_tmp, index=False)
#             self.exec_query(f'alter table {tmp_name} add primary key({stamp})')
#             return 
#             print("altered table") 
#             q = f'''
#                 update {sym} sym, {tmp_name} tmp 
#                 set sym.{col_name_ol} = tmp.{rets} 
#                 where sym.date = tmp.{stamp} and sym.{col_name_ol} = {default_val} ; 
#                 '''
#             self.exec_query(q)
#             print("updating sym")
#             self.exec_query(f'drop table {tmp_name}')
# 
#             nol_df = cur_df[[dte,col_name_k]]
#             nol_df.to_sql(tmp_name, self.engine, if_exists='replace', dtype=dt_tmp, index=False)
#             self.exec_query(f'alter table {tmp_name} add primary key({stamp})')
#             q = f'''
#                 update {sym} sym, {tmp_name} tmp 
#                 set sym.{col_name_ol} = tmp.{rets} 
#                 where sym.date = tmp.{stamp} and sym.{col_name_ol} = {default_val} ; 
#                 '''
#             self.exec_query(q)
# 
# 
#             return cur_df 
# 
#             ol_df  = [(int(x[TIMESTAMP]) , int(log_to_int * x[col_name_ol])) for i,x in cur_df.iterrows() if typecheck(x[col_name_ol])]
#             ol_df  = pd.DataFrame(ol_df , columns=[stamp, rets])
# 
#             nol_df = [(int(x[TIMESTAMP]) , int(log_to_int * x[col_name_k])) for i,x in cur_df.iterrows() if typecheck(x[col_name_k])]
#             nol_df = pd.DataFrame(nol_df , columns=[stamp, rets])
# 
#             self.exec_query(f'drop table {tmp_name}')
# 
#             for cname, cdf in [(col_name_k, nol_df), (col_name_ol, ol_df)]:
#                 tbl_create = f''' 
#                     create table if not exists {tmp_name} ( 
#                     {stamp} int not null, 
#                     {rets}  int not null, 
#                     primary key ({stamp})
#                     ) ENGINE = InnoDB '''
#                 self.exec_query(tbl_create)
#                 cdf.to_sql(tmp_name, self.engine, if_exists='append', dtype=dt, index=False)
#                 q = f'''
#                     update {sym.upper()} sym, {tmp_name} tmp 
#                     set sym.{cname} = tmp.{rets} 
#                     where sym.date = tmp.{stamp} and sym.{cname} = {default_val} ; 
#                     '''
#                 self.exec_query(q)
#                 self.exec_query(f'drop table {tmp_name}')
# 
#             

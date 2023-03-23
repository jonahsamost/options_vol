import requests
from dataclasses import dataclass,asdict,field
import json
import pandas as pd
import sys
from datetime import datetime,date, timedelta
from dateutil.relativedelta import relativedelta

from data_feed.Time import TimeFrame, sampleFreq, sampleType, remove_holidays
from data_feed.db import o,h,l,c,dte

from data_feed.from_csv import filter_by_time

TIINGO_API_KEY = ''

class Tiingo:
    def __init__(self):
        self.connected = False
        if self.connect():
            self.connected = True

    def get_api_key(self) :
        return TIINGO_API_KEY

    def connect(self):
        headers = {
                'Content-Type': 'application/json',
                'Authorization' : 'Token {}'.format(TIINGO_API_KEY)
                }
        r = requests.get("https://api.tiingo.com/api/test/", headers=headers)
        return json.loads(r.text)['message'] == 'You successfully sent a request'

    '''
    outputs pandas DataFrame in format:
                              date     open     high      low    close    volume
    0     2021-01-04T14:30:00.000Z  133.520  133.590  131.500  132.050  119667.0
    '''
    def get_data_with_symbol(self, symbol:str ,start ,end, sFreq:sampleFreq):
        assert type(start)==str and type(end)==str, 'start or end not str'

        if sFreq.stype.value == sampleType.DAY.value:
            base = 'https://api.tiingo.com/tiingo/daily/{}/prices?'.format(symbol)
        else:
            base = 'https://api.tiingo.com/iex/{}/prices?'.format(symbol)

        fstr = '%Y-%m-%d'
        ss = datetime.strptime(start,fstr).date()
        final = datetime.strptime(end,fstr).date()
        out_df = pd.DataFrame([])
        while ss <= final:
            ee = ss + relativedelta(months=3)
            form = f'''
                startDate={str(ss)}&
                endDate={str(ee)}&
                resampleFreq={sFreq}&
                afterHours=false&
                forceFill=false&
                format=json&
                includeIntradayVolume=true&
                columns=open,high,low,close,volume&
                token={TIINGO_API_KEY}'''.replace('\n','').replace(' ','')

            headers = { 'Content-Type': 'application/json' }
            r = requests.get(base + form , headers=headers)
            ss = ee + relativedelta(days=1)
            
            assert r.status_code == 200 , 'Status Code error: {}'.format(r.text)

            df = pd.DataFrame.from_dict(json.loads(r.text))
            if df.empty:
                continue 

            tmp = pd.to_datetime(df.date, format='%Y-%m-%dT%H:%M:%S.%fZ')
            tmp = tmp.dt.tz_localize('UTC')
            tmp = tmp.dt.tz_convert('US/Eastern')
            df.date = tmp 
            df = self.post_process_data(df,sFreq,trimFinalPeriod=True)
            out_df = out_df.append(df,ignore_index=True)

        out_df.reset_index(inplace=True)
        return out_df 

    def post_process_data(self, df, sFreq, trimFinalPeriod=False):
        df = remove_holidays(df)

        # change the final closing to reflect (actual?) closing price
        df = filter_by_time(df)
        return df

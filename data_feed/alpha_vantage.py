import requests
from dataclasses import dataclass,asdict,field
import pandas as pd
import sys, csv, json
from datetime import datetime,date, timedelta
from dateutil.relativedelta import relativedelta

from data_feed.Time import TimeFrame, sampleFreq, sampleType, remove_holidays

AV_API_KEY = ''

class AV:
    def __init__(self):
        pass


    def get_data(self):
        data = []
        with requests.Session() as s:
            for year in range(1,2+1):
                for month in range(1,12+1):
                    CSV_URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=5min&slice=year{year}month{month}&apikey={AV_API_KEY}'
                    download = s.get(CSV_URL)
                    decoded_content = download.content.decode('utf-8')
                    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                    data.append(list(cr))

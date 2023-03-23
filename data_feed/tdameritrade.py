import requests
from enum import Enum
from dataclasses import dataclass,asdict,field
import json
import math
import time
import urllib.parse as up
import pandas as pd
import multiprocessing
from datetime import datetime, timedelta

from timeit import default_timer as timer

from data_feed.Time import TimeFrame
from model.structures import *
import opt_plot.bs_opt_plot as bs

import logging
logger = logging.getLogger('VOL')

'''
directions: https://developer.tdameritrade.com/content/authentication-faq
'''

TD_AMERITRADE_API_KEY = ''
REDIRECT = ''
ACCESS_TOKEN=''
REFRESH_TOKEN = ''

class IncludeQuotes(Enum):
    TRUE  = 'TRUE'
    FALSE = 'FALSE'

class Strategy(Enum):
    SINGLE = 'SINGLE'
    ANALYTICAL = 'ANALYTICAL'
    COVERED = 'COVERED'
    VERTICAL = 'VERTICAL'
    CALENDAR = 'CALENDAR'
    STRANGLE = 'STRANGLE'
    STRADDLE = 'STRADDLE'
    BUTTERFLY = 'BUTTERFLY'
    CONDOR = 'CONDOR'
    DIAGONAL = 'DIAGONAL'
    COLLAR = 'COLLAR'
    ROLL = 'ROLL'

class Moneyness(Enum):
    ITM = 'ITM' # In-the-money
    NTM = 'NTM' # Near-the-money
    OTM = 'OTM' # Out-of-the-money
    SAK = 'SAK' # Strikes Above Market
    SBK = 'SBK' # Strikes Below Market
    SNK = 'SNK' # Strikes Near Market
    ALL = 'ALL' # All Strikes

@dataclass
class TD_CHAINS:
    apikey:             str = TD_AMERITRADE_API_KEY
    symbol:             str = ''
    contractType:       ContractType = ContractType.ALL
    strikeCount:        int = 25
    includeQuotes:      IncludeQuotes = IncludeQuotes.FALSE
    strategy:           Strategy = Strategy.SINGLE
    interval:           float = 0.0
    strike:             float = 0.0
    range:              Moneyness = Moneyness.ALL
    fromDate:           datetime.date = None
    toDate:             datetime.date = None
    volatility:         float = 0.0
    underlyingPrice:    float = 0.0
    interestRate:       float = 0.0
    daysToExpiration:   int = 0
    expMonth:           str = 'ALL'
    optionType:         str = 'ALL'

    def form(self):
        if not self.symbol: return None
        s = {}
        s['apikey'] = self.apikey
        s['symbol'] = self.symbol
        s['contractType'] = self.contractType.value
        s['strikeCount'] = self.strikeCount
        s['includeQuotes'] = self.includeQuotes.value
        s['strategy'] = self.strategy.value
        s['range'] = self.range.value
        if self.interval:   s['interval'] = self.interval
        if self.strike:     s['strike'] =   self.strike
        if self.expMonth:   s['expMonth'] = self.expMonth
        if self.optionType: s['optionType'] = self.optionType
        

        now = datetime.now().date()
        then = now + timedelta(days=3 * 32)

        s['fromDate'] = self.fromDate if self.fromDate else now
        s['toDate'] = self.toDate if self.toDate else then

        if self.strategy == Strategy.ANALYTICAL:
            s['volatility'] = self.volatility
            s['underlyingPrice'] = self.underlyingPrice
            s['interestRate'] = self.interestRate
            s['daysToExpiration'] = self.daysToExpiration

        return s


class TDAPI:

    def __init__(self):
        self.access_token = None
        self.account_id = None

    def get_options_chains(self, chain:TD_CHAINS):
        url = f'https://api.tdameritrade.com/v1/marketdata/chains' 
        headers= {'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Bearer {self.access_token}'}
        r = requests.get(url, headers=headers, params=chain.form())
        if r.status_code != 200:
            print(f'ERROR OPTIONS: {r.text}') 
            return None
        return json.loads(r.text)

    def get_account_id(self):
        if self.account_id is not None:
            return self.account_id

        url = 'https://api.tdameritrade.com/v1/accounts'
        headers= {'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Bearer {self.access_token}'}

        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f'ERROR OPTIONS: {r.text}') 
            return None
        return json.loads(r.text)

        if r.status_code != 200:
            print(f'ERROR GET_ACCOUNT: {r.text}') 
            return None

        txt = json.loads(r.text)
        self.account_id = txt[0]['securitiesAccount']['accountId']
        return self.account_id


    def refresh_access_token(self):
        if not self.access_token:
            if not self.get_access_token():
                return False
            time_check = time.time()
            return True

        now = time.time()
        if now - time_check > 60 * 29: # 30 mins
            self.get_access_token()
            time_check = time.time()
            return True

    def get_access_token(self):

        url = 'https://api.tdameritrade.com/v1/oauth2/token'
        data = {
            'grant_type':'refresh_token',
            'refresh_token':REFRESH_TOKEN,
            'client_id':TD_AMERITRADE_API_KEY , 
            'redirect_uri':up.quote_plus(REDIRECT)
        }
        r = requests.post(url,data=data)
        if r.status_code != 200:
            print('access token error: {}'.format(r.text))
            return False
        
        self.access_token = json.loads(r.text)['access_token']
        return True


class TD(TDAPI):
    def __init__(self, symbol):
        TDAPI.__init__(self)
        if not self.refresh_access_token():
            assert False, 'Access token failure'

        self.filter_by_moneyness = True
        self.symbol    = symbol
        self.chains    = []
        self.days_left = 15 
        self.atm_vol   = None

    def run(self):
        logger.info("")

        chain = TD_CHAINS(symbol=self.symbol)
        td_opts = self.get_options(chain)
        if td_opts['status'] == 'SUCCESS':
            self.chains = self.parse_options(td_opts) 

    def calc_atm_vols(self):
        '''
        calc bid and ask implied vol of each option, atm vol of each option chain, 
        and determine how much embedded premium some option has relative to its
        atm vol 
        '''
        
        ### is finding 'VXO' atm vol like this necessary?
        # first = second = None
        # for opt in self.chains:
        #     # if opt.iloc[0].dte >= 8 and opt.iloc[0].desc.lower().find('weekly') == -1:
        #     if opt.iloc[0].desc.lower().find('weekly') == -1:
        #         if first is None:
        #             first = opt
        #         elif second is None:
        #             second = opt
        #             break

        # if first is None and second is None:
        #     print(f"ERROR CALC ATM VOLS {self.symbol}")
        #     return 
        # 
        # self.atm_vol = bs.vxo_atm_vol(first,second)

        err = 0 
        for opt in self.chains:
            try:
                atm_vol = bs.get_option_atm_vol(opt)
                bs.get_premium_with_vol(opt, atm_vol)
                bs.find_bid_ask_vols(opt)
            except:
                err = 1
                return opt

        if err:
            print(f'ERROR OPTIONS CALC {self.symbol}')


    def get_options(self, chain):
        if type(chain) != TD_CHAINS:
            return None

        return self.get_options_chains(chain)

    '''
    input  : output from self.get_options 
    output : dataframe 
    '''
    def parse_options(self, opt):
        opt_dict = {}
        cur_price = float(opt['underlyingPrice'])
        puts =  opt['putExpDateMap']
        calls = opt['callExpDateMap']

        rows = []

        for t in [calls,puts]:
            for k,exp_dates in t.items():
                if type(exp_dates) != dict: continue 
                for strik,data in exp_dates.items():
                    for d in data:
                        put  = d['description'].lower().find('put') != -1
                        call = d['description'].lower().find('call') != -1 
                        
                        if self.filter_by_moneyness:
                            if put and float(d['strikePrice']) > cur_price + 10:
                                continue
                            if call and float(d['strikePrice']) < cur_price - 10:
                                continue

                        d = {
                            spot     : cur_price , 
                            strike   : float(d['strikePrice']),
                            delta    : float(d['delta']),
                            bid      : float(d['bid']),
                            ask      : float(d['ask']),
                            'td_vol' : float(d['volatility']),
                            #theo_vol : float(d['theoreticalVolatility']),
                            volume   : int(d['totalVolume']) ,
                            openInt  : int(d['openInterest']),
                            gamma    : float(d['gamma']),
                            theta    : float(d['theta']),
                            vega     : float(d['vega']),
                            #rho      : float(d['rho']),
                            dte      : int(d['daysToExpiration']),
                            desc     : d['description'],
                            # desc     : datetime.strptime(' '.join(descrip.split()[1:4]), '%b %d %Y'),
                            kind     : ContractType.PUT.value if put else ContractType.CALL.value
                        }
                        rows.append(d)

        df = pd.DataFrame(rows, columns=rows[0].keys()) 
        dfs = [x for _, x in df.groupby(dte)]
        return dfs 


def get_auth_url():
    uri = up.quote_plus(REDIRECT)        
    clid = TD_AMERITRADE_API_KEY
    url = 'https://auth.tdameritrade.com/auth?response_type=code&redirect_uri={}&client_id={}%40AMER.OAUTHAP'.format(uri,clid) 
    r = requests.get(url)
    print('MANUALLY GO TO THIS URL...get code parameter then `r = try_all(code)` ')
    return url

def try_all(code):
    '''
    https://developer.tdameritrade.com/authentication/apis/post/token-0
    '''
    code = code[code.find('code') + 5:]
    codes = [code, up.quote(code), up.unquote(code)]
    redi  = [REDIRECT, up.quote_plus(REDIRECT)]
    client_id = [TD_AMERITRADE_API_KEY, TD_AMERITRADE_API_KEY + '%40AMER.OAUTHAP']

    url = 'https://api.tdameritrade.com/v1/oauth2/token'
    for c in codes:
        for r in redi:
            for ci in client_id:

                data = {
                    'grant_type':'authorization_code',
                    'access_type':'offline',
                    'client_id':ci,
                    'redirect_uri':r,
                    'code':c
                }
                r = requests.post(url,data=data)
                if r.status_code == 200:
                    print('found')
                    return r


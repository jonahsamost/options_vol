from dataclasses import dataclass,asdict,field
from enum import Enum
from datetime import datetime

class ContractType(Enum):
    CALL = 'CALL'
    PUT  = 'PUT'
    ALL  = 'ALL'

@dataclass
class OptionInfo:
    contractType: ContractType = None
    spot        : float = 0.0
    bid         : float = 0.0
    ask         : float = 0.0
    last        : float = 0.0
    volatility  : float = 0.0
    delta       : float = 0.0
    gamma       : float = 0.0
    theta       : float = 0.0
    vega        : float = 0.0
    rho         : float = 0.0
    strike      : float = 0.0
    dte         : int   = 0
    desc        : datetime = None
    volume      : int   = 0 
    openInt     : int   = 0  


spot   =  'spot'
strike =  'strike'
delta =   'delta'
bid =     'bid'
ask =     'ask'
vol =     'vol'
volume =  'volume'
openInt = 'openInt'
gamma =   'gamma'
theta =   'theta'
vega =    'vega'
rho =     'rho'
dte =     'dte'
desc =    'desc'
kind =    'kind'
index =   'index'
theo_vol = 'theo_vol'
atm_vol  = 'atm_vol'

op = 'open'
hi = 'high'
lo = 'low'
cl = 'close'

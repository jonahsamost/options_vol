from dataclasses import dataclass,asdict,field
from enum import Enum
from datetime import datetime


@dataclass
class TimeFrame:
    year: int 
    month: int 
    day: int 

    def __str__(self):
        return '{}-{}-{}'.format('%04d'%self.year,'%02d'%self.month,'%02d'%self.day)

'''
@dataclass
class ExactTime(TimeFrame):
    hour: int 
    minute: int 
    second: int 

    def __str__(self):
        print(str(self.super))
        print('asdf')

'''

class sampleType(Enum):
    MIN   = 'min'
    HOUR  = 'hour'
    DAY   = 'daily'
    WEEK  = 'weekly'
    MONTH = 'monthly'
    YEAR  = 'annually'

@dataclass 
class sampleFreq:
    stype: sampleType = None
    freq: int = 1
    
    def __str__(self):
        if self.stype.value in [sampleType.DAY.value, sampleType.WEEK.value, sampleType.MONTH.value, sampleType.YEAR.value]:
            return self.stype.value
        else: 
            return str(self.freq) + self.stype.value

    def get_samples_per_day(self):
        # per trading day stats
        if self.stype == sampleType.MIN:
            mins = 390
            return int(mins / self.freq)
        elif self.stype == sampleType.DAY:
            return 1
        elif self.stype == sampleType.HOUR:
            if self.freq == 1:
                return 7
            elif self.freq == 2:
                return 3
            elif self.freq == 3:
                return 2
            else:
                assert False, f'Wrong hour type {self.freq}'

'''
input: pandas series
output: pandas series with holidays removed
'''
dlist = [ 
'2021-01-01',
'2021-01-18',
'2021-02-15',
'2021-04-02',
'2021-05-31',
'2021-07-05',
'2021-09-06',
'2021-11-25',
'2021-11-26',
'2021-12-24'
]
def remove_holidays(df):
    for d in dlist:
        df = df[df.date.dt.strftime('%Y-%m-%d') != d]
    return df


    

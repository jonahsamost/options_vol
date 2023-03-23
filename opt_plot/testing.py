
from math import e

from scipy.stats import norm 
cnd = norm.cdf


# Related third party imports
import numpy

def d1(S,K,t,r,sigma):  # see Hull, page 292
    
    sigma_squared = sigma*sigma
    numerator = numpy.log( S/float(K) ) + ( r + sigma_squared/2.) * t
    denominator = sigma * numpy.sqrt(t)

    return numerator/denominator

def d2(S,K,t,r,sigma):  # see Hull, page 292
    return d1(S, K, t, r, sigma) - sigma*numpy.sqrt(t)


def delta(flag, S, K, t, r, sigma):

    d_1 = d1(S, K, t, r, sigma)

    if flag == 'p':
        return cnd(d_1) - 1.0
    else:
        return cnd(d_1)

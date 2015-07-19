import numpy as np

from .constants import RES_LENGTH

TEST_FACTOR  = 0.05
MIN_SECTIONS = 10
MAX_SECTIONS = 100

def discretize_bezier(points, count=None):
    '''
    Arguments
    ----------
    points:  (o,d) list of points of the bezier. The first and last 
             points should be the start and end of the curve.  
             For a 2D cubic bezier, order o=3, dimension d=2

    Returns
    ----------
    discrete: (n,d) list of points, a polyline representation
              of the bezier curve which respects constants.RES_LENGTH
    '''
    def compute(t):
        # compute discrete points given a sampling t
        t_d      = 1.0 - t
        n        = len(points) - 1
        # binomial coefficents, i, and each point
        iterable = zip(binomial(n), np.arange(n+1), points)
        stacked  = [((t**i)*(t_d**(n-i))).reshape((-1,1))*p*c for c,i,p in iterable]
        discrete = np.sum(stacked, axis=0)
        return discrete
    
    # make sure we have a numpy array
    points  = np.array(points)

    if count is None:
        # how much distance does a small percentage of the curve take
        # this is so we can figure out how finely we have to sample t
        test   = np.sum(np.diff(compute(np.array([0.0, TEST_FACTOR])), 
                                axis = 0) ** 2) ** .5
        count  = np.ceil((test/TEST_FACTOR) / RES_LENGTH)
        count  = int(np.clip(count, MIN_SECTIONS, MAX_SECTIONS))
    result = compute(np.linspace(0.0, 1.0, count))
    return result

def binomial(n):
    '''
    Return all binomial coefficents for a given order.
    
    For n > 5, scipy.special.binom is used, below we hardcode
    to avoid the scipy.special dependancy. 
    '''
    if   n == 1: return [1,1]
    elif n == 2: return [1,2,1]
    elif n == 3: return [1,3,3,1]
    elif n == 4: return [1,4,6,4,1]
    elif n == 5: return [1,5,10,10,5,1]
    else:
        from scipy.special import binom
        return binom(n,np.arange(n+1))

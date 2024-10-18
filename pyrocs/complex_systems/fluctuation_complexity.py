import math
from collections import Counter
from functools import lru_cache

def fluctuation_complexity(A, L : int = 1) -> float:
    '''
    
    Fluctuating complexity extends the characterization of discrete entropy 
    to consider the ordering of states, by measuring the variation of probability 
    between adjacent events. Specifically, the fluctuating complexity measures 
    the probability of an event immediately following another event by 
    calculating the mean squared difference between the log(probability) of these events.
        
    The equation within the package follows the formulation from 
    :cite:p:`parrott_measuring_2010` as follows:
    
    .. math::
        C_F = - \\sum_{i,j=1}^n p_{L,ij} \\left(\\log_2\\frac{p_{L,i}}{p_{L,j}}\\right) ^2
    
    where :math:`C_F` is the fluctuating complexity, 
    :math:`p_{L,ij}` refers to the probability of observing event j 
    immediately following the word I in a series of 
    length L, and :math:`p_{L,i}` and :math:`p_{L,j}` correspond to the 
    respective frequencies of event :math:`i` and :math:`j` within the series. 
    
    Args:
        A (array): Sequence of symbols
        L (int): If > 1, groups symbols into short subsequences of length L.
    Returns:
        float
    '''
    if L > 1:
        A = [tuple(A[i: i + L]) for i in range(len(A) + 1 - L)]
        
    N = len(A)
    freqs = Counter(A)
    
    @lru_cache(maxsize=None)
    def square_log_freq_ratio(pair):
        a, b = pair
        log_freq_ratio = math.log2(freqs[a] / freqs[b])
        return log_freq_ratio * log_freq_ratio
    
    pairs = zip(A[:-1], A[1:])
    total_sqr_diff = sum(square_log_freq_ratio(p) for p in pairs)
    return total_sqr_diff / (N - 1)
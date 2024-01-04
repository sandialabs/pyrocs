import math
from collections import Counter
from functools import lru_cache

def fluctuation_complexity(A : list, L : int = 1):
    '''
    
    Fluctuating complexity extends the characterization of discrete entropy to consider the ordering of states, by measuring the variation of probability between adjacent events. Specifically, the fluctuating complexity measures the probability of an event immediately following another event by calculating the mean squared difference between the log(probability) of these events.
        
    The equation within the package follows the formulation from [Parrott, 2010](https://complexity-ok.sites.olt.ubc.ca/files/2014/09/Parrott_EcoInd_2010.pdf) as follows:
    CF = - ∑_(𝑖,𝑗=1)^𝑁▒〖"[" 𝑝_(𝐿,𝑖𝑗) "∗ " 〖"(" log⁡_2 𝑝_(𝐿,𝑖)/𝑝_(𝐿,𝑗)  ")" 〗^2 "]" 〗
    
    where CF is the fluctuating complexity, 𝑝_(𝐿,𝑖𝑗) refers to the probability of observing event j immediately following the word I in a series of length L, and 𝑝_(𝐿,𝑖) and 𝑝_(𝐿,𝑗) correspond to the respective frequencies of event i and j within the series. 
    
    Args:
        A: Sequence of symbols
        L: If > 1, groups symbols into short subsequences of length L.
    Returns:
        The Fluctuation Complexity of the sequence
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
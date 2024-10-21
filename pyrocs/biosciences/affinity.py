
from itertools import repeat
import numpy as np
from pandas import DataFrame

def affinity(data: np.ndarray, weights=None, to_bool=bool) -> float:
    """
    Returns the affinity between all pairs of columns in binary data.

    This metric evaluates the likelihood of two species to co-occur, 
    by evaluating the log odds ratio. Unlike other co-occurrence formulations, 
    the affinity model is insensitive to the relative prevalence of the two species. 
    The equation for Affinity is based on the formulation in :cite:p:`mainali_better_2022`.
    
    .. math::
    
        \\alpha = \\log((p_1/(1-p_1))/ (p_2/(1-p_2)))
    
    where :math:`\\alpha` is the affinity, :math:`p_1` and :math:`p_2` are the 
    probability of species 1 and species 2 respectively
    
    The normalization of each species probability by its complement (i.e., :math:`1-p`) 
    results in a binary implementation of affinity within this software.
    
    Args:
        data (array) 
        weights (optional array) 
        to_bool: function or type to convert array values to boolean
        
    Returns:
        float
    """
    
    num_cols = data.shape[1]

    # Deal with both DataFrames and NumPy Arrays
    if isinstance(data, DataFrame):
        rows = data.to_numpy()
    else:
        rows = data
        
    # Without weights, give all sites a weight of 1
    if weights is None:
        weights = repeat(1)
        
    # Count pairwise coincidences 
    counter = {}
    for row, weight in zip(rows, weights):
        for i in range(num_cols):
            i_val = to_bool(row[i])
            for j in range(i, num_cols):
                j_val = to_bool(row[j])
                key = (i, j, i_val, j_val)
                counter[key] = counter.get(key, 0) + weight
                
    # Transform counts into affinity matrix
    result = np.zeros((num_cols, num_cols))
                
    with np.errstate(divide='ignore'): # Ignore Divide-By-Zero Warning
        for i in range(num_cols):
            for j in range(i, num_cols):
                neither = np.float64(counter.get((i, j, False, False), 0))
                both = np.float64(counter.get((i, j, True, True), 0))
                i_without_j = np.float64(counter.get((i, j, True, False), 0))
                j_without_i = np.float64(counter.get((i, j, False, True), 0))

                i_given_j_odds_ratio = both / j_without_i
                i_given_not_j_odds_ratio = i_without_j / neither

                affinity = np.log(i_given_j_odds_ratio / i_given_not_j_odds_ratio)
                result[i][j] = affinity
                result[j][i] = affinity
            
    if isinstance(data, DataFrame):
        result = DataFrame(result, index = data.columns, columns = data.columns)
        
    return result

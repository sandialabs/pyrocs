
from collections import Counter
from collections.abc import Sequence

from scipy.stats import entropy
import numpy as np


def discrete_entropy(
        values: Sequence,
        counts: Sequence = None,
        base: int = 2) -> float:
    """
    Entropy is often used to measure the state of disorder/randomness in a system. 
    The general equation follows the form:
    
    .. math::
    
        H = - \\sum_{i=1}^N [p_i * \\log p_i]
    
    where :math:`H` = entropy, :math:`p` = discrete probability of the occurrence of an event from the :math:`i^{\mathrm{th}}` category, 
    and :math:`N` is the total number of categories. Low entropy values indicate a higher state of disorder 
    while higher entropy values indicate a well-ordered system. The maximum possible value of the
    entropy for a given system is :math:`log(N)`, and is thus varies by group size. Please see 
    :cite:p:`shannon_mathematical_1948` for more information.
    
    The function assumes users will either input an array of values or counts of values. 
    These are then normalized prior to calculating the entropy value. This metric builds 
    on the entropy function within the scipy package (including exposure of the specific base). 
    Various bases can selected based on user interests, including 2, 10, and e.

    For more details about entropy, please consult the 
    `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html>`_ as well as the references noted above. 

    Args:
        values (Sequence): Sequence of observed values from a random process
        counts (Sequence[int]): Number of times each value was observed
        base (int): Base of returned entropy (default returns number of bits)
    Returns:
        mutual information between x and y
    """
    
    if counts is None:
        counter = Counter(values)
    else:
        counter = Counter()
        for item, count in zip(values, counts):
            counter[item] += count
    array = np.array(list(counter.values()), dtype=float)
    array /= array.sum()
    return entropy(array, base=base)

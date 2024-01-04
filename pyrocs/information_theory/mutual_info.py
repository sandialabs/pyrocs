
from collections.abc import Sequence
import os
import sys
    
from pyrocs.information_theory import discrete_entropy


def mutual_info(
        x: Sequence,
        y: Sequence,
        counts: Sequence = None,
        base: int = 2) -> float:
    """
    Mutual information measures how much knowledge is gained about one random variable when another is observed. It is also a measure of mutual dependence between the random variables. 

    The equation within the package follows the formulations from Cover and Thomas [Cover & Thomas, 2005] (https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) using both individual and the joint entropies, 
        I(X;Y)=H(X)+H(Y)-H(X,Y)
    where 
    I(X;Y) is the mutual information of X and Y
    H(X) is the entropy for random variable X alone
    H(Y) is the entropy for random variable Y alone
    H(X,Y) is the joint entropy across both X and Y

    Mutual information ranges from 0 to the minimum of (H(X),H(Y)). Higher values indicate that more information is shared (i.e., mutual dependence is greater) between the two random variables, X and Y. Thus, higher values of mutual information indicate that more information can be gained about one variable when the other is observed. 

    
    Args:
        x,y (numpy.ndarray): arrays, discretized observations from random
            distributions x \in X and y \in Y
        counts (Sequence[int]): If present, the number of times each (x,y) pair was
            observed
        base (int): If present the base in which to return the entropy
        
    Returns:
      mutual information between x and y
    """
    x_entropy = discrete_entropy(x, counts, base)
    y_entropy = discrete_entropy(y, counts, base)
    joint_entropy = discrete_entropy(zip(x, y), counts, base)
    return x_entropy + y_entropy - joint_entropy

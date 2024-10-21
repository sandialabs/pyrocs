import numpy as np
from pyrocs.information_theory import discrete_entropy


def mutual_info(
        x: np.ndarray,
        y: np.ndarray,
        counts: np.ndarray = None,
        base: int = 2) -> float:
    """
    Mutual information measures how much knowledge is gained about one random variable when another is observed.
    It is also a measure of mutual dependence between the random variables.

    The equation within the package follows the formulations from 
    Cover and Thomas :cite:p:`cover_elements_2005`
    using both individual and the joint entropies,
    
    .. math::
    
        I(X;Y)=H(X)+H(Y)-H(X,Y)
    
    where :math:`I(X;Y)` is the mutual information of :math:`X` and :math:`Y`, 
    :math:`H(X)` is the entropy for random variable :math:`X` alone, 
    :math:`H(Y)` is the entropy for random variable :math:`Y` alone, 
    and :math:`H(X,Y)` is the joint entropy across both :math:`X` and :math:`Y`.

    Mutual information ranges from 0 to the minimum of :math:`(H(X),H(Y))`. 
    Higher values indicate that more information is shared 
    (i.e., mutual dependence is greater) between the two random 
    variables, :math:`X` and :math:`Y`. Thus, higher values of mutual information 
    indicate that more information can be gained about one variable 
    when the other is observed.

    Args:
        x (array): discretized observations from random
            distribution x \in X
        y (array): discretized observations from random
            distribution y \in Y
        counts (array[int]): If present, the number of times each (x,y) pair was
            observed
        base (int): If present the base in which to return the entropy

    Returns:
        float
    """
    x_entropy = discrete_entropy(x, counts, base)
    y_entropy = discrete_entropy(y, counts, base)
    joint_entropy = discrete_entropy(zip(x, y), counts, base)
    return x_entropy + y_entropy - joint_entropy

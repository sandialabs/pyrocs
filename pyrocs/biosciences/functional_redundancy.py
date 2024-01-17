import numpy as np

def functional_redundancy(p: np.array, delta: np.array) -> float:
    '''
    This metric evaluates how interchangeable groups within a population are based 
    on the specific function they perform. As a biological concept, 
    functional redundancy reflects the extent to which different species 
    within a community have the same ecological role.

    The equation within the package follows the formulation given 
    in :cite:p:`ricotta_measuring_2016`.
    
    .. math::
        
        R &= 1-(Q/D) \\\\
        &\\text{where} \\\\
        Q &= \\sum_i(p_i*(\\sum_j(p_j*δ_{ij})) \\\\
        D &= \\sum_i(p_i*(1-p_i))

    Args:
    ----------
    p : np.array
        Relative abundances p[i] (i = 1, 2,…,N) with 0 < p[i] ≤ 1 and where the constraint 0 < p[i]
        means that all calculations involve only those species that are actually present in 
        the assemblage with nonzero abundances.
    delta : np.array
        :math:`δ_{ij}` symmetric array of pairwise functional dissimilarities between species i and j 

    Returns:
    --------
    FR : float
        Functional Redundancy Score

    '''
   
    dim = len(p)
    assert delta.shape == (dim, dim)

    # Compute Rao's Quadratic Diversity Index, which is the mean dissimilarity
    # between two random items. This can be computed as the quadratic form:
    # Q = p * delta * p
    Q = np.linalg.multi_dot([p, delta, p])

    # Maximum possible value of Q given the relative abundances p
    D = p.dot(1 - p) #Ryan - we changed this to include a product of p, would be good to verify!

    FR = 1 - (Q / D)
    return FR

import math
import numpy as np


def hill_shannon(p: np.array) -> float:
    """
    The Hill-Shannon number is a specific instance (i.e. the Perplexity) of Hill Diversity, 
    which prioritizes neither common nor rare species. 
    
    The use of the geometric mean captures 
    the proportional difference from the mean of extreme values (rather than the absolute values). 
    The equation for the Hill-Simpson based on the formulation in 
    [Roswell et al., 2021](https://doi.org/10.1111/oik.07202)
    
    Hill Shannon (Perplexity): 
    
    .. math::
        H_q=e^{-\\sum(p_i*\\ln(p_i)}
	
    where :math:`q` approaches :math:`1` and the mean is the geometric mean
    
    Args:
        p: p[i] is the proportion of all individuals that belong to species i
    Returns:
        A metric for effective count of species (diversity)
    """
    entropy = -sum(x * np.log(x) for x in p if x > 0)
    return math.exp(entropy)


def hill_simpson(p: np.array) -> float:
    """
    The Hill-Simpson number is a specific instance (i.e. the Inverse Simpson Index) 
    of Hill Diversity that prioritizes the common species. 
    The use of an arithmetic mean gives more weight to more 
    frequently occurring species. The equation for the Hill-Simpson 
    based on the formulation in :cite:p:`roswell_conceptual_2021`.
    
    Hill Simpson (Inverse Simpson Index):
    
    .. math::
    
        H_2 = 1/\\sum p_i^2
    
    where :math:`q=2` and the mean is the usual arithmetic mean

    Args:
        p: p[i] is the proportion of all individuals that belong to species i
    Returns:
        A metric for effective count of species (diversity)
    """
    return 1.0 / p.dot(p)


def hill_diversity(p: np.array, q: float) -> float:
    """
    The Hill Numbers are a family of diversity metrics describing "effective number of species".
    
    For intuition, consider a distribution
    over N species, but only K of them have a "significant" share of the 
    distribution, the remaining species together having a "small" share 
    of the distribution. The simple count of number of species, N, is 
    sensitive to the presence of individuals from rare species. 
    However, a formula which somehow discards or discounts rare species, 
    returning a value close to K, is more robust and better reflects the 
    number of important species.
    
    To understand how the Hill Number achieves the above property, consider that 
    "number of species" and "mean probability" have an inverse relationship: 
    N = 1/p. Therefore, a way to compute the "effective number of species", is 
    to first compute an "effective mean probability" and then return the inverse. 
    Hill Numbers compute a mean probability using the generalized power mean, 
    weighted by the probabilities themselves. Using the probabilities as weights 
    discounts rare (i.e. low probability) species. Since the power mean is 
    parameterized, using different parameter values generates a family of Hill Numbers.

    The equations for the set of Hill metrics are based on the formulation in 
    :cite:p:`roswell_conceptual_2021`.
    
	Hill Diversity: 

    .. math::
    
        H_q = (\\sum p_i^q)^{1/(1-q)}

    where :math:`p_i` is the proportion of all individuals that belong to 
    species :math:`i`, :math:`q` is the exponent that determines the rarity scale on which the mean is taken
    
    Args:
        p: p[i] is the proportion of all individuals that belong to species i, 
        q: The exponent that determines the rarity scale on which the mean is taken.
            Species richness (q=0), Hill-Simpson diversity (q=2), Hill-Shannon diversity (q=1), 
    Returns:
        D: a metric for effective count of species (diversity) 
    """

    # Special cases
    if q == 2:
        return hill_simpson(p)
    elif q == 1:
        return hill_shannon(p)
    elif q == 0:
        return np.count_nonzero(p)

    # General case
    D = sum(x**q for x in p)
    D = D**(1/(1-q))

    return D

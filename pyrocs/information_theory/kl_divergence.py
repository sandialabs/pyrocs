import numpy as np

def kl_divergence(p: np.ndarray, q: np.ndarray, base: int = 2) -> float:
    """
    Sometimes called relative entropy, the Kullback-Leibler Divergence (KLD) 
    measures the similarity between two distributions 
    (one a sample and the other a reference). 
    In contrast to the continuous version available in 
    `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html>`_, 
    the formulation in this package uses a discrete form of the equation 
    following :cite:p:`jost_information_2021`:
    
    .. math::
    
        D(p||q) = - \\sum_{i=1}^N[p_i * \\log (p_i/q_i)]
        
    where :math:`D` is the KLD value, :math:`N` is the total number of categories, 
    and :math:`p_i` and :math:`q_i` reflect the discrete probability of the occurrence 
    of an event from the :math:`i^{\mathrm{th}}` category of the sample distribution and 
    reference distribution respectively.

    The function is able to calculate KLD for cases where not all categories from the reference distribution are present within the sample distribution. 

    Args:
        p (array): discrete probability distribution
        q (array): discrete probability distribution
        base (int): log base to compute from; base 2 (bits), base 10 (decimal/whole numbers), or base e (ecology, earth systems)

    Returns:
        float
    """

    assert p.shape == q.shape, 'p and q shapes must be identical'

    # Take ratio of p and q
    ratio = p / q
    
    # Replace 0 values in ratio, to prevent nan result
    ratio[ratio == 0] = 1
    
    logv = np.emath.logn(base, ratio)

    if len(p.shape) == 1:
        kl_div = (p * logv).sum()
    elif len(p.shape) == 2:
        kl_div = (p * logv).sum(axis=1)

    return kl_div


def novelty_transience_resonance(
    thetas_arr: np.ndarray, 
    window: int) -> tuple[np.ndarray]:
    """
    These three related metrics extend the Kullback-Leibler Divergence formulation to consider how 
    a distribution differs from past and future distributions within a sequence. Specifically, novelty 
    aims to measure how “new” information within a distribution is relative to what you knew about the 
    past sequence and transience focuses on how “new” current information based on what occurs in the 
    future sequence. In contrast, and resonance reflects the “stickiness” of “new” topics between the 
    past and the future; it is calculated by taking the difference between novelty and transience. 

    The equations for these calculations are sourced from Barron et al. 
    :cite:p:`barron_individuals_2018`. 
    
    .. math::
    
        N_w(p_i) &= (1/w)\sum(1 \\leq k \\leq w)[D(p_i || p_(i-k))]\\\\
        T_w(p_i) &= (1/w)\sum(1 \\leq k \\leq w)[D(p_i || p_(i+k))]\\\\
        R_w(p_i) &= N_w(p_i) - T_w(p_i)
        
    where :math:`N` is novelty, :math:`T` is transience, :math:`R` is resonance, 
    :math:`w` is the number of distributions to use either
    in the past or the future, :math:`p` is the proportion of entries that 
    belong to the ith category, 
    :math:`k` is the window of interest, and :math:`D` is the 
    equation for the KLD.

    Args:
        thetas_arr (array): rows are topic mixtures
        window (int): positive integer defining scale or scale size
    Returns:
        tuple(array) [novelties, transiences, resonances] 
    """

    # Find the first and last center speech offset, given window size.
    window_start = window

    # Calculate novelty, transience, resonance.
    novelties = []
    transiences = []
    resonances = []
    for j in range(window_start, thetas_arr.shape[0] - window, 1):

        center_theta = thetas_arr[j]

        # Define windows before and after center
        after_boxend = j + window + 1
        before_boxstart = j - window

        before_theta_arr = thetas_arr[before_boxstart:j]
        beforenum = before_theta_arr.shape[0]
        before_centertheta_arr = np.tile(center_theta, reps=(beforenum, 1))

        after_theta_arr = thetas_arr[j+1:after_boxend]
        afternum = after_theta_arr.shape[0]
        after_centertheta_arr = np.tile(center_theta, reps=(afternum, 1))

        # Calculate KLDs.
        before_KLDs = kl_divergence(before_theta_arr, before_centertheta_arr)
        after_KLDs = kl_divergence(after_theta_arr, after_centertheta_arr)

        # Calculate means of KLD.
        novelty = np.mean(before_KLDs)
        transience = np.mean(after_KLDs)

        # Final measures for this center speech.
        novelties.append(novelty)
        transiences.append(transience)
        resonances.append(novelty - transience)

    return novelties, transiences, resonances

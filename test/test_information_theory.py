from pyrocs.information_theory import kl_divergence, novelty_transience_resonance, discrete_entropy, mutual_info
import numpy as np
from scipy.stats import entropy
import pytest

def test_kl_divergence():
    p = np.array([1,2,3,4])
    q = np.array([4,3,2,1])
    kld = kl_divergence(p, q)
    assert(kld == 6.584962500721156)

def test_novelty_transience_resonance():
    theta = np.array([1,0,0])

    ntr = novelty_transience_resonance(theta, window=2)
    assert(ntr is not None)

def test_discrete_entropy():
    pk = np.array([1/2, 1/2])  # fair coin
    H = entropy(pk, base=2)
    assert(H == 1)

def test_mutual_info():
    # In discrete case computations are straightforward and can be done
    # by hand on given vectors.
    x = np.array([0, 1, 1, 0, 0])
    y = np.array([1, 0, 0, 0, 1])

    assert(mutual_info(x, y) == 0.419973094021975)



if __name__ == '__main__':
    test_kl_divergence()
    test_novelty_transience_resonance()
    test_discrete_entropy()
    test_mutual_info()


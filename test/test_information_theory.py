import numpy as np
import pytest
from pyrocs.information_theory import kl_divergence, novelty_transience_resonance, discrete_entropy, mutual_info, entropy


def test_kl_divergence():
    # Test with identical distributions
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert np.isclose(kl_divergence(p, q), 0)

    # Test with different distributions
    p = np.array([0.7, 0.3])
    q = np.array([0.4, 0.6])
    assert np.isclose(kl_divergence(p, q), 0.26514844544032273, atol=1e-5)

    # Test with 2D arrays
    p = np.array([[0.7, 0.3], [0.4, 0.6]])
    q = np.array([[0.4, 0.6], [0.7, 0.3]])
    assert np.allclose(kl_divergence(p, q), [0.26514845, 0.27705803], atol=1e-5)


def test_kl_divergence_base():
    # Test with different bases
    p = np.array([0.7, 0.3])
    q = np.array([0.4, 0.6])
    assert np.isclose(kl_divergence(p, q, base=10), 0.0798176353812117, atol=1e-5)
    assert np.isclose(kl_divergence(p, q, base=np.e), 0.18378689738681217, atol=1e-5)


def test_novelty_transience_resonance():
    # Test with simple example
    thetas_arr = np.array([[0.5, 0.5], [0.7, 0.3], [0.4, 0.6], [0.8, 0.2]])
    window = 1
    novelties, transiences, resonances = novelty_transience_resonance(thetas_arr, window)
    assert np.allclose(novelties, [0.12576938349798225, 0.26514844544032273])
    assert np.allclose(transiences, [0.2770580311769584, 0.4830074998557688])
    assert np.allclose(resonances, [-0.15128864767897612, -0.21785905441544606], atol=1e-5)


def test_discrete_entropy_values_only():
    values = [1, 2, 2, 3, 3, 3]
    result = discrete_entropy(values)
    assert np.isclose(result, 1.4591479170142856)


def test_discrete_entropy_values_and_counts():
    values = [1, 2, 3]
    counts = [2, 3, 4]
    result = discrete_entropy(values, counts)
    assert np.isclose(result, 1.5304930567574826)


def test_discrete_entropy_default_base():
    values = [1, 2, 2, 3, 3, 3]
    result = discrete_entropy(values)
    assert np.isclose(result, 1.4591479170142856)


def test_discrete_entropy_custom_base():
    values = [1, 2, 2, 3, 3, 3]
    result = discrete_entropy(values, base=10)
    assert np.isclose(result, 0.4392472911358187)


def test_discrete_entropy_single_element_input():
    values = [1]
    result = discrete_entropy(values)
    assert np.isclose(result, 0)


def test_discrete_entropy_all_elements_unique_input():
    values = [1, 2, 3, 4, 5]
    result = discrete_entropy(values)
    assert np.isclose(result, 2.321928094887362)


def test_discrete_entropy_all_elements_same_input():
    values = [1, 1, 1, 1, 1]
    result = discrete_entropy(values)
    assert np.isclose(result, 0)


def test_mutual_info():
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]

    assert(mutual_info(x, y) == 2.3219280948873626)

if __name__ == '__main__':
    test_kl_divergence()
    test_kl_divergence_base()
    test_novelty_transience_resonance()
    test_discrete_entropy_values_only()
    test_discrete_entropy_values_and_counts()
    test_discrete_entropy_default_base()
    test_discrete_entropy_custom_base()
    test_discrete_entropy_single_element_input()
    test_discrete_entropy_all_elements_unique_input()
    test_discrete_entropy_all_elements_same_input()
    test_mutual_info()



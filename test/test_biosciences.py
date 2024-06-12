from pyrocs.biosciences import affinity, functional_redundancy, hill_shannon, hill_simpson, hill_diversity
import numpy as np
import pytest
from pandas import DataFrame

epsilon = 1e-7

def test_hill_shannon():
    p = np.array([0.5, 0.3, 0.2])
    assert hill_shannon(p) > 1
    p = np.array([0.9, 0.1])
    assert hill_shannon(p) < 2
    p = np.array([0.99, 0.01])
    assert hill_shannon(p) < 2

def test_hill_simpson():
    p = np.array([0.5, 0.3, 0.2])
    assert hill_simpson(p) > 1
    p = np.array([0.9, 0.1])
    assert hill_simpson(p) < 2
    p = np.array([0.99, 0.01])
    assert hill_simpson(p) < 2

def test_hill_diversity():
    p = np.array([0.5, 0.3, 0.2])
    q = 1
    assert hill_diversity(p, q) > 1
    q = 2
    assert hill_diversity(p, q) < 3
    q = 0
    assert hill_diversity(p, q) == 3

def test_hill_diversity_edge_cases():
    p = np.array([1.0])
    q = 1
    assert hill_diversity(p, q) == 1
    p = np.array([0.5, 0.5])
    q = 2
    assert hill_diversity(p, q) == 2
    p = np.array([0.9, 0.1])
    q = 0
    assert hill_diversity(p, q) == 2

## TODO: add checking for negative p and q
# def test_hill_diversity_invalid_input():
#     p = np.array([-0.5, 0.3, 0.2])  # invalid input: probabilities should be non-negative
#     with pytest.raises(ValueError):
#         hill_diversity(p, 1)

#     p = np.array([0.5, 0.3, 0.2])  
#     q = -1  # invalid input: q should be non-negative
#     with pytest.raises(ValueError):
#         hill_diversity(p, q)


def test_affinity_numpy_array():
    # Test with NumPy array input
    data = np.array([[0, 1], [1, 1], [0, 0]])
    result = affinity(data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)

def test_affinity_pandas_dataframe():
    # Test with Pandas DataFrame input
    data = DataFrame({'A': [0, 1, 0], 'B': [1, 1, 0]})
    result = affinity(data)
    assert isinstance(result, DataFrame)
    assert result.shape == (2, 2)

def test_affinity_weights():
    # Test with weights input
    data = np.array([[0, 1], [1, 1], [0, 0]])
    weights = [1.0, 2.0, 3.0]
    result = affinity(data, weights)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)

def test_affinity_to_bool():
    # Test with custom to_bool function
    data = np.array([[0, 1], [1, 1], [0, 0]])
    def custom_to_bool(x):
        return x > 0.5
    result = affinity(data, to_bool=custom_to_bool)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)

def test_affinity_no_weights():
    # Test without weights input
    data = np.array([[0, 1], [1, 1], [0, 0]])
    result = affinity(data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)

def test_affinity_all_one_input():
    # Test with all-one input data
    data = np.array([['a', 'a'], ['a', 'b']])
    result = affinity(data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert np.isnan(result[0][0])

def test_affinity_invalid_input():
    # Test with invalid input data
    with pytest.raises(AttributeError):
        affinity("invalid input")


def test_functional_rednundancy():

    # Absolute abundances
    data = np.array([[30, 30, 30, 30, 15, 10, 10, 10, 30],
            [30, 0, 0, 0, 15, 10, 10, 10, 30],
            [30, 0, 0, 0, 0, 10, 10, 10, 30],
            [0, 0, 20, 30, 30, 30, 10, 10, 30],
            [0, 0, 0, 0, 0, 0, 10, 10, 30],
            [0, 0, 0, 0, 0, 0, 10, 10, 30],
            [0, 0, 10, 30, 30, 30, 30, 10, 30],
            [0, 0, 0, 0, 0, 0, 0, 10, 30],
            [0, 0, 0, 0, 0, 0, 0, 10, 30]])
    
    data = data/data.sum(axis=1)[:,None]

    # Pairwise species dissimilarities
    delta = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0]])
    

    
    assert(functional_redundancy(data[:, -1], delta))

def test_hill_shannon():
    species_freq = np.array([40, 20, 15, 8, 22])
    species_p = species_freq/species_freq.sum()

    H = hill_shannon(species_p)

    assert pytest.approx(H, epsilon) == 4.415461338250687

def test_hill_simpson():
    species_freq = np.array([40, 20, 15, 8, 22])
    species_p = species_freq/species_freq.sum()

    H = hill_simpson(species_p)

    assert pytest.approx(H, epsilon) == 3.975838442120448

def test_hill_diversity():
    species_freq = np.array([40, 20, 15, 8, 22])
    species_p = species_freq/species_freq.sum()

    H = hill_diversity(species_p, q=1)

    assert pytest.approx(H, epsilon) == 4.415461338250687


if __name__ == '__main__': 
    test_hill_simpson()
    test_functional_rednundancy()
    test_hill_shannon()
    test_hill_simpson()
    test_hill_diversity()
    test_affinity_numpy_array()
    test_affinity_invalid_input()
    test_affinity_pandas_dataframe()
    test_affinity_weights()
    test_affinity_to_bool()
    test_affinity_no_weights()
    test_affinity_all_one_input()
    # test_hill_diversity_invalid_input()
    test_hill_diversity_edge_cases()
    test_hill_diversity()
    test_hill_simpson()
    test_hill_shannon()
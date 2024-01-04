from pyrocs.biosciences import affinity, functional_redundancy, hill_shannon, hill_simpson, hill_diversity
import numpy as np
import pytest
epsilon = 1e-7


def test_affinity():
    magnirostris = np.array([0, 0, 1, 1, 1, 1, 1])
    fortis = np.array([1, 1, 1, 1, 1, 1, 1])
    fuliginosa = np.array([1, 1, 1, 1, 1, 1, 1])
    difficilis = np.array([0, 0, 1, 1, 1, 0, 0])
    scandens = np.array([1, 1, 1, 0, 1, 1, 1])
    conirostris = np.array([0, 0, 0, 0, 0, 0, 0])

    magnirostris = magnirostris/magnirostris.sum()
    fortis = fortis/fortis.sum()

    combined = (np.stack([fortis, magnirostris]))

    result = affinity(combined)
    assert(result is not None)

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
    test_affinity()
    test_hill_simpson()
    test_functional_rednundancy()
    test_hill_shannon()
    test_hill_simpson()
    test_hill_diversity()
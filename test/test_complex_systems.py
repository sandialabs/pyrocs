from pyrocs.complex_systems import cyclomatic_complexity, feedback_density, causal_complexity, grc, fluctuation_complexity
import numpy as np
import pytest
import networkx
import networkx as nx

########### MAIN FUNCTIONS ###########
def test_cyclomatic_complexity():
    # Test undirected graph
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert cyclomatic_complexity(A) == 1.0

    # Test directed graph
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert cyclomatic_complexity(A, directed=True) == 1.0

def test_feedback_density():
    # Test undirected graph with no feedback loops
    A = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    assert feedback_density(A, directed=True) == 0.0

    # Test directed graph with feedback loops
    A = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert feedback_density(A, directed=True) > 0.5

def test_causal_complexity():
    # Test undirected graph
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert causal_complexity(A) == 1.4

    # Test directed graph
    assert causal_complexity(A, directed=True) > 1.4

def test_fluctuation_complexity_default_L():
    A = [1, 2, 3, 4, 5]
    result = fluctuation_complexity(A)
    assert isinstance(result, float)

def test_fluctuation_complexity_L_gt_1():
    A = [1, 2, 3, 4, 5]
    L = 2
    result = fluctuation_complexity(A, L)
    assert isinstance(result, float)

def test_fluctuation_complexity_single_element_sequence():
    A = [1]
    with pytest.raises(ZeroDivisionError):
        result = fluctuation_complexity(A)

def test_fluctuation_complexity_L_eq_1():
    A = [1, 2, 3, 4, 5]
    L = 1
    result = fluctuation_complexity(A, L)
    assert isinstance(result, float)

def test_grc_undirected():
    # Test undirected graph with edges
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = grc(A, directed=False)
    assert result > 0

def test_grc_directed():
    # Test directed graph with edges
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    result = grc(A, directed=True)
    assert result > 0

def test_grc_no_edges():
    # Test graph with no edges
    A = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    result = grc(A, directed=False)
    assert result == 0.0

def test_grc_non_numpy_input():
    # Test non-Numpy array input
    A = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    with pytest.raises(AttributeError):
        grc(A, directed=False)

if __name__ == '__main__':
    test_cyclomatic_complexity()
    test_feedback_density()
    test_causal_complexity()
    test_fluctuation_complexity_default_L()
    test_fluctuation_complexity_L_gt_1()
    test_fluctuation_complexity_single_element_sequence()
    test_fluctuation_complexity_L_eq_1()
    test_grc_undirected()
    test_grc_directed()
    test_grc_no_edges()
    test_grc_non_numpy_input()

from pyrocs.complex_systems import cyclomatic_complexity, feedback_density, causal_complexity, grc
import numpy as np
import pytest


########## GLOBAL VARIABLES ##########
A = [[0, 0, 1], [0, 0, 1], [0, 0, 0]]

B = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]

C = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

D = [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]

E = np.array([
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 1,],
    [1, 0, 1, 0, 0, 0, 1, 1, 0, 1,],
    [0, 1, 0, 0, 1, 1, 1, 0, 1, 0,],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1,],
    [1, 0, 1, 0, 0, 1, 0, 1, 1, 0,],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0,],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 1,],
    [0, 1, 0, 0, 1, 1, 0, 0, 1, 1,],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 0,],
    [1, 1, 0, 1, 0, 0, 1, 1, 0, 0,]
])


########### MAIN FUNCTIONS ###########
def test_cyclomatic_complexity():
    '''
    Tests the cyclomatic_complexity() function from pyrocs/complex_systems/causal_complexity.py 
    '''

    g1 = np.array(A)
    assert cyclomatic_complexity(g1, directed=True) == 1.0

    g2 = np.array(B)
    assert cyclomatic_complexity(g2, directed=True) == 2.0

    g3 = np.array(C)
    assert cyclomatic_complexity(g3, directed=True) == 2.0

    g4 = np.array(D)
    assert cyclomatic_complexity(g4, directed=True) == 3.0


def test_feedback_density():
    '''
    Tests the feedback_density() function from pyrocs/complex_systems/causal_complexity.py 
    '''
    g1 = np.array(A)
    assert feedback_density(g1, directed=True) == 0.0

    g2 = np.array(B)
    assert feedback_density(g2, directed=True) == 0.0

    g3 = np.array(C)
    assert feedback_density(g3, directed=True) == 1.0

    g4 = np.array(D)
    assert feedback_density(g4, directed=True) == 0.8571428571428571


def test_causal_complexity():
    '''
    Tests the causal_complexity() function from pyrocs/complex_systems/causal_complexity.py 
    '''

    g1 = np.array(A)
    assert causal_complexity(g1, directed=True) == 1.0

    g2 = np.array(B)
    assert causal_complexity(g2, directed=True) == 2.0

    g3 = np.array(C)
    assert causal_complexity(g3, directed=True) == 4.0

    g4 = np.array(D)
    assert causal_complexity(g4, directed=True) == 5.571428571428571


def test_grc():
    '''
    Tests the grc() function from pyrocs/complex_systems/grc.py 
    '''

    E_results = grc(E, True)

    assert(E_results == 0.03703703703703704)

if __name__ == '__main__': 
    test_cyclomatic_complexity()
    test_feedback_density()
    test_causal_complexity()
    test_grc()
from pyrocs.complex_systems import cyclomatic_complexity, feedback_density, causal_complexity, grc
import numpy as np
import pytest


########## GLOBAL VARIABLES ##########
A = np.array([
    [0,0,1],
    [0,0,1],
    [0,0,0]
])

B = np.array([
    [0,1,1],
    [0,0,1],
    [0,0,0]
])

C = np.array([
    [0,1,0],
    [0,0,1],
    [1,0,0]
])

D = np.array([
    [0,1,0,0],
    [0,0,1,0],
    [1,0,0,0],
    [0,0,0,0]
])

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

    A_results = cyclomatic_complexity(A)
    B_results = cyclomatic_complexity(B)
    C_results = cyclomatic_complexity(C)
    D_results = cyclomatic_complexity(D)
    
    assert(A_results == 1)
    assert(B_results == 2)
    assert(C_results == 2)
    assert(D_results == 3)

    print(A_results)
    print(B_results)
    print(C_results)
    print(D_results)


def test_feedback_density():
    '''
    Tests the feedback_density() function from pyrocs/complex_systems/causal_complexity.py 
    '''

    A_results = feedback_density(A)
    B_results = feedback_density(B)
    C_results = feedback_density(C)
    D_results = feedback_density(D)

    assert(A_results == 0)
    assert(B_results == 0)
    assert(C_results == 1)
    assert(D_results == 0.8571428571428571)

    print(A_results)
    print(B_results)
    print(C_results)
    print(D_results)


def test_causal_complexity():
    '''
    Tests the causal_complexity() function from pyrocs/complex_systems/causal_complexity.py 
    '''

    A_results = causal_complexity(A)
    B_results = causal_complexity(B)
    C_results = causal_complexity(C)
    D_results = causal_complexity(D)

    assert(A_results == 1)
    assert(B_results == 2)
    assert(C_results == 4)
    assert(D_results == 5.571428571428571)

    print(A_results)
    print(B_results)
    print(C_results)
    print(D_results)


def test_grc():
    '''
    Tests the grc() function from pyrocs/complex_systems/grc.py 
    '''

    E_results = grc(E)

    assert(E_results == 0.03703703703703704)

    print(E_results)


if __name__ == '__main__': 
    test_cyclomatic_complexity()
    test_feedback_density()
    test_causal_complexity()
    test_grc()
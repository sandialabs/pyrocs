import networkx as nx
import numpy as np


def grc(A : np.ndarray, directed : bool, use_weights=False):
    """
    Global reaching centrality (GRC) measures the level of hierarchy within a network based on flow. 
    The equation within the package follows the formulations from 
    :cite:p:`mones_hierarchy_2012`, 
    who quantify GRC as the difference between the maximum and the average value of the
    local reach centralities of nodes within the network:
    
    .. math::
    
        GRC = \\frac{\\sum [C_R^\max - C_R(i)]}{N-1},
    
    where :math:`C_R` is the local reach centrality that reflects the 
    proportion of nodes that can be reached from a particular node 
    :math:`i` (:math:`C_R(i)`) or reflects the maximum value of local reach 
    centrality within the network (:math:`C_R^\\max`) and :math:`N` is the number 
    of nodes present within the network. :math:`GRC` values can range from 
    0 to 1, with lower values indicating lower hierarchy and vice 
    versa :cite:p:`lakkaraju_complexity_2019`.

    Args:
        A: Square matrix of adjacencies in the network
        directed (bool): If true, assume A represents a directed graph (row -> column).
            If false, assume A represents an undirected graph.
        use_weights (bool): If true, the nonzero values of A are
        used as edge weights for the computation of grc.
    Returns:
        Global reaching centrality of the graph 
    """
    if use_weights:
        # by default, from_numpy_array assigns the nonzero values of A to the edges under the key "weight"
        weight = "weight"
    else:
        weight = None

    if directed:
        G = nx.from_numpy_array(A, nx.DiGraph)
    else:
        G = nx.from_numpy_array(A)
        
    if G.number_of_edges() == 0:
        print("WARNING: Social network to compute GRC over has no edges!")
        return 0.0
    else:
        return nx.global_reaching_centrality(G, weight=weight)
